#!/usr/bin/env python3
"""
Transformer 時間漂移實驗（本機 GPU 版，Windows / RTX 5060 Laptop 8 GB）
使用 gpu_data 資料夾（由 amazon_electronics_drift.py 匯出）：
  train_2013_2016.parquet、test_*.parquet（與 TF-IDF 實驗完全相同的測試集）
設定與 TF-IDF 基準一致：標籤 0=neg 1=neu 2=pos，類別加權（等同 class_weight='balanced'），
macro-F1 附 bootstrap 95% CI。

快速測試（約 3–5 分鐘，先確認能跑）：
  python train_transformer.py --data_dir gpu_data --smoke
正式實驗（單一種子）：
  python train_transformer.py --data_dir gpu_data --seeds 42
"""
import argparse, os, sys, json, time, glob, math, platform, random
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, \
    get_linear_schedule_with_warmup

def get_args():
    a = argparse.ArgumentParser()
    a.add_argument("--data_dir", default="gpu_data")
    a.add_argument("--train_file", default="train_2013_2016.parquet", help="訓練檔名（位於 data_dir）")
    a.add_argument("--out_dir", default="transformer_out")
    a.add_argument("--model", default="distilbert/distilroberta-base")
    a.add_argument("--n_train", type=int, default=200_000)
    a.add_argument("--max_len", type=int, default=256)
    a.add_argument("--batch", type=int, default=32)
    a.add_argument("--grad_accum", type=int, default=1)
    a.add_argument("--epochs", type=int, default=1)
    a.add_argument("--lr", type=float, default=2e-5)
    a.add_argument("--seeds", type=int, nargs="+", default=[42])
    a.add_argument("--n_boot", type=int, default=1000)
    a.add_argument("--smoke", action="store_true", help="小量快速測試：1,000 筆訓練、每個測試集 300 筆")
    return a.parse_args()

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def log(m, t0):
    print(f"{time.strftime('%H:%M:%S')} [{(time.time()-t0)/60:6.1f} min] {m}", flush=True)

class Enc(torch.utils.data.Dataset):
    def __init__(self, enc, y): self.enc, self.y = enc, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        d = {k: v[i] for k, v in self.enc.items()}; d["labels"] = int(self.y[i]); return d

def encode(tok, texts, max_len):
    return tok(list(texts), truncation=True, max_length=max_len)

def f1_ci(y, yhat, n_boot, seed):
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    yt, yp = y[idx], yhat[idx]; fs = []
    for c in range(3):
        tp = ((yt == c) & (yp == c)).sum(1); fp = ((yt != c) & (yp == c)).sum(1); fn = ((yt == c) & (yp != c)).sum(1)
        fs.append(np.where(2*tp+fp+fn > 0, 2*tp / np.maximum(2*tp+fp+fn, 1), 0.0))
    b = np.mean(fs, 0); return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

@torch.no_grad()
def predict(model, loader, dev):
    model.eval(); probs = []
    for b in loader:
        b = {k: v.to(dev) for k, v in b.items() if k != "labels"}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**b).logits
        probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(probs)

def main():
    args = get_args(); t0 = time.time()
    if not torch.cuda.is_available(): sys.exit("找不到 CUDA GPU，請確認使用的是 sa-env 裡的 python.exe")
    dev = torch.device("cuda"); gpu = torch.cuda.get_device_name(0)
    if args.smoke:
        args.n_train, args.n_boot, args.out_dir = 1000, 100, args.out_dir + "_smoke"
    os.makedirs(args.out_dir, exist_ok=True); os.makedirs(os.path.join(args.out_dir, "preds"), exist_ok=True)
    log(f"GPU：{gpu}｜模型：{args.model}｜torch {torch.__version__}｜transformers {transformers.__version__}", t0)

    train_all = pd.read_parquet(os.path.join(args.data_dir, args.train_file))
    test_files = sorted(glob.glob(os.path.join(args.data_dir, "test_*.parquet")))
    if not test_files: sys.exit(f"在 {args.data_dir} 找不到 test_*.parquet")
    tests = {os.path.basename(f)[5:-8]: pd.read_parquet(f) for f in test_files}
    if args.smoke: tests = {k: v.sample(min(300, len(v)), random_state=0) for k, v in tests.items()}
    log(f"訓練池 {len(train_all):,} 筆；測試集 {len(tests)} 個", t0)

    tok = AutoTokenizer.from_pretrained(args.model)
    collate = DataCollatorWithPadding(tok)
    log("測試集斷詞...", t0)
    test_loaders = {k: DataLoader(Enc(encode(tok, v.text.fillna(""), args.max_len), v.y.to_numpy()),
                                  batch_size=args.batch * 2, collate_fn=collate) for k, v in tests.items()}
    res_path = os.path.join(args.out_dir, "transformer_results.csv")
    rows = pd.read_csv(res_path).to_dict("records") if os.path.exists(res_path) else []

    for seed in args.seeds:
        set_seed(seed)
        tr = train_all.sample(min(args.n_train, len(train_all)), random_state=seed)
        y_tr = tr.y.to_numpy()
        log(f"種子 {seed}：訓練 {len(tr):,} 筆，斷詞中...", t0)
        loader = DataLoader(Enc(encode(tok, tr.text.fillna(""), args.max_len), y_tr),
                            batch_size=args.batch, shuffle=True, collate_fn=collate,
                            generator=torch.Generator().manual_seed(seed))
        cnt = np.bincount(y_tr, minlength=3); w = torch.tensor(len(y_tr) / (3 * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
        loss_fn = torch.nn.CrossEntropyLoss(weight=w)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
        sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        model.train(); step = 0; t_train = time.time()
        for ep in range(args.epochs):
            pbar = tqdm(loader, desc=f"seed {seed} epoch {ep+1}/{args.epochs}", mininterval=10)
            for i, b in enumerate(pbar):
                b = {k: v.to(dev) for k, v in b.items()}; labels = b.pop("labels")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = loss_fn(model(**b).logits.float(), labels) / args.grad_accum
                loss.backward()
                if (i + 1) % args.grad_accum == 0 or i + 1 == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1
                    if step % 50 == 0: pbar.set_postfix(loss=f"{loss.item()*args.grad_accum:.3f}")
                if i == 200 and ep == 0:
                    est = (time.time() - t_train) / 201 * len(loader) * args.epochs / 60
                    log(f"  預估訓練時間：約 {est:.0f} 分鐘｜顯示記憶體峰值 {torch.cuda.max_memory_allocated()/1e9:.1f} GB", t0)
        train_min = (time.time() - t_train) / 60
        log(f"  訓練完成（{train_min:.1f} 分鐘），開始評估...", t0)
        for name, ld in test_loaders.items():
            te = tests[name]; y = te.y.to_numpy(); p = predict(model, ld, dev); yhat = p.argmax(1)
            lo, hi = f1_ci(y, yhat, args.n_boot, seed)
            pc = f1_score(y, yhat, labels=[0, 1, 2], average=None, zero_division=0)
            rows.append({"seed": seed, "test_set": name, "model": args.model.split("/")[-1], "n_train": len(tr),
                         "n_test": len(y), "macro_f1": round(f1_score(y, yhat, average="macro", zero_division=0), 4),
                         "ci95_low": round(lo, 4), "ci95_high": round(hi, 4), "accuracy": round(accuracy_score(y, yhat), 4),
                         "f1_neg": round(pc[0], 4), "f1_neu": round(pc[1], 4), "f1_pos": round(pc[2], 4),
                         "train_minutes": round(train_min, 1)})
            pd.DataFrame({"rid": te.rid.to_numpy(), "y": y, "pred": yhat, "p_neg": p[:, 0], "p_neu": p[:, 1],
                          "p_pos": p[:, 2]}).to_csv(os.path.join(args.out_dir, "preds", f"seed{seed}_{name}.csv"), index=False)
            log(f"  {name:20s} macro-F1 = {rows[-1]['macro_f1']} [{rows[-1]['ci95_low']}, {rows[-1]['ci95_high']}]", t0)
            pd.DataFrame(rows).to_csv(res_path, index=False)
        del model, opt; torch.cuda.empty_cache()

    json.dump({"model": args.model, "train_file": args.train_file, "n_train": args.n_train, "max_len": args.max_len, "batch": args.batch,
               "grad_accum": args.grad_accum, "epochs": args.epochs, "lr": args.lr, "seeds": args.seeds,
               "precision": "bf16 autocast", "class_weighting": "balanced", "smoke": args.smoke, "gpu": gpu,
               "torch": torch.__version__, "transformers": transformers.__version__,
               "python": platform.python_version(), "total_minutes": round((time.time()-t0)/60, 1)},
              open(os.path.join(args.out_dir, "run_meta.json"), "w"), indent=2)
    log(f"完成。結果在 {args.out_dir}", t0)

if __name__ == "__main__":
    main()
