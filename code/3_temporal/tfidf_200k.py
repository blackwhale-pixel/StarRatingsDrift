#!/usr/bin/env python3
"""
TF-IDF＋LR，訓練資料與 Transformer 完全相同（控制訓練資料量）
抽樣方式與 train_transformer.py 一字不差：
  train_all = pd.read_parquet(gpu_data/train_2013_2016.parquet)
  tr = train_all.sample(min(200000, len(train_all)), random_state=seed)
測試集：gpu_data/test_*.parquet（與兩個模型先前的測試完全相同）
輸出 tfidf200k_out：逐筆預測（preds/）、結果表、每個種子的訓練樣本 rid 清單（供重現）
用法（在 sa-project 資料夾）：
  C:\\Users\\black\\sa-env\\Scripts\\python.exe tfidf_200k.py --data_dir gpu_data
"""
import argparse, os, glob, json, time, shutil, platform
import numpy as np, pandas as pd, sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="gpu_data"); ap.add_argument("--train_file", default="train_2013_2016.parquet"); ap.add_argument("--out_dir", default="tfidf200k_out")
ap.add_argument("--n_train", type=int, default=200_000); ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
a = ap.parse_args()
os.makedirs(os.path.join(a.out_dir, "preds"), exist_ok=True); os.makedirs(os.path.join(a.out_dir, "train_rids"), exist_ok=True)
T0 = time.time()
def log(m): print(f"{time.strftime('%H:%M:%S')} [{(time.time()-T0)/60:5.1f} min] {m}", flush=True)

train_all = pd.read_parquet(os.path.join(a.data_dir, a.train_file))
tests = {os.path.basename(f)[5:-8]: pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(a.data_dir, "test_*.parquet")))}
log(f"訓練池 {len(train_all):,} 筆；測試集 {len(tests)} 個")
rows = []
for seed in a.seeds:
    tr = train_all.sample(min(a.n_train, len(train_all)), random_state=seed)   # 與 Transformer 相同
    tr[["rid"]].to_csv(os.path.join(a.out_dir, "train_rids", f"seed{seed}.csv"), index=False)
    log(f"種子 {seed}：訓練 {len(tr):,} 筆（前 3 個 rid：{tr.rid.head(3).tolist()}）")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=1_000_000, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(tr.text.fillna(""))
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X, tr.y.to_numpy())
    for name, te in tests.items():
        p = clf.predict_proba(vec.transform(te.text.fillna(""))); yhat = p.argmax(1); y = te.y.to_numpy()
        pd.DataFrame({"rid": te.rid.to_numpy(), "y": y, "pred": yhat, "p_neg": p[:, 0], "p_neu": p[:, 1], "p_pos": p[:, 2]}) \
          .to_csv(os.path.join(a.out_dir, "preds", f"seed{seed}_{name}.csv"), index=False)
        rows.append({"seed": seed, "test_set": name, "n_train": len(tr), "macro_f1": round(f1_score(y, yhat, average="macro"), 4)})
        log(f"  {name:20s} macro-F1 = {rows[-1]['macro_f1']}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "tfidf200k_results.csv"), index=False)
json.dump({"n_train": a.n_train, "train_file": a.train_file, "seeds": a.seeds, "sampling": "pandas DataFrame.sample(random_state=seed), same as train_transformer.py",
           "python": platform.python_version(), "pandas": pd.__version__, "sklearn": sklearn.__version__,
           "runtime_min": round((time.time()-T0)/60, 1)}, open(os.path.join(a.out_dir, "run_meta.json"), "w"), indent=2)
shutil.make_archive(a.out_dir, "zip", a.out_dir)
log(f"完成。請上傳 {a.out_dir}.zip")
