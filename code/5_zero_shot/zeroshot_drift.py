#!/usr/bin/env python3
"""
零樣本 LLM 時間漂移測試
邏輯：零樣本模型沒有用任何年份的資料訓練，不會「過時」。若它對照星數標籤的分數仍逐年下降，
代表「星數與文字的對應關係」在改變（概念漂移），而不是模型老化。

子指令：
  prepare   從 gpu_data 的 2013–2016 保留集與 2017–2023 分布對齊測試集，各抽 N 則（預設 2,000）
  run       用 Ollama 本機模型做三類零樣本分類（JSON schema 限制輸出、溫度 0），可中斷續跑
  evaluate  各年份 macro-F1、各類 F1、逐年斜率與 bootstrap 95% CI
用法（在 sa-project 資料夾）：
  python zeroshot_drift.py prepare --data_dir gpu_data
  python zeroshot_drift.py run --model mistral
  python zeroshot_drift.py evaluate
"""
import argparse, os, json, time, threading, datetime, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd

LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]          # 對應 y = 0, 1, 2
PROMPT = """You are a sentiment classifier for customer product reviews. Read the review and classify the overall sentiment the reviewer expresses about the purchase.
POSITIVE - overall favorable.
NEGATIVE - overall unfavorable.
NEUTRAL - neither clearly favorable nor clearly unfavorable, including mixed, balanced, lukewarm, or purely factual reviews.
Judge only from the text. Return only JSON: {"label": "POSITIVE" | "NEUTRAL" | "NEGATIVE"}"""
SETS = ["2013-2016_holdout"] + [f"{y}_matched" for y in range(2017, 2024)]

def cmd_prepare(a):
    parts = []
    for s in SETS:
        d = pd.read_parquet(os.path.join(a.data_dir, f"test_{s}.parquet"))
        parts.append(d.sample(min(a.n, len(d)), random_state=2026).assign(sample=s)[["sample", "rid", "y", "text"]])
    out = pd.concat(parts, ignore_index=True); out.to_csv(a.items, index=False, encoding="utf-8")
    print(f"共 {len(out):,} 則 → {a.items}"); print(out.groupby("sample").y.value_counts(normalize=True).unstack().round(3))

def cmd_run(a):
    items = pd.read_csv(a.items)
    out = os.path.join(a.runs, f"zs__{a.model.replace(':', '_').replace('/', '_')}.jsonl"); os.makedirs(a.runs, exist_ok=True)
    done = set()
    if os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            r = json.loads(l)
            if r.get("label"): done.add(r["rid"])
    todo = items[~items.rid.isin(done)]
    print(f"{out}｜已完成 {len(done):,}，待處理 {len(todo):,}", flush=True)
    schema = {"type": "object", "properties": {"label": {"type": "string", "enum": LABELS}}, "required": ["label"]}
    state, lock, cnt = {"think": True}, threading.Lock(), {"ok": 0, "fail": 0}
    def post(p):
        req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(p).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r: return json.loads(r.read())
    def work(row):
        lab, err = None, None
        for attempt in range(4):
            p = {"model": a.model, "stream": False, "format": schema, "options": {"temperature": 0, "num_predict": 30, "seed": 42},
                 "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": f'Review:\n"""{row.text}"""'}]}
            if state["think"]: p["think"] = False
            try:
                try: o = post(p)
                except urllib.error.HTTPError as e:
                    if "think" in e.read().decode(errors="ignore") and state["think"]:
                        state["think"] = False; p.pop("think"); o = post(p)
                    else: raise
                lab = str(json.loads(o["message"]["content"]).get("label", "")).upper()
                if lab in LABELS: break
                lab, err = None, "invalid label"
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:150]}"; time.sleep(3 * 2 ** attempt)
        rec = {"rid": int(row.rid), "sample": row.sample, "label": lab, "error": err, "model": a.model,
               "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
        with lock:
            open(out, "a", encoding="utf-8").write(json.dumps(rec) + "\n")
            cnt["ok" if lab else "fail"] += 1; k = cnt["ok"] + cnt["fail"]
            if k % 500 == 0 or k == len(todo): print(f"  {k:,}/{len(todo):,}（失敗 {cnt['fail']}）", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(work, r) for r in todo.itertuples()]): f.result()
    print("完成（有失敗的話，重跑同一指令即可補上）")

def mf1(y, p):
    fs = []
    for c in range(3):
        tp = ((y == c) & (p == c)).sum(-1); fp = ((y != c) & (p == c)).sum(-1); fn = ((y == c) & (p != c)).sum(-1)
        fs.append(2 * tp / np.maximum(2 * tp + fp + fn, 1))
    return np.mean(fs, 0), fs

def cmd_evaluate(a):
    items = pd.read_csv(a.items)[["sample", "rid", "y"]]; rng = np.random.default_rng(2026); rows = []
    for f in sorted(os.listdir(a.runs)):
        if not (f.startswith("zs__") and f.endswith(".jsonl")): continue
        r = pd.read_json(os.path.join(a.runs, f), lines=True); r = r[r.label.notna()].drop_duplicates("rid", keep="last")
        m = items.merge(r[["rid", "label"]], on="rid"); m["pred"] = m.label.map({l: i for i, l in enumerate(LABELS)})
        name = f[4:-6]; years = list(range(2017, 2024)); boots = []
        for s in SETS:
            x = m[m["sample"] == s]; y, p = x.y.to_numpy(), x.pred.to_numpy()
            f1, fs = mf1(y, p); idx = rng.integers(0, len(y), (a.n_boot, len(y)))
            if s != SETS[0]: boots.append(mf1(y[idx], p[idx])[0])
            rows.append({"model": name, "sample": s, "n": len(x), "macro_f1": round(float(f1), 4),
                         "f1_neg": round(float(fs[0]), 4), "f1_neu": round(float(fs[1]), 4), "f1_pos": round(float(fs[2]), 4),
                         "pred_neg": round(float((p == 0).mean()), 4), "pred_neu": round(float((p == 1).mean()), 4), "pred_pos": round(float((p == 2).mean()), 4)})
        pt = [r_["macro_f1"] for r_ in rows if r_["model"] == name and r_["sample"] != SETS[0]]
        sl = np.polyfit(years, np.array(boots), 1)[0]
        print(f"{name}: 年斜率 {np.polyfit(years, pt, 1)[0]:.4f}（95% CI {np.percentile(sl, 2.5):.4f} 至 {np.percentile(sl, 97.5):.4f}）"
              f"｜覆蓋 {len(m):,}/{len(items):,} 則")
    res = pd.DataFrame(rows); res.to_csv(os.path.join(a.runs, "zeroshot_results.csv"), index=False)
    print(res.to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("prepare"); s.add_argument("--data_dir", default="gpu_data"); s.add_argument("--n", type=int, default=2000)
    s.add_argument("--items", default="zs_items.csv")
    s = sp.add_parser("run"); s.add_argument("--model", required=True); s.add_argument("--items", default="zs_items.csv")
    s.add_argument("--runs", default="zs_runs"); s.add_argument("--workers", type=int, default=2)
    s = sp.add_parser("evaluate"); s.add_argument("--items", default="zs_items.csv"); s.add_argument("--runs", default="zs_runs")
    s.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args(); {"prepare": cmd_prepare, "run": cmd_run, "evaluate": cmd_evaluate}[a.cmd](a)
