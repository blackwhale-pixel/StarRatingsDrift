#!/usr/bin/env python3
"""
TF-IDF＋LR 逐筆預測（用於與 Transformer 做配對的衰退斜率檢定）
- 測試集：直接讀取 gpu_data/test_*.parquet（與 Transformer 完全相同的題目）
- 訓練集：種子 42 使用 gpu_data/train_2013_2016.parquet（與 Transformer 同一份抽樣）；
          種子 43、44 依漂移實驗相同條件從 pool_dedup.parquet 重新抽樣
- 模型設定與 amazon_electronics_drift.py 完全相同
用法：python tfidf_preds.py [資料夾]
"""
import sys, os, glob, json, time, shutil, platform
import numpy as np, pandas as pd, duckdb, sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/Downloads/amazon_reviews_2023/Electronics"
GPU = os.path.join(DATA_DIR, "gpu_data")
POOL_PQ = os.path.join(DATA_DIR, "pool_dedup.parquet")
OUT = os.path.join(DATA_DIR, "tfidf_preds_out")
N_TRAIN = int(os.environ.get("N_TRAIN", 500_000))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42,43,44").split(",")]
IN_TRAIN = "yr BETWEEN 2013 AND 2016 AND hash(rid) % 10 < 8"
os.makedirs(os.path.join(OUT, "preds"), exist_ok=True)
T0 = time.time()
def log(m): print(f"{time.strftime('%H:%M:%S')} [{(time.time()-T0)/60:6.1f} min] {m}", flush=True)

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")
con.execute(f"SET temp_directory='{os.path.join(DATA_DIR, 'duck_tmp')}'")
def read_pq(path): return con.execute(f"SELECT * FROM read_parquet('{path}')").df()

tests = {os.path.basename(f)[5:-8]: read_pq(f) for f in sorted(glob.glob(os.path.join(GPU, "test_*.parquet")))}
if not tests: sys.exit(f"在 {GPU} 找不到 test_*.parquet")
log(f"讀入 {len(tests)} 個測試集（與 Transformer 相同）")

if not os.path.exists(POOL_PQ) and any(s != 42 for s in SEEDS):
    log(f"找不到 {POOL_PQ}：只執行種子 42（使用 gpu_data 內的訓練集）")
    SEEDS = [42]
rows = []
for seed in SEEDS:
    if seed == 42:
        tr = read_pq(os.path.join(GPU, "train_2013_2016.parquet"))
    else:
        tr = con.execute(f"""SELECT rid, text, yr, y FROM (SELECT * FROM read_parquet('{POOL_PQ}') WHERE {IN_TRAIN})
                             USING SAMPLE reservoir({N_TRAIN} ROWS) REPEATABLE ({seed})""").df()
    log(f"種子 {seed}：訓練 {len(tr):,} 筆，TF-IDF＋LR 訓練中...")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=1_000_000, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(tr.text.fillna(""))
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X, tr.y.to_numpy())
    for name, te in tests.items():
        p = clf.predict_proba(vec.transform(te.text.fillna(""))); yhat = p.argmax(1); y = te.y.to_numpy()
        pd.DataFrame({"rid": te.rid.to_numpy(), "y": y, "pred": yhat, "p_neg": p[:, 0], "p_neu": p[:, 1], "p_pos": p[:, 2]}) \
          .to_csv(os.path.join(OUT, "preds", f"seed{seed}_{name}.csv"), index=False)
        rows.append({"seed": seed, "test_set": name, "macro_f1": round(f1_score(y, yhat, average="macro"), 4)})
        log(f"  {name:20s} macro-F1 = {rows[-1]['macro_f1']}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tfidf_results.csv"), index=False)
    del X, vec, clf, tr

json.dump({"n_train": N_TRAIN, "seeds": SEEDS, "seed42_train": "gpu_data/train_2013_2016.parquet",
           "tests": "gpu_data/test_*.parquet", "python": platform.python_version(), "duckdb": duckdb.__version__,
           "sklearn": sklearn.__version__, "runtime_min": round((time.time()-T0)/60, 1)},
          open(os.path.join(OUT, "run_meta.json"), "w"), indent=2)
shutil.make_archive(OUT, "zip", OUT)
log(f"完成。請下載 {OUT}.zip")
