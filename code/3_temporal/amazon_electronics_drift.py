#!/usr/bin/env python3
"""
Amazon Reviews 2023 (Electronics) — 逐年時間漂移實驗
沿用 baseline 產生的 pool_dedup.parquet（已去除整筆重複紀錄、限 2013–2023、標籤 0=neg 1=neu 2=pos）

內容：
1. 訓練：2013–2016 年（rid 雜湊前 80%），抽 50 萬筆，三個隨機種子
2. 測試：同期保留集（2013–2016 後 20%）＋ 2017–2023 每年 2 萬筆
   每年兩個版本：natural（自然分布）與 matched（依訓練集標籤比例分層抽樣，控制分布改變）
3. 模型：TF-IDF＋LR、只用長度、多數類；macro-F1 附 bootstrap 95% CI
4. 用語漂移指標：各年測試文字中，不在訓練詞彙表內的單字比例（OOV）
5. 匯出：GPU 實驗用的訓練集與測試集（gpu_data/）；三星評論盲標註樣本（annotation/）
用法：python amazon_electronics_drift.py [資料夾]
"""
import sys, os, json, time, platform, shutil
import numpy as np, pandas as pd, duckdb, sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/Downloads/amazon_reviews_2023/Electronics"
POOL_PQ = os.path.join(DATA_DIR, "pool_dedup.parquet")
OUT = os.path.join(DATA_DIR, "drift_out")
GPU = os.path.join(DATA_DIR, "gpu_data")
ANN = os.path.join(DATA_DIR, "annotation")
N_TRAIN = int(os.environ.get("N_TRAIN", 500_000))
N_YEAR = int(os.environ.get("N_YEAR", 20_000))
N_ANN3, N_ANN_CTRL = int(os.environ.get("N_ANN3", 2000)), int(os.environ.get("N_ANN_CTRL", 500))
SEEDS, TEST_SEED, N_BOOT = [42, 43, 44], 7, 1000
TRAIN_YEARS, TEST_YEARS = (2013, 2016), list(range(2017, 2024))
LABELS = ["neg", "neu", "pos"]
for d in (OUT, GPU, ANN): os.makedirs(d, exist_ok=True)

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")
con.execute(f"SET temp_directory='{os.path.join(DATA_DIR, 'duck_tmp')}'")
con.execute("SET preserve_insertion_order=false")
con.execute(f"CREATE VIEW p AS SELECT * FROM read_parquet('{POOL_PQ}')")
T0 = time.time()
def log(m): print(f"{time.strftime('%H:%M:%S')} [{(time.time()-T0)/60:6.1f} min] {m}", flush=True)
IN_TRAIN = f"yr BETWEEN {TRAIN_YEARS[0]} AND {TRAIN_YEARS[1]} AND hash(rid) % 10 < 8"
IN_HOLD = f"yr BETWEEN {TRAIN_YEARS[0]} AND {TRAIN_YEARS[1]} AND hash(rid) % 10 >= 8"
COLS = "rid, text, text_len, yr, y"

def save_pq(df, path):   # 用 DuckDB 寫 parquet，不需要 pyarrow
    con.register("_tmp_df", df); con.execute(f"COPY _tmp_df TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.unregister("_tmp_df")

def sample(cond, n, seed):
    return con.execute(f"SELECT {COLS} FROM (SELECT * FROM p WHERE {cond}) "
                       f"USING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed})").df()

# ---------- 0. 各年標籤分布（全量） ----------
log("各年標籤分布...")
con.execute(f"""COPY (SELECT yr, count(*) n, round(avg((y=0)::INT),4) neg, round(avg((y=1)::INT),4) neu,
   round(avg((y=2)::INT),4) pos, quantile_cont(text_len,0.5) median_len FROM p GROUP BY yr ORDER BY yr)
   TO '{os.path.join(OUT, "year_label_dist.csv")}' (HEADER)""")

# ---------- 1. 測試集（固定，所有種子共用） ----------
log("建立測試集...")
tests = {"2013-2016_holdout": sample(IN_HOLD, N_YEAR, TEST_SEED)}
tr_dist = con.execute(f"SELECT y, count(*) FROM p WHERE {IN_TRAIN} GROUP BY y ORDER BY y").fetchall()
tot = sum(c for _, c in tr_dist); prior = {y: c / tot for y, c in tr_dist}
for yr in TEST_YEARS:
    tests[f"{yr}_natural"] = sample(f"yr = {yr}", N_YEAR, TEST_SEED)
    parts = [sample(f"yr = {yr} AND y = {c}", int(round(N_YEAR * prior[c])), TEST_SEED + c) for c in range(3)]
    tests[f"{yr}_matched"] = pd.concat(parts, ignore_index=True)
for name in [k for k, d in tests.items() if len(d) == 0]:
    log(f"  警告：{name} 沒有資料，略過"); tests.pop(name)
for name, d in tests.items():
    save_pq(d[["rid", "text", "yr", "y"]], os.path.join(GPU, f"test_{name}.parquet"))

def f1_ci(y, yhat, rng):
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y))); yt, yp = y[idx], yhat[idx]; fs = []
    for c in range(3):
        tp = ((yt == c) & (yp == c)).sum(1); fp = ((yt != c) & (yp == c)).sum(1); fn = ((yt == c) & (yp != c)).sum(1)
        fs.append(np.where(2*tp+fp+fn > 0, 2*tp / np.maximum(2*tp+fp+fn, 1), 0.0))
    b = np.mean(fs, 0); return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

def len_x(s): x = np.log1p(s.to_numpy(float)); return np.c_[x, x**2]

rows, oov_rows = [], []
for seed in SEEDS:
    log(f"種子 {seed}：抽訓練集 {N_TRAIN:,} 筆...")
    tr = sample(IN_TRAIN, N_TRAIN, seed); ytr = tr.y.to_numpy()
    if seed == SEEDS[0]:
        save_pq(tr[["rid", "text", "yr", "y"]], os.path.join(GPU, "train_2013_2016.parquet"))
    maj = np.bincount(ytr).argmax()
    lm = LogisticRegression(class_weight="balanced", max_iter=1000).fit(len_x(tr.text_len), ytr)
    log("  TF-IDF＋LR 訓練...")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=1_000_000, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(tr.text.fillna(""))
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X, ytr)
    if seed == SEEDS[0]:
        uni = {k for k in vec.vocabulary_ if " " not in k}; an = vec.build_analyzer()
    rng = np.random.default_rng(seed)
    for name, te in tests.items():
        y = te.y.to_numpy()
        preds = {"majority": np.full_like(y, maj), "length_only": lm.predict(len_x(te.text_len)),
                 "tfidf_lr": clf.predict(vec.transform(te.text.fillna("")))}
        for model, yhat in preds.items():
            lo, hi = f1_ci(y, yhat, rng) if model == "tfidf_lr" else (np.nan, np.nan)
            pc = f1_score(y, yhat, labels=[0, 1, 2], average=None, zero_division=0)
            rows.append({"seed": seed, "test_set": name, "model": model, "n_test": len(y),
                         "macro_f1": round(f1_score(y, yhat, average="macro", zero_division=0), 4),
                         "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                         "accuracy": round(accuracy_score(y, yhat), 4),
                         "f1_neg": round(pc[0], 4), "f1_neu": round(pc[1], 4), "f1_pos": round(pc[2], 4)})
        if seed == SEEDS[0]:
            toks = [t for doc in te.text.fillna("").head(5000) for t in an(doc) if " " not in t]
            oov_rows.append({"test_set": name, "n_tokens": len(toks),
                             "oov_rate": round(sum(t not in uni for t in toks) / max(len(toks), 1), 5)})
        log(f"  {name:20s} tfidf macro-F1 = {rows[-1]['macro_f1']}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "drift_results.csv"), index=False)
    pd.DataFrame(oov_rows).to_csv(os.path.join(OUT, "oov_by_year.csv"), index=False)
    del X, vec, clf, tr

r = pd.DataFrame(rows)
r.groupby(["test_set", "model"]).macro_f1.agg(["mean", "std"]).round(4).reset_index() \
 .to_csv(os.path.join(OUT, "drift_summary_across_seeds.csv"), index=False)

# ---------- 2. 三星評論盲標註樣本 ----------
log("匯出三星盲標註樣本...")
ann = []
per = {3: N_ANN3, 2: N_ANN_CTRL, 4: N_ANN_CTRL}   # 3 星為主，2 星與 4 星各一組當對照
con.execute(f"""CREATE TEMP TABLE a AS
   SELECT user_id, asin, ts, text, rating::INT AS rating_star, year(to_timestamp(ts/1000)) AS yr
   FROM read_parquet('{os.path.join(DATA_DIR, "reviews.parquet")}')
   WHERE rating IN (2, 3, 4) AND text_len BETWEEN 20 AND 2000
     AND year(to_timestamp(ts/1000)) BETWEEN 2013 AND 2023""")
years = list(range(2013, 2024))
for star, n in per.items():
    k = int(np.ceil(n / len(years)))
    for yr in years:
        ann.append(con.execute(f"""SELECT * FROM (SELECT * FROM a WHERE rating_star = {star} AND yr = {yr})
             USING SAMPLE reservoir({k} ROWS) REPEATABLE (99)""").df())
ann = pd.concat(ann, ignore_index=True).drop_duplicates(["user_id", "asin", "ts"]).sample(frac=1, random_state=99).reset_index(drop=True)
ann["item_id"] = [f"R{i:05d}" for i in range(len(ann))]
ann[["item_id", "text"]].to_csv(os.path.join(ANN, "annotation_items_blind.csv"), index=False, encoding="utf-8-sig")
ann[["item_id", "user_id", "asin", "ts", "yr", "rating_star"]].to_csv(os.path.join(ANN, "annotation_key.csv"), index=False)

json.dump({"n_train": N_TRAIN, "n_year": N_YEAR, "seeds": SEEDS, "test_seed": TEST_SEED,
           "train_years": TRAIN_YEARS, "test_years": TEST_YEARS, "train_prior": prior,
           "annotation": {"three_star": N_ANN3, "control_each": N_ANN_CTRL, "n_final": len(ann)},
           "python": platform.python_version(), "duckdb": duckdb.__version__, "sklearn": sklearn.__version__,
           "runtime_min": round((time.time()-T0)/60, 1)}, open(os.path.join(OUT, "run_meta.json"), "w"), indent=2)
shutil.make_archive(OUT, "zip", OUT); shutil.make_archive(GPU, "zip", GPU); shutil.make_archive(ANN, "zip", ANN)
log("完成。請下載 drift_out.zip、gpu_data.zip、annotation.zip")
