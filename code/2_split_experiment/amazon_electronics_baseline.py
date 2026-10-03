#!/usr/bin/env python3
"""
Amazon Reviews 2023 (Electronics) — 基準實驗
階段 A（預設）：去除整筆重複紀錄後，四種切分各抽 50 萬訓練 / 10 萬測試，
               跑 TF-IDF + 邏輯迴歸、只用長度的模型、多數類基準，報告 macro-F1 與 bootstrap 95% CI。
階段 B（加 --full）：在「完整訓練池」上以 HashingVectorizer + SGD 串流訓練一個 epoch，
               用來檢驗抽樣是否稀釋了使用者／商品層級的效應。
標籤：1–2 星 = neg，3 星 = neu，4–5 星 = pos；rating = 0 剔除；只用 2013–2023 年資料。
用法：
  python amazon_electronics_baseline.py [資料夾]            # 只跑階段 A
  python amazon_electronics_baseline.py [資料夾] --full     # 階段 A + B
需求：pip install duckdb scikit-learn pandas numpy
"""
import sys, os, json, time, platform
import numpy as np, pandas as pd, duckdb, sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

args = [a for a in sys.argv[1:] if not a.startswith("--")]
DATA_DIR = args[0] if args else "/home/ubuntu/Downloads/amazon_reviews_2023/Electronics"
FULL = "--full" in sys.argv
REV_PQ = os.path.join(DATA_DIR, "reviews.parquet")
POOL_PQ = os.path.join(DATA_DIR, "pool_dedup.parquet")
OUT = os.path.join(DATA_DIR, "baseline_out")
N_TRAIN = int(os.environ.get('N_TRAIN', 500_000)); N_TEST = int(os.environ.get('N_TEST', 100_000))
SEED, N_BOOT = 42, 1000
LABELS = ["neg", "neu", "pos"]
os.makedirs(OUT, exist_ok=True)

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")
con.execute(f"SET temp_directory='{os.path.join(DATA_DIR, 'duck_tmp')}'")
con.execute("SET preserve_insertion_order=false")
T0 = time.time()
def log(m): print(f"{time.strftime('%H:%M:%S')} [{(time.time()-T0)/60:6.1f} min] {m}", flush=True)

# ---------- 1. 建立去重後的資料池 ----------
if not os.path.exists(POOL_PQ):
    log("建立去重資料池（去除整筆重複紀錄、剔除 rating=0 與空評論、限 2013–2023）...")
    con.execute(f"""COPY (
      SELECT row_number() OVER () AS rid, text, text_len, text_hash, user_id, parent_asin, yr,
             CASE WHEN rating <= 2 THEN 0 WHEN rating = 3 THEN 1 ELSE 2 END AS y
      FROM (SELECT *, year(to_timestamp(ts/1000)) AS yr FROM read_parquet('{REV_PQ}')
            WHERE rating BETWEEN 1 AND 5 AND text_len > 0)
      WHERE yr BETWEEN 2013 AND 2023
      QUALIFY row_number() OVER (PARTITION BY user_id, asin, ts, text_hash ORDER BY helpful_vote DESC) = 1
    ) TO '{POOL_PQ}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
con.execute(f"CREATE VIEW p AS SELECT * FROM read_parquet('{POOL_PQ}')")
pool_n = con.execute("SELECT count(*) FROM p").fetchone()[0]
log(f"資料池筆數：{pool_n:,}")

SPLITS = {  # (訓練條件, 測試條件)
    "random_row":        ("hash(rid) % 10 < 8",         "hash(rid) % 10 >= 8"),
    "group_parent_asin": ("hash(parent_asin) % 10 < 8", "hash(parent_asin) % 10 >= 8"),
    "group_user":        ("hash(user_id) % 10 < 8",     "hash(user_id) % 10 >= 8"),
    "temporal":          ("yr BETWEEN 2013 AND 2020",   "yr BETWEEN 2022 AND 2023"),
}

def sample(cond, n, seed):
    return con.execute(f"""SELECT rid, text, text_len, text_hash, user_id, parent_asin, y FROM
        (SELECT * FROM p WHERE {cond}) USING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed})""").df()

def macro_f1_ci(y, yhat, rng):
    n = len(y); k = len(LABELS)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    yt, yp = y[idx], yhat[idx]
    f1s = []
    for c in range(k):
        tp = ((yt == c) & (yp == c)).sum(1); fp = ((yt != c) & (yp == c)).sum(1); fn = ((yt == c) & (yp != c)).sum(1)
        f1s.append(np.where(2*tp+fp+fn > 0, 2*tp / np.maximum(2*tp+fp+fn, 1), 0.0))
    b = np.mean(f1s, axis=0)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

rows, rng = [], np.random.default_rng(SEED)
def record(tier, split, model, n_train, y, yhat, extra=None):
    lo, hi = macro_f1_ci(y, yhat, rng)
    pc = f1_score(y, yhat, labels=[0, 1, 2], average=None, zero_division=0)
    r = {"tier": tier, "split": split, "model": model, "n_train": n_train, "n_test": len(y),
         "macro_f1": round(f1_score(y, yhat, average="macro", zero_division=0), 4),
         "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
         "accuracy": round(accuracy_score(y, yhat), 4),
         "f1_neg": round(pc[0], 4), "f1_neu": round(pc[1], 4), "f1_pos": round(pc[2], 4)}
    if extra: r.update(extra)
    rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "results.csv"), index=False)
    cm = confusion_matrix(y, yhat, labels=[0, 1, 2])
    pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS]) \
      .to_csv(os.path.join(OUT, f"cm_{tier}_{split}_{model}.csv"))
    log(f"  {tier} {split:18s} {model:12s} macro-F1={r['macro_f1']} [{r['ci95_low']}, {r['ci95_high']}]")

def len_feats(s):
    x = np.log1p(s.to_numpy(dtype=float)); return np.c_[x, x**2]

# ---------- 2. 階段 A ----------
tests, dist = {}, []
for split, (tr_c, te_c) in SPLITS.items():
    log(f"階段 A：{split} 抽樣...")
    tr = sample(tr_c, N_TRAIN, SEED); te = sample(te_c, N_TEST, SEED + 1); tests[split] = (te_c, te)
    for name, d in [("train", tr), ("test", te)]:
        vc = d.y.value_counts(normalize=True)
        dist.append({"split": split, "set": name, "n": len(d), **{l: round(vc.get(i, 0), 4) for i, l in enumerate(LABELS)}})
    pd.DataFrame(dist).to_csv(os.path.join(OUT, "label_dist.csv"), index=False)
    # 抽樣後測試集中「文字在訓練樣本出現過」的比例（抽樣後的實際洩漏）
    leak = float(te.text_hash.isin(set(tr.text_hash)).mean())
    user_seen = float(te.user_id.isin(set(tr.user_id)).mean())
    extra = {"sampled_text_leak": round(leak, 5), "test_user_seen_in_train": round(user_seen, 5)}
    y_tr, y_te = tr.y.to_numpy(), te.y.to_numpy()

    record("A", split, "majority", len(tr), y_te, np.full_like(y_te, np.bincount(y_tr).argmax()), extra)
    lm = LogisticRegression(class_weight="balanced", max_iter=1000).fit(len_feats(tr.text_len), y_tr)
    record("A", split, "length_only", len(tr), y_te, lm.predict(len_feats(te.text_len)), extra)
    log("  TF-IDF 向量化與訓練...")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=1_000_000, sublinear_tf=True, dtype=np.float32)
    Xtr = vec.fit_transform(tr.text.fillna("")); Xte = vec.transform(te.text.fillna(""))
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xtr, y_tr)
    record("A", split, "tfidf_lr", len(tr), y_te, clf.predict(Xte), extra)
    del Xtr, Xte, vec, clf, tr

# ---------- 3. 階段 B（完整訓練池串流訓練） ----------
if FULL:
    hv = HashingVectorizer(n_features=2**22, ngram_range=(1, 2), alternate_sign=False, norm="l2")
    for split, (tr_c, te_c) in SPLITS.items():
        te = tests[split][1]; te_ids = set(te.rid)
        cnt = con.execute(f"SELECT y, count(*) FROM p WHERE {tr_c} GROUP BY y ORDER BY y").fetchall()
        tot = sum(c for _, c in cnt); w = {y: tot / (3 * c) for y, c in cnt}   # 等同 class_weight='balanced'
        log(f"階段 B：{split} 完整訓練池 {tot:,} 筆，開始串流訓練...")
        clf = SGDClassifier(loss="log_loss", alpha=1e-6, random_state=SEED)
        con.execute(f"SELECT rid, text, y FROM p WHERE {tr_c} ORDER BY hash(rid + {SEED})")
        seen, first = 0, True
        while True:
            ch = con.fetch_df_chunk(50)
            if ch is None or len(ch) == 0: break
            X = hv.transform(ch.text.fillna("")); yy = ch.y.to_numpy()
            sw = np.vectorize(w.get)(yy)
            if first: clf.partial_fit(X, yy, classes=np.array([0, 1, 2]), sample_weight=sw); first = False
            else: clf.partial_fit(X, yy, sample_weight=sw)
            seen += len(ch)
            if seen % 2_000_000 < len(ch): log(f"    已訓練 {seen:,} 筆")
        yhat = clf.predict(hv.transform(te.text.fillna("")))
        record("B", split, "hash_sgd_full", seen, te.y.to_numpy(), yhat)

meta = {"n_pool_after_dedup": pool_n, "n_train_A": N_TRAIN, "n_test": N_TEST, "seed": SEED, "n_boot": N_BOOT,
        "labels": "1-2=neg, 3=neu, 4-5=pos", "years": "2013-2023", "full_stage_B": FULL,
        "splits": SPLITS, "python": platform.python_version(), "duckdb": duckdb.__version__,
        "sklearn": sklearn.__version__, "runtime_min": round((time.time()-T0)/60, 1)}
json.dump(meta, open(os.path.join(OUT, "run_meta.json"), "w"), indent=2, ensure_ascii=False)
import shutil; shutil.make_archive(OUT, "zip", OUT)
log(f"完成。請上傳：{OUT}.zip")
