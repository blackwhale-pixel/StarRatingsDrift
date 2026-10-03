#!/usr/bin/env python3
"""
Amazon Reviews 2023 (Electronics, full raw) — 資料稽核腳本
輸出：全量統計、重複評論分析、切分洩漏估計、5 萬筆抽樣
需求：pip install duckdb
用法：python amazon_electronics_audit.py [資料夾路徑]
"""
import sys, os, json, time, shutil
import duckdb

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/Downloads/amazon_reviews_2023/Electronics"
REV_GZ = os.path.join(DATA_DIR, "Electronics.jsonl.gz")
META_GZ = os.path.join(DATA_DIR, "meta_Electronics.jsonl.gz")
OUT = os.path.join(DATA_DIR, "audit_out")
REV_PQ = os.path.join(DATA_DIR, "reviews.parquet")
META_PQ = os.path.join(DATA_DIR, "meta_min.parquet")
MIN_LEN = 50          # 重複分析只看 >= 50 字元的評論，排除 "Great product" 這類短句
SAMPLE_N = 50000
SEED = 42

os.makedirs(OUT, exist_ok=True)
con = duckdb.connect()
con.execute("SET memory_limit='8GB'")        # 機器記憶體較小可改 4GB
con.execute(f"SET temp_directory='{os.path.join(DATA_DIR, 'duck_tmp')}'")
con.execute("SET preserve_insertion_order=false")

def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)

def save(sql, name):
    con.execute(f"COPY ({sql}) TO '{os.path.join(OUT, name)}' (HEADER, DELIMITER ',')")
    log(f"  -> {name}")

# ---------- 1. 轉成 Parquet（只做一次，之後查詢快很多） ----------
if not os.path.exists(REV_PQ):
    log("轉換 Reviews 為 Parquet（最久的一步，可能 20–60 分鐘）...")
    con.execute(f"""
    COPY (
      SELECT rating, title, text, asin, parent_asin, user_id,
             "timestamp" AS ts, helpful_vote, verified_purchase,
             COALESCE(json_array_length(images), 0) AS n_images,
             md5(trim(regexp_replace(lower(COALESCE(text,'')), '\\s+', ' ', 'g'))) AS text_hash,
             length(COALESCE(text,'')) AS text_len
      FROM read_json('{REV_GZ}', format='newline_delimited', compression='gzip',
        columns={{'rating':'DOUBLE','title':'VARCHAR','text':'VARCHAR','images':'JSON',
                  'asin':'VARCHAR','parent_asin':'VARCHAR','user_id':'VARCHAR',
                  'timestamp':'BIGINT','helpful_vote':'BIGINT','verified_purchase':'BOOLEAN'}})
    ) TO '{REV_PQ}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
if not os.path.exists(META_PQ):
    log("轉換 Metadata（只取 parent_asin 與類別）...")
    con.execute(f"""
    COPY (
      SELECT parent_asin, main_category, categories, average_rating, rating_number
      FROM read_json('{META_GZ}', format='newline_delimited', compression='gzip',
        columns={{'parent_asin':'VARCHAR','main_category':'VARCHAR','categories':'VARCHAR[]',
                  'average_rating':'DOUBLE','rating_number':'BIGINT'}})
    ) TO '{META_PQ}' (FORMAT PARQUET, COMPRESSION ZSTD)""")

con.execute(f"CREATE VIEW r AS SELECT *, year(to_timestamp(ts/1000)) AS yr FROM read_parquet('{REV_PQ}')")
con.execute(f"CREATE VIEW m AS SELECT * FROM read_parquet('{META_PQ}')")

# ---------- 2. 基本統計 ----------
log("基本統計...")
summary = {}
row = con.execute(f"""SELECT count(*), count(DISTINCT user_id), count(DISTINCT asin),
   count(DISTINCT parent_asin), avg(verified_purchase::INT), min(yr), max(yr),
   sum((text_len=0)::INT), sum((text_len<{MIN_LEN})::INT) FROM r""").fetchone()
keys = ["n_reviews","n_users","n_asin","n_parent_asin","verified_ratio","year_min","year_max",
        "n_empty_text",f"n_text_shorter_than_{MIN_LEN}"]
summary["basic"] = dict(zip(keys, [float(x) if x is not None else None for x in row]))

save("SELECT rating, count(*) AS n, round(100.0*count(*)/sum(count(*)) OVER (),3) AS pct FROM r GROUP BY rating ORDER BY rating", "rating_dist.csv")
save("SELECT yr, count(*) AS n, round(avg(rating),3) AS mean_rating, round(avg(verified_purchase::INT),3) AS verified_ratio FROM r GROUP BY yr ORDER BY yr", "by_year.csv")
save("""SELECT rating, count(*) n,
   quantile_cont(text_len,0.1) p10, quantile_cont(text_len,0.5) p50, quantile_cont(text_len,0.9) p90,
   round(avg(helpful_vote),3) mean_helpful FROM r GROUP BY rating ORDER BY rating""", "length_by_rating.csv")

# ---------- 3. Metadata 串接覆蓋率 ----------
log("Metadata 串接覆蓋率...")
cov = con.execute("""SELECT avg((m.parent_asin IS NOT NULL)::INT)
   FROM r LEFT JOIN (SELECT DISTINCT parent_asin FROM m) m USING (parent_asin)""").fetchone()[0]
summary["meta_coverage_ratio"] = float(cov)
save("SELECT main_category, count(*) n FROM m GROUP BY 1 ORDER BY n DESC LIMIT 40", "meta_main_category.csv")

# ---------- 4. 重複評論分析（>= MIN_LEN 字元） ----------
log("重複評論分析...")
con.execute(f"""CREATE TEMP TABLE g AS
   SELECT text_hash, count(*) n_rows, count(DISTINCT user_id) n_users,
          count(DISTINCT parent_asin) n_parent, count(DISTINCT asin) n_asin,
          avg(rating) mean_rating, stddev_pop(rating) sd_rating, min(text_len) len
   FROM r WHERE text_len >= {MIN_LEN} GROUP BY text_hash""")
save("""SELECT CASE
     WHEN n_rows=1 THEN '0_unique'
     WHEN n_users=1 AND n_parent=1 THEN '1_same_user_same_parent (variant/repost)'
     WHEN n_users=1 AND n_parent>1 THEN '2_same_user_cross_product (copy-paste)'
     ELSE '3_cross_user (template/spam)' END AS dup_type,
   count(*) n_groups, sum(n_rows) n_rows,
   round(100.0*sum(n_rows)/sum(sum(n_rows)) OVER (),3) pct_rows
   FROM g GROUP BY 1 ORDER BY 1""", "dup_types.csv")
save("""SELECT CASE WHEN n_rows=1 THEN '1' WHEN n_rows<=5 THEN '2-5' WHEN n_rows<=20 THEN '6-20'
     WHEN n_rows<=100 THEN '21-100' ELSE '>100' END AS group_size,
   count(*) n_groups, sum(n_rows) n_rows FROM g GROUP BY 1 ORDER BY min(n_rows)""", "dup_group_size.csv")
save("""SELECT CASE WHEN sd_rating=0 THEN 'identical rating' WHEN sd_rating<1 THEN 'sd<1' ELSE 'sd>=1' END rating_agreement,
   count(*) n_groups, sum(n_rows) n_rows FROM g WHERE n_rows>1 GROUP BY 1""", "dup_rating_consistency.csv")
# 同一使用者、同時間、同內容，掛在多個 asin 下（規格變體共用評論）
vr = con.execute(f"""SELECT count(*), sum(k) FROM (
   SELECT count(*) k FROM r WHERE text_len >= {MIN_LEN}
   GROUP BY user_id, ts, text_hash HAVING count(*)>1)""").fetchone()
summary["variant_shared_groups"] = {"n_groups": vr[0], "n_rows": float(vr[1] or 0)}
save("""SELECT g.n_rows, g.n_users, g.n_parent, g.n_asin, round(g.mean_rating,2) mean_rating,
   round(g.sd_rating,2) sd_rating, left(any_value(r.text),300) AS text_preview
   FROM g JOIN r USING (text_hash) WHERE g.n_rows>1
   GROUP BY g.text_hash, g.n_rows, g.n_users, g.n_parent, g.n_asin, g.mean_rating, g.sd_rating
   ORDER BY g.n_rows DESC LIMIT 300""", "top_dup_groups.csv")

# ---------- 5. 切分洩漏估計：隨機 80/20 vs 依 parent_asin vs 依 user ----------
log("切分洩漏估計...")
leak = {}
for name, key in [("random_row", "user_id || '|' || CAST(ts AS VARCHAR) || '|' || asin"),
                  ("group_parent_asin", "parent_asin"),
                  ("group_user", "user_id")]:
    q = f"""WITH s AS (SELECT text_hash, (hash({key}) % 10) < 8 AS is_train
                       FROM r WHERE text_len >= {MIN_LEN}),
            tr AS (SELECT DISTINCT text_hash FROM s WHERE is_train)
            SELECT count(*), sum((tr.text_hash IS NOT NULL)::INT)
            FROM s LEFT JOIN tr USING (text_hash) WHERE NOT s.is_train"""
    n_test, n_leak = con.execute(q).fetchone()
    leak[name] = {"n_test": n_test, "n_test_text_seen_in_train": n_leak,
                  "leak_pct": round(100.0 * (n_leak or 0) / n_test, 4) if n_test else None}
    log(f"  {name}: {leak[name]}")
summary["split_leakage_exact_text"] = leak

# ---------- 6. 抽樣 5 萬筆（給 Claude 做近似重複與標籤雜訊分析） ----------
log("抽樣...")
con.execute(f"""COPY (SELECT rating, title, text, asin, parent_asin, user_id, ts, yr,
   helpful_vote, verified_purchase, n_images, text_hash, text_len
   FROM r USING SAMPLE reservoir({SAMPLE_N} ROWS) REPEATABLE ({SEED}))
   TO '{os.path.join(OUT, "sample_50k.parquet")}' (FORMAT PARQUET, COMPRESSION ZSTD)""")

summary["params"] = {"min_len_for_dup": MIN_LEN, "sample_n": SAMPLE_N, "seed": SEED,
                     "split": "hash(key) % 10 < 8 => train", "duckdb": duckdb.__version__}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
shutil.make_archive(OUT, "zip", OUT)
log(f"完成。請上傳：{OUT}.zip")
