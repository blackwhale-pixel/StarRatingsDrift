#!/usr/bin/env python3
"""
第二輪稽核（沿用第一輪產生的 reviews.parquet，不需重新轉檔）
1. 整筆重複紀錄（同使用者、同 asin、同時間、同文字）有多少
2. 真正「依列隨機切分」（等同 sklearn train_test_split）的洩漏率
3. 寬鬆比對的近似重複洩漏：去除標點、數字、HTML 後比對，以及只比對前 120 字元
用法：python amazon_electronics_audit2.py [資料夾路徑]
"""
import sys, os, json, time, shutil
import duckdb

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/Downloads/amazon_reviews_2023/Electronics"
REV_PQ = os.path.join(DATA_DIR, "reviews.parquet")
OUT = os.path.join(DATA_DIR, "audit_out2")
MIN_LEN = 50
os.makedirs(OUT, exist_ok=True)

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")
con.execute(f"SET temp_directory='{os.path.join(DATA_DIR, 'duck_tmp')}'")
con.execute("SET preserve_insertion_order=false")
def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)

res = {}
log("1. 整筆重複紀錄...")
row = con.execute(f"""SELECT count(*), sum(k), sum(k-1) FROM (
   SELECT count(*) k FROM read_parquet('{REV_PQ}')
   GROUP BY user_id, asin, ts, text_hash HAVING count(*)>1)""").fetchone()
res["exact_record_duplicates_all_lengths"] = {"n_groups": row[0], "n_rows_in_groups": float(row[1] or 0),
                                             "n_redundant_rows": float(row[2] or 0)}
log(f"  {res['exact_record_duplicates_all_lengths']}")

log("2. 建立雜湊表（>= 50 字元）...")
con.execute(f"""CREATE TABLE h AS
 WITH b AS (SELECT row_number() OVER () AS rid, user_id, parent_asin, text_hash,
     trim(regexp_replace(regexp_replace(lower(COALESCE(text,'')), '<[^>]*>', ' ', 'g'),
          '[^a-z]+', ' ', 'g')) AS loose
   FROM read_parquet('{REV_PQ}') WHERE text_len >= {MIN_LEN})
 SELECT rid, user_id, parent_asin,
        hash(text_hash) AS h_exact, hash(loose) AS h_loose, hash(left(loose, 120)) AS h_prefix,
        length(loose) AS loose_len
 FROM b""")
n = con.execute("SELECT count(*), count(DISTINCT h_exact), count(DISTINCT h_loose), count(DISTINCT h_prefix) FROM h").fetchone()
res["distinct_counts"] = {"n_rows": n[0], "exact": n[1], "loose": n[2], "prefix120": n[3]}
log(f"  {res['distinct_counts']}")

log("3. 各切分方式 × 各比對方式的洩漏率...")
splits = {"random_row_true": "hash(rid)", "group_parent_asin": "hash(parent_asin)", "group_user": "hash(user_id)"}
rows = []
for sname, skey in splits.items():
    con.execute(f"CREATE OR REPLACE TEMP TABLE s AS SELECT *, ({skey} % 10) < 8 AS is_train FROM h")
    for mname in ["h_exact", "h_loose", "h_prefix"]:
        n_test, n_leak = con.execute(f"""WITH tr AS (SELECT DISTINCT {mname} k FROM s WHERE is_train)
            SELECT count(*), sum((tr.k IS NOT NULL)::INT)
            FROM s LEFT JOIN tr ON s.{mname}=tr.k WHERE NOT s.is_train""").fetchone()
        pct = round(100.0 * n_leak / n_test, 4)
        rows.append({"split": sname, "match": mname, "n_test": n_test, "n_leak": n_leak, "leak_pct": pct})
        log(f"  {sname:18s} {mname:9s} {pct}%")
res["leakage"] = rows

log("4. 寬鬆比對下才會合併的重複群組範例...")
LOOSE = "trim(regexp_replace(regexp_replace(lower(COALESCE(text,'')), '<[^>]*>', ' ', 'g'), '[^a-z]+', ' ', 'g'))"
con.execute("""CREATE TEMP TABLE top AS
  SELECT h_loose, count(*) n_rows, count(DISTINCT h_exact) n_exact_variants,
         count(DISTINCT user_id) n_users, count(DISTINCT parent_asin) n_parent
  FROM h GROUP BY h_loose HAVING count(DISTINCT h_exact) > 1
  ORDER BY n_rows DESC LIMIT 200""")
con.execute(f"""COPY (
  SELECT t.n_rows, t.n_exact_variants, t.n_users, t.n_parent,
         left(any_value(p.text), 250) AS example_text
  FROM read_parquet('{REV_PQ}') p JOIN top t ON hash({LOOSE}) = t.h_loose
  WHERE p.text_len >= {MIN_LEN}
  GROUP BY ALL ORDER BY t.n_rows DESC) TO '{os.path.join(OUT, "loose_only_groups.csv")}' (HEADER)""")
g = con.execute("""SELECT count(*), sum(n) FROM (SELECT count(*) n FROM h GROUP BY h_loose
                   HAVING count(DISTINCT h_exact) > 1)""").fetchone()
res["loose_groups_with_multiple_exact_variants"] = {"n_groups": g[0], "n_rows": float(g[1] or 0)}

res["params"] = {"min_len": MIN_LEN, "split_rule": "hash(key) % 10 < 8 => train",
                 "loose": "lowercase, strip HTML tags, keep letters a-z only",
                 "prefix": "first 120 chars of loose text", "duckdb": duckdb.__version__}
with open(os.path.join(OUT, "summary2.json"), "w") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
shutil.make_archive(OUT, "zip", OUT)
log(f"完成。請上傳：{OUT}.zip")
