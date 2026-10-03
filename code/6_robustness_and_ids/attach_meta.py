#!/usr/bin/env python3
"""
穩健性檢查：商品組成是否隨年份改變（用 Metadata 控制商品類別）
輸入（同一資料夾）：reviews.parquet、meta_min.parquet、pool_dedup.parquet、gpu_data/、annotation_key.csv
輸出 meta_out.zip：
  annotation_key_meta.csv   標註樣本逐筆附上商品類別、驗證購買、有用票數
  test_items_meta.csv       15 個測試集逐筆附上商品類別（rid 對應）
  category_by_year.csv      全資料池各年份的第一層子類別占比（前 15 名＋其他）
  star3_by_category_year.csv  各子類別、各年份的三星比例與平均評分
用法：python attach_meta.py [資料夾]
"""
import sys, os, glob, time, shutil
import duckdb

D = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/Downloads/amazon_reviews_2023/Electronics"
OUT = os.path.join(D, "meta_out"); os.makedirs(OUT, exist_ok=True)
for f in ["reviews.parquet", "meta_min.parquet", "pool_dedup.parquet", "annotation_key.csv"]:
    if not os.path.exists(os.path.join(D, f)): sys.exit(f"找不到 {f}")
con = duckdb.connect()
con.execute("SET memory_limit='6GB'")
con.execute(f"SET temp_directory='{os.path.join(D, 'duck_tmp')}'")
T0 = time.time()
def log(m): print(f"{time.strftime('%H:%M:%S')} [{(time.time()-T0)/60:5.1f} min] {m}", flush=True)
def q(p): return os.path.join(D, p)

# 子類別：categories 路徑的第 2 層（第 1 層通常是 "Electronics"）；缺值標為 (none)
con.execute(f"""CREATE TEMP TABLE m AS SELECT parent_asin, main_category,
   COALESCE(NULLIF(categories[2], ''), '(none)') AS cat_l1,
   COALESCE(NULLIF(categories[3], ''), '(none)') AS cat_l2
   FROM read_parquet('{q("meta_min.parquet")}')""")

log("1. 標註樣本附上商品資訊...")
con.execute(f"""COPY (
  SELECT k.*, r.parent_asin, r.verified_purchase, r.helpful_vote, m.main_category, m.cat_l1, m.cat_l2
  FROM read_csv_auto('{q("annotation_key.csv")}', header=true, all_varchar=true) k
  LEFT JOIN (SELECT DISTINCT user_id, asin, CAST(ts AS VARCHAR) AS ts, parent_asin, verified_purchase, helpful_vote
             FROM read_parquet('{q("reviews.parquet")}')
             WHERE asin IN (SELECT asin FROM read_csv_auto('{q("annotation_key.csv")}', header=true, all_varchar=true))) r
    ON k.user_id = r.user_id AND k.asin = r.asin AND k.ts = r.ts
  LEFT JOIN m USING (parent_asin)
) TO '{os.path.join(OUT, "annotation_key_meta.csv")}' (HEADER)""")
n, miss = con.execute(f"""SELECT count(*), sum((cat_l1 IS NULL)::INT)
   FROM read_csv_auto('{os.path.join(OUT, "annotation_key_meta.csv")}')""").fetchone()
log(f"  {n} 筆，未對到類別 {miss} 筆")

log("2. 測試集附上商品類別...")
files = sorted(glob.glob(q("gpu_data/test_*.parquet")))
union = " UNION ALL ".join(f"SELECT '{os.path.basename(f)[5:-8]}' AS test_set, rid FROM read_parquet('{f}')" for f in files)
con.execute(f"""COPY (
  SELECT t.test_set, t.rid, p.parent_asin, m.cat_l1, m.cat_l2
  FROM ({union}) t JOIN read_parquet('{q("pool_dedup.parquet")}') p USING (rid) LEFT JOIN m USING (parent_asin)
) TO '{os.path.join(OUT, "test_items_meta.csv")}' (HEADER)""")

log("3. 全資料池的類別組成與三星比例（依年份）...")
con.execute(f"""CREATE TEMP TABLE pc AS SELECT p.yr, p.y, COALESCE(m.cat_l1, '(none)') AS cat_l1
   FROM read_parquet('{q("pool_dedup.parquet")}') p LEFT JOIN m USING (parent_asin)""")
con.execute("CREATE TEMP TABLE top AS SELECT cat_l1 FROM pc GROUP BY 1 ORDER BY count(*) DESC LIMIT 15")
con.execute("""CREATE TEMP TABLE pc2 AS SELECT yr, y,
   CASE WHEN cat_l1 IN (SELECT cat_l1 FROM top) THEN cat_l1 ELSE '(other)' END AS cat FROM pc""")
con.execute(f"""COPY (
  WITH c AS (SELECT yr, cat, count(*) AS n FROM pc2 GROUP BY yr, cat)
  SELECT yr, cat, n, round(n / sum(n) OVER (PARTITION BY yr), 5) AS share FROM c ORDER BY yr, n DESC
) TO '{os.path.join(OUT, "category_by_year.csv")}' (HEADER)""")
con.execute(f"""COPY (
  SELECT yr, cat, count(*) AS n, round(avg((y = 1)::INT), 5) AS star3_share, round(avg((y = 0)::INT), 5) AS neg_share
  FROM pc2 GROUP BY yr, cat ORDER BY yr, cat) TO '{os.path.join(OUT, "star3_by_category_year.csv")}' (HEADER)""")

shutil.make_archive(OUT, "zip", OUT)
log(f"完成。請下載 {OUT}.zip")
