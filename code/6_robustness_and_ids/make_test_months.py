#!/usr/bin/env python3
"""
相同月份比較：為 15 個測試集的每一則評論找出發表月份
做法：依 rid 從 pool_dedup.parquet 取得 user_id、parent_asin、text_hash、yr，
     再到 reviews.parquet 找出同一使用者、同一商品、同一文字、同一年份的評論時間戳記（多筆取最早）。
用法：python make_test_months.py [資料夾]
輸出：test_months_out.zip（test_months.csv：rid, ts, month；以及對照率統計）
"""
import sys, os, glob, json, shutil
import duckdb
D = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/Downloads/amazon_reviews_2023/Electronics"
con = duckdb.connect(); con.execute("SET memory_limit='4GB'")
con.execute(f"SET temp_directory='{os.path.join(D, 'duck_tmp')}'")
tests = sorted(glob.glob(os.path.join(D, "gpu_data", "test_*.parquet")))
union = " UNION ALL ".join(f"SELECT DISTINCT rid FROM read_parquet('{f}')" for f in tests)
con.execute(f"CREATE TEMP TABLE t AS SELECT DISTINCT rid FROM ({union})")
con.execute(f"""CREATE TEMP TABLE k AS SELECT p.rid, p.user_id, p.parent_asin, p.text_hash, p.yr
               FROM read_parquet('{os.path.join(D, "pool_dedup.parquet")}') p JOIN t USING (rid)""")
con.execute(f"""CREATE TEMP TABLE m AS
  SELECT k.rid, min(r.ts) AS ts FROM k JOIN read_parquet('{os.path.join(D, "reviews.parquet")}') r
    ON r.user_id = k.user_id AND r.parent_asin = k.parent_asin AND r.text_hash = k.text_hash
   AND year(to_timestamp(r.ts / 1000)) = k.yr
  GROUP BY k.rid""")
od = os.path.join(D, "test_months_out"); os.makedirs(od, exist_ok=True)
con.execute(f"""COPY (SELECT rid, ts, month(to_timestamp(ts / 1000)) AS month FROM m ORDER BY rid)
               TO '{os.path.join(od, "test_months.csv")}' (HEADER)""")
n_t, n_m = con.execute("SELECT (SELECT count(*) FROM t), (SELECT count(*) FROM m)").fetchone()
meta = {"test_items": n_t, "matched_with_timestamp": n_m, "match_rate": round(n_m / n_t, 6), "duckdb": duckdb.__version__}
print(json.dumps(meta, indent=2)); json.dump(meta, open(os.path.join(od, "test_months_meta.json"), "w"), indent=2)
shutil.make_archive(od, "zip", od); print("完成：", od + ".zip")
