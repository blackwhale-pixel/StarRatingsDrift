#!/usr/bin/env python3
"""
找出研究樣本中「使用者＋商品＋文字雜湊」不唯一的評論，並檢查加上標籤後能否唯一對應。
省記憶體版：只針對研究樣本的鍵值計數，不對整個資料池分組。
用法：python check_nonunique_keys.py [資料夾]
輸出：nonunique_out.zip（nonunique_items.csv、nonunique_candidates.csv、摘要 JSON）
"""
import sys, os, glob, json, shutil
import duckdb
D = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/Downloads/amazon_reviews_2023/Electronics"
tmp = os.path.join(D, "duck_tmp"); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp, exist_ok=True)
con = duckdb.connect()
for q in ("SET memory_limit='3GB'", "SET threads=2", "SET preserve_insertion_order=false", f"SET temp_directory='{tmp}'"):
    con.execute(q)
pool = os.path.join(D, "pool_dedup.parquet"); rev = os.path.join(D, "reviews.parquet")
files = sorted(glob.glob(os.path.join(D, "gpu_data", "*.parquet")))
union = " UNION ALL ".join(f"SELECT rid FROM read_parquet('{f}')" for f in files)
# 1. 研究樣本的鍵值（約 130 萬列）
con.execute(f"""CREATE TEMP TABLE sk AS
  SELECT p.rid, p.user_id, p.parent_asin, p.text_hash, p.y,
         hash(p.user_id, p.parent_asin, p.text_hash) AS k
  FROM read_parquet('{pool}') p WHERE p.rid IN (SELECT DISTINCT rid FROM ({union}))""")
con.execute("CREATE TEMP TABLE keys AS SELECT DISTINCT k FROM sk")
# 2. 資料池中與這些鍵值相同的列（只保留雜湊與標籤，資料量很小）
con.execute(f"""CREATE TEMP TABLE cand AS
  SELECT hash(user_id, parent_asin, text_hash) AS k, y FROM read_parquet('{pool}')
  WHERE hash(user_id, parent_asin, text_hash) IN (SELECT k FROM keys)""")
con.execute("CREATE TEMP TABLE ck AS SELECT k, count(*) n_key FROM cand GROUP BY k HAVING count(*) > 1")
con.execute("CREATE TEMP TABLE cky AS SELECT k, y, count(*) n_key_and_label FROM cand WHERE k IN (SELECT k FROM ck) GROUP BY k, y")
con.execute("""CREATE TEMP TABLE items AS
  SELECT s.rid AS item_id, s.user_id, s.parent_asin, s.text_hash, s.y, c.n_key, e.n_key_and_label
  FROM sk s JOIN ck c USING (k) JOIN cky e USING (k, y)""")
od = os.path.join(D, "nonunique_out"); os.makedirs(od, exist_ok=True)
con.execute(f"COPY (SELECT item_id, n_key, n_key_and_label FROM items ORDER BY item_id) TO '{os.path.join(od, 'nonunique_items.csv')}' (HEADER)")
con.execute(f"""COPY (
  SELECT i.item_id, r.asin, r.ts, r.rating
  FROM items i JOIN read_parquet('{rev}') r
    ON r.user_id = i.user_id AND r.parent_asin = i.parent_asin AND r.text_hash = i.text_hash
  WHERE r.rating BETWEEN 1 AND 5 ORDER BY i.item_id, r.ts) TO '{os.path.join(od, 'nonunique_candidates.csv')}' (HEADER)""")
n, u = con.execute("SELECT count(*), sum((n_key_and_label = 1)::INT) FROM items").fetchone()
meta = {"sample_items_with_nonunique_key": n, "made_unique_by_adding_label": int(u or 0),
        "still_ambiguous_after_label": n - int(u or 0),
        "note": "ambiguous items share user, product family, text and rating class with another pool row"}
print(json.dumps(meta, indent=2)); json.dump(meta, open(os.path.join(od, "nonunique_meta.json"), "w"), indent=2)
shutil.rmtree(tmp, ignore_errors=True)
shutil.make_archive(od, "zip", od); print("完成：", od + ".zip")
