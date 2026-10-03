#!/usr/bin/env python3
"""
匯出「可公開、不含評論全文」的樣本識別清單，供期刊資料可用性使用。
內部編號 rid 是處理過程中產生的列號，別人無法重建；這支腳本把 rid 換成任何人都能
從官方 Amazon Reviews 2023 原始檔重建的鍵值：user_id + parent_asin + text_hash。
text_hash = md5( trim( 把連續空白壓成單一空白( lower(text) ) ) )，與稽核腳本相同。

輸出 portable_ids.zip：
  rid_map.csv        本研究所有用到的 rid → user_id, parent_asin, text_hash, yr, label
  sample_index.csv   每個 rid 屬於哪個樣本（train_seed42 或 test_2017_matched 等）
  key_uniqueness.txt 鍵值在資料池中是否唯一（可重建性檢查）
用法：python export_ids.py [資料夾]
"""
import sys, os, glob, shutil, time
import duckdb

D = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/Downloads/amazon_reviews_2023/Electronics"
OUT = os.path.join(D, "portable_ids"); os.makedirs(OUT, exist_ok=True)
con = duckdb.connect()
con.execute("SET memory_limit='6GB'")
con.execute(f"SET temp_directory='{os.path.join(D, 'duck_tmp')}'")
def log(m): print(time.strftime("%H:%M:%S"), m, flush=True)

files = sorted(glob.glob(os.path.join(D, "gpu_data", "*.parquet")))
if not files: sys.exit("找不到 gpu_data/*.parquet")
union = " UNION ALL ".join(
    f"SELECT '{os.path.basename(f)[:-8].replace('train_2013_2016', 'train_seed42')}' AS sample, rid FROM read_parquet('{f}')"
    for f in files)
con.execute(f"CREATE TEMP TABLE s AS {union}")
con.execute(f"COPY (SELECT * FROM s ORDER BY sample, rid) TO '{os.path.join(OUT, 'sample_index.csv')}' (HEADER)")
log(f"樣本清單：{con.execute('SELECT count(*), count(DISTINCT sample) FROM s').fetchone()}")

pool = os.path.join(D, "pool_dedup.parquet")
con.execute(f"""COPY (
  SELECT p.rid, p.user_id, p.parent_asin, p.text_hash, p.yr,
         CASE p.y WHEN 0 THEN 'neg' WHEN 1 THEN 'neu' ELSE 'pos' END AS label
  FROM read_parquet('{pool}') p WHERE p.rid IN (SELECT DISTINCT rid FROM s) ORDER BY p.rid
) TO '{os.path.join(OUT, 'rid_map.csv')}' (HEADER)""")

dup = con.execute(f"""SELECT count(*), sum((k > 1)::INT), sum(CASE WHEN k > 1 THEN k ELSE 0 END) FROM (
   SELECT count(*) k FROM read_parquet('{pool}') GROUP BY user_id, parent_asin, text_hash)""").fetchone()
used = con.execute(f"""WITH k AS (SELECT user_id, parent_asin, text_hash, count(*) n FROM read_parquet('{pool}')
   GROUP BY ALL HAVING count(*) > 1)
   SELECT count(*) FROM read_parquet('{pool}') p JOIN k USING (user_id, parent_asin, text_hash)
   WHERE p.rid IN (SELECT DISTINCT rid FROM s)""").fetchone()[0]
txt = (f"資料池鍵值組數：{dup[0]:,}\n重複鍵值組數：{dup[1]:,}（涵蓋 {dup[2]:,} 列）\n"
       f"本研究樣本中鍵值不唯一的列數：{used:,}\n")
open(os.path.join(OUT, "key_uniqueness.txt"), "w", encoding="utf-8").write(txt); log(txt)
shutil.make_archive(OUT, "zip", OUT)
log(f"完成。請下載 {OUT}.zip")
