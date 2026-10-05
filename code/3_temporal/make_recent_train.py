#!/usr/bin/env python3
"""
近期資料重訓實驗：從去重資料池抽取 2019–2020 年的訓練樣本（50 萬筆）
- 排除所有已用於測試的評論（gpu_data/test_*.parquet 的 rid），避免訓練與測試重疊
- 輸出格式與 gpu_data/train_2013_2016.parquet 相同（rid, text, yr, y）
用法：python make_recent_train.py [資料夾]
輸出：gpu_data/train_2019_2020.parquet、recent_train_out.zip（含該檔與說明）
"""
import sys, os, glob, json, shutil
import duckdb
D = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/Downloads/amazon_reviews_2023/Electronics"
N, SEED = 500_000, 42
con = duckdb.connect(); con.execute("SET memory_limit='4GB'")
con.execute(f"SET temp_directory='{os.path.join(D, 'duck_tmp')}'")
tests = sorted(glob.glob(os.path.join(D, "gpu_data", "test_*.parquet")))
union = " UNION ALL ".join(f"SELECT rid FROM read_parquet('{f}')" for f in tests)
con.execute(f"CREATE TEMP TABLE test_rids AS SELECT DISTINCT rid FROM ({union})")
out = os.path.join(D, "gpu_data", "train_2019_2020.parquet")
con.execute(f"""COPY (
  SELECT rid, text, yr, y FROM (
    SELECT * FROM read_parquet('{os.path.join(D, "pool_dedup.parquet")}')
    WHERE yr BETWEEN 2019 AND 2020 AND rid NOT IN (SELECT rid FROM test_rids))
  USING SAMPLE reservoir({N} ROWS) REPEATABLE ({SEED})
) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
n, ov = con.execute(f"SELECT count(*), sum((rid IN (SELECT rid FROM test_rids))::INT) FROM read_parquet('{out}')").fetchone()
dist = con.execute(f"SELECT y, count(*) FROM read_parquet('{out}') GROUP BY y ORDER BY y").fetchall()
yrs = con.execute(f"SELECT yr, count(*) FROM read_parquet('{out}') GROUP BY yr ORDER BY yr").fetchall()
meta = {"n": n, "overlap_with_test_sets": ov, "label_counts(0=neg,1=neu,2=pos)": dict(dist), "year_counts": dict(yrs),
        "years": "2019-2020", "seed": SEED, "excluded": "all rids in gpu_data/test_*.parquet", "duckdb": duckdb.__version__}
print(json.dumps(meta, indent=2))
od = os.path.join(D, "recent_train_out"); os.makedirs(od, exist_ok=True)
shutil.copy(out, od); json.dump(meta, open(os.path.join(od, "recent_train_meta.json"), "w"), indent=2)
shutil.make_archive(od, "zip", od); print("完成：", od + ".zip")
