# Star Ratings Drift: code and derived data

Code and derived data for the paper *Star Ratings Drift: Temporal Concept Drift in Large-Scale Review Sentiment Classification* (Jun He, Open University of Kaohsiung).

The study uses the Electronics category of Amazon Reviews 2023. **This repository contains no review text.** Every review is identified by a key that anyone with the official data can rebuild, so all samples, labels and predictions can be matched back to the source.

## Contents

| Folder | What it holds |
|---|---|
| `code/` | All scripts, numbered in the order they were run |
| `data/keys/` | Public keys for every reviewed item used in training and testing |
| `data/annotation/` | LLM annotation sample (IDs, year, star rating), labels, codebook, SemEval-2014 derived gold labels |
| `results/` | Result tables, per-item predictions of every model and seed, run metadata |
| `figures/` | Figures 1–4 (PDF, PNG) and `make_figures.py` |
| `environment/` | Library versions recorded at run time, cloud environment snapshot |

## Getting the source data

Download `Electronics.jsonl.gz` and `meta_Electronics.jsonl.gz` from https://amazon-reviews-2023.github.io (Hou et al., 2024, arXiv:2403.03952). The files used in this study were downloaded on 30 September 2026:

| File | Size (bytes) | SHA-256 |
|---|---|---|
| Electronics.jsonl.gz | 6,474,438,619 | `17b2c5f3736d4c0cb874859076436ebd4513f4c2396c528440a63204084e6a28` |
| meta_Electronics.jsonl.gz | 1,312,900,427 | `a4a196a1c8e443e0942d8a5c79b5a5d2d68e29d483f2badec1126690b4b2790d` |

The same values are in `environment/source_data_SHA256.txt`. Check a download with `sha256sum -c environment/source_data_SHA256.txt` (Linux/macOS) or `Get-FileHash <file> -Algorithm SHA256` (Windows). If the hashes differ, the source files have changed since this study.

## Matching keys to reviews

`data/keys/item_keys.csv.gz` maps each `item_id` to `user_id`, `parent_asin`, `text_hash`, `year` and the three-class `label`.

- `text_hash` = MD5 of the review text after lowercasing, collapsing runs of whitespace to one space and trimming.
- `label`: `neg` = 1–2 stars, `neu` = 3 stars, `pos` = 4–5 stars.
- `item_id` is an internal row number of the deduplicated pool; it carries no meaning outside this repository.
- For 401 of 799,258 items (0.05%) the triple (user_id, parent_asin, text_hash) is not unique in the pool, because the same user posted the same text under different variants of one product.

`data/keys/sample_membership.csv.gz` lists which items belong to each sample: `train_seed42` (500,000 reviews, 2013–2016), the 2013–2016 holdout and the 14 yearly test sets (`test_<year>_matched`, `test_<year>_natural`, 20,000 each). `data/keys/train_200k_by_seed.csv.gz` lists the 200,000-review training samples used by DistilRoBERTa, RoBERTa-base and the 200k TF-IDF model for seeds 42–44.

The annotation sample (`data/annotation/annotation_key.csv`) is keyed by `user_id`, `asin` and `ts` (timestamp in ms), which identify a review directly in the source file.

## Running the pipeline

Scripts take a data folder as their first argument; analysis scripts in `code/7_analysis` have input paths at the top that must be edited.

| Step | Script | Produces |
|---|---|---|
| 1 | `code/1_audit/amazon_electronics_audit.py` | `reviews.parquet`, duplicate and leakage audit (Section 4.1) |
| 2 | `code/1_audit/amazon_electronics_audit2.py` | exact, loose and prefix leakage table (Table 2) |
| 3 | `code/2_split_experiment/amazon_electronics_baseline.py` | deduplicated pool `pool_dedup.parquet`; split experiment (Table 3) |
| 4 | `code/3_temporal/amazon_electronics_drift.py` | temporal test sets, TF-IDF 500k drift run, annotation sample |
| 5 | `code/3_temporal/tfidf_preds.py` | TF-IDF 500k per-item predictions on the shared test sets |
| 6 | `code/3_temporal/train_transformer.py` | DistilRoBERTa (`--model distilbert/distilroberta-base`) and RoBERTa-base (`--model FacebookAI/roberta-base --batch 16 --grad_accum 2`) |
| 7 | `code/3_temporal/tfidf_200k.py` | TF-IDF on the same 200,000 reviews as the transformers |
| 8 | `code/4_annotation/llm_annotate.py` | LLM annotation, SemEval check, agreement (Section 4.4) |
| 9 | `code/5_zero_shot/zeroshot_drift.py` | zero-shot test (Section 4.5) |
| 10 | `code/6_robustness_and_ids/attach_meta.py`, `export_ids.py` | product categories; public keys |
| 11 | `code/7_analysis/*.py` | paired slope tests, category-controlled trends, reweighting |
| 12 | `figures/make_figures.py` | Figures 1–4 |

Because DuckDB reservoir sampling is not guaranteed to repeat exactly, a rerun of steps 3–5 may draw slightly different samples. Use the key files in `data/keys/` to rebuild the exact samples analysed in the paper.

## Where each reported number comes from

| Paper | Number | Source |
|---|---|---|
| 4.1 | 478,014 redundant record copies | `results/1_audit/summary.json` and step 2 log |
| Table 2 | leakage rates | `results/1_audit/summary_leakage.json` |
| Table 3 | split macro-F1 | `results/2_split_experiment/results.csv` |
| Table 4 | yearly slopes and intervals | `code/7_analysis/multi_slope.py`, `slope_test.py` on `results/3_temporal/*/preds` |
| Table 5 | paired slope differences | `code/7_analysis/multi_slope.py`, `slope_test.py`, `slope_test2.py` |
| 4.2 | vocabulary shift, label shares | `results/3_temporal/tfidf_500k_drift_run/oov_by_year.csv`, `year_label_dist.csv` |
| 4.4 | kappa, SemEval accuracy, label shares | `results/4_annotation/agreement_report.json`, `*_dist_by_star.csv` |
| Table 6 | odds ratios with category controls | `code/7_analysis/meta_analysis.py` |
| 4.5 | zero-shot slopes | `code/7_analysis/zs_paired2.py` on `results/5_zero_shot/` |
| 4.6 | product-mix reweighting | `code/7_analysis/meta_analysis.py`, `results/6_product_mix/` |
| Figures 1–4 | — | `figures/make_figures.py` |

## Models and environment

- LLM annotators (Ollama 0.35.0): `qwen3:8b` (500a1f067a9f), `llama3.1:8b` (46e0c10c039e), `mistral:latest` (6577803aa9a0); temperature 0, seed 42, JSON-schema output. Codebook SHA-256 prefix `5ebe279f4a50`.
- Laptop: AMD Ryzen AI 9 HX 370, 32 GB RAM, NVIDIA GeForce RTX 5060 Laptop GPU (8 GB); Python 3.11.9, PyTorch 2.11.0 (CUDA 12.8), Transformers 5.17.0.
- Cloud sandbox: Intel Xeon @ 2.50 GHz, 6 cores, 6–11 GiB RAM depending on session; Python 3.12.3, DuckDB 1.5.6, scikit-learn 1.9.1.
- Library versions per run: `environment/library_versions_by_run.csv`. The cloud sandbox was reconfigured after the analyses, so later package snapshots do not describe the analysis environment.

## Not included

Review texts and product data from Amazon Reviews 2023, intermediate Parquet files containing text, SemEval-2014 sentences, and the free-text reasons returned by the LLM annotators (they may quote reviews). These are kept by the author and can be made available to reviewers through the journal.

## License

Code: MIT (`LICENSE`). Derived data, results and figures: CC BY 4.0 (`LICENSE-DATA`). The Amazon Reviews 2023 and SemEval-2014 data remain under their own terms.

## Citation

See `CITATION.cff`.
