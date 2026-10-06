# Star Ratings Drift: code and derived data

Code and derived data for the paper *Star Ratings Drift: Temporal Concept Drift in Large-Scale Review Sentiment Classification* (Jun-He Yang, Fang-Kai Tang, Chia-Pang Chan and Chiung-Hui Tsai).

The study uses the Electronics category of Amazon Reviews 2023. **This repository contains no review text.** Every review is identified by a key that anyone with the official data can rebuild, so all samples, labels and predictions can be matched back to the source.

## Contents

| Folder | What it holds |
|---|---|
| `code/` | All scripts, numbered in the order they were run |
| `data/keys/` | Public keys for every reviewed item used in training and testing |
| `data/annotation/` | LLM annotation sample (IDs, year, star rating), labels, codebook, SemEval-2014 derived gold labels, item attributes (English flag, text length, overlap with training and test samples) |
| `results/` | Result tables, per-item predictions of every model and seed (including predictions on the annotation sample), unrounded macro-F1 by seed, run metadata |
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
- For 504 of 1,299,258 items (0.04%) the triple (user_id, parent_asin, text_hash) is not unique in the pool. In 503 of them the same user posted the same text, with the same rating class, under 2 to 10 variants of one product, so any candidate row gives the same model input and label; for the remaining one, `label` identifies the row. The original row itself (its variant ASIN and timestamp) cannot always be recovered: candidates share the parent product, so subcategory analyses are unaffected, and `test_item_months.csv` uses the earliest timestamp among candidates from the item's year. `data/keys/nonunique_key_items.csv` lists these items (`n_key` = pool rows sharing the triple; `n_key_and_label` = rows also sharing the label), and `data/keys/nonunique_key_candidates.csv` lists each candidate's `asin`, `ts` and `rating` in the source file (made with `code/6_robustness_and_ids/check_nonunique_keys.py`).

`data/keys/sample_membership.csv.gz` lists which items belong to each sample: `train_seed42` (500,000 reviews, 2013–2016), `train_2019_2020` (500,000 reviews used for retraining, none of them in any test set), the 2013–2016 holdout and the 14 yearly test sets (`test_<year>_matched`, `test_<year>_natural`, 20,000 each). `data/keys/train_200k_by_seed.csv.gz` lists the 200,000-review training samples used by DistilRoBERTa, RoBERTa-base and the 200k TF-IDF model for seeds 42–44; `data/keys/train_recent_2019_2020_200k_by_seed.csv.gz` lists the corresponding samples for the retrained models.

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
| 4.7 (Table 7) | slopes without 2023; binary task without neutral items | `code/7_analysis/robustness_2023_binary.py` on `results/3_temporal/*/preds` |
| 3.7, 4.2, 4.4, Appendix B | per-seed slopes, product/user cluster bootstrap, neutral-or-mixed trend, no-majority sensitivity | `code/7_analysis/robustness_cluster_seed_annotation.py` |
| 3.8, 4.4 (human validation table) | human–LLM agreement, human three-star trend, sensitivity | `code/7_analysis/human_validation.py`; labels in `data/annotation/human_validation_labels.csv` (item IDs match `annotation_key.csv`) |
| 4.8 (retraining table) | gains from training on 2019–2020 | `code/3_temporal/make_recent_train.py`, `tfidf_200k.py` / `train_transformer.py --train_file train_2019_2020.parquet`, `code/7_analysis/retraining_comparison.py` |
| 4.7 (month-matched check) | slopes using January–September reviews only | `code/6_robustness_and_ids/make_test_months.py`, `code/7_analysis/robustness_month_matched.py`; months in `data/keys/test_item_months.csv` |
| 3.7, 4.4 (length control) | log-length control, length tertiles, four-star trend, human period odds ratio with length | `code/7_analysis/length_control.py` on `data/annotation/` (text length in `annotation_item_attributes.csv`) |
| 4.9, Table 10, Table B6 | shift-share decomposition of three-star neutral recall; prediction distribution of negative-text three-star reviews | `code/7_analysis/shift_share_decomposition.py` → `results/4_annotation/shift_share_results.csv`; predictions in `results/4_annotation/annotation_preds/` |
| Table 4, Table 9, Table B4 | unrounded macro-F1 for every model, test set and seed | `results/3_temporal/macro_f1_unrounded_by_seed.csv`, `macro_f1_unrounded_seed_means.csv` |
| 4.2 | holdout composition by year and class | `results/3_temporal/holdout_year_by_class.csv` |
| Figures 1–4 | — | `figures/make_figures.py` |

### Notes on the added files (version 1.1.0)

- `annotation_item_attributes.csv`: `english` (False for the 12 non-English reviews), `text_len` (characters), and whether the review appears in the 2013–2016 500,000-review training file (`in_train_2013_2016_500k`, 54 reviews; every seed's 200,000-review sample is drawn from this file), in any seed's 200,000-review sample (`in_train_200k_any_seed`, 43) or in any test set (`in_any_test_set`, 17).
- `annotation_preds/<model>/seed<k>.csv`: `item_id`, predicted class (`pred`: 0 = negative, 1 = neutral, 2 = positive) and class probabilities for the 3,014 annotated reviews, from the models trained on 2013–2016 reviews.
- Table 4 and Table 9 average the unrounded seed values; averaging values rounded to three decimals can differ by 0.001.
- The analysis scripts need `pandas`, `numpy` and `statsmodels`, and are run from the repository root.

## Models and environment

- LLM annotators (Ollama 0.35.0): `qwen3:8b` (500a1f067a9f), `llama3.1:8b` (46e0c10c039e), `mistral:latest` (6577803aa9a0); temperature 0, seed 42, JSON-schema output. Codebook SHA-256 prefix `5ebe279f4a50`.
- Laptop: AMD Ryzen AI 9 HX 370, 32 GB RAM, NVIDIA GeForce RTX 5060 Laptop GPU (8 GB); Python 3.11.9, PyTorch 2.11.0 (CUDA 12.8), Transformers 5.17.0.
- Cloud sandbox: Intel Xeon @ 2.50 GHz, 6 cores, 6–11 GiB RAM depending on session; Python 3.12.3, DuckDB 1.5.6, scikit-learn 1.9.1.
- Library versions per run: `environment/library_versions_by_run.csv`. The cloud sandbox was reconfigured after the analyses, so later package snapshots do not describe the analysis environment.

## Not included

Review texts and product data from Amazon Reviews 2023, intermediate Parquet files containing text, SemEval-2014 sentences, and the free-text reasons returned by the LLM annotators (they may quote reviews). These are kept by the authors and can be made available to reviewers through the journal.

## License

Code: MIT (`LICENSE`). Derived data, results and figures: CC BY 4.0 (`LICENSE-DATA`). The Amazon Reviews 2023 and SemEval-2014 data remain under their own terms.

## Citation

See `CITATION.cff`.
