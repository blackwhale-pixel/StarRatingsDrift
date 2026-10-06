# Changelog

## 1.1.0 (2026-10-06)

Added for the revised manuscript:
- `code/7_analysis/length_control.py`: review-length control for the three-star and four-star negative-share trends (Sections 3.7 and 4.4).
- `code/7_analysis/shift_share_decomposition.py` and `results/4_annotation/shift_share_results.csv`: decomposition of three-star neutral recall into composition, within-class and interaction terms with bootstrap intervals, and the prediction distribution of negative-text three-star reviews (Section 4.9, Table 10, Table B6).
- `results/4_annotation/annotation_preds/`: per-item predictions of TF-IDF (200k) and DistilRoBERTa, trained on 2013–2016 reviews with seeds 42–44, on the 3,014 annotated reviews.
- `data/annotation/annotation_item_attributes.csv`: English flag, text length and training/test overlap flags for the annotated reviews.
- `results/3_temporal/macro_f1_unrounded_by_seed.csv`, `macro_f1_unrounded_seed_means.csv`: unrounded macro-F1 (Table 4, Table 9, Table B4).
- `results/3_temporal/holdout_year_by_class.csv`: 2013–2016 holdout by year and class.

No review text is included. Code, data and result files from version 1.0.0 are unchanged; README.md, CITATION.cff and .zenodo.json were updated.

## 1.0.0

First public release.
