# Initial 5-Model Comparison Report

## Dataset

- Dataset used: `data/processed/modeling_dataset_ge500_noaa_v1.csv`
- Main modeling years: 1989-2024
- Main modeling rows: 1800
- Cells: 50
- Target: `decline_event_next`, indicating whether next-year relative canopy falls below the cell-specific 1984-2013 baseline 25th percentile.

## Temporal Split

- Train: 1989-2016
- Validation: 2017-2020
- Test: 2021-2024

## Feature Sets

- `canopy_only`: current canopy/status variables plus lagged relative canopy and lagged canopy change.
- `oisst_only`: NOAA OISST thermal stress variables.
- `canopy_noaa`: canopy, OISST, CUTI/BEUTI, one-hot encoded `region_group`, and spatial controls `center_lat`, `center_lon`.

## Leakage Variables Excluded

- `baseline_p25_relative_canopy_1984_2013`
- `decline_50pct_next`
- `decline_event_next`
- `decline_event_next_p25_full`
- `next_year_kelp_area_m2`
- `next_year_relative_canopy`
- `p25_relative_canopy_full_history`
- `relative_canopy_change_next`
- `relative_canopy_pct_change_next`

## Models Compared

- Logistic Regression
- SVM
- Random Forest
- XGBoost
- LightGBM

## Metrics

Validation and test metrics include PR-AUC, ROC-AUC, recall, precision, F1, accuracy, and confusion matrix counts. Final model comparison prioritizes test PR-AUC, recall, and F1.

## Best Test Models

- Best by test PR-AUC: canopy_only / Random Forest (PR-AUC=0.8974, Recall=0.5896, F1=0.7149)
- Best by test Recall: canopy_only / Logistic Regression (Recall=0.7985, PR-AUC=0.8135, F1=0.7456)
- Best Canopy + NOAA PR-AUC: 0.8459
- Best Canopy-only PR-AUC: 0.8974
- Canopy + NOAA improves over Canopy-only by best test PR-AUC: False

## Limitations

- The number of spatial cells is small for generalizable machine-learning inference.
- The final test period is limited to four years, 2021-2024.
- NOAA variables are environmental exposure proxies, not direct ecological mechanisms.
- OISST uses nearest valid ocean grid assignment in Version 1.
- CUTI/BEUTI are latitude-bin proxies.
