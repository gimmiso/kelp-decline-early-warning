# CRW 5 km Composite SST Feature Report

## Purpose

This report adds a NOAA Coral Reef Watch 5 km monthly-composite SST exposure layer.
It does not replace the existing OISST V1/V2 workflow or the optional daily CRW point-cache path.

## Data Access

- Daily CRW point-cache extraction was tested previously but was too slow for the current environment.
- The ERDDAP monthly bbox path was also inconsistent for full 1988-2024 extraction in this environment.
- This run streams predictable NOAA STAR monthly NetCDF files, extracts retained Kelpwatch cell points, and deletes raw NetCDF files unless `--keep-raw-cache` is used.
- NOAA STAR monthly root: https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/5km/v3.1_op/nc/v1.0/monthly/
- CRW 5 km composite product page: https://coralreefwatch.noaa.gov/product/5km/index_5km_composite.php

## Spatial Matching

- Baseline extraction uses the nearest valid CRW 5 km ocean grid cell to each retained Kelpwatch 10 km cell centroid.
- The compact extracted cache stores only point-level monthly SST/SSTA values, source filenames, and extraction status.
- CRW 5 km remains satellite SST exposure, not true in-situ nearshore temperature.

## Feature Construction Summary

- Monthly files successfully processed: `444` month pairs
- Extracted monthly point rows: `22200`
- Annual CRW feature rows built: `1800`
- Unique Kelpwatch cells: `50`
- Mean CRW feature missingness: `0.0000`
- Mean nearest-grid distance: `2.815` km
- Max nearest-grid distance: `7.720` km
- Processed months this run: `444`
- Cached months reused: `0`
- Failed months this run: `0`

Composite features summarize monthly mean SST and SSTA. They cannot fully reproduce daily hot-day counts, cumulative heat stress, or short marine-heatwave duration metrics.

## Model Comparison

- Computed model-comparison rows: `40`

Best result per target:

| target_definition | feature_family | model | pr_auc | recall | precision | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_decline_drop | canopy_only | Logistic Regression L2 | 0.579 | 0.441 | 0.556 | 0.492 | 0.874 |
| at_risk_original_gt005 | canopy_only | Random Forest | 0.602 | 0.422 | 0.613 | 0.500 | 0.610 |
| new_decline_transition | canopy_only | Logistic Regression L2 | 0.407 | 0.462 | 0.429 | 0.444 | 0.775 |
| original_decline | canopy_only | Random Forest | 0.879 | 0.604 | 0.844 | 0.704 | 0.769 |

Best row by target and feature family:

| target_definition | feature_family | model | pr_auc | recall | precision | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_decline_drop | canopy_only | Logistic Regression L2 | 0.579 | 0.441 | 0.556 | 0.492 | 0.874 |
| actionable_decline_drop | canopy_plus_crw_composite | Logistic Regression L2 | 0.396 | 0.412 | 0.333 | 0.368 | 0.760 |
| actionable_decline_drop | crw_composite_only | Logistic Regression L2 | 0.180 | 0.647 | 0.202 | 0.308 | 0.553 |
| actionable_decline_drop | oisst_only | Logistic Regression L2 | 0.543 | 1.000 | 0.405 | 0.576 | 0.870 |
| actionable_decline_drop | oisst_plus_crw_composite | Logistic Regression L2 | 0.434 | 0.735 | 0.309 | 0.435 | 0.788 |
| at_risk_original_gt005 | canopy_only | Random Forest | 0.602 | 0.422 | 0.613 | 0.500 | 0.610 |
| at_risk_original_gt005 | canopy_plus_crw_composite | Logistic Regression L2 | 0.483 | 0.422 | 0.413 | 0.418 | 0.425 |
| at_risk_original_gt005 | crw_composite_only | Logistic Regression L2 | 0.465 | 0.467 | 0.412 | 0.438 | 0.418 |
| at_risk_original_gt005 | oisst_only | Logistic Regression L2 | 0.548 | 0.222 | 0.500 | 0.308 | 0.563 |
| at_risk_original_gt005 | oisst_plus_crw_composite | Logistic Regression L2 | 0.470 | 0.622 | 0.491 | 0.549 | 0.421 |
| new_decline_transition | canopy_only | Logistic Regression L2 | 0.407 | 0.462 | 0.429 | 0.444 | 0.775 |
| new_decline_transition | canopy_plus_crw_composite | Logistic Regression L2 | 0.303 | 0.462 | 0.235 | 0.312 | 0.565 |
| new_decline_transition | crw_composite_only | Logistic Regression L2 | 0.212 | 0.385 | 0.089 | 0.145 | 0.386 |
| new_decline_transition | oisst_only | Logistic Regression L2 | 0.148 | 0.269 | 0.137 | 0.182 | 0.556 |
| new_decline_transition | oisst_plus_crw_composite | Logistic Regression L2 | 0.231 | 0.462 | 0.146 | 0.222 | 0.589 |
| original_decline | canopy_only | Random Forest | 0.879 | 0.604 | 0.844 | 0.704 | 0.769 |
| original_decline | canopy_plus_crw_composite | Logistic Regression L2 | 0.845 | 0.642 | 0.782 | 0.705 | 0.706 |
| original_decline | crw_composite_only | Logistic Regression L2 | 0.828 | 0.634 | 0.780 | 0.700 | 0.683 |
| original_decline | oisst_only | Logistic Regression L2 | 0.714 | 0.261 | 0.714 | 0.383 | 0.520 |
| original_decline | oisst_plus_crw_composite | Logistic Regression L2 | 0.775 | 0.545 | 0.702 | 0.613 | 0.603 |

## Interpretation

CRW composite features should be interpreted as a higher-resolution satellite SST exposure alternative to OISST. They are not local in-situ temperature measurements.
Improvements should be claimed only where the computed comparison table supports them. If CRW composite features do not improve transition-oriented targets, the result still supports the broader interpretation that abrupt kelp transitions likely require local ecological covariates in addition to satellite SST exposure.

## Output Files

- `data/processed/crw5km_composite_features.csv`
- `results/tables/crw5km_composite_feature_diagnostics.csv`
- `results/tables/crw5km_composite_model_comparison.csv`
- `outputs/diagnostics/crw5km_composite_feature_report.md`