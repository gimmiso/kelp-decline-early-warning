# Multi-Horizon Actionable Warning Report

## Purpose

The project mostly used next-year decline labels. This experiment tests whether actionable sharp canopy-drop risk can be screened at two warning horizons: next year and within the next two years.
The goal is risk-screening evidence for the main machine-learning framing, not a claim of operational early warning.

## Label Definitions

- `actionable_decline_drop_next_1year`: current relative canopy > 0.05 and proportional drop from year `t` to `t+1` is at least 30%.
- `actionable_decline_drop_next_2year`: current relative canopy > 0.05 and proportional drop from year `t` to the minimum canopy in `t+1` or `t+2` is at least 30%.
- Rows without required future canopy observations are excluded from that horizon's evaluation.
- Future canopy values are used only to define labels, not as predictors.

## Test Event Counts

| horizon | rows | positive_count | event_rate | year_min | year_max |
| --- | --- | --- | --- | --- | --- |
| 1year | 150 | 30 | 0.200 | 2021 | 2023 |
| 2year | 100 | 54 | 0.540 | 2021 | 2022 |

Full horizon-valid label counts by split:

| horizon | split | rows | positive_count | event_rate | year_min | year_max |
| --- | --- | --- | --- | --- | --- | --- |
| 1year | all | 1750 | 466 | 0.266 | 1989 | 2023 |
| 1year | train | 1400 | 395 | 0.282 | 1989 | 2016 |
| 1year | validation | 200 | 41 | 0.205 | 2017 | 2020 |
| 1year | test | 150 | 30 | 0.200 | 2021 | 2023 |
| 2year | all | 1700 | 696 | 0.409 | 1989 | 2022 |
| 2year | train | 1400 | 578 | 0.413 | 1989 | 2016 |
| 2year | validation | 200 | 64 | 0.320 | 2017 | 2020 |
| 2year | test | 100 | 54 | 0.540 | 2021 | 2022 |

## Best Horizon-Valid Test Results

| horizon | test_year_min | test_year_max | test_positive_count | test_event_rate | best_feature_family | best_model | best_pr_auc | best_recall | best_precision | best_f2 | best_false_negatives | trajectory_minus_current_pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1year | 2021 | 2023 | 30 | 0.200 | canopy_current_only | Logistic Regression | 0.602 | 0.867 | 0.520 | 0.765 | 4 | -0.091 |
| 2year | 2021 | 2022 | 54 | 0.540 | canopy_current_plus_trajectory | Random Forest | 0.975 | 1.000 | 0.900 | 0.978 | 0 | 0.018 |

## Common-Year Comparison

The common-year comparison uses years where both one-year and two-year labels are available.

| feature_family | model | pr_auc_1year | pr_auc_2year | pr_auc_2year_minus_1year | recall_1year | recall_2year | f2_1year | f2_2year | false_negatives_1year | false_negatives_2year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| canopy_current_plus_trajectory | Random Forest | 0.523 | 0.975 | 0.451 | 0.833 | 1.000 | 0.740 | 0.978 | 5 | 0 |
| canopy_current_plus_trajectory | XGBoost | 0.520 | 0.970 | 0.450 | 0.867 | 1.000 | 0.751 | 0.978 | 4 | 0 |
| canopy_current_plus_trajectory_plus_environment | Random Forest | 0.417 | 0.961 | 0.544 | 1.000 | 1.000 | 0.833 | 0.978 | 0 | 0 |
| canopy_current_plus_trajectory | LightGBM | 0.519 | 0.960 | 0.442 | 0.733 | 1.000 | 0.667 | 0.978 | 8 | 0 |
| canopy_current_plus_trajectory_plus_environment | XGBoost | 0.399 | 0.960 | 0.560 | 0.833 | 1.000 | 0.714 | 0.978 | 5 | 0 |
| canopy_current_only | Logistic Regression | 0.614 | 0.957 | 0.343 | 0.867 | 0.981 | 0.778 | 0.964 | 4 | 1 |
| canopy_current_plus_trajectory | Logistic Regression | 0.516 | 0.953 | 0.437 | 0.700 | 0.963 | 0.648 | 0.935 | 9 | 2 |
| canopy_current_only | XGBoost | 0.595 | 0.950 | 0.355 | 1.000 | 1.000 | 0.833 | 0.978 | 0 | 0 |

## Interpretation

The two-year label produced higher best PR-AUC in the horizon-valid test comparison.
The two-year target has a wider event window and a higher test event rate, so higher apparent performance should be read as broader horizon risk screening rather than a cleaner operational warning system.
Trajectory features are considered helpful when `canopy_current_plus_trajectory` improves PR-AUC over `canopy_current_only`; this varies by horizon and should be interpreted as persistence-aware screening rather than ecological mechanism discovery.
This experiment remains an actionable risk-screening diagnostic. It does not prove operational early warning, causal drivers, or spatial transferability.
