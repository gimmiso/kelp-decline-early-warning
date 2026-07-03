# Threshold Trade-off Summary

## Purpose

This analysis evaluates how alert thresholds change the balance between missed declines and false alerts. It is intended as a persistence-aware decision-support diagnostic rather than evidence of fully operational early-warning skill.

## Inputs

- Prediction file: `outputs/metadata/model_comparison_test_predictions.csv`
- Prediction rows: `3,000`
- Label targets evaluated: `decline_event_next`
- Feature sets evaluated: `canopy_noaa, canopy_only, oisst_only`
- Models evaluated: `LightGBM, Logistic Regression, Random Forest, SVM, XGBoost`
- Splits evaluated: `test`

## Threshold Grid

- Thresholds: `0.00` to `1.00` in increments of `0.01`.
- Lower thresholds increase recall but increase alert burden.
- Higher thresholds reduce false alerts but increase missed declines.

## Overview Figure Group

The overview figures use the highest PR-AUC model group in the input prediction file:

- Label target: `decline_event_next`
- Feature set: `canopy_only`
- Model: `Random Forest`
- Split: `test`
- PR-AUC: `0.897`
- ROC-AUC: `0.791`

## Selected Thresholds for Overview Group

| threshold | precision | recall | f1 | f2 | alerts_per_100_cell_years | false_alerts_per_100_cell_years | missed_declines_per_100_true_declines | tp | fp | fn | tn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.100 | 0.670 | 1.000 | 0.802 | 0.910 | 100.000 | 33.000 | 0.000 | 134 | 66 | 0 | 0 |
| 0.200 | 0.694 | 0.963 | 0.806 | 0.893 | 93.000 | 28.500 | 3.731 | 129 | 57 | 5 | 9 |
| 0.300 | 0.753 | 0.910 | 0.824 | 0.874 | 81.000 | 20.000 | 8.955 | 122 | 40 | 12 | 26 |
| 0.500 | 0.908 | 0.590 | 0.715 | 0.634 | 43.500 | 4.000 | 41.045 | 79 | 8 | 55 | 58 |

## Best F2 Operating Point for Overview Group

- Threshold: `0.11`
- Precision: `0.673`
- Recall: `1.000`
- F2: `0.912`
- Alerts per 100 cell-years: `99.5`
- False alerts per 100 cell-years: `32.5`
- Missed declines per 100 true decline cases: `0.0`

## Interpretation

- Threshold selection reveals a practical trade-off between missed declines and false alerts.
- The curves help distinguish broad risk-state screening from stricter operational early-warning claims.
- The analysis supports decision-threshold selection rather than proving causal early-warning skill.
- Apparent performance should still be interpreted alongside persistence baselines, at-risk subsets, transition labels, and actionable-drop labels.

## Outputs

- `outputs/tables/threshold_tradeoff_metrics.csv`
- `outputs/figures/threshold_tradeoff_recall_vs_alerts.png`
- `outputs/figures/threshold_tradeoff_precision_recall.png`
- `outputs/figures/threshold_tradeoff_false_alerts_vs_missed.png`
- `outputs/figures/threshold_tradeoff_f2_vs_threshold.png`
