# Alert-Budget Top-k Prioritisation Summary

## Purpose

Top-k evaluation measures whether model risk scores can prioritise follow-up monitoring under fixed alert budgets. It helps distinguish broad risk prioritisation from stricter operational early-warning claims.

## Inputs

- Prediction file: `outputs/metadata/model_comparison_test_predictions.csv`
- Prediction rows evaluated: `3,000`
- Label targets evaluated: `decline_event_next`
- Feature sets evaluated: `canopy_noaa, canopy_only, oisst_only`
- Models evaluated: `LightGBM, Logistic Regression, Random Forest, SVM, XGBoost`
- Splits evaluated: `test`

## Budget Definitions

- Annual fixed-k budgets: `1, 3, 5, 10, 15, 20` cells per year.
- Pooled fixed-k budgets: `5, 10, 20, 50, 100` grid-year observations.
- Percent budgets: top `10%`, `20%`, `30%`, `40%`, and `50%` of available candidates.
- Percentage budgets are converted to integer k using ceiling and capped at the number of candidates.

## Main Figure Group

- Label target: `decline_event_next`
- Feature set: `canopy_only`
- Model: `Random Forest`
- Split: `test`
- PR-AUC: `0.897`
- ROC-AUC: `0.791`

## Selected Annual Budgets

| annual_top_k | mean_precision_at_k | mean_recall_at_k | mean_lift_at_k | mean_hits | mean_false_alerts | mean_missed_declines | total_hits | total_false_alerts | n_years |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | 1.000 | 0.031 | 1.560 | 1.000 | 0.000 | 32.500 | 4 | 0 | 4 |
| 3.000 | 1.000 | 0.094 | 1.560 | 3.000 | 0.000 | 30.500 | 12 | 0 | 4 |
| 5.000 | 1.000 | 0.156 | 1.560 | 5.000 | 0.000 | 28.500 | 20 | 0 | 4 |
| 10.000 | 0.975 | 0.301 | 1.507 | 9.750 | 0.250 | 23.750 | 39 | 1 | 4 |
| 20.000 | 0.900 | 0.551 | 1.377 | 18.000 | 2.000 | 15.500 | 72 | 8 | 4 |

## Selected Pooled Budgets

| pooled_top_k | precision_at_k | recall_at_k | lift_at_k | hits | false_alerts | missed_declines | expected_random_hits | enrichment_over_random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10.000 | 1.000 | 0.075 | 1.493 | 10 | 0 | 124 | 6.700 | 1.493 |
| 20.000 | 1.000 | 0.149 | 1.493 | 20 | 0 | 114 | 13.400 | 1.493 |
| 50.000 | 0.980 | 0.366 | 1.463 | 49 | 1 | 85 | 33.500 | 1.463 |
| 100.000 | 0.860 | 0.642 | 1.284 | 86 | 14 | 48 | 67.000 | 1.284 |

## Interpretation

- Fixed alert budgets evaluate prioritisation rather than binary classification.
- Small budgets may have high precision but low recall.
- Larger budgets capture more true declines but create more false alerts.
- Lift@k compares model prioritisation against random selection at the same budget.
- Results should be interpreted alongside persistence baselines, transition labels, and threshold trade-off curves.
- Top-k performance does not prove causal or fully operational early-warning skill.

## Outputs

- `outputs/tables/alert_budget_topk_metrics.csv`
- `outputs/tables/alert_budget_topk_annual_summary.csv`
- `outputs/figures/topk_recall_vs_budget.png`
- `outputs/figures/topk_precision_vs_budget.png`
- `outputs/figures/topk_lift_vs_budget.png`
- `outputs/figures/topk_cumulative_gain.png`
- `outputs/figures/topk_hits_false_alerts_by_budget.png`

## Assumptions and Warnings

- No label target column found; assigned `decline_event_next` to all rows.
