# Zero-Persistence and At-Risk Early-Warning Diagnostic

## Purpose

This diagnostic checks whether model performance is partly explained by persistence of already-zero or already-low canopy states. It distinguishes three cases: true early warning of future decline, persistence of already-low canopy states, and trivial zero-to-zero prediction.

## Zero-Persistence Transition Rates

| canopy_threshold | zero_to_zero_persistence_rate | nonzero_to_zero_transition_rate |
| --- | --- | --- |
| 0.000 | 0.143 | 0.003 |
| 0.010 | 0.630 | 0.079 |
| 0.050 | 0.669 | 0.190 |
| 0.100 | 0.746 | 0.299 |

## Original Label, Highest Test F1 Model by Evaluation Subset

| evaluation_subset | feature_set | model | n_observations | n_positive_events | event_prevalence | precision | recall | f1 | pr_auc | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_canopy_gt_0 | canopy_only | Logistic Regression | 199 | 133 | 0.668 | 0.697 | 0.797 | 0.744 | 0.806 | 0.675 |
| current_canopy_gt_0.01 | canopy_only | Logistic Regression | 138 | 74 | 0.536 | 0.550 | 0.446 | 0.493 | 0.601 | 0.515 |
| current_canopy_gt_0.05 | canopy_only | Random Forest | 93 | 45 | 0.484 | 0.600 | 0.400 | 0.480 | 0.633 | 0.615 |
| current_canopy_gt_0.1 | canopy_only | Random Forest | 61 | 29 | 0.475 | 0.667 | 0.345 | 0.455 | 0.616 | 0.589 |
| full_sample | canopy_only | Logistic Regression | 200 | 134 | 0.670 | 0.699 | 0.799 | 0.746 | 0.813 | 0.686 |

## New-Decline Transition Label, Highest Test F1 Model by Evaluation Subset

| evaluation_subset | feature_set | model | n_observations | n_positive_events | event_prevalence | precision | recall | f1 | pr_auc | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_canopy_gt_0 | canopy_only | Logistic Regression | 199 | 26 | 0.131 | 0.364 | 0.462 | 0.407 | 0.390 | 0.804 |
| current_canopy_gt_0.01 | canopy_only | Logistic Regression | 138 | 25 | 0.181 | 0.423 | 0.440 | 0.431 | 0.407 | 0.758 |
| current_canopy_gt_0.05 | canopy_only | Logistic Regression | 93 | 18 | 0.194 | 0.476 | 0.556 | 0.513 | 0.447 | 0.756 |
| current_canopy_gt_0.1 | canopy_only | Logistic Regression | 61 | 17 | 0.279 | 0.500 | 0.471 | 0.485 | 0.485 | 0.668 |
| full_sample | canopy_only | Logistic Regression | 200 | 26 | 0.130 | 0.375 | 0.462 | 0.414 | 0.388 | 0.803 |

## Interpretation

- Full-sample original-label best PR-AUC: `canopy_only / Random Forest` with PR-AUC=0.897, recall=0.590, and F1=0.715.
- At-risk original-label best PR-AUC for `current_canopy > 0.05`: `canopy_only / Random Forest` with PR-AUC=0.633, recall=0.400, and F1=0.480.
- Full-sample new-decline-label best PR-AUC: `canopy_only / Random Forest` with PR-AUC=0.401, recall=0.423, and F1=0.386.

If performance is much stronger in the full sample than in current-nonzero or new-decline-transition evaluations, the model should be interpreted primarily as detecting canopy-state persistence rather than robust early warning. If useful skill remains in at-risk and new-decline-transition subsets, that is preliminary evidence that the model may capture warning signals before visible collapse.

## Output Files

- `outputs/diagnostics/zero_persistence_transition_counts.csv`
- `outputs/diagnostics/zero_persistence_transition_rates.csv`
- `outputs/diagnostics/zero_persistence_transition_rates.png`
- `outputs/diagnostics/at_risk_subset_model_performance.csv`
- `outputs/diagnostics/new_decline_transition_model_performance.csv`
