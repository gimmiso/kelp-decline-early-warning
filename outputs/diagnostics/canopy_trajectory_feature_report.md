# Canopy Trajectory and Instability Proxy Feature Report

## Purpose

This report adds leakage-safe canopy trajectory and time-series instability proxy features.
The features are a diagnostic extension of the persistence baseline, not an environmental driver layer.

## Leakage Audit

- Every generated feature row stores `trajectory_max_source_year_used`.
- The audit verifies `trajectory_max_source_year_used <= year` for every model-period row.
- Leakage audit status: `passed`; violating rows: `0`.
- First year where all trajectory features are available in the full panel: `1988`.

## Feature Definitions

- 3-year windows use year `t`, `t-1`, and `t-2` only.
- 5-year windows use year `t`, `t-1`, `t-2`, `t-3`, and `t-4` only.
- `instability_score_5yr_t` is mean absolute annual canopy change over the past/current 5-year window.
- `presence_frequency_5yr_t` is the share of years in the 5-year window with `relative_canopy > 0.01`.
- These are time-series instability proxies, not true spatial fragmentation metrics because patch geometry is not used.

## Label and Threshold Notes

- The default `decline_event_next` label uses the existing `baseline_p25_relative_canopy_1984_2013` threshold.
- The source dataset also contains full-history p25 fields, but this script does not use them for the default model comparison.
- The optional high-canopy subgroup uses the 1984-2013 cell-specific p75 threshold.

## Feature Construction Summary

- Feature rows built: `1800`
- Cells represented: `50`
- Model-period years: `1989-2024`
- Maximum feature missingness: `0.0000`

## Model Comparison

- Computed model-comparison rows: `75`

Best result per target:

| target_definition | feature_family | model | pr_auc | recall | precision | f1 | false_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_decline_drop | existing_canopy_only | Logistic Regression L2 | 0.579 | 0.765 | 0.531 | 0.627 | 8 |
| at_risk_original_gt005 | habitat_only | Logistic Regression L2 | 0.764 | 0.556 | 0.781 | 0.649 | 20 |
| high_canopy_original_decline | canopy_trajectory_plus_habitat | Logistic Regression L2 | 0.657 | 0.571 | 0.444 | 0.500 | 3 |
| new_decline_transition | canopy_trajectory_plus_crw_plus_habitat | Logistic Regression L2 | 0.419 | 0.038 | 1.000 | 0.074 | 25 |
| original_decline | naive_persistence_baseline | current_low_canopy_score | 0.893 | 0.858 | 0.732 | 0.790 | 19 |

Best row by target and feature family:

| target_definition | feature_family | model | pr_auc | recall | precision | f1 | false_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_decline_drop | canopy_trajectory_only | Random Forest | 0.533 | 0.824 | 0.444 | 0.577 | 6 |
| actionable_decline_drop | canopy_trajectory_plus_crw | Logistic Regression L2 | 0.474 | 0.206 | 0.500 | 0.292 | 27 |
| actionable_decline_drop | canopy_trajectory_plus_crw_plus_habitat | Logistic Regression L2 | 0.477 | 0.382 | 0.565 | 0.456 | 21 |
| actionable_decline_drop | canopy_trajectory_plus_habitat | Logistic Regression L2 | 0.503 | 0.559 | 0.487 | 0.521 | 15 |
| actionable_decline_drop | crw_composite_only | Logistic Regression L2 | 0.184 | 0.676 | 0.240 | 0.354 | 11 |
| actionable_decline_drop | existing_canopy_only | Logistic Regression L2 | 0.579 | 0.765 | 0.531 | 0.627 | 8 |
| actionable_decline_drop | habitat_only | Random Forest | 0.171 | 1.000 | 0.170 | 0.291 | 0 |
| actionable_decline_drop | naive_persistence_baseline | current_low_canopy_score | 0.099 | 0.971 | 0.167 | 0.284 | 1 |
| at_risk_original_gt005 | canopy_trajectory_only | Logistic Regression L2 | 0.650 | 0.844 | 0.667 | 0.745 | 7 |
| at_risk_original_gt005 | canopy_trajectory_plus_crw | Logistic Regression L2 | 0.638 | 0.333 | 0.577 | 0.423 | 30 |
| at_risk_original_gt005 | canopy_trajectory_plus_crw_plus_habitat | Logistic Regression L2 | 0.655 | 0.356 | 0.615 | 0.451 | 29 |
| at_risk_original_gt005 | canopy_trajectory_plus_habitat | Logistic Regression L2 | 0.671 | 0.689 | 0.660 | 0.674 | 14 |
| at_risk_original_gt005 | crw_composite_only | Logistic Regression L2 | 0.549 | 0.333 | 0.517 | 0.405 | 30 |
| at_risk_original_gt005 | existing_canopy_only | Random Forest | 0.602 | 0.578 | 0.565 | 0.571 | 19 |
| at_risk_original_gt005 | habitat_only | Logistic Regression L2 | 0.764 | 0.556 | 0.781 | 0.649 | 20 |
| at_risk_original_gt005 | naive_persistence_baseline | current_low_canopy_score | 0.573 | 0.978 | 0.484 | 0.647 | 1 |
| high_canopy_original_decline | canopy_trajectory_only | Logistic Regression L2 | 0.617 | 0.571 | 0.444 | 0.500 | 3 |
| high_canopy_original_decline | canopy_trajectory_plus_crw | Logistic Regression L2 | 0.499 | 0.429 | 0.375 | 0.400 | 4 |
| high_canopy_original_decline | canopy_trajectory_plus_crw_plus_habitat | Logistic Regression L2 | 0.518 | 0.429 | 0.333 | 0.375 | 4 |
| high_canopy_original_decline | canopy_trajectory_plus_habitat | Logistic Regression L2 | 0.657 | 0.571 | 0.444 | 0.500 | 3 |
| high_canopy_original_decline | crw_composite_only | Logistic Regression L2 | 0.218 | 0.429 | 0.188 | 0.261 | 4 |
| high_canopy_original_decline | existing_canopy_only | Random Forest | 0.602 | 0.143 | 0.333 | 0.200 | 6 |
| high_canopy_original_decline | habitat_only | Logistic Regression L2 | 0.656 | 0.571 | 0.667 | 0.615 | 3 |
| high_canopy_original_decline | naive_persistence_baseline | current_low_canopy_score | 0.194 | 0.857 | 0.250 | 0.387 | 1 |
| new_decline_transition | canopy_trajectory_only | Random Forest | 0.359 | 0.692 | 0.310 | 0.429 | 8 |
| new_decline_transition | canopy_trajectory_plus_crw | Logistic Regression L2 | 0.364 | 0.038 | 1.000 | 0.074 | 25 |
| new_decline_transition | canopy_trajectory_plus_crw_plus_habitat | Logistic Regression L2 | 0.419 | 0.038 | 1.000 | 0.074 | 25 |
| new_decline_transition | canopy_trajectory_plus_habitat | Logistic Regression L2 | 0.386 | 0.577 | 0.326 | 0.417 | 11 |
| new_decline_transition | crw_composite_only | Logistic Regression L2 | 0.195 | 0.308 | 0.083 | 0.131 | 18 |
| new_decline_transition | existing_canopy_only | Logistic Regression L2 | 0.407 | 0.308 | 0.471 | 0.372 | 18 |
| new_decline_transition | habitat_only | Random Forest | 0.141 | 0.577 | 0.129 | 0.211 | 11 |
| new_decline_transition | naive_persistence_baseline | current_low_canopy_score | 0.082 | 0.962 | 0.126 | 0.223 | 1 |
| original_decline | canopy_trajectory_only | Logistic Regression L2 | 0.809 | 0.948 | 0.743 | 0.833 | 7 |
| original_decline | canopy_trajectory_plus_crw | Logistic Regression L2 | 0.848 | 0.642 | 0.819 | 0.720 | 48 |
| original_decline | canopy_trajectory_plus_crw_plus_habitat | Logistic Regression L2 | 0.854 | 0.672 | 0.811 | 0.735 | 44 |
| original_decline | canopy_trajectory_plus_habitat | Logistic Regression L2 | 0.841 | 0.955 | 0.744 | 0.837 | 6 |
| original_decline | crw_composite_only | Logistic Regression L2 | 0.852 | 0.619 | 0.783 | 0.692 | 51 |
| original_decline | existing_canopy_only | Random Forest | 0.879 | 0.739 | 0.818 | 0.776 | 35 |
| original_decline | habitat_only | Random Forest | 0.889 | 0.821 | 0.809 | 0.815 | 24 |
| original_decline | naive_persistence_baseline | current_low_canopy_score | 0.893 | 0.858 | 0.732 | 0.790 | 19 |

## Diagnostic Answers

- Do trajectory features improve original broad decline prediction?
  For `original_decline`, `canopy_trajectory_only` did not improve over `existing_canopy_only` by PR-AUC (0.809 vs 0.879; delta -0.069).
- Do trajectory features improve at-risk or transition-oriented prediction?
  For `at_risk_original_gt005`, `canopy_trajectory_only` improved over `existing_canopy_only` by PR-AUC (0.650 vs 0.602; delta +0.047).
  For `new_decline_transition`, `canopy_trajectory_only` did not improve over `existing_canopy_only` by PR-AUC (0.359 vs 0.407; delta -0.047).
  For `actionable_decline_drop`, `canopy_trajectory_only` did not improve over `existing_canopy_only` by PR-AUC (0.533 vs 0.579; delta -0.046).
- Do improvements mainly strengthen persistence-based risk-state screening?
  Interpret gains cautiously: trajectory features use current and recent canopy states, so improvements mostly refine persistence/risk-state screening unless they hold for stricter transition and actionable labels.
- Do trajectory features reduce false negatives for actionable decline?
  For actionable decline, canopy trajectory features had 6 false negatives versus 1 for the naive persistence score.
- Do CRW and/or habitat features still add value after trajectory features are included?
  For `at_risk_original_gt005`, `canopy_trajectory_plus_crw` did not improve over `canopy_trajectory_only` by PR-AUC (0.638 vs 0.650; delta -0.011).
  For `at_risk_original_gt005`, `canopy_trajectory_plus_habitat` improved over `canopy_trajectory_only` by PR-AUC (0.671 vs 0.650; delta +0.021).
  For `actionable_decline_drop`, `canopy_trajectory_plus_crw_plus_habitat` did not improve over `canopy_trajectory_only` by PR-AUC (0.477 vs 0.533; delta -0.056).