# Rare-Event Alert Learning Report

## Purpose

This experiment tests whether rare transition/actionable kelp decline events can be detected more sensitively using training-only resampling, class weighting, hard-negative sampling, and validation-selected thresholds. It does not create new ecological events, does not resample validation or test rows, and does not replace the existing V1/V2 workflows.

## Event-Count Diagnosis

| target_definition | target_column | split | rows | positive_events | positive_rate | cells_with_positive_events | years_with_positive_events | filter_column |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_drop | actionable_decline_drop_next | train | 1400 | 395 | 0.282143 | 49 | 27 |  |
| actionable_drop | actionable_decline_drop_next | validation | 200 | 41 | 0.205000 | 32 | 4 |  |
| actionable_drop | actionable_decline_drop_next | test | 200 | 34 | 0.170000 | 31 | 3 |  |
| actionable_drop | actionable_decline_drop_next | all | 1800 | 470 | 0.261111 | 50 | 34 |  |
| new_transition | new_decline_event_next | train | 1400 | 270 | 0.192857 | 50 | 22 |  |
| new_transition | new_decline_event_next | validation | 200 | 21 | 0.105000 | 18 | 3 |  |
| new_transition | new_decline_event_next | test | 200 | 26 | 0.130000 | 25 | 3 |  |
| new_transition | new_decline_event_next | all | 1800 | 317 | 0.176111 | 50 | 28 |  |
| at_risk_original | decline_event_next | train | 955 | 258 | 0.270157 | 47 | 22 | at_risk_original_eligible |
| at_risk_original | decline_event_next | validation | 111 | 38 | 0.342342 | 20 | 4 | at_risk_original_eligible |
| at_risk_original | decline_event_next | test | 93 | 45 | 0.483871 | 23 | 4 | at_risk_original_eligible |
| at_risk_original | decline_event_next | all | 1159 | 341 | 0.294219 | 47 | 30 | at_risk_original_eligible |

## Hard-Negative Strategy

Hard negatives are non-event rows that still represent plausible decline-risk cases: canopy-present rows, rows with recent decline or instability, and rows with thermal or trajectory risk. These are more informative than easy negatives because they resemble rows that might plausibly trigger an alert.

| target_definition | split | hard_negative_rule | negative_rows | hard_negative_rows | hard_negative_share_of_negatives | status |
| --- | --- | --- | --- | --- | --- | --- |
| actionable_drop | train | hard_negative_canopy_present | 1005 | 560 | 0.557214 | computed |
| actionable_drop | train | hard_negative_recent_decline | 1005 | 595 | 0.592040 | computed |
| actionable_drop | train | hard_negative_thermal_or_trajectory_risk | 1005 | 775 | 0.771144 | computed |
| actionable_drop | train | hard_negative_any | 1005 | 938 | 0.933333 | computed |
| new_transition | train | hard_negative_canopy_present | 1130 | 738 | 0.653097 | computed |
| new_transition | train | hard_negative_recent_decline | 1130 | 589 | 0.521239 | computed |
| new_transition | train | hard_negative_thermal_or_trajectory_risk | 1130 | 802 | 0.709735 | computed |
| new_transition | train | hard_negative_any | 1130 | 1072 | 0.948673 | computed |
| at_risk_original | train | hard_negative_canopy_present | 697 | 697 | 1.000000 | computed |
| at_risk_original | train | hard_negative_recent_decline | 697 | 253 | 0.362984 | computed |
| at_risk_original | train | hard_negative_thermal_or_trajectory_risk | 697 | 429 | 0.615495 | computed |
| at_risk_original | train | hard_negative_any | 697 | 697 | 1.000000 | computed |

## Main Results

### actionable_drop

- Best PR-AUC: `canopy_trajectory_plus_OISST_plus_habitat` / `Logistic Regression` / `positive_oversampling_train_only` (pr_auc = `0.576`, precision = `0.538`, recall = `0.618`, FN = `13`)
- Best F2: `canopy_only` / `Logistic Regression` / `class_weighted_threshold_tuned` (f2 = `0.764`, precision = `0.463`, recall = `0.912`, FN = `3`)
- Best recall with precision >= 0.40: `canopy_only` / `Random Forest` / `class_weighted_threshold_tuned` (recall = `0.971`, precision = `0.407`, recall = `0.971`, FN = `1`)
- Lowest false negatives: `canopy_trajectory_plus_OISST_plus_habitat` / `Random Forest` / `class_weighted_threshold_tuned` (false_negatives = `0.000`, precision = `0.386`, recall = `1.000`, FN = `0`)
- Best top-k alert result: `canopy_trajectory_plus_CRW` / `LightGBM` / `positive_oversampling_train_only` / `top_10_cell_years` (mean recall = `0.782`, mean precision = `0.375`, captured = `15`)

### new_transition

- Best PR-AUC: `canopy_trajectory_plus_OISST_plus_habitat` / `XGBoost` / `positive_oversampling_train_only` (pr_auc = `0.482`, precision = `0.667`, recall = `0.231`, FN = `20`)
- Best F2: `canopy_trajectory_plus_OISST_plus_habitat` / `XGBoost` / `positive_oversampling_threshold_tuned` (f2 = `0.676`, precision = `0.309`, recall = `0.962`, FN = `1`)
- Best recall with precision >= 0.40: `canopy_trajectory` / `LightGBM` / `positive_oversampling_train_only` (recall = `0.577`, precision = `0.556`, recall = `0.577`, FN = `11`)
- Lowest false negatives: `canopy_trajectory_plus_OISST_plus_habitat` / `LightGBM` / `class_weighted_threshold_tuned` (false_negatives = `0.000`, precision = `0.234`, recall = `1.000`, FN = `0`)
- Best top-k alert result: `canopy_trajectory_plus_OISST_plus_habitat` / `LightGBM` / `class_weighted` / `top_10_cell_years` (mean recall = `0.759`, mean precision = `0.375`, captured = `15`)

### at_risk_original

- Best PR-AUC: `bathymetry_habitat` / `Logistic Regression` / `class_weighted` (pr_auc = `0.764`, precision = `0.786`, recall = `0.489`, FN = `23`)
- Best F2: `bathymetry_habitat` / `Random Forest` / `class_weighted_threshold_tuned` (f2 = `0.837`, precision = `0.558`, recall = `0.956`, FN = `2`)
- Best recall with precision >= 0.40: `canopy_trajectory_plus_CRW_plus_habitat` / `Logistic Regression` / `positive_oversampling_threshold_tuned` (recall = `1.000`, precision = `0.489`, recall = `1.000`, FN = `0`)
- Lowest false negatives: `canopy_trajectory_plus_CRW_plus_habitat` / `Logistic Regression` / `positive_oversampling_threshold_tuned` (false_negatives = `0.000`, precision = `0.489`, recall = `1.000`, FN = `0`)
- Best top-k alert result: `bathymetry_habitat` / `XGBoost` / `hard_negative_threshold_tuned` / `top_10_cell_years` (mean recall = `0.748`, mean precision = `0.700`, captured = `84`)

## Comparison to Claim Gates

- Actionable-drop Gate 3 interpretation after rare-event learning: `transition_recall_oriented_sensitivity_only`.
- New-transition Gate 3 interpretation after rare-event learning: `transition_early_warning_not_supported`.
- Overall Gate 3 interpretation: `transition_recall_oriented_sensitivity_only`.

These labels are diagnostic. They should not be interpreted as operational early-warning success unless precision, recall, F2, PR-AUC, and false-negative reduction are jointly acceptable.

## Precision-Recall Tradeoff

Rare-event learning improved sensitivity for some model/target combinations, especially through class weighting and threshold tuning. Where recall increased without PR-AUC improvement over canopy-only baselines, the result is best described as recall-oriented sensitivity improvement. Top-k gains are alert-prioritization support, not full early-warning success.

## Recommended Next Step

Wave exposure remains the next priority because rare-event learning alone does not establish robust transition/actionable early-warning support.
