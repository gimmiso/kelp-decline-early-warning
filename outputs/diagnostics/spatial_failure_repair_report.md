# Spatial Failure Repair Report

## Purpose

This diagnostic treats spatial-validation weakness as a model-debugging signal rather than only a limitation.
It tests whether replacing or supplementing absolute predictors with cell-relative, anomaly-based, and percentile-style features improves latitude-band spatial transfer.

## Spatial Failure Anatomy

- Target with largest PR-AUC loss under existing spatial validation: `at_risk_original`
- Target with largest recall loss under existing spatial validation: `new_transition`
- Worst three-band held-out band by mean best-row PR-AUC: `band_3_north`

Best existing model row by target and held-out band:

| heldout_band | target_definition | feature_family | model | positive_rate | pr_auc | recall | precision | false_negatives | failure_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| band_1_south | actionable_drop | trajectory_crw_habitat | LightGBM | 0.304 | 0.706 | 0.333 | 0.775 | 124 | threshold_failure |
| band_1_south | at_risk_original | trajectory_crw_habitat_wave | LightGBM | 0.234 | 0.553 | 0.221 | 0.793 | 81 | threshold_failure |
| band_1_south | new_transition | trajectory_crw_habitat_wave | Random Forest | 0.175 | 0.488 | 0.150 | 0.842 | 91 | threshold_failure |
| band_1_south | original_decline | trajectory_crw_habitat | LightGBM | 0.248 | 0.528 | 0.533 | 0.497 | 71 | no_major_failure_in_best_row |
| band_2_central | actionable_drop | trajectory_crw_habitat_wave | LightGBM | 0.264 | 0.586 | 0.467 | 0.577 | 81 | threshold_failure |
| band_2_central | at_risk_original | trajectory_crw_habitat_wave | LightGBM | 0.372 | 0.602 | 0.218 | 0.745 | 147 | threshold_failure |
| band_2_central | new_transition | trajectory_crw_habitat_wave | XGBoost | 0.198 | 0.592 | 0.605 | 0.476 | 45 | no_major_failure_in_best_row |
| band_2_central | original_decline | trajectory_crw_habitat_wave | Random Forest | 0.401 | 0.672 | 0.385 | 0.730 | 142 | threshold_failure |
| band_3_north | actionable_drop | trajectory_crw_habitat_wave | Random Forest | 0.216 | 0.757 | 0.250 | 0.805 | 99 | threshold_failure |
| band_3_north | at_risk_original | trajectory_crw_habitat | Random Forest | 0.234 | 0.358 | 0.082 | 0.800 | 45 | ranking_and_threshold_failure |
| band_3_north | new_transition | trajectory_crw_habitat_wave | Random Forest | 0.157 | 0.304 | 0.000 | 0.000 | 96 | ranking_and_threshold_failure |
| band_3_north | original_decline | naive_persistence | current_low_canopy_score | 0.449 | 0.738 | 1.000 | 0.449 | 0 | no_major_failure_in_best_row |

## Feature Distribution Shift

Largest three-band standardized mean shifts:

| heldout_band | feature | standardized_mean_difference | median_difference | missingness_difference |
| --- | --- | --- | --- | --- |
| band_3_north | annual_mean_sst_crw5km | -2.122 | -1.577 | 0.000 |
| band_1_south | winter_max_wave_height_cdip_model | -1.550 | -2.376 | 0.000 |
| band_1_south | annual_mean_sst_crw5km | 1.480 | 0.925 | 0.000 |
| band_2_central | slope_mean | 1.462 | 0.044 | 0.000 |
| band_3_north | relative_canopy | -1.223 | -0.123 | 0.000 |
| band_3_north | presence_frequency_5yr_t | -1.163 | -0.200 | 0.000 |
| band_3_north | instability_score_5yr_t | -1.117 | -0.049 | 0.000 |
| band_2_central | relative_canopy | 0.992 | 0.142 | 0.000 |
| band_2_central | presence_frequency_5yr_t | 0.951 | 0.000 | 0.000 |
| band_3_north | slope_mean | -0.930 | -0.008 | 0.000 |
| band_2_central | mean_depth_m | 0.756 | 19.259 | 0.000 |
| band_2_central | winter_max_wave_height_cdip_model | 0.736 | 1.595 | 0.000 |
| band_3_north | mean_depth_m | -0.643 | -6.720 | 0.000 |
| band_1_south | slope_mean | -0.622 | -0.005 | 0.000 |
| band_3_north | winter_max_wave_height_cdip_model | 0.555 | 1.680 | 0.000 |

## Repair Features

- Canopy repair features are expanding or rolling within-cell summaries available up to year `t`.
- Environmental repair features are training-period local z-scores or percentiles from existing CRW and CDIP wave variables.
- These features do not add new external data sources.
- The default run uses Logistic Regression and Random Forest for runtime stability; optional boosting models can be enabled with `--include-boosting`.

- Repair features built: `22`
- Maximum repair-feature missingness: `0.333`

## Spatial Repair Results

Best three-band repair model by mean spatial PR-AUC:

| target_definition | feature_family | model | mean_pr_auc | min_pr_auc | mean_recall | min_recall | mean_precision | mean_f2 | total_false_negatives | valid_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_drop | relative_dynamic_plus_habitat_context | Random Forest | 0.607 | 0.518 | 0.451 | 0.364 | 0.616 | 0.474 | 255 | 3 |
| at_risk_original | relative_dynamic_combined | Random Forest | 0.501 | 0.414 | 0.281 | 0.082 | 0.513 | 0.294 | 235 | 3 |

Best three-band repair model by worst-band PR-AUC:

| target_definition | feature_family | model | mean_pr_auc | min_pr_auc | mean_recall | min_recall | mean_f2 | total_false_negatives | valid_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_drop | relative_dynamic_combined | Random Forest | 0.593 | 0.538 | 0.408 | 0.386 | 0.436 | 278 | 3 |
| at_risk_original | relative_dynamic_plus_habitat_context | Random Forest | 0.487 | 0.429 | 0.239 | 0.102 | 0.259 | 251 | 3 |

Top stability-ranked three-band repair models for primary targets:

| target_definition | feature_family | model | stability_score | mean_pr_auc | min_pr_auc | mean_recall | min_recall | pr_auc_std | recall_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_drop | relative_dynamic_combined | Random Forest | 0.638 | 0.593 | 0.538 | 0.408 | 0.386 | 0.039 | 0.020 |
| at_risk_original | relative_dynamic_plus_habitat_context | Random Forest | 0.454 | 0.487 | 0.429 | 0.239 | 0.102 | 0.056 | 0.108 |

Precision-floor threshold recalibration highlights aggregated across three-band holdouts:

Precision floors are enforced during spatial-training threshold selection; held-out precision can still fall below the requested floor.

| target_definition | precision_floor | feature_family | model | mean_recall | mean_precision | min_precision | mean_f2 | total_false_negatives | total_false_positives | valid_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_drop | 0.300 | canopy_relative | Logistic Regression | 0.992 | 0.326 | 0.293 | 0.702 | 4.000 | 963.000 | 3 |
| actionable_drop | 0.400 | trajectory_absolute | Logistic Regression | 0.947 | 0.419 | 0.314 | 0.749 | 27.000 | 664.000 | 3 |
| actionable_drop | 0.500 | canopy_relative | Random Forest | 0.797 | 0.466 | 0.345 | 0.691 | 95.000 | 472.000 | 3 |
| at_risk_original | 0.300 | canopy_relative | Logistic Regression | 0.939 | 0.305 | 0.237 | 0.649 | 26.000 | 695.000 | 3 |
| at_risk_original | 0.400 | relative_dynamic_combined | Logistic Regression | 0.720 | 0.381 | 0.275 | 0.593 | 83.000 | 431.000 | 3 |
| at_risk_original | 0.500 | thermal_relative | Random Forest | 0.616 | 0.382 | 0.296 | 0.522 | 152.000 | 321.000 | 3 |

## Interpretation

Spatial repair features should be interpreted as internal robustness diagnostics, not as proof of external spatial generalization.
If repair features improve PR-AUC but recall remains low, the result supports better ranking under spatial transfer but not operational early warning.
If recall improves only with low precision, the result supports recall-oriented alert sensitivity rather than robust spatially transferable prediction.

## Next Step

The project can stop adding feature layers for the current V1 portfolio and frame external region validation as the next methodological step.
