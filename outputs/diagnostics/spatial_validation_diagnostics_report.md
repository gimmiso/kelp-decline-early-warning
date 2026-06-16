# Spatial Validation Diagnostics Report

## Purpose

This diagnostic checks whether kelp decline risk-screening performance is dependent on spatial autocorrelation among neighboring 10 km retained Kelpwatch cells.
The existing workflow already uses temporal train/validation/test splits; this layer asks whether models transfer across latitude-defined coastal bands within the same study domain.

## Retained Cell and Target Summary

- Retained cells: `50`
- Latitude range: `34.460` to `39.443`
- Longitude range: `-123.780` to `-120.272`
- Year range: `1989` to `2024`

Target event counts across the full model period:

| target_definition | positive_events |
| --- | --- |
| at_risk_original | 341 |
| actionable_drop | 470 |
| original_decline | 658 |
| new_transition | 317 |

## Spatial Fold Design

- `three_band_holdout`: retained cells are split into south, central, and north latitude bands; train on two bands and test on the held-out band.
- `five_band_holdout`: retained cells are split into five latitude bands; leave one band out at a time.
- Classification thresholds are selected using training bands only.
- The primary diagnostic is `three_band_holdout`; five-band results are reported as underpowered when held-out positives are too sparse.

## Event-Count Feasibility

- Five-band feasibility status: `usable_with_caution` (`0` underpowered target-band combinations out of `20`).

| heldout_band | target_definition | test_rows | positive_events | event_prevalence | feasibility |
| --- | --- | --- | --- | --- | --- |
| band_1_south | at_risk_original | 445 | 104 | 0.234 | usable |
| band_1_south | actionable_drop | 612 | 186 | 0.304 | usable |
| band_1_south | original_decline | 612 | 152 | 0.248 | usable |
| band_1_south | new_transition | 612 | 107 | 0.175 | usable |
| band_2_central | at_risk_original | 505 | 188 | 0.372 | usable |
| band_2_central | actionable_drop | 576 | 152 | 0.264 | usable |
| band_2_central | original_decline | 576 | 231 | 0.401 | usable |
| band_2_central | new_transition | 576 | 114 | 0.198 | usable |
| band_3_north | at_risk_original | 209 | 49 | 0.234 | usable |
| band_3_north | actionable_drop | 612 | 132 | 0.216 | usable |
| band_3_north | original_decline | 612 | 275 | 0.449 | usable |
| band_3_north | new_transition | 612 | 96 | 0.157 | usable |

## Results

- Computed fold-model rows: `544`

Best three-band summary row per target:

| target_definition | feature_family | mean_pr_auc | mean_recall | mean_precision | mean_f2 | total_false_negatives | valid_folds | underpowered_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_drop | trajectory_crw_habitat_wave | 0.676 | 0.339 | 0.716 | 0.373 | 310 | 3 | 0 |
| at_risk_original | trajectory_crw_habitat_wave | 0.493 | 0.167 | 0.624 | 0.195 | 274 | 3 | 0 |
| new_transition | trajectory_crw_habitat_wave | 0.461 | 0.252 | 0.439 | 0.251 | 232 | 3 | 0 |
| original_decline | trajectory_crw_habitat_wave | 0.625 | 0.444 | 0.629 | 0.464 | 369 | 3 | 0 |

Primary target summaries:

| target_definition | feature_family | mean_pr_auc | mean_recall | mean_precision | mean_f2 | total_false_negatives | valid_folds | underpowered_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_drop | trajectory_crw_habitat_wave | 0.676 | 0.339 | 0.716 | 0.373 | 310 | 3 | 0 |
| actionable_drop | trajectory_crw_habitat | 0.660 | 0.353 | 0.702 | 0.387 | 303 | 3 | 0 |
| actionable_drop | canopy_only | 0.577 | 0.701 | 0.523 | 0.614 | 152 | 3 | 0 |
| actionable_drop | canopy_trajectory | 0.576 | 0.637 | 0.535 | 0.603 | 162 | 3 | 0 |
| actionable_drop | naive_persistence | 0.166 | 0.998 | 0.261 | 0.634 | 1 | 3 | 0 |
| at_risk_original | trajectory_crw_habitat_wave | 0.493 | 0.167 | 0.624 | 0.195 | 274 | 3 | 0 |
| at_risk_original | trajectory_crw_habitat | 0.470 | 0.273 | 0.632 | 0.277 | 242 | 3 | 0 |
| at_risk_original | canopy_trajectory | 0.338 | 0.206 | 0.373 | 0.221 | 264 | 3 | 0 |
| at_risk_original | naive_persistence | 0.321 | 0.970 | 0.283 | 0.640 | 17 | 3 | 0 |
| at_risk_original | canopy_only | 0.311 | 0.281 | 0.283 | 0.270 | 262 | 3 | 0 |

## Comparison to Temporal Split

| target_definition | feature_family | best_temporal_pr_auc | spatial_mean_pr_auc | spatial_minus_temporal_pr_auc | best_temporal_recall | spatial_mean_recall | spatial_minus_temporal_recall | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_drop | canopy_only | 0.650 | 0.577 | -0.073 | 1.000 | 0.701 | -0.299 | spatial_holdout_moderately_lower |
| actionable_drop | canopy_trajectory | 0.576 | 0.576 | 0.001 | 1.000 | 0.637 | -0.363 | spatial_holdout_similar_to_temporal |
| actionable_drop | naive_persistence | 0.396 | 0.166 | -0.230 | 0.971 | 0.998 | 0.028 | spatial_holdout_substantially_lower |
| actionable_drop | trajectory_crw_habitat | 0.479 | 0.660 | 0.182 | 1.000 | 0.353 | -0.647 | spatial_holdout_similar_to_temporal |
| actionable_drop | trajectory_crw_habitat_wave | 0.448 | 0.676 | 0.227 | 0.794 | 0.339 | -0.455 | spatial_holdout_similar_to_temporal |
| at_risk_original | canopy_only | 0.897 | 0.311 | -0.585 | 1.000 | 0.281 | -0.719 | spatial_holdout_substantially_lower |
| at_risk_original | canopy_trajectory | 0.671 | 0.338 | -0.333 | 1.000 | 0.206 | -0.794 | spatial_holdout_substantially_lower |
| at_risk_original | naive_persistence | 0.807 | 0.321 | -0.486 | 0.978 | 0.970 | -0.008 | spatial_holdout_substantially_lower |
| at_risk_original | trajectory_crw_habitat | 0.655 | 0.470 | -0.186 | 1.000 | 0.273 | -0.727 | spatial_holdout_substantially_lower |
| at_risk_original | trajectory_crw_habitat_wave | 0.655 | 0.493 | -0.162 | 0.400 | 0.167 | -0.233 | spatial_holdout_substantially_lower |
| new_transition | canopy_only | 0.582 | 0.286 | -0.295 | 1.000 | 0.575 | -0.425 | spatial_holdout_substantially_lower |
| new_transition | canopy_trajectory | 0.610 | 0.305 | -0.305 | 1.000 | 0.161 | -0.839 | spatial_holdout_substantially_lower |
| new_transition | naive_persistence | 0.291 | 0.122 | -0.169 | 0.962 | 0.994 | 0.033 | spatial_holdout_substantially_lower |
| new_transition | trajectory_crw_habitat | 0.419 | 0.424 | 0.005 | 0.885 | 0.203 | -0.682 | spatial_holdout_similar_to_temporal |
| new_transition | trajectory_crw_habitat_wave | 0.319 | 0.461 | 0.142 | 0.154 | 0.252 | 0.098 | spatial_holdout_similar_to_temporal |
| original_decline | canopy_only | 0.897 | 0.468 | -0.429 | 0.896 | 0.532 | -0.363 | spatial_holdout_substantially_lower |
| original_decline | canopy_trajectory | 0.841 | 0.483 | -0.358 | 0.955 | 0.463 | -0.493 | spatial_holdout_substantially_lower |
| original_decline | naive_persistence | 0.893 | 0.523 | -0.370 | 0.858 | 0.534 | -0.324 | spatial_holdout_substantially_lower |
| original_decline | trajectory_crw_habitat | 0.880 | 0.612 | -0.269 | 0.769 | 0.506 | -0.263 | spatial_holdout_substantially_lower |
| original_decline | trajectory_crw_habitat_wave | 0.788 | 0.625 | -0.163 | 0.522 | 0.444 | -0.078 | spatial_holdout_substantially_lower |

## Interpretation

If spatial holdout performance is stable, the workflow shows some internal spatial transferability within the retained California cells.
If spatial performance drops or is unstable, the model is better interpreted as regional risk screening within the studied domain rather than robust spatially transferable early warning.
In this repository, spatial validation should be treated as a robustness diagnostic and not as external region validation.

## Limitations

- Only 50 retained cells are available.
- Spatial folds can have low positive event counts, especially for stricter transition/actionable labels.
- Latitude bands are an approximate coastal spatial blocking strategy.
- External region validation remains future work.
