# Wave Exposure Feature Report

## Purpose

This report adds a physical disturbance / wave-exposure layer inspired by kelp persistence studies.
The goal is to test whether wave-related covariates improve transition/actionable kelp decline prediction beyond canopy persistence, CRW thermal exposure, bathymetry/habitat, and canopy trajectory features.

## CDIP-Specific Access Diagnostic

- CDIP data access documentation reachable: `True`.
- CDIP MOP THREDDS catalog reachable: `True`.
- CDIP MOP OPeNDAP test URL: `https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/model/MOP_alongshore/SN220_hindcast.nc`.
- CDIP MOP `waveHs` accessible: `True`.
- CDIP MOP test time coverage: `2000-01-01 00:00:00` to `2025-03-31 23:00:00`.
- CDIP buoy historic observation URL tested: `https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/067p1/067p1_historic.nc`.
- CDIP buoy `waveHs` accessible: `True`.
- Spatial modeled product close to the kelp-persistence literature accessible: `True`.

## Data Source Decision

- Selected working source: `CDIP MOP alongshore modeled wave products`.
- CDIP MOP is closer to the kelp persistence literature than NDBC because it is a modeled alongshore nearshore wave product rather than a generic nearest-buoy proxy.
- NDBC remains fallback code only and was not selected when CDIP MOP access succeeded.
- CDIP MOP hindcast data begin in 2000 in the tested files, so 1989-1999 wave rows are missing and handled by model imputation.

Candidate source inventory:

| source_name | access_method | compatibility_with_10km_kelp_cells | selected_for_implementation | limitations |
| --- | --- | --- | --- | --- |
| CDIP MOP alongshore modeled wave products | CDIP THREDDS / OPeNDAP MOP_alongshore hindcast NetCDF | Nearest modeled alongshore point to each retained 10 km cell centroid | True | Hindcast begins in 2000, so 1989-1999 model rows have missing CDIP MOP wave values |
| CDIP buoy observations | CDIP THREDDS / OPeNDAP archived station historic NetCDF | Nearest CDIP station proxy if modeled products fail | False | Less literature-matched than MOP alongshore modeled exposure; station aggregation can be complex |
| NDBC buoy observations | Annual gzipped standard meteorological text files | Nearest-buoy proxy only | False | Not equivalent to CDIP nearshore wave-propagation exposure |
| ERA5 wave reanalysis | Copernicus Climate Data Store API | Nearest-grid or coastal-buffer proxy | False | Requires CDS credentials and may miss reef-scale nearshore transformation |

## Feature Construction

- Response unit: retained 10 km Kelpwatch cell-year.
- Spatial matching: each cell centroid is assigned to the nearest sampled CDIP MOP alongshore hindcast point.
- Wave variable: `waveHs`, significant wave height in meters.
- Winter definition: Dec(t-1), Jan(t), Feb(t).
- Interaction features are exploratory and should not be interpreted as mechanistic proof.

## Diagnostics

- Retained cells with wave features: `50`
- Cell-year wave rows: `1800`
- Year coverage: `1989-2024`
- Unique CDIP MOP points used: `16`
- Mean nearest MOP-point distance: `9.5` km
- Maximum nearest MOP-point distance: `26.4` km

Feature missingness:

| feature | value |
| --- | --- |
| winter_max_wave_height_cdip_model | 0.3055555555555556 |
| winter_mean_wave_height_cdip_model | 0.3055555555555556 |
| annual_max_wave_height_cdip_model | 0.3055555555555556 |
| annual_mean_wave_height_cdip_model | 0.3055555555555556 |
| lag1_winter_max_wave_height_cdip_model | 0.3333333333333333 |
| lag1_winter_mean_wave_height_cdip_model | 0.3333333333333333 |
| wave_height_anomaly_cdip_model | 0.3055555555555556 |
| storm_month_count_cdip_model | 0.3055555555555556 |
| distance_to_cdip_model_point_km | 0.0 |

## Model Results

- Computed model-comparison rows: `176`

Best result per target:

| target_definition | feature_family | model | pr_auc | recall | precision | f2 | false_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_drop | trajectory_only | Random Forest | 0.533 | 0.824 | 0.444 | 0.704 | 6 |
| at_risk_original | wave_only | Random Forest | 0.806 | 0.533 | 0.857 | 0.577 | 21 |
| new_transition | trajectory_plus_wave | Random Forest | 0.427 | 0.115 | 1.000 | 0.140 | 23 |
| original_decline | habitat_only | XGBoost | 0.903 | 0.716 | 0.857 | 0.741 | 38 |

Best row by target and feature family:

| target_definition | feature_family | model | pr_auc | recall | precision | f2 | false_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_drop | crw_only | Logistic Regression | 0.184 | 0.676 | 0.240 | 0.496 | 11 |
| actionable_drop | crw_plus_habitat_plus_wave | XGBoost | 0.285 | 0.912 | 0.228 | 0.570 | 3 |
| actionable_drop | crw_plus_wave | XGBoost | 0.283 | 0.912 | 0.237 | 0.581 | 3 |
| actionable_drop | habitat_only | Random Forest | 0.173 | 1.000 | 0.170 | 0.506 | 0 |
| actionable_drop | habitat_plus_wave | Random Forest | 0.383 | 0.794 | 0.221 | 0.523 | 7 |
| actionable_drop | trajectory_only | Random Forest | 0.533 | 0.824 | 0.444 | 0.704 | 6 |
| actionable_drop | trajectory_plus_crw_plus_habitat_plus_wave | Logistic Regression | 0.448 | 0.412 | 0.560 | 0.435 | 20 |
| actionable_drop | trajectory_plus_crw_plus_wave | XGBoost | 0.487 | 0.794 | 0.458 | 0.692 | 7 |
| actionable_drop | trajectory_plus_oisst_plus_habitat_plus_wave | Logistic Regression | 0.508 | 0.382 | 0.619 | 0.414 | 21 |
| actionable_drop | trajectory_plus_wave | Random Forest | 0.498 | 0.765 | 0.473 | 0.681 | 8 |
| actionable_drop | wave_only | XGBoost | 0.333 | 0.941 | 0.176 | 0.503 | 2 |
| at_risk_original | crw_only | Logistic Regression | 0.549 | 0.333 | 0.517 | 0.359 | 30 |
| at_risk_original | crw_plus_habitat_plus_wave | Logistic Regression | 0.637 | 0.289 | 0.650 | 0.325 | 32 |
| at_risk_original | crw_plus_wave | Logistic Regression | 0.638 | 0.289 | 0.619 | 0.323 | 32 |
| at_risk_original | habitat_only | Logistic Regression | 0.764 | 0.556 | 0.781 | 0.590 | 20 |
| at_risk_original | habitat_plus_wave | Random Forest | 0.661 | 0.267 | 0.632 | 0.302 | 33 |
| at_risk_original | trajectory_only | Logistic Regression | 0.650 | 0.844 | 0.667 | 0.802 | 7 |
| at_risk_original | trajectory_plus_crw_plus_habitat_plus_wave | Logistic Regression | 0.655 | 0.244 | 0.786 | 0.284 | 34 |
| at_risk_original | trajectory_plus_crw_plus_wave | Logistic Regression | 0.656 | 0.244 | 0.786 | 0.284 | 34 |
| at_risk_original | trajectory_plus_oisst_plus_habitat_plus_wave | Logistic Regression | 0.646 | 0.222 | 0.667 | 0.256 | 35 |
| at_risk_original | trajectory_plus_wave | Random Forest | 0.695 | 0.489 | 0.667 | 0.516 | 23 |
| at_risk_original | wave_only | Random Forest | 0.806 | 0.533 | 0.857 | 0.577 | 21 |
| new_transition | crw_only | Logistic Regression | 0.195 | 0.308 | 0.083 | 0.200 | 18 |
| new_transition | crw_plus_habitat_plus_wave | Logistic Regression | 0.134 | 0.038 | 0.100 | 0.044 | 25 |
| new_transition | crw_plus_wave | XGBoost | 0.146 | 0.038 | 0.077 | 0.043 | 25 |
| new_transition | habitat_only | Random Forest | 0.138 | 0.192 | 0.104 | 0.164 | 21 |
| new_transition | habitat_plus_wave | Random Forest | 0.211 | 0.115 | 0.600 | 0.138 | 23 |
| new_transition | trajectory_only | XGBoost | 0.405 | 0.538 | 0.378 | 0.496 | 12 |
| new_transition | trajectory_plus_crw_plus_habitat_plus_wave | XGBoost | 0.319 | 0.154 | 0.222 | 0.164 | 22 |
| new_transition | trajectory_plus_crw_plus_wave | XGBoost | 0.269 | 0.115 | 0.375 | 0.134 | 23 |
| new_transition | trajectory_plus_oisst_plus_habitat_plus_wave | LightGBM | 0.267 | 0.154 | 0.308 | 0.171 | 22 |
| new_transition | trajectory_plus_wave | Random Forest | 0.427 | 0.115 | 1.000 | 0.140 | 23 |
| new_transition | wave_only | Random Forest | 0.227 | 0.077 | 0.400 | 0.092 | 24 |
| original_decline | crw_only | Logistic Regression | 0.852 | 0.619 | 0.783 | 0.646 | 51 |
| original_decline | crw_plus_habitat_plus_wave | Logistic Regression | 0.793 | 0.515 | 0.793 | 0.554 | 65 |
| original_decline | crw_plus_wave | Logistic Regression | 0.790 | 0.515 | 0.793 | 0.554 | 65 |
| original_decline | habitat_only | XGBoost | 0.903 | 0.716 | 0.857 | 0.741 | 38 |
| original_decline | habitat_plus_wave | Logistic Regression | 0.743 | 0.642 | 0.735 | 0.658 | 48 |
| original_decline | trajectory_only | Logistic Regression | 0.809 | 0.948 | 0.743 | 0.898 | 7 |
| original_decline | trajectory_plus_crw_plus_habitat_plus_wave | XGBoost | 0.788 | 0.410 | 0.764 | 0.452 | 79 |
| original_decline | trajectory_plus_crw_plus_wave | Logistic Regression | 0.783 | 0.455 | 0.803 | 0.498 | 73 |
| original_decline | trajectory_plus_oisst_plus_habitat_plus_wave | XGBoost | 0.767 | 0.440 | 0.766 | 0.481 | 75 |
| original_decline | trajectory_plus_wave | Random Forest | 0.754 | 0.403 | 0.740 | 0.443 | 80 |
| original_decline | wave_only | Random Forest | 0.842 | 0.910 | 0.735 | 0.869 | 12 |

Best actionable-drop wave-related result:

| target_definition | feature_family | model | pr_auc | recall | precision | f2 | false_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_drop | trajectory_plus_oisst_plus_habitat_plus_wave | Logistic Regression | 0.508 | 0.382 | 0.619 | 0.414 | 21 |

Lowest false-negative actionable-drop wave-related result:

| target_definition | feature_family | model | pr_auc | recall | precision | f2 | false_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_drop | habitat_plus_wave | XGBoost | 0.365 | 0.941 | 0.239 | 0.593 | 2 |

## Early-Warning Interpretation

- Broad risk-state screening: For `original_decline`, `crw_plus_wave` did not improve over `crw_only` by PR-AUC (0.790 vs 0.852; delta -0.062).
- At-risk screening: For `at_risk_original`, `trajectory_plus_wave` improved over `trajectory_only` by PR-AUC (0.695 vs 0.650; delta +0.045).
- New-transition target: For `new_transition`, `trajectory_plus_wave` improved over `trajectory_only` by PR-AUC (0.427 vs 0.405; delta +0.022).
- Actionable-drop target: For `actionable_drop`, `trajectory_plus_wave` did not improve over `trajectory_only` by PR-AUC (0.498 vs 0.533; delta -0.035).
- Gate 3 should only change after rerunning integrated results and claim gates. If gains mainly improve recall/F2 with limited precision, the correct interpretation remains recall-oriented alert sensitivity.

## Required Explicit Answers

- Did we successfully use CDIP modeled wave exposure? `Yes`.
- If not, did we use CDIP buoy observations? `Not needed; CDIP buoy access was tested and remains fallback`.
- If not, why did we fall back to NDBC? `No NDBC fallback was used because CDIP MOP access succeeded`.
- How close is the chosen wave proxy to the kelp persistence literature? `Closer than NDBC because it uses CDIP modeled alongshore nearshore significant wave height, but still aggregated to 10 km Kelpwatch cell-year units and not identical to Cavanaugh et al.'s exact extraction`.

## Limitations

- CDIP MOP hindcast starts in 2000 for tested files, so early model-period wave values are missing.
- Nearest sampled MOP-point matching approximates, but does not exactly reproduce, the original kelp-persistence extraction.
- Wave exposure may act through interactions with canopy condition, SST stress, habitat, or grazing pressure.
- Lack of improvement would not prove waves are unimportant; it would only limit this particular proxy layer.
- Large raw wave data are not committed; compact extracted caches remain under ignored external data paths.

## Next Steps

- Densify the CDIP MOP metadata cache if nearest-point distances are too large.
- Test true spatial fragmentation if patch geometry becomes available.
- Add urchin/grazer pressure as a separate ecological case-study layer.
- Continue external/spatial validation before operational early-warning claims.
