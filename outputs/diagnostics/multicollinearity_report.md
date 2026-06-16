# Multicollinearity Diagnostic Report

## Purpose

This diagnostic checks whether the Version 1 model features contain strong pairwise or multivariate redundancy. It is intended to support cautious interpretation of linear coefficients, SHAP summaries, and feature-importance narratives. It does not change the modeling dataset or retrain the main models.

## Data and Feature Scope

- Modeling rows evaluated: `1800`.
- Numeric model features evaluated in the combined correlation matrix: `30`.
- High-correlation threshold: `abs(r) >= 0.80`.
- VIF caution thresholds: moderate `>= 5.0`, high `>= 10.0`.
- Heatmap: `outputs/figures/multicollinearity_correlation_heatmap.png`.

Categorical region variables are not included in the numeric correlation matrix. The VIF table is calculated separately for each model feature set using numeric predictors only.

## Main Findings

- High-correlation feature pairs: `19`.
- High-VIF rows: `37`.
- Moderate-VIF rows: `9`.

### Top High-Correlation Pairs

| feature_1 | feature_2 | correlation | abs_correlation |
| --- | --- | --- | --- |
| count_cells_historic_footprint | historical_footprint_area_m2 | 1.000 | 1.000 |
| count_cells_historic_footprint | count_cells_no_clouds | 1.000 | 1.000 |
| count_cells_no_clouds | historical_footprint_area_m2 | 1.000 | 1.000 |
| center_lat | center_lon | -0.996 | 0.996 |
| hot_days_p90 | hot_days_p95 | 0.963 | 0.963 |
| count_cells_kelp | kelp_area_m2 | 0.910 | 0.910 |
| lag1_annual_mean_sst_anomaly | lag1_hot_days_p90 | 0.886 | 0.886 |
| annual_mean_sst_anomaly | hot_days_p90 | 0.882 | 0.882 |
| summer_mean_beuti | summer_mean_cuti | 0.870 | 0.870 |
| annual_mean_beuti | spring_mean_beuti | 0.869 | 0.869 |
| annual_max_sst | annual_mean_sst | 0.869 | 0.869 |
| annual_max_sst_anomaly | hot_days_p90 | 0.858 | 0.858 |

### Top VIF Values

| feature_set | feature | vif | r_squared_with_other_features |
| --- | --- | --- | --- |
| canopy_noaa | relative_canopy | inf | 1.000 |
| canopy_noaa | count_cells_historic_footprint | inf | 1.000 |
| canopy_noaa | historical_footprint_area_m2 | inf | 1.000 |
| canopy_noaa | lag1_relative_canopy | inf | 1.000 |
| canopy_noaa | relative_canopy_change_lag1 | inf | 1.000 |
| canopy_noaa | count_cells_no_clouds | 1227.658 | 0.999 |
| canopy_noaa | annual_mean_beuti | 271.844 | 0.996 |
| canopy_noaa | annual_mean_sst | 218.058 | 0.995 |
| canopy_noaa | annual_max_sst | 201.298 | 0.995 |
| canopy_noaa | center_lat | 164.971 | 0.994 |
| canopy_noaa | center_lon | 141.266 | 0.993 |
| canopy_noaa | annual_max_sst_anomaly | 120.336 | 0.992 |
| canopy_noaa | annual_mean_sst_anomaly | 78.309 | 0.987 |
| canopy_noaa | annual_mean_cuti | 77.498 | 0.987 |
| canopy_noaa | beuti_anomaly | 71.093 | 0.986 |

### Condition Numbers by Feature Set

| feature_set | condition_number_feature_set |
| --- | --- |
| canopy_noaa | inf |
| canopy_only | inf |
| oisst_only | 45.875 |

## Interpretation

Several canopy-size variables are structurally related, especially `kelp_area_m2`, `count_cells_kelp`, and `relative_canopy`. Several NOAA thermal variables are also expected to be correlated because annual mean, maximum, anomaly, and hot-day metrics are derived from the same OISST time series. CUTI and BEUTI seasonal and anomaly summaries can likewise be redundant within a small spatial domain.

This does not invalidate the tree-based screening models, but it means coefficient-level interpretation from Logistic Regression and feature-level importance narratives should be treated cautiously. For paper framing, the safer interpretation is feature-group-level evidence: canopy state, OISST thermal exposure, CUTI upwelling proxy, and BEUTI nitrate-flux proxy, rather than isolated claims about one highly correlated predictor.

## Recommended Use

- Keep Random Forest, XGBoost, and LightGBM as prediction benchmarks because tree models can tolerate correlated predictors, while still sharing importance across redundant variables.
- Use Logistic Regression mainly as a transparent baseline, not as definitive evidence about individual variable effects when VIF is high.
- Prefer grouped SHAP or grouped feature-set ablation over single-feature causal wording.
- Consider a reduced predictor set in sensitivity analysis, selecting one representative from each highly correlated group.
