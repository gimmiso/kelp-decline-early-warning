# SHAP Interpretation Report

## Models Interpreted

- Model A: `canopy_only / Random Forest`.
- Model B: `canopy_noaa / Random Forest`.

## Model Selection Rationale

The canopy-only Random Forest was selected because it was the best overall model by test PR-AUC in the initial model comparison.

The best canopy+NOAA model by PR-AUC was SVM (test PR-AUC=0.8459). For SHAP, `Random Forest` was used among the tree-based canopy+NOAA models (test PR-AUC=0.8307) because TreeExplainer is faster and more stable than full Kernel SHAP. SVM Kernel SHAP is left as an optional future refinement.

## Top Canopy-Only Features

- `kelp_area_m2` (canopy): mean |SHAP| = 0.0515
- `relative_canopy` (canopy): mean |SHAP| = 0.0513
- `lag1_relative_canopy` (canopy): mean |SHAP| = 0.0331
- `relative_canopy_change_lag1` (canopy): mean |SHAP| = 0.0243
- `count_cells_kelp` (canopy): mean |SHAP| = 0.0226
- `count_cells_no_clouds` (canopy): mean |SHAP| = 0.0144
- `count_cells_historic_footprint` (canopy): mean |SHAP| = 0.0118
- `historical_footprint_area_m2` (canopy): mean |SHAP| = 0.0108

## Top Canopy+NOAA Features

- `beuti_anomaly` (BEUTI): mean |SHAP| = 0.0590
- `annual_sst_std` (OISST): mean |SHAP| = 0.0275
- `hot_days_p90` (OISST): mean |SHAP| = 0.0207
- `cuti_anomaly` (CUTI): mean |SHAP| = 0.0182
- `annual_mean_sst_anomaly` (OISST): mean |SHAP| = 0.0166
- `lag1_cuti_anomaly` (CUTI): mean |SHAP| = 0.0163
- `relative_canopy` (canopy): mean |SHAP| = 0.0148
- `hot_days_p95` (OISST): mean |SHAP| = 0.0142

## NOAA Environmental Feature Rankings

- `beuti_anomaly` (BEUTI): mean |SHAP| = 0.0590
- `annual_sst_std` (OISST): mean |SHAP| = 0.0275
- `hot_days_p90` (OISST): mean |SHAP| = 0.0207
- `cuti_anomaly` (CUTI): mean |SHAP| = 0.0182
- `annual_mean_sst_anomaly` (OISST): mean |SHAP| = 0.0166
- `lag1_cuti_anomaly` (CUTI): mean |SHAP| = 0.0163
- `hot_days_p95` (OISST): mean |SHAP| = 0.0142
- `lag1_annual_mean_sst_anomaly` (OISST): mean |SHAP| = 0.0140

## Grouped Importance

- canopy_noaa / Random Forest / OISST: mean |SHAP| = 0.1378 (38.6%)
- canopy_noaa / Random Forest / BEUTI: mean |SHAP| = 0.0907 (25.4%)
- canopy_noaa / Random Forest / CUTI: mean |SHAP| = 0.0640 (18.0%)
- canopy_noaa / Random Forest / canopy: mean |SHAP| = 0.0555 (15.6%)
- canopy_noaa / Random Forest / spatial: mean |SHAP| = 0.0077 (2.2%)
- canopy_noaa / Random Forest / region: mean |SHAP| = 0.0009 (0.3%)
- canopy_only / Random Forest / canopy: mean |SHAP| = 0.2198 (100.0%)

## Dependence Plot Interpretation

Dependence plots were used to inspect how the interpreted canopy+NOAA Random Forest used environmental variables when predicting decline risk. These plots summarize model behavior and should not be interpreted as causal ecological effect estimates.

The SST anomaly variables showed relatively clear positive model associations with predicted decline risk. In contrast, hot-day exceedance counts and CUTI/BEUTI anomalies showed more mixed or context-dependent patterns. For example, some hot-day variables had negative SHAP associations in the interpreted Random Forest, and BEUTI/CUTI effects were not uniformly aligned with a simple monotonic ecological expectation.

These results suggest that the canopy+NOAA model uses environmental variables in nonlinear and interaction-dependent ways. Therefore, NOAA variables are interpreted as environmental exposure context rather than direct causal drivers of kelp decline.

Model-behavior diagnostics:

- `annual_mean_sst_anomaly`: In this fitted model, higher feature values have a positive SHAP association. (Spearman r=0.896; high-minus-low mean SHAP=0.0287).
- `annual_max_sst_anomaly`: In this fitted model, higher feature values have a positive SHAP association. (Spearman r=0.342; high-minus-low mean SHAP=0.0102).
- `hot_days_p90`: In this fitted model, higher feature values have a negative SHAP association. (Spearman r=-0.139; high-minus-low mean SHAP=-0.0084).
- `hot_days_p95`: In this fitted model, higher feature values have a negative SHAP association. (Spearman r=-0.265; high-minus-low mean SHAP=-0.0023).
- `beuti_anomaly`: In this fitted model, higher feature values have a positive SHAP association. (Spearman r=0.323; high-minus-low mean SHAP=0.0032).
- `cuti_anomaly`: In this fitted model, higher feature values have a negative SHAP association. (Spearman r=-0.756; high-minus-low mean SHAP=-0.0327).

## Local High-Risk Case Examples

- Local explanation rows created: 12.
- The local table reports top positive and negative SHAP features for high-risk true positives and canopy-only false-negative cases where available.

## Final Interpretation

Current canopy condition dominates short-term prediction in the aggregate model comparison. The canopy-only SHAP results confirm that current and lagged canopy-condition variables are the main biological-state signals. In the canopy+NOAA model, OISST, CUTI, and BEUTI variables carry substantial internal SHAP importance, indicating that NOAA environmental exposure indicators provide useful stress-context information.

However, because some SHAP dependence patterns are nonlinear or directionally mixed, these variables should be interpreted as contextual environmental indicators rather than direct causal drivers. The results support a two-layer interpretation: biological state monitoring provides the strongest short-term predictive signal, while NOAA environmental variables help characterize thermal and upwelling/nitrate-flux exposure context.

Interpretation caution: SHAP values explain how the fitted model used features for prediction. They do not establish ecological causality. Directional patterns should be interpreted alongside known data limitations, including OISST grid resolution, CUTI/BEUTI latitude-bin assignment, missing biotic drivers such as grazing pressure, and the limited number of test years.

## Limitations

- SHAP explains model behavior, not causal mechanisms.
- NOAA variables are exposure proxies.
- OISST uses nearest valid ocean grid in Version 1.
- CUTI is a coastal upwelling transport proxy, and BEUTI is a nitrate-flux proxy.
- CUTI/BEUTI are latitude-bin environmental exposure proxies, not cell-specific in situ measurements.
- No direct grazing or urchin variables are included.
- The analysis has a small number of cells and limited test years.
