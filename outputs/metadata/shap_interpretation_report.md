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

Dependence plots were generated to inspect whether SST anomalies and hot-day counts increase predicted decline risk, and whether lower CUTI/BEUTI anomalies correspond to increased risk in the interpreted canopy+NOAA tree model. These plots should be read as model-behavior summaries, not causal effect estimates.

- `annual_mean_sst_anomaly`: Higher feature values tend to increase model-predicted decline risk. (Spearman r=0.896; high-minus-low mean SHAP=0.0287).
- `annual_max_sst_anomaly`: Higher feature values tend to increase model-predicted decline risk. (Spearman r=0.342; high-minus-low mean SHAP=0.0102).
- `hot_days_p90`: Higher feature values tend to decrease model-predicted decline risk. (Spearman r=-0.139; high-minus-low mean SHAP=-0.0084).
- `hot_days_p95`: Higher feature values tend to decrease model-predicted decline risk. (Spearman r=-0.265; high-minus-low mean SHAP=-0.0023).
- `beuti_anomaly`: Higher feature values tend to increase model-predicted decline risk. (Spearman r=0.323; high-minus-low mean SHAP=0.0032).
- `cuti_anomaly`: Higher feature values tend to decrease model-predicted decline risk. (Spearman r=-0.756; high-minus-low mean SHAP=-0.0327).

## Local High-Risk Case Examples

- Local explanation rows created: 12.
- The local table reports top positive and negative SHAP features for high-risk true positives and canopy-only false-negative cases where available.

## Final Interpretation

Current canopy condition dominates short-term prediction in the aggregate model comparison, while NOAA environmental variables provide interpretable thermal and upwelling/nitrate-flux exposure context. In the interpreted canopy+NOAA tree model, environmental proxies carry substantial internal SHAP importance, but this should be read as model explanation rather than evidence that NOAA variables replace direct canopy observations.

## Limitations

- SHAP explains model behavior, not causal mechanisms.
- NOAA variables are exposure proxies.
- OISST uses nearest valid ocean grid in Version 1.
- CUTI/BEUTI use latitude-bin assignment.
- No direct grazing or urchin variables are included.
- The analysis has a small number of cells and limited test years.
