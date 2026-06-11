# Model Comparison Diagnostics Report

## Initial Model Comparison Summary

- Best overall test PR-AUC: canopy_only / Random Forest (0.8974)
- Best canopy-only test PR-AUC: 0.8974
- Best canopy+NOAA test PR-AUC: 0.8459
- Canopy-only outperformed canopy+NOAA by best test PR-AUC: True

## Same-Model NOAA Improvements

- SVM: delta recall=0.2463, delta F1=0.3837, delta false negatives=-33

## Leakage Audit

- Leakage variables included in feature sets: False
- Canopy-only features are current-year canopy/status variables plus lagged canopy features.
- If only current-year and lagged canopy variables are used, strong canopy-only performance is likely due to temporal persistence and autocorrelation in kelp canopy condition, not target leakage.

## Why Canopy-Only May Be Strong

- Current canopy state is temporally persistent.
- The decline label is defined from next-year canopy condition.
- Canopy history is a direct biological response signal, while NOAA features are environmental exposure proxies.

## Environmental Signal Interpretation

- `cuti_anomaly` is lower in decline rows (standardized difference=-0.339).
- `annual_mean_sst_anomaly` is higher in decline rows (standardized difference=0.264).
- `annual_mean_beuti` is higher in decline rows (standardized difference=0.264).

NOAA variables still provide environmental context for interpretation, even when they do not outperform direct canopy observations in test PR-AUC. They should be compared through SHAP explanations rather than judged only as replacements for canopy variables.

## Recommended Next Steps

- Run SHAP for the best canopy-only model.
- Run SHAP for the best canopy+NOAA model.
- Compare explanations to determine whether NOAA variables clarify environmental exposure patterns even when predictive improvement is limited.

## Limitations

- Small number of cells.
- Limited test years.
- OISST nearest valid grid Version 1.
- CUTI/BEUTI latitude-bin proxy.
- No direct grazing or urchin variables.
