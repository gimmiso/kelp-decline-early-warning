# Canopy Persistence and Environmental Context Analysis

## Research Question

How should the initial model finding be interpreted when canopy-only models outperform NOAA-enhanced models by aggregate PR-AUC?

## Why Canopy-Only Can Be Strong

Current canopy condition can be a strong short-term predictor because kelp canopy state is temporally persistent. The project target is next-year decline, so current-year canopy observations provide direct biological state information that may already integrate recent environmental stress, disturbance, and recovery history.

## Canopy Persistence Results

- Rows analyzed: 1800 across 50 cells from 1989-2024.
- Overall next-year decline rate: 0.366.
- Correlation between current relative canopy and next-year relative canopy: 0.610.
- Lowest current-canopy quintile decline rate: 0.633.
- Highest current-canopy quintile decline rate: 0.303.

## Decline vs Non-Decline NOAA Signal Results

- `cuti_anomaly` is lower in decline rows (standardized difference=-0.339).
- `annual_mean_sst_anomaly` is higher in decline rows (standardized difference=0.268).
- `annual_mean_beuti` is higher in decline rows (standardized difference=0.260).

These comparisons test whether NOAA variables show directional environmental differences even when they do not replace direct canopy observations as the highest-performing aggregate predictors.

## Stratified Analysis by Canopy Condition

Environmental stress indicators were compared within low, medium, and high current-canopy groups. This asks whether stress variables provide context beyond current biological state.

- `hot_days_p90` in `low_canopy`: high-stress decline rate minus low-stress rate = 0.038.
- `hot_days_p90` in `medium_canopy`: high-stress decline rate minus low-stress rate = 0.039.
- `hot_days_p90` in `high_canopy`: high-stress decline rate minus low-stress rate = 0.043.
- `annual_mean_sst_anomaly` in `low_canopy`: high-stress decline rate minus low-stress rate = 0.171.
- `annual_mean_sst_anomaly` in `medium_canopy`: high-stress decline rate minus low-stress rate = 0.095.
- `annual_mean_sst_anomaly` in `high_canopy`: high-stress decline rate minus low-stress rate = 0.118.
- `beuti_anomaly` in `low_canopy`: high-stress decline rate minus low-stress rate = -0.059.
- `beuti_anomaly` in `medium_canopy`: high-stress decline rate minus low-stress rate = 0.012.

## Canopy-Only False-Negative Environmental Profile

Canopy-only false-negative profile used Random Forest predictions.

- Canopy-only false negatives have lower `annual_mean_sst_anomaly` than all test rows (standardized difference=-0.342).
- Canopy-only false negatives have lower `annual_max_sst_anomaly` than all test rows (standardized difference=-0.097).
- Canopy-only false negatives have lower `hot_days_p90` than all test rows (standardized difference=-0.121).
- Canopy-only false negatives have lower `hot_days_p95` than all test rows (standardized difference=-0.079).
- Canopy-only false negatives have higher `beuti_anomaly` than all test rows (standardized difference=0.057).
- Canopy-only false negatives have higher `cuti_anomaly` than all test rows (standardized difference=0.163).

## Feature-Set Role Interpretation

- `canopy_only`: biological state monitoring.
- `oisst_only`: thermal exposure screening.
- `canopy_noaa`: biological state plus environmental context.
- `CUTI/BEUTI`: upwelling and nitrate-flux proxy context.

## Final Interpretation

Current canopy condition was the strongest short-term predictor of next-year kelp decline, reflecting temporal persistence in canopy state. NOAA environmental indicators did not outperform canopy-only models in aggregate prediction performance, but SST stress and CUTI/BEUTI variables provide interpretable environmental context. Therefore, NOAA variables are better interpreted as environmental-risk context rather than replacements for direct canopy observations.
