# Environmental Covariate Diagnostic Report

## OISST Spatial Matching Distance

- Mean distance to assigned OISST source grid: 10.16 km
- Median distance: 10.63 km
- Maximum distance: 22.18 km
- Cells > 20 km: 1
- Cells > 40 km: 0

## Nearest vs Cached 3x3 OISST Sensitivity

| feature | cells | mean_cached_3x3_grid_count | mean_abs_difference | mean_difference | max_abs_difference | median_correlation |
| --- | --- | --- | --- | --- | --- | --- |
| annual_hot_days_p90 | 50.0000 | 4.1600 | 2.6495 | 0.2779 | 20.0000 | 0.9967 |
| annual_hot_days_p95 | 50.0000 | 4.1600 | 1.9079 | 0.2911 | 19.0000 | 0.9975 |
| annual_mean_sst | 50.0000 | 4.1600 | 0.0710 | -0.0020 | 0.3985 | 0.9988 |
| annual_sst_anomaly | 50.0000 | 4.1600 | 0.0264 | 0.0002 | 0.1883 | 0.9988 |
| lag1_sst_anomaly | 50.0000 | 4.1600 | 0.0256 | -0.0001 | 0.1883 | 0.9989 |

## Incremental Value by Feature Set, Full Original Label

| feature_set | model_family | precision | recall | f1 | f2 | pr_auc | false_negatives | false_positives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| canopy_current_only | Logistic Regression | 0.7273 | 0.8955 | 0.8027 | 0.8559 | 0.8539 | 14.0000 | 45.0000 |
| canopy_current_plus_environment | Logistic Regression | 0.7377 | 0.6716 | 0.7031 | 0.6839 | 0.7944 | 44.0000 | 32.0000 |
| canopy_current_plus_trajectory | Logistic Regression | 0.7027 | 0.5821 | 0.6367 | 0.6028 | 0.7792 | 56.0000 | 33.0000 |
| canopy_current_plus_trajectory_plus_environment | Logistic Regression | 0.7500 | 0.6716 | 0.7087 | 0.6860 | 0.7834 | 44.0000 | 30.0000 |
| canopy_trajectory_only | Logistic Regression | 0.7200 | 0.5373 | 0.6154 | 0.5660 | 0.7689 | 62.0000 | 28.0000 |
| environment_only | Logistic Regression | 0.7355 | 0.6642 | 0.6980 | 0.6773 | 0.7913 | 45.0000 | 32.0000 |

## Best Results by Evaluation Context

| evaluation_context | target | feature_set | model_family | precision | recall | f2 | pr_auc | false_negatives | false_positives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_drop_next | actionable_decline_drop_next | canopy_current_plus_trajectory | XGBoost | 0.4906 | 0.7647 | 0.6878 | 0.4310 | 8.0000 | 27.0000 |
| actionable_low_next | actionable_decline_low_next | canopy_current_only | XGBoost | 0.4737 | 0.6000 | 0.5696 | 0.6054 | 18.0000 | 30.0000 |
| at_risk_current_canopy_gt_0.05 | decline_event_next | environment_only | Logistic Regression | 0.5319 | 0.5556 | 0.5507 | 0.5543 | 20.0000 | 22.0000 |
| full_original | decline_event_next | canopy_current_only | Logistic Regression | 0.7273 | 0.8955 | 0.8559 | 0.8539 | 14.0000 | 45.0000 |
| new_decline_transition | new_decline_event_next | canopy_current_only | XGBoost | 0.3404 | 0.6154 | 0.5298 | 0.4468 | 10.0000 | 31.0000 |

## Interpretation

NOAA environmental covariates did not improve full-sample performance over current-canopy-only models under the current feature construction. This does not imply that environmental variables are irrelevant. The OISST matching distances, partial cached-neighborhood averaging, CUTI/BEUTI latitude-bin assignment, and annual/seasonal aggregation all indicate that the present covariates are coarse relative to nearshore kelp canopy dynamics.

The most defensible interpretation is that the current NOAA covariates provide useful environmental context but limited incremental predictive value once current canopy state is included. Environmental variables may become more useful with finer coastal SST matching, event-window features, wave/grazing/disease covariates, or region-specific validation.
