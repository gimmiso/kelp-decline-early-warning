# Explainable Early-Warning Modeling for Kelp Canopy Decline Using Kelpwatch and NOAA Environmental Data

## Overview

This project builds a reproducible research-stage early-warning screening workflow for next-year kelp canopy decline. It combines Kelpwatch satellite-derived kelp canopy observations with NOAA environmental exposure indicators from OISST, CUTI, and BEUTI.

The workflow creates a 10 km fishnet cell-year dataset, constructs next-year decline labels, engineers NOAA thermal and upwelling-related proxy features, compares five supervised machine-learning models, and uses diagnostics plus SHAP interpretation to explain model behavior.

Main claim: this repository evaluates whether Kelpwatch satellite-derived kelp canopy time series and NOAA environmental covariates can support a recall-oriented early-warning screening workflow for next-year kelp canopy decline. The results suggest useful risk-state prediction performance, but stricter diagnostics show that near-low-canopy persistence contributes substantially to apparent full-sample performance.

## Research Questions

- Can satellite-derived kelp canopy time series define useful next-year decline events?
- How strongly does current canopy condition persist into next-year canopy condition?
- Do NOAA OISST, CUTI, and BEUTI exposure proxies add environmental context beyond direct canopy monitoring?
- How do model results change across linear, margin-based, random-forest, and boosted-tree classifiers?
- Can SHAP help separate biological-state signals from environmental-context signals without making causal claims?

## Data Sources

- **Kelpwatch:** satellite-derived kelp canopy area time series for user-defined AOIs.
- **NOAA OISST:** daily gridded sea surface temperature used for annual SST summaries, anomalies, and hot-day exceedance indicators.
- **NOAA CUTI:** coastal upwelling transport proxy.
- **NOAA BEUTI:** nitrate-flux proxy.

CUTI and BEUTI are interpreted as latitude-bin environmental exposure proxies. They are not cell-specific in situ nutrient measurements.

## Spatial Design

The study region is the Northern and Central California coastal corridor. Initial broad Kelpwatch regional exports were too coarse for cell-level early-warning modeling, so the project uses a regular 10 km x 10 km fishnet grid.

Current spatial design summary:

```text
Candidate fishnet cells: 285
Exploratory retained cells with count_cells_historic_footprint > 0: 74
Main modeling cells with count_cells_historic_footprint >= 500: 50
```

Each grid cell is uploaded to Kelpwatch as a single-feature GeoJSON because Kelpwatch accepts one feature per uploaded geometry. The GeoJSON AOIs and validation files are stored under:

```text
geometries/regular_10km_fishnet/
```

## Target Definition

The response variable is a next-year decline label:

```text
decline_event_next = 1
if next-year growing-season maximum relative canopy
falls below the cell-specific historical 25th percentile
```

The main baseline period is 1984-2013. The final year is excluded from label construction because next-year canopy is unavailable.

## NOAA Environmental Feature Engineering

Version 1 uses NOAA environmental predictors assigned to each 10 km Kelpwatch cell:

- OISST features are assigned using the nearest valid OISST ocean grid point to the cell centroid.
- CUTI and BEUTI features are assigned using the nearest available latitude bin.
- OISST anomalies and hot-day exceedance indicators use the 1984-2013 baseline period.

Example generated features include:

```text
annual_mean_sst
annual_max_sst
annual_min_sst
annual_sst_std
annual_mean_sst_anomaly
annual_max_sst_anomaly
hot_days_p90
hot_days_p95
lag1_annual_mean_sst_anomaly
lag1_hot_days_p90
annual_mean_cuti
cuti_anomaly
annual_mean_beuti
beuti_anomaly
lag1_cuti_anomaly
lag1_beuti_anomaly
```

`hot_days_p90` and `hot_days_p95` are hot-day exceedance indicators, not a full marine heatwave intensity analysis.

## Modeling Framework

The main modeling subset uses complete-feature years 1989-2024:

```text
Train: 1989-2016
Validation: 2017-2020
Test: 2021-2024
```

Five supervised classifiers were compared:

- Logistic Regression: transparent linear baseline.
- SVM: nonlinear margin-based classifier for a relatively small tabular dataset.
- Random Forest: robust nonlinear tree benchmark and SHAP-compatible model.
- XGBoost: boosted-tree benchmark.
- LightGBM: efficient boosted-tree benchmark.

Three feature sets were compared within each model class:

- `canopy_only`: direct biological state monitoring.
- `oisst_only`: thermal exposure variables alone.
- `canopy_noaa`: canopy state plus NOAA environmental exposure context.

Performance was evaluated using PR-AUC, recall, F1, F2, and false negatives because the task is framed as research-stage early-warning screening rather than balanced classification.

## Results

### 1. Canopy State Is a Strong Short-Term Signal

![Current relative canopy vs next-year relative canopy](outputs/figures/canopy_persistence_scatter.png)

Current relative canopy was strongly associated with next-year canopy condition. The current-to-next-year relative canopy correlation was `0.610`, supporting the interpretation that canopy state has strong temporal persistence.

![Next-year decline rate by current canopy quintile](outputs/figures/canopy_quantile_decline_rate.png)

The lowest current-canopy quintile had the highest next-year decline rate (`0.633`), while the highest quintile had a lower decline rate (`0.303`). This helps explain why canopy-only models performed strongly.

### 2. Canopy-Only Models Performed Best in Aggregate Prediction

![Model performance comparison](outputs/figures/model_performance_comparison.png)

The best aggregate test PR-AUC came from `canopy_only / Random Forest`:

```text
PR-AUC: 0.8974
Recall: 0.5896
F1: 0.7149
False negatives: 55
```

The best canopy+NOAA PR-AUC came from `canopy_noaa / SVM`:

```text
PR-AUC: 0.8459
Recall: 0.2463
F1: 0.3837
False negatives: 101
```

Within the tested algorithms, canopy+NOAA did not improve PR-AUC over canopy-only. However, for SVM at the default threshold, canopy+NOAA improved recall and F1 and reduced false negatives relative to canopy-only SVM. OISST-only models were generally weaker than canopy-only models, while canopy+NOAA models generally improved over OISST-only models.

### Recall-Oriented Threshold Tuning

Because kelp decline prediction is framed as a recall-oriented screening task, the default `0.50` decision threshold may be too conservative. Thresholds were selected on the validation period and applied unchanged to the test period. This analysis evaluates whether false negatives can be reduced while preserving a reasonable precision-recall trade-off. Threshold tuning changes the classifier operating point; it does not change PR-AUC.

The main threshold-tuned model is `canopy_only / Random Forest` at threshold `0.30`, which balances high recall with reasonable precision:

```text
Selection role: main threshold-tuned model
Recall: 0.910
Precision: 0.753
F1: 0.824
F2: 0.874
False negatives: 12
Default-threshold false negatives: 55
```

The high-sensitivity screening scenario is `canopy_noaa / SVM` at threshold `0.05`. This setting achieves very high recall, but it uses a much lower threshold and may produce more warnings:

```text
Selection role: high-sensitivity screening scenario
Recall: 1.000
Precision: 0.670
F1: 0.802
F2: 0.910
False negatives: 0
Default-threshold false negatives: 101
```

The threshold analysis now reports five selection rules: default `0.50`, max F1, max F2, recall >= 0.70 then max F1, and max recall subject to precision >= 0.65. If the precision floor is too strict for a model, the rule falls back to precision >= 0.50 and records that fallback. These thresholds were selected using the 2017-2020 validation period only, then fixed for the 2021-2024 test period to avoid test-set leakage.

### Early-Warning Validity Diagnostics

An additional methodological robustness check evaluates whether model performance is partly driven by zero-state or near-zero-state persistence. This matters because a model can appear useful for early warning if it mostly identifies locations that are already degraded and likely to remain degraded, rather than detecting transition into future low-canopy conditions.

The diagnostic measures current-to-next-year canopy transitions using `relative_canopy` and `next_year_relative_canopy` under zero and near-zero thresholds:

```text
current_zero -> next_zero
current_zero -> next_nonzero
current_nonzero -> next_zero
current_nonzero -> next_nonzero
```

Exact zero-to-zero persistence was limited in this dataset, but near-zero persistence was substantial:

```text
Threshold 0.00: zero -> zero persistence = 0.143
Threshold 0.01: zero -> zero persistence = 0.630
Threshold 0.05: zero -> zero persistence = 0.669
Threshold 0.10: zero -> zero persistence = 0.746
```

At-risk subset evaluation shows that original-label performance declines when already-low canopy states are removed. For `current_canopy > 0.05`, the best original-label test result was `canopy_only / Random Forest`:

```text
PR-AUC: 0.633
Recall: 0.400
F1: 0.480
Positive events: 45 / 93
```

A stricter transition label was also tested:

```text
new_decline_event_next = 1
if current relative canopy is at or above the cell-specific 1984-2013 p25 baseline
and next-year relative canopy falls below that p25 baseline
```

This label captures transition into a low-canopy state rather than persistence of an already-low state. Under this stricter target, performance was more modest; the best full-sample PR-AUC was `0.401` for `canopy_only / Random Forest`, while at-risk PR-AUC values were generally in the `0.39-0.51` range depending on the threshold and model.

These diagnostics suggest that the current Version 1 model is strongest at detecting canopy-state persistence and already-low or near-low canopy conditions. There is some preliminary signal in at-risk and new-decline-transition settings, but the present results should be described as a research-stage early-warning validity evaluation rather than a deployed monitoring workflow.

### How to Interpret the Results

Full-sample PR-AUC can look strong because it includes persistent low-canopy states. At-risk subset performance is more relevant for early warning because it asks whether the model can identify future decline among locations that still have nonzero or moderate current canopy. The stricter `new_decline_event_next` label better captures new transition into low canopy, but it is harder to predict because it removes already-low persistence from the positive class.

High recall is important because missing actual decline events is costly in a screening workflow. Precision still matters because too many false alarms reduce practical usefulness. The most defensible interpretation is therefore that the model provides a reproducible robustness check for distinguishing preliminary early-warning signal from canopy-state persistence.

### Recall-Oriented Modeling Extensions

The next modeling extension adds cost-sensitive learning, actionable decline labels, canopy trajectory features, environmental stress interactions, feature-set ablations, and extended validation-based threshold tuning. The purpose is to improve recall-oriented risk screening while avoiding a trivial "predict everything as decline" result.

False negatives are costly in early-warning screening because missed decline events reduce the value of an alerting workflow. Class-weighted and positive-class-weighted models were added to increase sensitivity to decline events while keeping unweighted models as baselines. Threshold tuning is still selected on the 2017-2020 validation period and fixed on the 2021-2024 test period to avoid test-set leakage.

Two actionable labels were added:

```text
actionable_decline_low_next:
current_canopy > 0.05
and next_canopy < historical_25th_percentile

actionable_decline_drop_next:
current_canopy > 0.05
and proportional next-year canopy drop >= 0.30
```

These labels are more practical for actionable decline screening than the original label alone because they require currently observable canopy before a low-canopy or sharp-drop outcome. They are also less strict than `new_decline_event_next`, which only captures transition from at-or-above the historical p25 threshold into below-p25 canopy.

The extension also adds leakage-safe canopy trajectory features such as `canopy_lag1`, `canopy_lag2`, `canopy_2yr_change`, `canopy_3yr_slope`, `canopy_3yr_cv`, `canopy_drop_from_3yr_max`, and `years_since_last_high_canopy`. Environmental stress features include lagged SST anomaly, two-year SST anomaly mean, lagged hot-day exposure, lagged CUTI/BEUTI anomalies, and simple thermal-stress interactions.

Key results:

```text
Original decline label, default threshold:
Best F2 = 0.857
Model = canopy_current_only / Logistic Regression / cost_sensitive
Recall = 0.896
Precision = 0.732

Original decline label, balanced threshold tuning:
Model = canopy_current_only / XGBoost / unweighted
Threshold = 0.25
Recall = 0.813
Precision = 0.779
F2 = 0.806

Original decline label, high-sensitivity threshold tuning:
Model = canopy_current_only / LightGBM / unweighted
Threshold = 0.05
Recall = 1.000
Precision = 0.677
F2 = 0.913

Actionable drop label:
Best default-threshold F2 = 0.699
Model = canopy_trajectory_only / XGBoost / cost_sensitive

Actionable drop label, threshold tuned:
Model = canopy_current_plus_trajectory / SVM / cost_sensitive
Threshold = 0.20
Recall = 0.941
Precision = 0.516
F2 = 0.808
```

Trajectory features did not outperform current-canopy-only models for the original `decline_event_next` label, so the original full-sample task still appears strongly influenced by canopy-state persistence. However, trajectory features were useful for `actionable_decline_drop_next`, providing preliminary evidence that recent canopy trajectory can support actionable decline screening when the target is defined as a sharp future drop rather than persistence of already-low canopy.

### 3. NOAA Variables Provide Environmental Exposure Context

![Environmental signal comparison between decline and non-decline rows](outputs/figures/environmental_signal_decline_vs_nondecline.png)

Decline rows showed directional differences in several NOAA variables. For example, `annual_mean_sst_anomaly` was higher in decline rows, while `cuti_anomaly` was lower. BEUTI-related signals also differed, but they should be interpreted cautiously as proxy-based and context-dependent.

NOAA variables are environmental exposure proxies. They provide context for interpreting decline risk but do not replace direct canopy monitoring and do not establish causal mechanisms.

### 4. SHAP Separates Biological-State Signals from Environmental-Context Signals

![Grouped SHAP importance](outputs/figures/shap_grouped_importance.png)

SHAP interpretation was performed on:

```text
canopy_only / Random Forest
canopy_noaa / Random Forest
```

Although SVM had the best canopy+NOAA PR-AUC, Kernel SHAP for SVM was not used because it can be slow and less stable. A Random Forest canopy+NOAA model was used for TreeExplainer-based interpretation.

Grouped SHAP importance showed:

- `canopy_only / Random Forest`: canopy-state variables accounted for `100%` of grouped SHAP importance.
- `canopy_noaa / Random Forest`: OISST, BEUTI, and CUTI variables carried substantial internal SHAP importance, while canopy variables remained part of the model explanation.

These SHAP values explain fitted model behavior, not ecological causality.

## Reproducible Workflow

The completed workflow is:

1. Northern/Central California 10 km fishnet spatial design.
2. Kelpwatch cell export and validation.
3. Historical-footprint cell filtering.
4. Annual growing-season maximum canopy panel construction.
5. Next-year decline label construction.
6. NOAA OISST + CUTI/BEUTI feature engineering.
7. Final modeling dataset validation.
8. Five-model comparison.
9. Threshold tuning.
10. Zero-persistence and at-risk validity diagnostics.
11. Recall-oriented modeling extensions.
12. Model diagnostics.
13. Canopy persistence and environmental-context analysis.
14. SHAP interpretation.
15. Within-model feature-set comparison.

Main scripts:

```bash
python scripts/filter_kelpwatch_cells.py
python scripts/build_kelpwatch_panel.py
python scripts/construct_decline_labels.py
python scripts/build_noaa_environmental_features.py
python scripts/train_model_comparison.py
python scripts/tune_decision_thresholds.py
python scripts/diagnose_zero_persistence.py
python scripts/run_recall_oriented_modeling_extensions.py
python scripts/diagnose_model_results.py
python scripts/analyze_canopy_environment_context.py
python scripts/interpret_models_shap.py
```

Run `python scripts/diagnose_zero_persistence.py` after the main modeling pipeline and threshold tuning, but before final interpretation. This makes the final narrative distinguish risk-state prediction, near-low-canopy persistence, and stricter transition-into-decline performance.

Run `python scripts/run_recall_oriented_modeling_extensions.py` after the validity diagnostics when testing cost-sensitive models, actionable labels, trajectory features, environmental stress interactions, and extended threshold tuning.

Raw Kelpwatch exports, processed datasets, and NOAA cache files are intentionally ignored by Git. The repository tracks scripts, GeoJSON AOIs, validation metadata, diagnostic reports, selected model-result summaries, reproducibility reports, and figures.

`outputs/diagnostics/` contains zero-persistence transition tables, at-risk subset evaluation, stricter new-decline label performance, actionable-label summaries, and diagnostic plots/reports. `outputs/model_results/` contains compact model-result outputs such as threshold tuning grids, threshold-selection summaries, cost-sensitive model comparisons, actionable-label performance, and feature-ablation results.

## Repository Structure

```text
kelp-decline-early-warning/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── geometries/
│   └── regular_10km_fishnet/
├── scripts/
├── docs/
│   └── maps/
├── notebooks/
├── src/
└── outputs/
    ├── figures/
    ├── diagnostics/
    ├── maps/
    ├── metadata/
    └── model_results/
```

## Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use Python 3.10-3.13. The SHAP dependency stack currently pulls `numba`, which does not support Python 3.14.

Environment check:

```bash
python -c "import pandas, sklearn, xgboost, lightgbm, shap; print('OK')"
```

On macOS, if importing XGBoost or LightGBM fails with `libomp.dylib` missing, install OpenMP:

```bash
brew install libomp
```

## Limitations

- Small number of retained modeling cells (`50` main cells).
- Limited final test years (`2021-2024`).
- The temporal split does not fully test spatial generalization.
- OISST uses nearest valid ocean-grid assignment in Version 1.
- CUTI/BEUTI use nearest latitude-bin assignment.
- No direct cell-level nutrient measurements are included.
- No grazing, urchin, sea star wasting disease, or direct biotic pressure variables are included.
- Environmental interpretation is proxy-based.
- The workflow supports early-warning screening, not causal attribution.
- Early-warning validity depends on separating transition-into-decline skill from persistence of already-low or near-zero canopy states.

## Future Work

- Test spatial or grouped cross-validation, including leave-region-out validation.
- Compare nearest-grid OISST assignment with coastal-buffer average sensitivity analyses.
- Add ecological covariates such as grazing pressure, urchin observations, wave disturbance, and disease context where available.
- Estimate uncertainty using bootstrap confidence intervals for model metrics.
- Develop an optional Streamlit dashboard only if it can be polished and connected to tracked reproducibility outputs.
