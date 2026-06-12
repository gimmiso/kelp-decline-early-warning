# kelp-decline-early-warning

## Project Title

**Explainable Early-Warning Modeling for Kelp Canopy Decline Using Kelpwatch and NOAA Environmental Data**

## Research Objective

This project develops an explainable machine learning workflow for detecting early-warning signals of kelp canopy decline. It integrates Kelpwatch satellite-derived kelp canopy observations with NOAA Optimum Interpolation Sea Surface Temperature (OISST), CUTI, and BEUTI data to build features, construct decline labels, train predictive models, and interpret environmental exposure context for decline risk.

The project is designed as a reproducible research workflow for kelp monitoring and early-warning model interpretation.

## Core Research Questions

- Can satellite-derived kelp canopy time series be used to define robust decline events?
- Do SST anomalies, hot-day exceedance indicators, CUTI/BEUTI proxies, and recent canopy trends provide early-warning information before decline?
- Which temperature, temporal, and site-level features are most associated with elevated decline risk?
- Can explainable AI methods such as SHAP make model outputs interpretable for ecological monitoring?

## Data Sources

- **Kelpwatch:** satellite-derived kelp canopy area or extent time series.
- **NOAA OISST:** daily gridded sea surface temperature for thermal anomaly and hot-day exceedance feature engineering.
- **NOAA CUTI/BEUTI:** coastal upwelling transport and nitrate-flux proxy data for environmental exposure context.
- **Optional external covariates:** coastline, ecoregion, bathymetry, exposure, or site metadata if needed for model interpretation.

## Spatial Sampling Strategy

Initial Kelpwatch aggregate CSV exports were found to be too coarse for site-level early-warning modeling. Therefore, this project uses a regular 10 km x 10 km fishnet grid across the Northern and Central California coastal corridor.

The number of candidate cells is determined by the spatial grid design rather than predefined manually. The generated candidate grid contains 285 cells. Adjacent cells share boundaries but do not overlap by area.

Because Kelpwatch accepts only single-feature geometry uploads, each grid cell is stored as an individual GeoJSON file and uploaded separately to Kelpwatch.

Candidate cells will be retained for modeling only if Kelpwatch reports positive historical kelp footprint. A stricter robustness filter may use `count_cells_historic_footprint >= 500`, following the logic of previous Kelpwatch-based studies.

After Kelpwatch export validation, candidate cells are filtered using historical kelp footprint. The exploratory dataset retains cells with `count_cells_historic_footprint > 0`, while the main modeling dataset retains cells with `count_cells_historic_footprint >= 500`. Based on the current Kelpwatch export summary, this leaves 74 exploratory cells and 50 main modeling cells from the original 285 candidate fishnet cells.

## Decline Label Construction

The early-warning target is defined using next-year kelp canopy condition. For each 10 km fishnet cell, the main decline threshold is the cell-specific 25th percentile of `relative_canopy` during the 1984-2013 baseline period. A row is labeled as a decline event if the following year's growing-season maximum relative canopy falls below this baseline threshold.

The final year is excluded from modeling because next-year canopy is unavailable. Robustness labels include a full-history 25th percentile label and a 50% next-year decline label.

## NOAA Environmental Features

Version 1 uses NOAA environmental predictors. The core environmental feature set combines NOAA OISST thermal exposure indicators with NOAA CUTI/BEUTI upwelling and nitrate-flux proxy variables.

OISST features are assigned to each 10 km Kelpwatch fishnet cell using the nearest OISST grid point to the cell centroid. Annual SST summaries, SST anomalies, and hot-day indicators are computed using the 1984-2013 baseline period.

Because OISST is coarser than the 10 km Kelpwatch fishnet, the nearest-grid assignment is treated as the Version 1 primary workflow. If the nearest coastal grid point has no valid SST values, the workflow falls back to the nearest valid neighboring OISST ocean grid point and records the source coordinates. A later sensitivity analysis will compare nearest-grid assignment with a small coastal-buffer average around each cell, following the logic of prior kelp remote-sensing studies.

CUTI and BEUTI features are used as environmental exposure proxies. CUTI is treated as a coastal upwelling transport proxy, and BEUTI is treated as a nitrate-flux proxy. Each cell is assigned to the nearest available CUTI/BEUTI latitude bin from the NOAA/PFEG ERDDAP service, so these variables are not interpreted as cell-specific in situ measurements.

Chlorophyll-a and wave disturbance variables are reserved for future extensions.

## Initial Model Comparison

The first modeling workflow compares Logistic Regression, SVM, Random Forest, XGBoost, and LightGBM using a temporal split rather than a random split. The main modeling subset uses complete-feature years 1989-2024, with training on 1989-2016, validation on 2017-2020, and final test evaluation on 2021-2024.

Three feature sets are compared: canopy-only, OISST-only, and canopy plus NOAA environmental predictors. Final model comparison prioritizes PR-AUC, recall, and F1 because the task is framed as early-warning screening for future kelp canopy decline events.

Key model-comparison result: the best aggregate test PR-AUC came from `canopy_only / Random Forest`, while the best canopy+NOAA PR-AUC came from `canopy_noaa / SVM`. SHAP interpretation uses `canopy_only / Random Forest` and `canopy_noaa / Random Forest`; SVM Kernel SHAP is left as a future refinement because it is slower and less stable for this workflow.

### Initial Model Diagnostics

The first temporal model comparison showed that the canopy-only baseline achieved the highest test PR-AUC. This suggests that current canopy condition is a strong short-term early-warning signal for next-year decline. NOAA OISST and CUTI/BEUTI variables did not outperform the best canopy-only model in the first split, but they are retained as environmental exposure indicators for interpretation and SHAP-based comparison.

### Canopy Persistence and Environmental Context

The initial model comparison showed that canopy-only models achieved the strongest aggregate performance. To interpret this result, we analyzed canopy persistence and NOAA environmental signals separately. Current relative canopy was compared with next-year canopy condition, and NOAA OISST/CUTI/BEUTI variables were compared between decline and non-decline rows. Stratified analysis by current canopy condition was used to examine whether environmental stress indicators provide context beyond biological state monitoring.

### SHAP Interpretation

SHAP results are interpreted as model-behavior explanations rather than causal mechanisms. The canopy-only model relied primarily on current and lagged canopy variables, while the canopy+NOAA model assigned substantial importance to OISST, CUTI, and BEUTI features. Some dependence patterns were nonlinear or directionally mixed, so NOAA variables are interpreted as environmental exposure context rather than simple causal drivers.

Interpretation caution: SHAP values explain how the fitted model used features for prediction. They do not establish ecological causality. Directional patterns should be interpreted alongside known data limitations, including OISST grid resolution, CUTI/BEUTI latitude-bin assignment, missing biotic drivers such as grazing pressure, and the limited number of test years.

## Workflow

1. **Kelpwatch 10 km fishnet spatial design**
   - Define regular 10 km x 10 km candidate cells across the Northern and Central California coastal corridor.

2. **Kelpwatch cell filtering**
   - Retain main modeling cells using historical kelp footprint thresholds.

3. **Annual canopy panel construction**
   - Build a cell-year panel from growing-season maximum Kelpwatch canopy exports.

4. **Next-year decline label construction**
   - Label rows using next-year canopy relative to a cell-specific historical baseline threshold.

5. **NOAA environmental feature engineering**
   - Add OISST thermal metrics and CUTI/BEUTI environmental exposure proxies.

6. **Dataset validation**
   - Check row counts, year coverage, missingness, suspicious values, OISST fallback, and CUTI/BEUTI latitude-bin assignment.

7. **Five-model comparison**
   - Compare Logistic Regression, SVM, Random Forest, XGBoost, and LightGBM across canopy-only, OISST-only, and canopy+NOAA feature sets.

8. **Model diagnostics**
   - Audit leakage, compare feature sets, inspect false negatives, and summarize temporal/environmental patterns.

9. **Canopy persistence and environmental context analysis**
   - Separate canopy-state persistence from NOAA environmental signal interpretation.

10. **SHAP interpretation**
    - Explain the best canopy-only model and an interpretable tree-based canopy+NOAA model.

## Repository Structure

```text
kelp-decline-early-warning/
├── README.md
├── requirements.txt
├── .gitignore
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
│   ├── 01_kelpwatch_panel_construction.ipynb
│   ├── 02_decline_label_construction.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_model_diagnostics.ipynb
│   ├── 06_canopy_environment_context_analysis.ipynb
│   └── 07_shap_interpretation.ipynb
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── labeling.py
│   ├── modeling.py
│   └── visualization.py
└── outputs/
    ├── figures/
    ├── maps/
    └── model_results/
```

## Setup

Create and activate a Python virtual environment, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use Python 3.10-3.13 for this environment. The SHAP dependency stack currently pulls `numba`, which does not support Python 3.14.

On Windows, activate the environment with:

```bat
.venv\Scripts\activate
```

Run a quick environment check:

```bash
python -c "import pandas, sklearn, xgboost, lightgbm, shap; print('OK')"
```

On macOS, if the environment check fails with `libomp.dylib` missing while importing XGBoost or LightGBM, install the OpenMP runtime:

```bash
brew install libomp
```

## Reproducible Script Workflow

The main analysis scripts are designed to be run in order after local Kelpwatch CSV exports and NOAA cache files are available:

```bash
python scripts/filter_kelpwatch_cells.py
python scripts/build_kelpwatch_panel.py
python scripts/construct_decline_labels.py
python scripts/build_noaa_environmental_features.py
python scripts/validate_kelpwatch_exports.py
python scripts/train_model_comparison.py
python scripts/diagnose_model_results.py
python scripts/analyze_canopy_environment_context.py
python scripts/interpret_models_shap.py
```

Raw Kelpwatch exports, processed modeling datasets, and NOAA cache files are intentionally ignored by Git. The repository tracks scripts, reproducibility metadata, figures, GeoJSON AOIs, and written reports.

## Expected Outputs

- Kelpwatch 10 km fishnet AOI design and validation files.
- Annual kelp canopy panel metadata and decline-label summaries.
- NOAA environmental feature summaries and modeling-dataset validation reports.
- Five-model comparison tables and figures.
- Model diagnostics for feature sets, false negatives, and environmental signal context.
- Canopy persistence and environmental-context figures.
- SHAP feature-importance, grouped-importance, dependence, and local explanation outputs.

## Scope and Interpretation

This project is an early-warning and interpretability workflow. Model predictions should be interpreted as risk indicators for monitoring and prioritization, not as proof of ecological causation. Field observations and ecological expertise remain important for validating decline mechanisms and management decisions.

## Remaining Improvements

- Add a polished README figure panel for portfolio presentation.
- Add direct links from README sections to the most important reports and figures.
- Add sensitivity analysis comparing nearest-grid OISST assignment with a coastal-buffer average.
- Add additional ecological covariates where available, especially grazing pressure, urchin observations, wave disturbance, and disease-related context.
- Consider spatial or grouped cross-validation in addition to the current temporal split.
- Add a concise results-summary document for readers who want the main findings without opening every metadata report.
