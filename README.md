# kelp-decline-early-warning

## Project Title

**Explainable Early-Warning Modeling for Kelp Canopy Decline Using Kelpwatch and NOAA OISST**

## Research Objective

This project develops an explainable machine learning workflow for detecting early-warning signals of kelp canopy decline. It integrates Kelpwatch satellite-derived kelp canopy observations with NOAA Optimum Interpolation Sea Surface Temperature (OISST) data to build features, construct decline labels, train predictive models, and interpret environmental drivers of decline risk.

The project is designed as a reproducible research workflow and prototype decision-support system for kelp monitoring.

## Core Research Questions

- Can satellite-derived kelp canopy time series be used to define robust decline events?
- Do SST anomalies, marine heatwave indicators, and recent canopy trends provide early-warning information before decline?
- Which temperature, temporal, and site-level features are most associated with elevated decline risk?
- Can explainable AI methods such as SHAP make model outputs interpretable for ecological monitoring?

## Planned Data Sources

- **Kelpwatch:** satellite-derived kelp canopy area or extent time series.
- **NOAA OISST:** daily gridded sea surface temperature for thermal anomaly and marine heatwave feature engineering.
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

Version 1 uses NOAA-only environmental predictors. The core environmental feature set combines NOAA OISST thermal stress indicators with NOAA CUTI/BEUTI upwelling variables.

OISST features are assigned to each 10 km Kelpwatch fishnet cell using the nearest OISST grid point to the cell centroid. Annual SST summaries, SST anomalies, and hot-day indicators are computed using the 1984-2013 baseline period.

Because OISST is coarser than the 10 km Kelpwatch fishnet, the nearest-grid assignment is treated as the Version 1 primary workflow. If the nearest coastal grid point has no valid SST values, the workflow falls back to the nearest valid neighboring OISST ocean grid point and records the source coordinates. A later sensitivity analysis will compare nearest-grid assignment with a small coastal-buffer average around each cell, following the logic of prior kelp remote-sensing studies.

CUTI and BEUTI features are used as physical and biologically effective upwelling proxies. Each cell is assigned to the nearest available CUTI/BEUTI latitude bin from the NOAA/PFEG ERDDAP service, and daily values are aggregated into annual, spring, and summer summaries.

Chlorophyll-a and wave disturbance variables are reserved for future extensions.

## Initial Model Comparison

The first modeling workflow compares Logistic Regression, SVM, Random Forest, XGBoost, and LightGBM using a temporal split rather than a random split. The main modeling subset uses complete-feature years 1989-2024, with training on 1989-2016, validation on 2017-2020, and final test evaluation on 2021-2024.

Three feature sets are compared: canopy-only, OISST-only, and canopy plus NOAA environmental predictors. Final model comparison prioritizes PR-AUC, recall, and F1 because the task is framed as early-warning screening for future kelp canopy decline events.

### Initial Model Diagnostics

The first temporal model comparison showed that the canopy-only baseline achieved the highest test PR-AUC. This suggests that current canopy condition is a strong short-term early-warning signal for next-year decline. NOAA OISST and CUTI/BEUTI variables did not outperform the best canopy-only model in the first split, but they are retained as environmental exposure indicators for interpretation and SHAP-based comparison.

### Canopy Persistence and Environmental Context

The initial model comparison showed that canopy-only models achieved the strongest aggregate performance. To interpret this result, we analyzed canopy persistence and NOAA environmental signals separately. Current relative canopy was compared with next-year canopy condition, and NOAA OISST/CUTI/BEUTI variables were compared between decline and non-decline rows. Stratified analysis by current canopy condition was used to examine whether environmental stress indicators provide context beyond biological state monitoring.

## Workflow

1. **Kelpwatch data exploration**
   - Load kelp canopy time series, inspect spatial and temporal coverage, and identify candidate study sites.

2. **OISST SST feature engineering**
   - Extract SST time series near kelp sites and compute anomalies, rolling means, cumulative heat stress, and marine heatwave indicators.

3. **Decline label construction**
   - Define decline events using canopy loss thresholds, baseline periods, rolling windows, and persistence criteria.

4. **Modeling with XGBoost and SHAP**
   - Train early-warning models to predict decline risk and interpret feature contributions with SHAP.

5. **Dashboard prototype**
   - Build a Streamlit prototype for exploring kelp canopy trends, SST stress indicators, model predictions, and interpretability outputs.

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
│   ├── 01_kelpwatch_data_exploration.ipynb
│   ├── 02_oisst_sst_feature_engineering.ipynb
│   ├── 03_decline_label_construction.ipynb
│   ├── 04_modeling_xgboost_shap.ipynb
│   └── 05_dashboard_prototype.ipynb
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── labeling.py
│   ├── modeling.py
│   └── visualization.py
├── app/
│   └── streamlit_app.py
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

## Expected Outputs

- Cleaned kelp canopy time series.
- Site-level SST feature tables.
- Decline event labels for model training and evaluation.
- XGBoost model outputs and validation summaries.
- SHAP feature-importance and explanation plots.
- Prototype Streamlit dashboard for visual exploration.

## Scope and Interpretation

This project is an early-warning and interpretability workflow. Model predictions should be interpreted as risk indicators for monitoring and prioritization, not as proof of ecological causation. Field observations and ecological expertise remain important for validating decline mechanisms and management decisions.

## Current Development Priorities

- Identify initial Kelpwatch data access format and candidate regions.
- Build the first kelp canopy time-series loader.
- Prototype OISST extraction for selected kelp sites.
- Define an initial decline-labeling rule.
- Train a baseline XGBoost model.
- Create first SHAP summary plots and dashboard mockup.
