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
