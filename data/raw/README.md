# Raw Data

This directory is used to store raw data files for the kelp canopy decline early-warning project.

Raw data files are **not committed to this GitHub repository** because they may be large, externally maintained, or subject to different usage conditions. Instead, this repository documents the data sources, download procedures, and expected file structure so that the workflow can be reproduced.

## 1. Kelpwatch Data

### Source

Kelpwatch
Website: https://kelpwatch.org/

Kelpwatch provides satellite-derived kelp canopy data that can be visualized and downloaded for selected geographic and temporal extents. In this project, Kelpwatch data will be used as the main ecological response variable for constructing kelp canopy decline labels.

### Planned Use in This Project

The Kelpwatch data will be used to construct a panel dataset with the following general structure:

```text
site_id x year x season x kelp_canopy_area
```

The main target variable will be derived from future kelp canopy conditions, such as whether canopy area declines below a site-specific historical threshold in the next period.

Example target variable:

```text
decline_event_next = 1 if next-period canopy is below the site-specific historical 25th percentile
decline_event_next = 0 otherwise
```

### Download Procedure

1. Go to the Kelpwatch website.
2. Select the study region or area of interest.
3. Select the relevant time range and seasonal aggregation option, if available.
4. Download the aggregated kelp canopy data as a CSV file.
5. Save the downloaded file in this directory.

Example local file path:

```text
data/raw/kelpwatch_sample.csv
```

### Notes

The raw Kelpwatch CSV file should include temporal information such as year, season, or date, as well as kelp canopy-related measurements. Depending on the export settings, the file may also include geographic identifiers or area-of-interest information.

The exact column names will be inspected during the preprocessing step.

## 2. NOAA OISST Data

### Source

NOAA Optimum Interpolation Sea Surface Temperature
Website: https://www.ncei.noaa.gov/products/optimum-interpolation-sst

NOAA OISST provides daily gridded sea surface temperature data. In the next stage of this project, NOAA OISST will be used to derive marine heat stress indicators and merge them with the Kelpwatch canopy dataset.

### Planned Use in This Project

NOAA OISST will be used to generate SST-based predictor variables, including:

```text
mean_sst
max_sst
sst_anomaly
marine_heatwave_days
marine_heatwave_intensity
lag1_sst_anomaly
lag1_marine_heatwave_days
```

These variables will be used as environmental predictors in the kelp canopy decline early-warning models.

### Expected Storage

Downloaded or processed NOAA OISST files should be stored separately under:

```text
data/external/
```

or, after preprocessing:

```text
data/processed/
```

Raw OISST files should not be committed to GitHub.

## 3. Data Management Policy

This repository does not track raw data files directly. The following files are intentionally ignored by Git:

```text
data/raw/*
data/external/*
data/processed/*
```

Only documentation files such as `README.md` and placeholder files such as `.gitkeep` are committed.

This approach keeps the repository lightweight while preserving reproducibility through clear documentation of data sources and preprocessing steps.

## 4. Planned Workflow

The raw data workflow is:

```text
Kelpwatch CSV
    -> canopy preprocessing
    -> decline-event label construction

NOAA OISST
    -> SST and marine heatwave feature engineering
    -> merge with Kelpwatch decline labels

Merged dataset
    -> five-model comparison
    -> SHAP interpretation
    -> early-warning dashboard
```
