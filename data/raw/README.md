# Raw Data

This directory is used to store raw data files for the kelp canopy decline early-warning project.

Raw data files are **not committed to this GitHub repository** because they may be large, externally maintained, or subject to different usage conditions. Instead, this repository documents the data sources, download procedures, and expected file structure so that the workflow can be reproduced.

## 1. Kelpwatch Data

### Source

Kelpwatch
Website: [https://kelpwatch.org/](https://kelpwatch.org/)

Kelpwatch provides satellite-derived kelp canopy data that can be visualized and downloaded for user-selected geographic and temporal extents. In this project, Kelpwatch data are used as the main ecological response variable for constructing kelp canopy decline labels.

### Study Region and Spatial Unit

The study region for the first version of this project is the Northern and Central California coastal corridor.

This region was selected because broad regional Kelpwatch exports for Northern California and Central California were initially tested, but they were found to be too coarse for site-level early-warning modeling. Therefore, the project now uses a finer spatial sampling design within the same coastal region.

The spatial unit is a regular 10 km x 10 km fishnet grid cell.

Each grid cell is treated as an individual observation unit and is assigned a unique `cell_id`. The modeling dataset will therefore be constructed at the following spatial-temporal scale:

```text
cell_id x year
```

For raw Kelpwatch exports, the full structure is:

```text
cell_id x year x quarter x kelp_area_m2
```

For the first annual modeling version, only the `quarter = max` rows are used, resulting in:

```text
cell_id x year x growing_season_max_kelp_area
```

Adjacent grid cells share boundaries but do not overlap by area. The number of candidate cells is determined by the regular fishnet grid design, not by a manually predefined number of AOIs.

### Role in This Project

Because Kelpwatch accepts only single-feature geometry uploads, each fishnet cell is stored as an individual GeoJSON file and uploaded separately to Kelpwatch.

For the first version of the modeling workflow, only the `quarter = max` rows will be used. In Kelpwatch, `max` represents the maximum quarterly kelp canopy value observed within the growing season.

### Target Variable

The main prediction target will be derived from future kelp canopy conditions.

Example target variable:

```text
decline_event_next = 1 if next-year growing-season maximum canopy is below the cell-specific historical 25th percentile
decline_event_next = 0 otherwise
```

In other words, a decline event is defined as a year in which the next-year kelp canopy falls below a historically low threshold for the same grid cell.

### Spatial Sampling Design

The candidate spatial units are generated from a regular 10 km x 10 km fishnet grid. The number of candidate cells is determined by the spatial grid design rather than predefined manually.

The current candidate grid is stored under:

```text
geometries/regular_10km_fishnet/
```

Relevant files include:

```text
geometries/regular_10km_fishnet/aoi_inventory_regular_10km_fishnet.csv
geometries/regular_10km_fishnet/grid_validation_regular_10km_fishnet.txt
geometries/regular_10km_fishnet/kelpwatch_regular_10km_fishnet_preview.geojson
geometries/regular_10km_fishnet/single_cell_geojsons/
```

Adjacent grid cells share boundaries but do not overlap by area. Candidate cells will be retained for modeling only if Kelpwatch reports a positive historical kelp footprint.

Initial filtering rule:

```text
count_cells_historic_footprint > 0
```

A stricter robustness filter may also be tested:

```text
count_cells_historic_footprint >= 500
```

### Download Procedure

1. Open the Kelpwatch website.
2. Upload one single-feature GeoJSON file from:

```text
geometries/regular_10km_fishnet/single_cell_geojsons/
```

3. Select the time variable:

```text
Growing Season Max
```

4. Download the aggregated Kelpwatch CSV for that grid cell.
5. Rename the downloaded file using the cell ID.
6. Save the file locally under:

```text
data/raw/kelpwatch_aoi/
```

Expected local file naming convention:

```text
data/raw/kelpwatch_aoi/kelpwatch_cell_001.csv
data/raw/kelpwatch_aoi/kelpwatch_cell_002.csv
data/raw/kelpwatch_aoi/kelpwatch_cell_003.csv
...
```

### Automated Download Test

The Kelpwatch web app request pattern can be automated through the public upload and aggregate endpoints. The repository includes:

```text
scripts/download_kelpwatch_cell_exports.py
scripts/validate_kelpwatch_exports.py
docs/kelpwatch_api_investigation.md
```

The download script defaults to a limited 3-cell test:

```bash
python3 scripts/download_kelpwatch_cell_exports.py
python3 scripts/validate_kelpwatch_exports.py
```

Do not run the full 285-cell download until the 3-cell test workflow is confirmed.

### Expected Raw CSV Fields

The downloaded Kelpwatch CSV files are expected to include the following fields:

```text
year
quarter
kelp_area_m2
count_cells_kelp
count_cells_no_clouds
count_cells_historic_footprint
```

Field meanings:

- `year`: year when imagery was collected
- `quarter`: annual quarter or `max` growing-season maximum row
- `kelp_area_m2`: total emergent kelp canopy area in square meters within the selected cell
- `count_cells_kelp`: number of 30 m x 30 m cells containing kelp canopy
- `count_cells_no_clouds`: number of cloud-free 30 m x 30 m cells within unoccupied kelp habitat
- `count_cells_historic_footprint`: number of 30 m x 30 m cells where kelp canopy was observed at least once across the full observation period

### Data Management Policy

Raw Kelpwatch CSV exports are not committed to this repository. They are stored locally under `data/raw/kelpwatch_aoi/` and excluded from Git to keep the repository lightweight.

The repository tracks only reproducibility files such as:

```text
GeoJSON AOI files
AOI inventory files
grid validation files
documentation
processing scripts
```

Raw CSV files should remain local and should not be staged or committed.

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
