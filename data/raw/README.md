# Raw Data

This directory is used for local raw data files for the kelp canopy decline early-warning project.

Raw data files are **not committed to this GitHub repository** because they may be large, externally maintained, or subject to different usage conditions. The repository instead tracks data-source documentation, spatial AOI definitions, scripts, validation metadata, reproducibility reports, and figures.

## 1. Kelpwatch Data

### Source

Kelpwatch

Website: [https://kelpwatch.org/](https://kelpwatch.org/)

Kelpwatch provides satellite-derived kelp canopy data that can be visualized and downloaded for user-selected geographic and temporal extents. In this project, Kelpwatch data are used as the ecological response variable for constructing annual canopy panels and next-year kelp canopy decline labels.

### Study Region and Spatial Unit

The study region for the current workflow is the Northern and Central California coastal corridor.

Broad regional Kelpwatch exports for Northern California and Central California were initially tested, but they were too coarse for cell-level early-warning modeling. The project therefore uses a finer regular fishnet sampling design within the same coastal region.

The spatial unit is a regular 10 km x 10 km fishnet grid cell. Each grid cell is treated as an individual observation unit and is assigned a unique `cell_id`.

The modeling dataset is constructed at the following spatial-temporal scale:

```text
cell_id x year
```

For raw Kelpwatch exports, the full structure is:

```text
cell_id x year x quarter x kelp_area_m2
```

The current annual modeling workflow uses only `quarter = max` rows. In Kelpwatch, `max` represents the maximum quarterly kelp canopy value observed within the growing season. This produces an annual growing-season maximum canopy dataset:

```text
cell_id x year x growing_season_max_kelp_area
```

Adjacent grid cells share boundaries but do not overlap by area. The number of candidate cells is determined by the regular fishnet grid design, not by a manually predefined number of AOIs.

### Spatial Sampling and Filtering Summary

The candidate spatial units are generated from a regular 10 km x 10 km fishnet grid. The current fishnet design and validation files are stored under:

```text
geometries/regular_10km_fishnet/
```

Relevant files include:

```text
geometries/regular_10km_fishnet/README_grid_method.md
geometries/regular_10km_fishnet/aoi_inventory_regular_10km_fishnet.csv
geometries/regular_10km_fishnet/grid_validation_regular_10km_fishnet.txt
geometries/regular_10km_fishnet/kelpwatch_regular_10km_fishnet_preview.geojson
geometries/regular_10km_fishnet/single_cell_geojsons/
```

Current filtering summary:

```text
Candidate fishnet cells: 285
Exploratory retained cells with count_cells_historic_footprint > 0: 74
Main modeling cells with count_cells_historic_footprint >= 500: 50
```

The main modeling workflow uses the `count_cells_historic_footprint >= 500` retained-cell list.

### Download Procedure

Because Kelpwatch accepts only single-feature geometry uploads, each fishnet cell is stored as an individual GeoJSON file and uploaded separately to Kelpwatch.

Manual download procedure:

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

### Automated Download Workflow

The Kelpwatch web app request pattern can be automated through the public upload and aggregate endpoints. The repository includes:

```text
scripts/download_kelpwatch_cell_exports.py
scripts/validate_kelpwatch_exports.py
docs/kelpwatch_api_investigation.md
```

The automated Kelpwatch download workflow should be tested on a small number of cells before larger batch exports. This keeps the workflow reproducible and avoids unnecessary repeated requests.

Example commands:

```bash
python3 scripts/download_kelpwatch_cell_exports.py
python3 scripts/validate_kelpwatch_exports.py
```

### Expected Raw CSV Fields

Downloaded Kelpwatch CSV files are expected to include:

```text
year
quarter
kelp_area_m2
count_cells_kelp
count_cells_no_clouds
count_cells_historic_footprint
```

Field meanings:

- `year`: year when imagery was collected.
- `quarter`: annual quarter or `max` growing-season maximum row.
- `kelp_area_m2`: total emergent kelp canopy area in square meters within the selected cell.
- `count_cells_kelp`: number of 30 m x 30 m cells containing kelp canopy.
- `count_cells_no_clouds`: number of cloud-free 30 m x 30 m cells within unoccupied kelp habitat.
- `count_cells_historic_footprint`: number of 30 m x 30 m cells where kelp canopy was observed at least once across the full observation period.

### Target Variable

The main prediction target is derived from next-year canopy condition:

```text
decline_event_next = 1 if next-year growing-season maximum canopy is below the cell-specific historical 25th percentile
decline_event_next = 0 otherwise
```

In other words, a decline event is defined as a year in which the next-year kelp canopy falls below a historically low threshold for the same grid cell.

## 2. NOAA Environmental Data: OISST, CUTI, and BEUTI

### Sources

NOAA Optimum Interpolation Sea Surface Temperature (OISST)

Website: [https://www.ncei.noaa.gov/products/optimum-interpolation-sst](https://www.ncei.noaa.gov/products/optimum-interpolation-sst)

NOAA/PFEG ERDDAP for CUTI and BEUTI

Base URL: [https://upwell.pfeg.noaa.gov/erddap](https://upwell.pfeg.noaa.gov/erddap)

### Use in This Project

The current workflow uses NOAA environmental data to create cell-year exposure features that are merged with Kelpwatch decline labels:

- NOAA OISST daily sea surface temperature.
- NOAA CUTI coastal upwelling transport proxy.
- NOAA BEUTI nitrate-flux proxy.

OISST features are assigned to each 10 km Kelpwatch fishnet cell using the nearest valid OISST ocean grid point to the cell centroid. CUTI and BEUTI features are assigned by the nearest available latitude bin.

CUTI/BEUTI are interpreted as environmental exposure proxies, not cell-specific in situ measurements. CUTI is interpreted as a coastal upwelling transport proxy. BEUTI is interpreted as a nitrate-flux proxy.

### Generated Environmental Feature Examples

The current NOAA feature-engineering workflow creates variables including:

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
spring_mean_cuti
summer_mean_cuti
cuti_anomaly
lag1_cuti_anomaly
annual_mean_beuti
spring_mean_beuti
summer_mean_beuti
beuti_anomaly
lag1_beuti_anomaly
```

`hot_days_p90` and `hot_days_p95` are hot-day exceedance indicators based on historical OISST thresholds. The current Version 1 workflow does not claim to implement a full marine heatwave intensity analysis.

### Expected Local Storage

Downloaded or cached NOAA source files are stored locally under:

```text
data/external/noaa/
```

Processed environmental feature tables are stored locally under:

```text
data/processed/
```

NOAA cache files, NetCDF files, and processed datasets are not committed to GitHub.

## 3. Data Management Policy

This repository does not track raw, external, or processed data files directly. The following file groups are intentionally ignored by Git:

```text
data/raw/*
data/external/*
data/processed/*
NOAA cache files
raw Kelpwatch CSV exports
```

Only documentation files such as `README.md` and placeholder files such as `.gitkeep` are committed inside data directories.

The repository tracks reproducibility files such as:

```text
scripts
GeoJSON AOI definitions
AOI inventory files
grid validation files
validation metadata
reproducibility reports
figures
documentation
```

This approach keeps the repository lightweight while preserving reproducibility through clear documentation of data sources, expected local file structure, and preprocessing steps.

## 4. Current Workflow

The completed data workflow is:

```text
Kelpwatch cell exports
    -> export validation
    -> historical-footprint filtering
    -> annual growing-season maximum canopy panel
    -> next-year decline label construction

NOAA OISST + CUTI/BEUTI
    -> environmental feature engineering
    -> feature validation
    -> merge with Kelpwatch decline labels

Merged modeling dataset
    -> temporal train/validation/test split
    -> five-model comparison
    -> model diagnostics
    -> canopy persistence and environmental-context analysis
    -> SHAP interpretation
```

An interactive app is not part of the current core workflow.
