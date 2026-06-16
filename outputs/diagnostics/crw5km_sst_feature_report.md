# CRW 5 km SST Feature Report

## Purpose

This report adds NOAA Coral Reef Watch CoralTemp 5 km SST as a candidate exposure family.
It does not remove or overwrite the existing OISST V1/V2 workflow.

## Data Access Feasibility

- Run mode: `dry_run_no_local_crw_cache`
- Expected local CRW cache directory: `data/external/noaa/cache/crw5km`
- ERDDAP dataset ID: `dhw_5km`
- CRW SST variable: `CRW_SST`
- CRW SST anomaly variable: `CRW_SSTANOMALY`
- Grid resolution: 0.05 degree, approximately 5 km.
- Time coverage in the ERDDAP metadata begins on 1985-04-01 for this operational griddap endpoint.
- Sample point CSV request: `https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dhw_5km.csv?CRW_SST[(1985-04-01T12:00:00Z):1:(2024-12-31T12:00:00Z)][(39.425)][(-123.775)]`

Source pages:

- CRW product page: https://coralreefwatch.noaa.gov/product/5km/index_5km_sst.php
- CRW methodology page: https://coralreefwatch.noaa.gov/product/5km/methodology.php
- NOAA/NCEI metadata page: https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc%3ACRW-5km-HeatStressProducts

## Spatial Matching Strategy

- Baseline: nearest CRW 5 km ocean grid cell to each Kelpwatch 10 km cell centroid.
- Sensitivity: 10 km and 25 km buffer means if sufficient local CRW grid-point caches are available.
- Diagnostics record `distance_to_crw_grid_km` and the number of CRW grid points used.
- CRW 5 km SST is interpreted as a higher-resolution satellite SST exposure layer, not true local in-situ nearshore temperature.

## Current Run Summary

- Kelpwatch cells inspected: `50`
- Mean distance to theoretical nearest CRW grid center: `1.940` km
- Max distance to theoretical nearest CRW grid center: `3.549` km
- CRW feature rows built: `0`
- CRW nearest feature columns built: `0`
- Computed model-comparison rows: `0`

## Dry-Run Interpretation

No local CRW 5 km daily point cache was found, so feature construction and model comparison were not run.
The diagnostic tables were still written to document required inputs and access steps.

Expected local daily point-cache naming pattern:

```text
data/external/noaa/cache/crw5km/crw5km_lat39.425_lonm123.775_daily.csv
```

Each CSV should include at least:

```text
time, CRW_SST
```

`CRW_SSTANOMALY` may also be included if downloaded from ERDDAP.

## Output Files

- `results/tables/crw5km_vs_oisst_feature_diagnostics.csv`
- `results/tables/crw5km_model_comparison.csv`
- `outputs/diagnostics/crw5km_sst_feature_report.md`

## Interpretation

CRW 5 km SST should be treated as a higher-resolution satellite SST exposure alternative to OISST, not as true local in-situ temperature. The goal is to test whether a less coarse SST product improves at-risk and transition-oriented kelp decline prediction.

If CRW features do not improve transition-oriented targets, the result still helps support the interpretation that abrupt kelp transitions require local ecological drivers such as grazing pressure, predator/community state, wave disturbance, substrate, and disease context.