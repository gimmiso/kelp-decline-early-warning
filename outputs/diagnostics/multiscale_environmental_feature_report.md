# Multi-Scale Environmental Feature Construction Report

## Purpose

This V2 layer constructs source-aware OISST exposure variables around each retained Kelpwatch cell. It keeps the Version 1 nearest-grid assignment as the baseline, adds IDW-interpolated OISST exposure at kelp-cell centroids, and adds broader coastal-neighborhood buffer summaries to evaluate support mismatch.

## Implementation

- OISST cache directory: `data/external/noaa/cache/oisst`.
- Nearest-grid assignment is retained as the baseline.
- IDW interpolation uses k = `4, 8` and power = `2.0`.
- IDW is source-aware interpolation from a coarse 0.25-degree gridded SST field to kelp cell centroids; it is not ordinary missing-value imputation and does not create true 10 km SST.
- Bilinear interpolation included: `False`.
- Bilinear complete cells: `0`; incomplete cells: `50`.
- Buffer scales: `10 km, 25 km, 30 km, 50 km, 75 km`.
- Distance operations use projected CRS `EPSG:3310` rather than degree buffers.
- OISST cached grid cells are treated as point supports at their grid centroids.
- CUTI and BEUTI remain latitude-bin proxies in Version 1 and are not converted to radial buffer supports here.

## Feature Coverage

| scale | feature_columns | rows | rows_with_any_missing | mean_grid_points | min_grid_points | max_grid_points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nearest | 10 | 1900 | 0 | 1.00 | 1 | 1 |
| idw_k4 | 10 | 1900 | 0 | 4.00 | 4 | 4 |
| idw_k8 | 10 | 1900 | 0 | 8.00 | 8 | 8 |
| 10km | 9 | 1900 | 1064 | 1.00 | 1 | 1 |
| 25km | 9 | 1900 | 0 | 2.18 | 1 | 3 |
| 30km | 9 | 1900 | 0 | 2.58 | 1 | 4 |
| 50km | 9 | 1900 | 0 | 4.36 | 2 | 6 |
| 75km | 9 | 1900 | 0 | 6.46 | 2 | 9 |

## Interpretation Notes

IDW-interpolated OISST exposure at kelp-cell centroids is the main practical interpolation method in this V2 layer because 10 km buffers are under-supported relative to the 0.25-degree OISST grid spacing. Buffer aggregation reduces sensitivity to a single nearest OISST grid cell, but it does not create true nearshore in-situ temperature. The current local cache contains OISST points previously needed by the Version 1 workflow, so this V2 output should be interpreted as a reproducible multi-scale prototype. A publication-grade run should cache all OISST grid points intersecting each candidate buffer.
