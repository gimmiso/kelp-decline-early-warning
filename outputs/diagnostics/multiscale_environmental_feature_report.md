# Multi-Scale Environmental Feature Construction Report

## Purpose

This V2 layer constructs OISST exposure variables at multiple spatial supports around each retained Kelpwatch cell. It keeps the Version 1 nearest-grid assignment as the baseline and adds buffer-based summaries to evaluate support mismatch.

## Implementation

- OISST cache directory: `data/external/noaa/cache/oisst`.
- Buffer scales: `10 km, 25 km, 30 km, 50 km, 75 km`.
- Distance operations use projected CRS `EPSG:3310` rather than degree buffers.
- OISST cached grid cells are treated as point supports at their grid centroids.
- CUTI and BEUTI remain latitude-bin proxies in Version 1 and are not converted to radial buffer supports here.

## Feature Coverage

| scale | feature_columns | rows | rows_with_any_missing | mean_grid_points | min_grid_points | max_grid_points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nearest | 10 | 1900 | 0 | 1.00 | 1 | 1 |
| 10km | 9 | 1900 | 1064 | 1.00 | 1 | 1 |
| 25km | 9 | 1900 | 0 | 1.54 | 1 | 2 |
| 30km | 9 | 1900 | 0 | 1.88 | 1 | 3 |
| 50km | 9 | 1900 | 0 | 2.86 | 2 | 4 |
| 75km | 9 | 1900 | 0 | 4.08 | 2 | 5 |

## Interpretation Notes

Buffer aggregation reduces sensitivity to a single nearest OISST grid cell, but it does not create true nearshore in-situ temperature. The current local cache contains OISST points previously needed by the Version 1 workflow, so this V2 output should be interpreted as a reproducible multi-scale prototype. A publication-grade run should cache all OISST grid points intersecting each candidate buffer.
