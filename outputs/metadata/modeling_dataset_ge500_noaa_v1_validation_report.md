# NOAA V1 Modeling Dataset Validation Report

## Dataset Summary

- Input file: `data/processed/modeling_dataset_ge500_noaa_v1.csv`
- Row count: 2050
- Unique cells: 50
- Year range: 1984-2024
- Duplicate `cell_id x year` rows: 0

## Target Distribution

- `decline_event_next = 0`: 1354
- `decline_event_next = 1`: 696
- Decline-event rate: 0.3395

## Missing Values

- `lag1_beuti_anomaly`: 250 (0.1220)
- `lag1_cuti_anomaly`: 250 (0.1220)
- `spring_mean_beuti`: 200 (0.0976)
- `annual_mean_beuti`: 200 (0.0976)
- `beuti_anomaly`: 200 (0.0976)
- `summer_mean_beuti`: 200 (0.0976)
- `cuti_anomaly`: 200 (0.0976)
- `summer_mean_cuti`: 200 (0.0976)
- `spring_mean_cuti`: 200 (0.0976)
- `annual_mean_cuti`: 200 (0.0976)
- `beuti_lat_bin`: 200 (0.0976)
- `cuti_lat_bin`: 200 (0.0976)
- `lag1_hot_days_p90`: 50 (0.0244)
- `lag1_annual_mean_sst_anomaly`: 50 (0.0244)
- `relative_canopy_pct_change_next`: 10 (0.0049)

## Missingness Pattern Checks

- CUTI/BEUTI missing values occur only in 1984-1987: True
- CUTI/BEUTI lag missing values occur only in 1984-1988: True
- SST lag missing values occur only in 1984: True

CUTI/BEUTI missing years by feature:

- `cuti_lat_bin`: [1984, 1985, 1986, 1987]
- `beuti_lat_bin`: [1984, 1985, 1986, 1987]
- `annual_mean_cuti`: [1984, 1985, 1986, 1987]
- `spring_mean_cuti`: [1984, 1985, 1986, 1987]
- `summer_mean_cuti`: [1984, 1985, 1986, 1987]
- `cuti_anomaly`: [1984, 1985, 1986, 1987]
- `annual_mean_beuti`: [1984, 1985, 1986, 1987]
- `spring_mean_beuti`: [1984, 1985, 1986, 1987]
- `summer_mean_beuti`: [1984, 1985, 1986, 1987]
- `beuti_anomaly`: [1984, 1985, 1986, 1987]

Lag missing years by feature:

- `lag1_cuti_anomaly`: [1984, 1985, 1986, 1987, 1988]
- `lag1_beuti_anomaly`: [1984, 1985, 1986, 1987, 1988]
- `lag1_annual_mean_sst_anomaly`: [1984]
- `lag1_hot_days_p90`: [1984]

## Suspicious Value Checks

- `annual_mean_sst <= 0`: 0
- `annual_max_sst <= 0`: 0
- `annual_min_sst <= 0`: 0
- `annual_sst_std < 0`: 0
- `hot_days_p90 < 0`: 0
- `hot_days_p95 < 0`: 0
- `hot_days_p90 > 366`: 0
- `hot_days_p95 > 366`: 0
- `hot_days_p95 <= hot_days_p90` violations: 0
- `annual_min_sst <= annual_mean_sst <= annual_max_sst` violations: 0

## OISST Fallback Check

- Rows using fallback to neighboring valid OISST ocean grid: 41
- Cells using fallback: 1

| cell_id | nearest_lat | nearest_lon | source_lat | source_lon |
|---|---:|---:|---:|---:|
| cell_278 | 34.625 | -120.375 | 34.375 | -120.375 |

## CUTI/BEUTI Latitude Bin Assignment

CUTI latitude bins:
- `34.0`: 3 cells
- `35.0`: 7 cells
- `36.0`: 18 cells
- `37.0`: 7 cells
- `38.0`: 2 cells
- `39.0`: 13 cells

BEUTI latitude bins:
- `34.0`: 3 cells
- `35.0`: 7 cells
- `36.0`: 18 cells
- `37.0`: 7 cells
- `38.0`: 2 cells
- `39.0`: 13 cells

## Modeling Readiness

The NOAA V1 modeling dataset is ready for initial modeling. Remaining missingness is expected from CUTI/BEUTI availability beginning in 1988 and one-year lag construction. Models should either use algorithms that handle missing values or apply an explicit imputation strategy documented in the modeling workflow.

## Feature Range Output

Numeric feature ranges are written to `outputs/metadata/modeling_dataset_ge500_noaa_v1_feature_ranges.csv`.
