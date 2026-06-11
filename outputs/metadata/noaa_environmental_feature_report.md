# NOAA Environmental Feature Report

## Summary

Number of cells: 50
Year range: 1984-2024
Number of unique nearest OISST grid points: 26
Number of CUTI/BEUTI latitude bins: 6
Merged dataset row count: 2050
Baseline period: 1984-2013
Runtime seconds: 2.33

## NOAA Data Access Notes

noaa_data_access_notes: OISST: SST_OI_DAILY_1981_PRESENT_HL via https://erddap.aoml.noaa.gov/hdb/erddap/griddap/SST_OI_DAILY_1981_PRESENT_HL.nc; CUTI: erdCUTIdaily via https://upwell.pfeg.noaa.gov/erddap; BEUTI: erdBEUTIdaily via https://upwell.pfeg.noaa.gov/erddap; CUTI/BEUTI daily values are aggregated to annual, spring, and summer summaries.

OISST: SST_OI_DAILY_1981_PRESENT_HL via https://erddap.aoml.noaa.gov/hdb/erddap/griddap/SST_OI_DAILY_1981_PRESENT_HL.nc; CUTI: erdCUTIdaily via https://upwell.pfeg.noaa.gov/erddap; BEUTI: erdBEUTIdaily via https://upwell.pfeg.noaa.gov/erddap; CUTI/BEUTI daily values are aggregated to annual, spring, and summer summaries.

## Feature List

- `nearest_oisst_lat`
- `nearest_oisst_lon`
- `oisst_source_lat`
- `oisst_source_lon`
- `oisst_observed_days`
- `oisst_gap_filled`
- `annual_mean_sst`
- `annual_max_sst`
- `annual_min_sst`
- `annual_sst_std`
- `annual_mean_sst_anomaly`
- `annual_max_sst_anomaly`
- `hot_days_p90`
- `hot_days_p95`
- `lag1_annual_mean_sst_anomaly`
- `lag1_hot_days_p90`
- `cuti_lat_bin`
- `beuti_lat_bin`
- `annual_mean_cuti`
- `spring_mean_cuti`
- `summer_mean_cuti`
- `cuti_anomaly`
- `lag1_cuti_anomaly`
- `annual_mean_beuti`
- `spring_mean_beuti`
- `summer_mean_beuti`
- `beuti_anomaly`
- `lag1_beuti_anomaly`

## Target Distribution After Merge

- decline_event_next = 0: 1354
- decline_event_next = 1: 696

## Missing Value Counts

- `nearest_oisst_lat`: 0 (0.0000)
- `nearest_oisst_lon`: 0 (0.0000)
- `oisst_source_lat`: 0 (0.0000)
- `oisst_source_lon`: 0 (0.0000)
- `oisst_observed_days`: 0 (0.0000)
- `oisst_gap_filled`: 0 (0.0000)
- `annual_mean_sst`: 0 (0.0000)
- `annual_max_sst`: 0 (0.0000)
- `annual_min_sst`: 0 (0.0000)
- `annual_sst_std`: 0 (0.0000)
- `annual_mean_sst_anomaly`: 0 (0.0000)
- `annual_max_sst_anomaly`: 0 (0.0000)
- `hot_days_p90`: 0 (0.0000)
- `hot_days_p95`: 0 (0.0000)
- `lag1_annual_mean_sst_anomaly`: 50 (0.0244)
- `lag1_hot_days_p90`: 50 (0.0244)
- `cuti_lat_bin`: 200 (0.0976)
- `beuti_lat_bin`: 200 (0.0976)
- `annual_mean_cuti`: 200 (0.0976)
- `spring_mean_cuti`: 200 (0.0976)
- `summer_mean_cuti`: 200 (0.0976)
- `cuti_anomaly`: 200 (0.0976)
- `lag1_cuti_anomaly`: 250 (0.1220)
- `annual_mean_beuti`: 200 (0.0976)
- `spring_mean_beuti`: 200 (0.0976)
- `summer_mean_beuti`: 200 (0.0976)
- `beuti_anomaly`: 200 (0.0976)
- `lag1_beuti_anomaly`: 250 (0.1220)

## Progress Log Summary

- cells processed: 50
- unique OISST grid points: 26
- CUTI/BEUTI latitude bins: 6
- downloaded files: 0
- cached files reused: 29

## Notes on Limitations

- OISST is assigned using the nearest 0.25-degree grid point. A coastal-buffer average is reserved for sensitivity analysis.
- 2024 OISST annual features use data available through 2024-11-27; rerun after NOAA archive updates for complete 2024 daily coverage.
- CUTI/BEUTI begin in 1988, so 1984-1987 rows have missing CUTI/BEUTI values in Version 1.
- Daily CUTI/BEUTI are aggregated to annual, spring, and summer summaries; daily-window features can be added later.
