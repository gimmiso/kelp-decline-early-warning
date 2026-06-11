# Kelpwatch Fishnet Cell Filtering Report

## Summary

Total candidate cells: 285
Cells with historical footprint > 0: 74
Cells with historical footprint >= 500: 50

## Counts by Region Group

### Central California

- footprint > 0: 40
- footprint >= 500: 34

### Northern California

- footprint > 0: 34
- footprint >= 500: 16

## Filtering Logic

The regular 10 km fishnet generated 285 candidate cells across the Northern and Central California coastal corridor. After querying Kelpwatch, many candidate cells reported zero historical kelp footprint, indicating that no kelp canopy was observed within those cells during the historical observation period. These cells are excluded before modeling.

The exploratory dataset retains all cells with `count_cells_historic_footprint > 0`. The main modeling dataset applies a stricter threshold of `count_cells_historic_footprint >= 500`, following the logic of previous Kelpwatch-based studies that excluded 10 km cells with very low potential kelp habitat.

## Outputs

- `aoi_inventory_regular_10km_fishnet_with_filters.csv`
- `filtered_cells_historic_footprint_gt0.csv`
- `filtered_cells_historic_footprint_ge500.csv`
