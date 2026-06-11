# Regular 10 km Fishnet AOI Design

This folder contains a regular 10 km × 10 km fishnet grid for a pilot Kelpwatch-based early-warning modeling workflow across Northern and Central California.

## Design Logic

The number of cells is not pre-defined. Instead, a continuous 10 km × 10 km fishnet grid is generated across a simplified Northern/Central California coastal corridor. Cells are retained if they intersect a 15 km buffer around the coastal transect.

This avoids the earlier problem of selecting an arbitrary number of AOIs manually.

## Source of Truth and Reproducibility

The committed AOI files in this directory are package-derived files from `kelpwatch_regular_10km_fishnet_package.zip`. The package is treated as the source of truth for the accepted 285-cell design.

The repository includes two scripts for reproducibility:

- `scripts/extract_regular_10km_fishnet_package.py`: extracts the accepted package into the canonical repository structure and validates the expected outputs.
- `scripts/verify_regular_10km_fishnet_design.py`: checks the committed files without modifying or regenerating the grid.

Do not overwrite the committed GeoJSON files with regenerated output unless the regenerated files exactly reproduce the accepted package-derived design.

## Files

- `single_cell_geojsons/`: single-feature GeoJSON files for Kelpwatch upload
- `kelpwatch_regular_10km_fishnet_preview.geojson`: multi-feature file for visualization only
- `aoi_inventory_regular_10km_fishnet.csv`: inventory of all generated cells
- `grid_validation_regular_10km_fishnet.txt`: overlap validation
- `kelpwatch_regular_10km_fishnet_preview_map.html`: interactive preview map

## Kelpwatch Upload Rule

Kelpwatch rejects multi-feature uploads. Therefore, upload each file in `single_cell_geojsons/` individually.

## Modeling Filter

After downloading Kelpwatch CSV files, retain only cells with historical kelp footprint:

```text
count_cells_historic_footprint > 0
```

A stricter robustness filter may follow the Kelpwatch paper logic:

```text
count_cells_historic_footprint >= 500
```

## Generated Cell Count

Generated cells: 285
