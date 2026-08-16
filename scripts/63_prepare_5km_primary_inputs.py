"""Prepare the locked 5-km coastwide panel and matched NOAA feature table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


OISST = [
    "annual_mean_sst_anomaly",
    "annual_max_sst",
    "hot_weeks_p90",
    "winter_mean_sst_anomaly",
    "summer_mean_sst_anomaly",
    "lag1_annual_mean_sst_anomaly",
]
UPWELLING = [
    "annual_mean_cuti",
    "cuti_anomaly",
    "winter_mean_cuti",
    "winter_cuti_anomaly",
    "spring_mean_cuti",
    "spring_cuti_anomaly",
    "upwelling_season_mean_cuti",
    "upwelling_season_cuti_anomaly",
    "annual_mean_beuti",
    "beuti_anomaly",
    "winter_mean_beuti",
    "winter_beuti_anomaly",
    "spring_mean_beuti",
    "spring_beuti_anomaly",
    "upwelling_season_mean_beuti",
    "upwelling_season_beuti_anomaly",
    "lag1_cuti_anomaly",
    "lag1_beuti_anomaly",
]
SUPPORTED_REGIONS = {
    "Southern California",
    "Central California",
    "Northern California",
    "Oregon",
    "Washington outer coast",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-panels", type=Path, required=True)
    parser.add_argument("--source-environment", type=Path, required=True)
    parser.add_argument("--panel-output", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--spec-id", default="g05_area_scaled_o00")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.grid_panels)
    panel = source.loc[source["spec_id"].eq(args.spec_id)].copy()
    if panel.empty:
        raise ValueError(f"No rows for {args.spec_id}")
    if panel.duplicated(["cell_id", "year"]).any():
        raise ValueError("5-km source panel has duplicate cell-year keys")

    panel = panel.rename(
        columns={
            "annual_area_m2": "annual_max_kelp_area_m2",
            "annual_complete": "annual_complete_enough",
            "pre2005_footprint_pixels": "count_cells_pre2005_footprint",
        }
    )
    panel_columns = [
        "cell_id",
        "region_group",
        "center_lat",
        "center_lon",
        "p95_annual_area_pre2005",
        "count_cells_pre2005_footprint",
        "year",
        "annual_max_kelp_area_m2",
        "valid_quarters",
        "annual_complete_enough",
        "relative_canopy",
        "next_year_area_m2",
        "relative_drop_next",
        "event",
        "canopy_lag1",
        "canopy_lag2",
        "canopy_2yr_change",
        "canopy_3yr_change",
        "canopy_3yr_mean",
        "canopy_3yr_std",
        "canopy_3yr_slope",
        "canopy_drop_from_3yr_max",
        "eligible",
    ]
    panel_out = panel[panel_columns].copy()

    environment = panel[["cell_id", "year", *OISST]].copy()
    environment["oisst_supported"] = panel["oisst_supported"].fillna(False).astype(bool)
    environment.loc[~environment["oisst_supported"], OISST] = np.nan

    source_environment = pd.read_csv(args.source_environment)
    upwelling_columns = ["upwelling_lat_bin", "year", *UPWELLING]
    upwelling_source = source_environment.loc[
        source_environment["upwelling_lat_bin"].notna(), upwelling_columns
    ].copy()
    consistency = upwelling_source.groupby(["upwelling_lat_bin", "year"])[UPWELLING].nunique(dropna=False)
    if consistency.gt(1).any().any():
        raise ValueError("Upwelling features are not unique by latitude bin and year")
    upwelling_source = upwelling_source.drop_duplicates(["upwelling_lat_bin", "year"])

    cells = panel[["cell_id", "center_lat", "region_group"]].drop_duplicates("cell_id")
    cells["upwelling_supported"] = (
        cells["center_lat"].between(31.0, 47.0, inclusive="both")
        & cells["region_group"].isin(SUPPORTED_REGIONS)
    )
    cells["upwelling_lat_bin"] = np.where(
        cells["upwelling_supported"], cells["center_lat"].round(), np.nan
    )
    keys = panel[["cell_id", "year"]].copy()
    upwelling = keys.merge(
        cells[["cell_id", "upwelling_supported", "upwelling_lat_bin"]],
        on="cell_id",
        how="left",
        validate="many_to_one",
    ).merge(
        upwelling_source,
        on=["upwelling_lat_bin", "year"],
        how="left",
        validate="many_to_one",
    )
    environment = environment.merge(
        upwelling,
        on=["cell_id", "year"],
        how="left",
        validate="one_to_one",
    )
    if len(environment) != len(panel_out) or environment.duplicated(["cell_id", "year"]).any():
        raise ValueError("Environment construction changed the 5-km panel grain")

    args.panel_output.parent.mkdir(parents=True, exist_ok=True)
    args.environment_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    panel_out.to_csv(args.panel_output, index=False)
    environment.to_csv(args.environment_output, index=False)

    forecast = panel_out.loc[
        panel_out["year"].between(2005, 2024)
        & panel_out["relative_canopy"].gt(0.05)
        & panel_out["relative_drop_next"].notna()
    ]
    audit = {
        "spec_id": args.spec_id,
        "cells": int(panel_out["cell_id"].nunique()),
        "cell_years": int(len(panel_out)),
        "year_min": int(panel_out["year"].min()),
        "year_max": int(panel_out["year"].max()),
        "duplicate_panel_keys": int(panel_out.duplicated(["cell_id", "year"]).sum()),
        "duplicate_environment_keys": int(environment.duplicated(["cell_id", "year"]).sum()),
        "eligible_forecast_rows": int(len(forecast)),
        "forecast_events": int(forecast["relative_drop_next"].ge(0.30).sum()),
        "oisst_supported_cells": int(panel.loc[panel["oisst_supported"].fillna(False), "cell_id"].nunique()),
        "upwelling_supported_cells": int(cells["upwelling_supported"].sum()),
        "source_sha256": sha256(args.grid_panels),
        "source_environment_sha256": sha256(args.source_environment),
        "panel_sha256": sha256(args.panel_output),
        "environment_sha256": sha256(args.environment_output),
    }
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
