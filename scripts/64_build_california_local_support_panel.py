"""Build 300-m and 1-km KelpWatch panels around Giraldo field sites."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


AXES = [
    "grazing_log1p",
    "rock_probability",
    "depth_m",
    "kelp_species_contrast",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--netcdf", type=Path, required=True)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rolling_slope(values: np.ndarray) -> float:
    if len(values) != 3 or np.isnan(values).any():
        return np.nan
    return float(np.polyfit(np.arange(3), values, 1)[0])


def add_trajectory(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.sort_values(["support_m", "site_id", "year"]).copy()
    keys = ["support_m", "site_id"]
    grouped = out.groupby(keys, group_keys=False)["relative_canopy"]
    out["canopy_lag1"] = grouped.shift(1)
    out["canopy_lag2"] = grouped.shift(2)
    lag3 = grouped.shift(3)
    out["canopy_2yr_change"] = out["relative_canopy"] - out["canopy_lag2"]
    out["canopy_3yr_change"] = out["relative_canopy"] - lag3
    rolling = grouped.rolling(3, min_periods=3)
    out["canopy_3yr_mean"] = rolling.mean().reset_index(level=keys, drop=True)
    out["canopy_3yr_std"] = rolling.std().reset_index(level=keys, drop=True)
    out["canopy_3yr_slope"] = rolling.apply(rolling_slope, raw=True).reset_index(level=keys, drop=True)
    rolling_max = rolling.max().reset_index(level=keys, drop=True)
    out["canopy_drop_from_3yr_max"] = (
        rolling_max - out["relative_canopy"]
    ) / np.maximum(rolling_max, 1e-9)
    return out


def prepare_field(field: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "site_campus_unique_ID",
        "survey_year",
        "latitude",
        "longitude",
        "region",
        "transect",
        "den_STRPURAD",
        "den_MESFRAAD",
        "den_MACPYRAD",
        "den_NERLUE",
        "mean_prob_of_rock",
        "mean_depth",
    }
    missing = sorted(required - set(field.columns))
    if missing:
        raise ValueError(f"Missing Giraldo columns: {missing}")
    data = field[list(required)].copy()
    data["site_id"] = data["site_campus_unique_ID"].astype(str)
    data["grazing_log1p"] = np.log1p(
        data[["den_STRPURAD", "den_MESFRAAD"]].sum(axis=1, min_count=1)
    )
    data["rock_probability"] = data["mean_prob_of_rock"]
    data["depth_m"] = -data["mean_depth"]
    data["kelp_species_contrast"] = np.log1p(data["den_NERLUE"]) - np.log1p(data["den_MACPYRAD"])
    region_map = {"nc": "Northern California", "cc": "Central California", "sc": "Southern California"}
    data["region_group"] = data["region"].str.lower().map(region_map)
    site_year = (
        data.groupby(["site_id", "survey_year"], as_index=False)
        .agg(
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            region_group=("region_group", "first"),
            grazing_log1p=("grazing_log1p", "median"),
            rock_probability=("rock_probability", "median"),
            depth_m=("depth_m", "median"),
            kelp_species_contrast=("kelp_species_contrast", "median"),
            n_transects=("transect", "size"),
        )
        .rename(columns={"survey_year": "year"})
    )
    sites = (
        site_year.groupby("site_id", as_index=False)
        .agg(
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            region_group=("region_group", "first"),
            field_years=("year", "nunique"),
        )
    )
    return sites, site_year


def aggregate_support(
    support_m: int,
    sites: pd.DataFrame,
    site_xy: np.ndarray,
    source_xy: np.ndarray,
    source_indices: np.ndarray,
    prepositive: np.ndarray,
    dataset: Dataset,
    time_indices: np.ndarray,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tree = cKDTree(source_xy[prepositive])
    local_source_indices = source_indices[prepositive]
    pixel_lists = tree.query_ball_point(site_xy, r=float(support_m))
    footprint_counts = np.asarray([len(values) for values in pixel_lists], dtype=int)
    minimum_pixels = int(config["cohort"]["minimum_prepositive_pixels"][str(support_m)])
    candidate = footprint_counts >= minimum_pixels
    candidate_sites = sites.loc[candidate].reset_index(drop=True)
    candidate_xy = site_xy[candidate]
    candidate_lists = [pixel_lists[index] for index in np.flatnonzero(candidate)]
    candidate_counts = footprint_counts[candidate]

    clusters = DBSCAN(eps=float(2 * support_m), min_samples=1).fit_predict(candidate_xy)
    site_metadata = candidate_sites.copy()
    site_metadata["support_m"] = support_m
    site_metadata["overlap_cluster"] = [f"s{support_m}_c{value:03d}" for value in clusters]
    site_metadata["pre2005_footprint_pixels"] = candidate_counts

    n_times = len(time_indices)
    n_sites = len(candidate_sites)
    area = np.zeros((n_times, n_sites), dtype=float)
    valid_count = np.zeros((n_times, n_sites), dtype=int)
    area_variable = dataset.variables["area"]
    for local_time, source_time in enumerate(time_indices):
        full_row = area_variable[source_time, :]
        selected_all = full_row[local_source_indices]
        mask_all = np.ma.getmaskarray(selected_all)
        values_all = np.asarray(selected_all.filled(0), dtype=float)
        for site_index, local_pixels in enumerate(candidate_lists):
            local_mask = mask_all[local_pixels]
            local_values = values_all[local_pixels]
            valid_count[local_time, site_index] = int((~local_mask).sum())
            area[local_time, site_index] = float(local_values[~local_mask].sum())
        if (local_time + 1) % 20 == 0 or local_time + 1 == n_times:
            print(
                f"KelpWatch local support {support_m} m: "
                f"{local_time + 1}/{n_times} quarters, {n_sites} candidate sites",
                flush=True,
            )
        del full_row, selected_all, mask_all, values_all

    years = np.asarray(dataset.variables["year"][:], dtype=int)[time_indices]
    quarters = np.asarray(dataset.variables["quarter"][:], dtype=int)[time_indices]
    unique_years = np.arange(1984, 2026)
    if not np.array_equal(years, np.repeat(unique_years, 4)) or not np.array_equal(quarters, np.tile(np.arange(1, 5), len(unique_years))):
        raise ValueError("Expected complete ordered 1984-2025 quarterly KelpWatch data")
    area = area.reshape(len(unique_years), 4, n_sites)
    valid_count = valid_count.reshape(len(unique_years), 4, n_sites)
    valid_fraction = valid_count / candidate_counts[np.newaxis, np.newaxis, :]
    quarter_valid = valid_fraction >= float(config["annual_observation_rule"]["minimum_valid_habitat_fraction"])
    complete = quarter_valid.sum(axis=1) >= int(config["annual_observation_rule"]["minimum_valid_quarters"])
    if bool(config["annual_observation_rule"]["require_q3"]):
        complete &= quarter_valid[:, 2, :]
    masked_area = np.where(quarter_valid, area, -np.inf)
    annual_area = masked_area.max(axis=1)
    selected_quarter = masked_area.argmax(axis=1) + 1
    annual_area[~complete] = np.nan
    selected_quarter[~complete] = 0
    ref = unique_years <= int(config["cohort"]["reference_end_year"])
    positive_years = np.sum(np.isfinite(annual_area[ref]) & (annual_area[ref] > 0), axis=0)
    stable = positive_years >= int(config["cohort"]["minimum_positive_reference_years"])
    site_metadata["positive_reference_years"] = positive_years
    site_metadata["stable_support"] = stable
    site_metadata = site_metadata.loc[stable].reset_index(drop=True)
    annual_area = annual_area[:, stable]
    complete = complete[:, stable]
    selected_quarter = selected_quarter[:, stable]
    valid_quarters = quarter_valid.sum(axis=1)[:, stable]
    p95 = np.nanquantile(annual_area[ref], 0.95, axis=0)

    rows: list[pd.DataFrame] = []
    for index, site in site_metadata.iterrows():
        frame = pd.DataFrame(
            {
                "site_id": site.site_id,
                "support_m": support_m,
                "overlap_cluster": site.overlap_cluster,
                "region_group": site.region_group,
                "latitude": site.latitude,
                "longitude": site.longitude,
                "year": unique_years,
                "annual_area_m2": annual_area[:, index],
                "annual_complete": complete[:, index],
                "valid_quarters": valid_quarters[:, index],
                "selected_quarter": selected_quarter[:, index],
                "pre2005_footprint_pixels": site.pre2005_footprint_pixels,
                "positive_reference_years": site.positive_reference_years,
                "p95_annual_area_pre2005": p95[index],
            }
        )
        rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    panel["relative_canopy"] = panel["annual_area_m2"] / panel["p95_annual_area_pre2005"]
    panel["next_year_area_m2"] = panel.groupby(["support_m", "site_id"])["annual_area_m2"].shift(-1)
    panel["relative_drop_next"] = (panel["annual_area_m2"] - panel["next_year_area_m2"]) / panel["annual_area_m2"]
    panel.loc[panel["annual_area_m2"].le(0), "relative_drop_next"] = np.nan
    panel["event"] = panel["relative_drop_next"].ge(float(config["outcome"]["decline_fraction"])).astype(int)
    panel["eligible"] = (
        panel["annual_complete"]
        & panel["relative_canopy"].gt(float(config["outcome"]["eligibility_relative_canopy_gt"]))
        & panel["relative_drop_next"].notna()
    )
    return panel, site_metadata


def main() -> None:
    args = parse_args()
    if (args.output_dir / "panel_manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed panel: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "panel_command.txt").write_text(
        shlex.join(sys.argv) + "\n", encoding="utf-8"
    )
    started = time.time()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    field = pd.read_csv(args.field, low_memory=False)
    sites_all, site_year = prepare_field(field)
    missing_coordinates = sites_all["latitude"].isna() | sites_all["longitude"].isna()
    sites = sites_all.loc[~missing_coordinates].reset_index(drop=True)
    transformer = Transformer.from_crs(config["source_crs"], config["analysis_crs"], always_xy=True)
    site_x, site_y = transformer.transform(sites["longitude"].to_numpy(), sites["latitude"].to_numpy())
    site_xy = np.column_stack([site_x, site_y])

    panels = []
    metadata = []
    with Dataset(args.netcdf) as dataset:
        longitude = np.asarray(dataset.variables["longitude"][:], dtype=float)
        latitude = np.asarray(dataset.variables["latitude"][:], dtype=float)
        coordinate_ok = np.isfinite(longitude) & np.isfinite(latitude)
        source_indices = np.flatnonzero(coordinate_ok)
        source_x, source_y = transformer.transform(longitude[coordinate_ok], latitude[coordinate_ok])
        source_xy = np.column_stack([source_x, source_y])
        max_support = max(int(value) for value in config["spatial_supports_m"])
        nearby_lists = cKDTree(source_xy).query_ball_point(site_xy, r=float(max_support))
        nearby_local = np.unique(np.concatenate([np.asarray(values, dtype=int) for values in nearby_lists if values]))
        restricted_source_indices = source_indices[nearby_local]
        restricted_xy = source_xy[nearby_local]
        years_all = np.asarray(dataset.variables["year"][:], dtype=int)
        reference_indices = np.flatnonzero(
            years_all <= int(config["cohort"]["reference_end_year"])
        )
        if not np.all(np.diff(reference_indices) == 1):
            raise ValueError("Expected contiguous pre-forecast reference quarters")
        prepositive = np.zeros(len(restricted_source_indices), dtype=bool)
        reference_chunk_size = 8
        for chunk_start in range(0, len(reference_indices), reference_chunk_size):
            chunk = reference_indices[
                chunk_start : chunk_start + reference_chunk_size
            ]
            full_block = dataset.variables["area"][
                int(chunk[0]) : int(chunk[-1]) + 1, :
            ]
            selected_block = full_block[:, restricted_source_indices]
            block_mask = np.ma.getmaskarray(selected_block)
            block_values = np.asarray(selected_block.filled(0), dtype=float)
            prepositive |= np.any(
                (~block_mask) & (block_values > 0), axis=0
            )
            print(
                f"KelpWatch reference footprint: {chunk_start + len(chunk)}/"
                f"{len(reference_indices)} quarters",
                flush=True,
            )
            del full_block, selected_block, block_mask, block_values
        print(
            f"KelpWatch reference footprint: {len(reference_indices)} quarters, "
            f"{len(restricted_source_indices)} nearby pixels, "
            f"{int(prepositive.sum())} pre-2005 positive pixels",
            flush=True,
        )
        time_indices = np.flatnonzero(years_all <= 2025)
        for support_m in config["spatial_supports_m"]:
            panel, site_meta = aggregate_support(
                int(support_m),
                sites,
                site_xy,
                restricted_xy,
                restricted_source_indices,
                prepositive,
                dataset,
                time_indices,
                config,
            )
            panels.append(panel)
            metadata.append(site_meta)

    panel = add_trajectory(pd.concat(panels, ignore_index=True))
    site_metadata = pd.concat(metadata, ignore_index=True)
    if panel.duplicated(["support_m", "site_id", "year"]).any():
        raise ValueError("Local support panel keys are not unique")
    site_year = site_year.merge(
        sites[["site_id"]], on="site_id", how="inner", validate="many_to_one"
    )
    outputs = {
        "local_support_panel.csv": panel,
        "local_support_site_metadata.csv": site_metadata,
        "field_site_year_ecology.csv": site_year,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    coverage_rows = []
    for support_m, group in panel.groupby("support_m"):
        eligible = group.loc[group["year"].between(2005, 2021) & group["eligible"]]
        ecology_keys = site_year[["site_id", "year"]].drop_duplicates()
        matched = eligible.merge(ecology_keys, on=["site_id", "year"], how="inner")
        coverage_rows.append(
            {
                "support_m": int(support_m),
                "stable_sites": int(group["site_id"].nunique()),
                "overlap_clusters": int(group["overlap_cluster"].nunique()),
                "eligible_forecast_rows": int(len(eligible)),
                "forecast_events": int(eligible["event"].sum()),
                "field_matched_rows": int(len(matched)),
                "field_matched_sites": int(matched["site_id"].nunique()),
                "field_matched_events": int(matched["event"].sum()),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(args.output_dir / "support_coverage.csv", index=False)
    manifest = {
        "status": "complete",
        "protocol": config["protocol"],
        "runtime_seconds": round(time.time() - started, 2),
        "inputs": {
            "config": {"path": str(args.config), "sha256": sha256(args.config)},
            "netcdf": {"path": str(args.netcdf), "sha256": sha256(args.netcdf)},
            "field": {"path": str(args.field), "sha256": sha256(args.field)},
        },
        "raw_field_sites": int(field["site_campus_unique_ID"].nunique()),
        "sites_with_coordinates": int(len(sites)),
        "sites_missing_coordinates": int(missing_coordinates.sum()),
        "nearby_source_pixels": int(len(restricted_source_indices)),
        "pre2005_positive_nearby_pixels": int(prepositive.sum()),
        "coverage": coverage.to_dict(orient="records"),
        "output_sha256": {
            "panel_command.txt": sha256(args.output_dir / "panel_command.txt"),
            **{name: sha256(args.output_dir / name) for name in outputs},
        },
    }
    (args.output_dir / "panel_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["coverage"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
