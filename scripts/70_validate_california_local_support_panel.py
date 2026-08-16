"""Validate the local KelpWatch support panel, including raw NetCDF spot checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pyproj import Transformer
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--netcdf", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add(rows: list[dict[str, object]], check: str, passed: bool, evidence: object, severity: str = "blocker") -> None:
    rows.append({"check": check, "passed": bool(passed), "severity_if_failed": severity, "evidence": str(evidence)})


def selected_sites(metadata: pd.DataFrame) -> pd.DataFrame:
    stable = metadata.loc[metadata["stable_support"].astype(str).str.lower().eq("true")].copy()
    rows = []
    for _, group in stable.groupby("support_m", sort=True):
        ordered = group.sort_values("site_id").reset_index(drop=True)
        positions = sorted({0, len(ordered) // 2, len(ordered) - 1})
        rows.append(ordered.iloc[positions])
    return pd.concat(rows, ignore_index=True)


def raw_spot_checks(
    config: dict[str, object],
    netcdf_path: Path,
    panel: pd.DataFrame,
    sites: pd.DataFrame,
) -> pd.DataFrame:
    transformer = Transformer.from_crs(config["source_crs"], config["analysis_crs"], always_xy=True)
    site_x, site_y = transformer.transform(sites["longitude"].to_numpy(), sites["latitude"].to_numpy())
    site_xy = np.column_stack([site_x, site_y])
    years_to_check = [2005, 2014, 2021]
    rows: list[dict[str, object]] = []
    with Dataset(netcdf_path) as dataset:
        longitude = np.asarray(dataset.variables["longitude"][:], dtype=float)
        latitude = np.asarray(dataset.variables["latitude"][:], dtype=float)
        coordinate_ok = np.isfinite(longitude) & np.isfinite(latitude)
        source_indices = np.flatnonzero(coordinate_ok)
        source_x, source_y = transformer.transform(longitude[coordinate_ok], latitude[coordinate_ok])
        source_xy = np.column_stack([source_x, source_y])
        tree = cKDTree(source_xy)
        all_lists = [
            np.asarray(tree.query_ball_point(point, r=float(site.support_m)), dtype=int)
            for point, site in zip(site_xy, sites.itertuples(index=False), strict=True)
        ]
        union_local = np.unique(np.concatenate(all_lists))
        union_global = source_indices[union_local]
        years_all = np.asarray(dataset.variables["year"][:], dtype=int)
        reference_indices = np.flatnonzero(years_all <= int(config["cohort"]["reference_end_year"]))
        prepositive_union = np.zeros(len(union_global), dtype=bool)
        for start in range(0, len(reference_indices), 8):
            chunk = reference_indices[start : start + 8]
            block = dataset.variables["area"][int(chunk[0]) : int(chunk[-1]) + 1, :]
            selected = block[:, union_global]
            mask = np.ma.getmaskarray(selected)
            values = np.asarray(selected.filled(0), dtype=float)
            prepositive_union |= np.any((~mask) & (values > 0), axis=0)
        prepositive_global = union_global[prepositive_union]
        global_to_position = {int(value): index for index, value in enumerate(prepositive_global)}
        check_time = np.flatnonzero(np.isin(years_all, years_to_check))
        check_values: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for time_index in check_time:
            full = dataset.variables["area"][int(time_index), :]
            selected = full[prepositive_global]
            check_values[int(time_index)] = (
                np.asarray(selected.filled(0), dtype=float),
                np.ma.getmaskarray(selected),
            )
        for list_local, site in zip(all_lists, sites.itertuples(index=False), strict=True):
            site_global = source_indices[list_local]
            positions = np.asarray([global_to_position[int(value)] for value in site_global if int(value) in global_to_position], dtype=int)
            footprint_pixels = len(positions)
            for year in years_to_check:
                time_indices = np.flatnonzero(years_all == year)
                quarterly_area = []
                quarterly_valid = []
                for time_index in time_indices:
                    values, mask = check_values[int(time_index)]
                    local_mask = mask[positions]
                    quarterly_area.append(float(values[positions][~local_mask].sum()))
                    quarterly_valid.append(int((~local_mask).sum()))
                quarterly_area_array = np.asarray(quarterly_area)
                valid_fraction = np.asarray(quarterly_valid) / footprint_pixels
                quarter_valid = valid_fraction >= float(config["annual_observation_rule"]["minimum_valid_habitat_fraction"])
                annual_complete = bool(quarter_valid.sum() >= int(config["annual_observation_rule"]["minimum_valid_quarters"]))
                if bool(config["annual_observation_rule"]["require_q3"]):
                    annual_complete &= bool(quarter_valid[2])
                masked = np.where(quarter_valid, quarterly_area_array, -np.inf)
                area = float(masked.max()) if annual_complete else np.nan
                quarter = int(masked.argmax() + 1) if annual_complete else 0
                saved = panel.loc[
                    panel["support_m"].eq(site.support_m)
                    & panel["site_id"].astype(str).eq(str(site.site_id))
                    & panel["year"].eq(year)
                ].iloc[0]
                rows.append(
                    {
                        "support_m": int(site.support_m),
                        "site_id": str(site.site_id),
                        "year": year,
                        "raw_footprint_pixels": footprint_pixels,
                        "saved_footprint_pixels": int(saved.pre2005_footprint_pixels),
                        "raw_annual_area_m2": area,
                        "saved_annual_area_m2": saved.annual_area_m2,
                        "raw_annual_complete": annual_complete,
                        "saved_annual_complete": bool(saved.annual_complete),
                        "raw_valid_quarters": int(quarter_valid.sum()),
                        "saved_valid_quarters": int(saved.valid_quarters),
                        "raw_selected_quarter": quarter,
                        "saved_selected_quarter": int(saved.selected_quarter),
                    }
                )
    result = pd.DataFrame(rows)
    result["area_matches"] = np.isclose(result["raw_annual_area_m2"], result["saved_annual_area_m2"], rtol=1e-12, atol=1e-6, equal_nan=True)
    result["all_rules_match"] = (
        result["area_matches"]
        & result["raw_footprint_pixels"].eq(result["saved_footprint_pixels"])
        & result["raw_annual_complete"].eq(result["saved_annual_complete"])
        & result["raw_valid_quarters"].eq(result["saved_valid_quarters"])
        & result["raw_selected_quarter"].eq(result["saved_selected_quarter"])
    )
    return result


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    panel = pd.read_csv(args.panel)
    metadata = pd.read_csv(args.metadata)
    field = pd.read_csv(args.field)
    rows: list[dict[str, object]] = []
    add(rows, "panel_keys_unique", not panel.duplicated(["support_m", "site_id", "year"]).any(), int(panel.duplicated(["support_m", "site_id", "year"]).sum()))
    add(rows, "metadata_keys_unique", not metadata.duplicated(["support_m", "site_id"]).any(), int(metadata.duplicated(["support_m", "site_id"]).sum()))
    add(rows, "field_keys_unique", not field.duplicated(["site_id", "year"]).any(), int(field.duplicated(["site_id", "year"]).sum()))
    expected_rows = int(panel["site_id"].groupby(panel["support_m"]).nunique().sum() * panel["year"].nunique())
    add(rows, "complete_site_year_rectangle", len(panel) == expected_rows, f"expected={expected_rows}; rows={len(panel)}")
    stable_keys = metadata.loc[metadata["stable_support"].astype(str).str.lower().eq("true"), ["support_m", "site_id"]].sort_values(["support_m", "site_id"]).reset_index(drop=True)
    panel_keys = panel[["support_m", "site_id"]].drop_duplicates().sort_values(["support_m", "site_id"]).reset_index(drop=True)
    add(rows, "panel_contains_exact_stable_support_sites", stable_keys.equals(panel_keys), f"metadata={len(stable_keys)}; panel={len(panel_keys)}")
    expected_event = panel["relative_drop_next"].ge(float(config["outcome"]["decline_fraction"])).astype(int)
    add(rows, "event_rule_recomputed", expected_event.eq(panel["event"]).all(), int(expected_event.ne(panel["event"]).sum()))
    expected_eligible = panel["annual_complete"].astype(bool) & panel["relative_canopy"].gt(float(config["outcome"]["eligibility_relative_canopy_gt"])) & panel["relative_drop_next"].notna()
    add(rows, "eligibility_rule_recomputed", expected_eligible.eq(panel["eligible"].astype(bool)).all(), int(expected_eligible.ne(panel["eligible"].astype(bool)).sum()))
    reference = panel.loc[panel["year"].between(int(config["cohort"]["reference_start_year"]), int(config["cohort"]["reference_end_year"]))]
    p95 = reference.groupby(["support_m", "site_id"])["annual_area_m2"].quantile(0.95).rename("p95_recheck").reset_index()
    saved_p95 = panel[["support_m", "site_id", "p95_annual_area_pre2005"]].drop_duplicates()
    p95_check = p95.merge(saved_p95, on=["support_m", "site_id"], validate="one_to_one")
    p95_error = float((p95_check["p95_recheck"] - p95_check["p95_annual_area_pre2005"]).abs().max())
    add(rows, "reference_p95_recomputed", p95_error < 1e-6, f"max_error={p95_error:.3g}")
    spot = raw_spot_checks(config, args.netcdf, panel, selected_sites(metadata))
    spot.to_csv(args.output_dir / "panel_raw_spot_checks.csv", index=False)
    add(rows, "raw_netcdf_spot_checks_match", bool(spot["all_rules_match"].all()), f"passed={int(spot['all_rules_match'].sum())}/{len(spot)}")
    checks = pd.DataFrame(rows)
    checks.to_csv(args.output_dir / "panel_validation_recheck.csv", index=False)
    blockers = checks.loc[~checks["passed"] & checks["severity_if_failed"].eq("blocker")]
    assessment = "Share with caveats" if blockers.empty else "Do not share"
    report = [
        "# California local-support panel 독립 검증",
        "",
        f"- 판정: **{assessment}**",
        f"- 통과: {int(checks['passed'].sum())}/{len(checks)}",
        f"- raw NetCDF spot checks: {int(spot['all_rules_match'].sum())}/{len(spot)}",
        "",
        "300 m와 1 km에서 각각 세 site, 2005·2014·2021년을 골라 고정 pre-2005 positive footprint, 분기 유효성, annual maximum과 선택분기를 공식 NetCDF에서 다시 계산했다.",
        "",
        checks.to_markdown(index=False),
    ]
    (args.output_dir / "panel_validation_report_ko.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "assessment": assessment,
        "checks": len(checks),
        "passed": int(checks["passed"].sum()),
        "blockers": len(blockers),
        "panel_sha256": sha256(args.panel),
        "netcdf_sha256": sha256(args.netcdf),
        "spot_check_sha256": sha256(args.output_dir / "panel_raw_spot_checks.csv"),
    }
    (args.output_dir / "panel_validation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(checks.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not blockers.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
