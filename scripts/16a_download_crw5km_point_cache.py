"""Download local NOAA CRW CoralTemp 5 km daily point caches.

The main CRW feature script expects local point CSV files under
``data/external/noaa/cache/crw5km``. This helper identifies retained Kelpwatch
cell centroids, starts from the snapped 0.05-degree CRW grid cell, searches for
the nearest valid ocean pixel when the snapped point is land/missing, and writes
one daily CSV per unique CRW source point.

Downloaded cache files are external data and are ignored by Git.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests


CRW_SCRIPT = Path("scripts/16_build_crw_5km_sst_features.py")
DEFAULT_CACHE_DIR = Path("data/external/noaa/cache/crw5km")
DEFAULT_LOG_OUTPUT = Path("outputs/metadata/crw5km_point_cache_download_log.csv")
DEFAULT_POINT_INVENTORY_OUTPUT = Path("outputs/metadata/crw5km_point_cache_inventory.csv")
CRW_ENDPOINTS = {
    "pacioos": "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dhw_5km.csv",
    "coastwatch": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW.csv",
}
START_DATE = "1985-04-01T12:00:00Z"
END_DATE = "2024-12-31T12:00:00Z"
PROBE_START_DATE = "2021-08-01T12:00:00Z"
PROBE_END_DATE = "2021-08-14T12:00:00Z"
GRID_STEP = 0.05
DOWNLOAD_CHUNKS = [
    ("1985-04-01T12:00:00Z", "1989-12-31T12:00:00Z"),
    ("1990-01-01T12:00:00Z", "1994-12-31T12:00:00Z"),
    ("1995-01-01T12:00:00Z", "1999-12-31T12:00:00Z"),
    ("2000-01-01T12:00:00Z", "2004-12-31T12:00:00Z"),
    ("2005-01-01T12:00:00Z", "2009-12-31T12:00:00Z"),
    ("2010-01-01T12:00:00Z", "2014-12-31T12:00:00Z"),
    ("2015-01-01T12:00:00Z", "2019-12-31T12:00:00Z"),
    ("2020-01-01T12:00:00Z", "2024-12-31T12:00:00Z"),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download CRW 5 km daily point cache files.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--log-output", type=Path, default=DEFAULT_LOG_OUTPUT)
    parser.add_argument("--point-inventory-output", type=Path, default=DEFAULT_POINT_INVENTORY_OUTPUT)
    parser.add_argument("--limit-cells", type=int, default=None)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument("--max-search-km", type=float, default=35.0)
    parser.add_argument("--endpoint", choices=sorted(CRW_ENDPOINTS), default="pacioos")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_crw_module():
    """Load the CRW feature module despite its numeric filename prefix."""
    spec = importlib.util.spec_from_file_location("crw_features", CRW_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {CRW_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["crw_features"] = module
    spec.loader.exec_module(module)
    return module


def safe_coord(value: float) -> str:
    """Encode a coordinate for stable CRW cache filenames."""
    return f"{value:.3f}".replace("-", "m")


def cache_path(cache_dir: Path, lat: float, lon: float) -> Path:
    """Return expected local CRW point-cache path."""
    return cache_dir / f"crw5km_lat{safe_coord(lat)}_lon{safe_coord(lon)}_daily.csv"


def erddap_url(base_url: str, lat: float, lon: float, start_date: str, end_date: str, variables: list[str]) -> str:
    """Build a CRW ERDDAP griddap CSV URL for one point and date range."""
    parts = [
        f"{variable}[({start_date}):1:({end_date})][({lat:.3f})][({lon:.3f})]"
        for variable in variables
    ]
    return f"{base_url}?{quote(','.join(parts), safe='?,=&[]():')}"


def request_csv(url: str, timeout: int = 240) -> pd.DataFrame:
    """Request a CSV from ERDDAP and parse it with the units row skipped."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    text = response.text
    if text.startswith("Error") or "Error {" in text[:500]:
        raise RuntimeError(text[:500])
    from io import StringIO

    return pd.read_csv(StringIO(text), skiprows=[1])


def has_valid_sst(frame: pd.DataFrame) -> bool:
    """Return whether a probe/full response contains at least one valid SST."""
    if "CRW_SST" not in frame.columns:
        return False
    return pd.to_numeric(frame["CRW_SST"], errors="coerce").notna().any()


def candidate_points(crw, cell: pd.Series, max_search_km: float) -> pd.DataFrame:
    """Generate nearby CRW grid candidates sorted by distance to a cell centroid."""
    max_steps = int(np.ceil(max_search_km / 4.5))
    rows = []
    for dlat_step in range(-max_steps, max_steps + 1):
        for dlon_step in range(-max_steps, max_steps + 1):
            lat = round(float(cell["nearest_crw_lat"]) + dlat_step * GRID_STEP, 3)
            lon = round(float(cell["nearest_crw_lon"]) + dlon_step * GRID_STEP, 3)
            distance = float(
                crw.haversine_km(
                    np.array([float(cell["center_lat"])]),
                    np.array([float(cell["center_lon"])]),
                    np.array([lat]),
                    np.array([lon]),
                )[0]
            )
            if distance <= max_search_km:
                rows.append({"candidate_lat": lat, "candidate_lon": lon, "distance_to_cell_km": distance})
    return pd.DataFrame(rows).sort_values("distance_to_cell_km").drop_duplicates(
        ["candidate_lat", "candidate_lon"]
    )


def choose_valid_point(
    crw,
    cell: pd.Series,
    cache_dir: Path,
    base_url: str,
    max_search_km: float,
    delay_seconds: float,
) -> dict[str, object]:
    """Choose the nearest CRW candidate with valid SST values."""
    for candidate in candidate_points(crw, cell, max_search_km).itertuples(index=False):
        path = cache_path(cache_dir, candidate.candidate_lat, candidate.candidate_lon)
        if path.exists():
            try:
                cached = pd.read_csv(path, nrows=20)
                if has_valid_sst(cached) or len(cached) > 20:
                    return {
                        "selected_lat": candidate.candidate_lat,
                        "selected_lon": candidate.candidate_lon,
                        "distance_to_cell_km": candidate.distance_to_cell_km,
                        "probe_status": "cached_existing",
                    }
            except Exception:
                pass
        probe_url = erddap_url(
            base_url,
            candidate.candidate_lat,
            candidate.candidate_lon,
            PROBE_START_DATE,
            PROBE_END_DATE,
            ["CRW_SST"],
        )
        try:
            probe = request_csv(probe_url, timeout=90)
            if has_valid_sst(probe):
                return {
                    "selected_lat": candidate.candidate_lat,
                    "selected_lon": candidate.candidate_lon,
                    "distance_to_cell_km": candidate.distance_to_cell_km,
                    "probe_status": "valid_ocean_pixel",
                }
        except Exception as exc:
            last_error = str(exc)[:250]
        else:
            last_error = "probe_returned_no_valid_sst"
        time.sleep(delay_seconds)
    return {
        "selected_lat": np.nan,
        "selected_lon": np.nan,
        "distance_to_cell_km": np.nan,
        "probe_status": f"no_valid_pixel_within_{max_search_km:g}km: {last_error}",
    }


def download_full_point(
    lat: float,
    lon: float,
    path: Path,
    base_url: str,
    endpoint_name: str,
    force: bool,
    delay_seconds: float,
) -> dict[str, object]:
    """Download one full daily CRW point time series if needed."""
    if path.exists() and not force:
        cached = pd.read_csv(path)
        valid_rows = int(pd.to_numeric(cached.get("CRW_SST", pd.Series(dtype=float)), errors="coerce").notna().sum())
        return {
            "status": "reused_cache",
            "endpoint": endpoint_name,
            "rows": int(len(cached)),
            "valid_sst_rows": valid_rows,
            "error_message": "",
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frames = []
        for chunk_start, chunk_end in DOWNLOAD_CHUNKS:
            url = erddap_url(base_url, lat, lon, chunk_start, chunk_end, ["CRW_SST", "CRW_SSTANOMALY"])
            frames.append(request_csv(url, timeout=180))
            time.sleep(delay_seconds)
        frame = pd.concat(frames, ignore_index=True).drop_duplicates(["time", "latitude", "longitude"])
        frame.to_csv(path, index=False)
        valid_rows = int(pd.to_numeric(frame.get("CRW_SST", pd.Series(dtype=float)), errors="coerce").notna().sum())
        status = "downloaded" if valid_rows > 0 else "downloaded_no_valid_sst"
        return {
            "status": status,
            "endpoint": endpoint_name,
            "rows": int(len(frame)),
            "valid_sst_rows": valid_rows,
            "error_message": "",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "endpoint": endpoint_name,
            "rows": 0,
            "valid_sst_rows": 0,
            "error_message": str(exc)[:500],
        }


def main() -> None:
    """Download CRW 5 km point caches for retained Kelpwatch cells."""
    args = parse_args()
    crw = load_crw_module()
    input_path = args.input if args.input is not None else crw.INPUT_DATASET
    base_url = CRW_ENDPOINTS[args.endpoint]
    data = crw.load_base_rows(input_path, args.limit_cells)
    cells = crw.cell_metadata(data)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.log_output.parent.mkdir(parents=True, exist_ok=True)
    args.point_inventory_output.parent.mkdir(parents=True, exist_ok=True)

    selections = []
    for cell in cells.itertuples(index=False):
        cell_series = pd.Series(cell._asdict())
        selected = choose_valid_point(
            crw,
            cell_series,
            args.cache_dir,
            base_url,
            args.max_search_km,
            args.delay_seconds,
        )
        selected.update(
            {
                "cell_id": cell.cell_id,
                "center_lat": cell.center_lat,
                "center_lon": cell.center_lon,
                "snapped_lat": cell.nearest_crw_lat,
                "snapped_lon": cell.nearest_crw_lon,
                "snapped_distance_km": cell.distance_to_crw_grid_km,
                "endpoint": args.endpoint,
            }
        )
        selections.append(selected)
        print(
            f"{cell.cell_id}: selected {selected['selected_lat']}, {selected['selected_lon']} "
            f"({selected['probe_status']})",
            flush=True,
        )

    inventory = pd.DataFrame(selections)
    valid_inventory = inventory.dropna(subset=["selected_lat", "selected_lon"]).copy()
    point_inventory = valid_inventory[
        ["selected_lat", "selected_lon", "distance_to_cell_km"]
    ].drop_duplicates(["selected_lat", "selected_lon"])
    point_inventory.to_csv(args.point_inventory_output, index=False)

    log_rows = []
    for point in point_inventory.itertuples(index=False):
        path = cache_path(args.cache_dir, float(point.selected_lat), float(point.selected_lon))
        result = download_full_point(
            float(point.selected_lat),
            float(point.selected_lon),
            path,
            base_url,
            args.endpoint,
            args.force,
            args.delay_seconds,
        )
        log_rows.append(
            {
                "selected_lat": point.selected_lat,
                "selected_lon": point.selected_lon,
                "cache_file": str(path),
                **result,
            }
        )
        print(
            f"{path.name}: {result['status']} rows={result['rows']} valid_sst={result['valid_sst_rows']}",
            flush=True,
        )

    log = pd.DataFrame(log_rows)
    merged_log = inventory.merge(log, on=["selected_lat", "selected_lon"], how="left")
    merged_log.to_csv(args.log_output, index=False)

    downloaded_or_reused = log["status"].isin(["downloaded", "reused_cache"]).sum() if not log.empty else 0
    total_rows = int(log["rows"].sum()) if not log.empty else 0
    print(f"Cells processed: {cells['cell_id'].nunique()}")
    print(f"Unique CRW points selected: {len(point_inventory)}")
    print(f"Downloaded/reused point caches: {downloaded_or_reused}")
    print(f"Daily records cached: {total_rows}")
    print(f"Wrote log: {args.log_output}")


if __name__ == "__main__":
    main()
