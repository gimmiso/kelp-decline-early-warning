"""Download Kelpwatch cell-level CSV exports through the public web API.

The script uses the same request pattern observed in the Kelpwatch web app:

1. POST a single-feature GeoJSON to the shp2json upload service.
2. Use the returned upload ID to request aggregate Kelpwatch CSV data.
3. Save raw CSV files locally under data/raw/kelpwatch_aoi/.

By default this script runs only the first three test cells. Use --all only
after the test workflow has been confirmed.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests


UPLOAD_URL = "https://shp2json.codefornature.org/api/upload?multi=true"
UPLOAD_STATUS_URL = "https://shp2json.codefornature.org/api/upload/{upload_id}"
AGGREGATE_URL = "https://kelp-production-agg.kelpwatch.org/aggregate/id"
DEFAULT_CELLS = ("001", "002", "003")
REQUIRED_COLUMNS = (
    "year",
    "quarter",
    "kelp_area_m2",
    "count_cells_kelp",
    "count_cells_no_clouds",
    "count_cells_historic_footprint",
)


@dataclass
class DownloadResult:
    """Result row for the download log."""

    cell_id: str
    geojson_file: str
    csv_file: str
    status: str
    http_status: int | None
    download_time: str
    error_message: str = ""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download Kelpwatch cell-level CSV exports.")
    parser.add_argument(
        "--geojson-dir",
        type=Path,
        default=Path("geometries/regular_10km_fishnet/single_cell_geojsons"),
        help="Directory containing cell_XXX.geojson files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/kelpwatch_aoi"),
        help="Local raw CSV output directory.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("outputs/metadata/kelpwatch_download_log.csv"),
        help="Download log CSV path.",
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=list(DEFAULT_CELLS),
        help="Cell numbers or IDs to download, for example 001 002 003 or cell_001.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all cell_*.geojson files. Do not use until the 3-cell test is confirmed.",
    )
    parser.add_argument("--start-year", type=int, default=1984)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--source", default="landsat")
    parser.add_argument("--delay-seconds", type=float, default=2.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--upload-poll-seconds", type=float, default=1.0)
    parser.add_argument("--upload-poll-attempts", type=int, default=30)
    return parser.parse_args()


def normalize_cell_id(cell: str) -> str:
    """Return a canonical cell_id like cell_001."""
    value = cell.strip()
    if value.startswith("cell_"):
        return value
    return f"cell_{int(value):03d}"


def iter_geojson_files(geojson_dir: Path, cells: Iterable[str], include_all: bool) -> list[Path]:
    """Return GeoJSON paths for requested cells."""
    if include_all:
        return sorted(geojson_dir.glob("cell_*.geojson"))
    cell_ids = [normalize_cell_id(cell) for cell in cells]
    return [geojson_dir / f"{cell_id}.geojson" for cell_id in cell_ids]


def upload_geojson(session: requests.Session, geojson_path: Path, timeout: float) -> str:
    """Upload a GeoJSON file and return the Kelpwatch upload ID."""
    with geojson_path.open("rb") as file:
        response = session.post(
            UPLOAD_URL,
            files={"geoupload": (geojson_path.name, file, "application/geo+json")},
            timeout=timeout,
        )
    response.raise_for_status()
    payload = response.json()
    upload_id = payload.get("id")
    if not upload_id:
        raise ValueError(f"Upload response did not include an id: {payload}")
    return upload_id


def wait_for_upload_success(
    session: requests.Session,
    upload_id: str,
    timeout: float,
    poll_seconds: float,
    poll_attempts: int,
) -> None:
    """Poll the upload status endpoint until the geometry is ready."""
    url = UPLOAD_STATUS_URL.format(upload_id=upload_id)
    for _ in range(poll_attempts):
        response = session.get(url, timeout=timeout)
        if response.status_code in {500, 502, 503, 504}:
            time.sleep(poll_seconds)
            continue
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        if status == "Success":
            return
        if status == "Error":
            raise ValueError(f"Upload processing failed: {payload}")
        time.sleep(poll_seconds)
    raise TimeoutError(f"Upload {upload_id} did not reach Success after {poll_attempts} polls.")


def request_csv(
    session: requests.Session,
    upload_id: str,
    start_year: int,
    end_year: int,
    source: str,
    timeout: float,
) -> requests.Response:
    """Request CSV aggregate data for an uploaded geometry."""
    response = session.get(
        AGGREGATE_URL,
        params={
            "id": upload_id,
            "start": start_year,
            "end": end_year,
            "source": source,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response


def csv_has_required_shape(text: str) -> bool:
    """Check that a downloaded CSV has the expected columns and max rows."""
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        return False
    if not set(REQUIRED_COLUMNS).issubset(rows[0].keys()):
        return False
    return any(row.get("quarter") == "max" for row in rows)


def download_one_cell(
    session: requests.Session,
    geojson_path: Path,
    output_dir: Path,
    start_year: int,
    end_year: int,
    source: str,
    retries: int,
    timeout: float,
    poll_seconds: float,
    poll_attempts: int,
) -> DownloadResult:
    """Upload one cell and download its Kelpwatch CSV."""
    cell_id = geojson_path.stem
    csv_path = output_dir / f"kelpwatch_{cell_id}.csv"
    now = datetime.now(timezone.utc).isoformat()
    http_status = None

    if not geojson_path.exists():
        return DownloadResult(cell_id, str(geojson_path), str(csv_path), "error", None, now, "GeoJSON not found")

    for attempt in range(1, retries + 1):
        try:
            upload_id = upload_geojson(session, geojson_path, timeout)
            wait_for_upload_success(session, upload_id, timeout, poll_seconds, poll_attempts)
            response = request_csv(session, upload_id, start_year, end_year, source, timeout)
            http_status = response.status_code
            content_type = response.headers.get("content-type", "")
            text = response.text
            if "text/csv" not in content_type and not text.startswith("year,"):
                raise ValueError(f"Unexpected response content type: {content_type}")
            if not csv_has_required_shape(text):
                raise ValueError("CSV missing required columns or quarter=max rows.")
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path.write_text(text)
            return DownloadResult(cell_id, str(geojson_path), str(csv_path), "success", http_status, now)
        except Exception as exc:
            if attempt == retries:
                return DownloadResult(cell_id, str(geojson_path), str(csv_path), "error", http_status, now, str(exc))
            time.sleep(min(2**attempt, 30))

    return DownloadResult(cell_id, str(geojson_path), str(csv_path), "error", http_status, now, "Unknown error")


def write_log(results: list[DownloadResult], log_path: Path) -> None:
    """Write a CSV download log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "cell_id",
                "geojson_file",
                "csv_file",
                "status",
                "http_status",
                "download_time",
                "error_message",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def main() -> None:
    """Run downloads for requested cells."""
    args = parse_args()
    geojson_paths = iter_geojson_files(args.geojson_dir, args.cells, args.all)
    if not args.all:
        print("Running limited test download only:", ", ".join(path.stem for path in geojson_paths))
    else:
        print(f"Running full download for {len(geojson_paths)} cells.")

    session = requests.Session()
    results = []
    for index, geojson_path in enumerate(geojson_paths):
        result = download_one_cell(
            session=session,
            geojson_path=geojson_path,
            output_dir=args.output_dir,
            start_year=args.start_year,
            end_year=args.end_year,
            source=args.source,
            retries=args.retries,
            timeout=args.timeout,
            poll_seconds=args.upload_poll_seconds,
            poll_attempts=args.upload_poll_attempts,
        )
        results.append(result)
        print(f"{result.cell_id}: {result.status} ({result.http_status}) {result.error_message}")
        if index < len(geojson_paths) - 1:
            time.sleep(args.delay_seconds)

    write_log(results, args.log_path)
    print(f"Wrote log: {args.log_path}")


if __name__ == "__main__":
    main()
