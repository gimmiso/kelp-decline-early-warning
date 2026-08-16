"""Fetch and verify the official Giraldo-Ospina et al. (2025) code release."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import requests


RECORD_API = "https://zenodo.org/api/records/14590514"
EXPECTED_KEY = "anitas-giraldo/Drivers_of_kelp_dynamics_California-v1.zip"
EXPECTED_SIZE = 1_214_954
EXPECTED_MD5 = "d595752ff2ff59aac60c86d60a941bf8"
REQUIRED_MEMBERS = {
    "No_survey_years_per_site.csv",
    "bull_kelp_site_region_metadata.csv",
    "giant_kelp_site_region_metadata.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "Drivers_of_kelp_dynamics_California-v1.zip"

    metadata_response = requests.get(RECORD_API, timeout=60)
    metadata_response.raise_for_status()
    metadata = metadata_response.json()
    matches = [item for item in metadata["files"] if item["key"] == EXPECTED_KEY]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one official archive, found {len(matches)}")
    source = matches[0]
    if source["size"] != EXPECTED_SIZE or source["checksum"] != f"md5:{EXPECTED_MD5}":
        raise RuntimeError("Zenodo file metadata differs from the locked release")

    if not archive.exists() or archive.stat().st_size != EXPECTED_SIZE:
        with requests.get(source["links"]["self"], stream=True, timeout=120) as response:
            response.raise_for_status()
            with archive.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    if archive.stat().st_size != EXPECTED_SIZE or digest(archive, "md5") != EXPECTED_MD5:
        raise RuntimeError("Downloaded Zenodo archive failed size or MD5 verification")

    extracted: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(archive) as bundle:
        members = {
            Path(name).name: name
            for name in bundle.namelist()
            if Path(name).name in REQUIRED_MEMBERS
        }
        if set(members) != REQUIRED_MEMBERS:
            raise RuntimeError(f"Missing required cohort files: {sorted(REQUIRED_MEMBERS - set(members))}")
        for short_name, member in members.items():
            target = args.output_dir / short_name
            target.write_bytes(bundle.read(member))
            extracted[short_name] = {
                "archive_member": member,
                "size": target.stat().st_size,
                "sha256": digest(target, "sha256"),
            }

    manifest = {
        "status": "complete",
        "record_api": RECORD_API,
        "concept_doi": "10.5281/zenodo.14590514",
        "version_doi": "10.5281/zenodo.14590515",
        "archive": {
            "path": str(archive),
            "key": EXPECTED_KEY,
            "size": archive.stat().st_size,
            "md5": digest(archive, "md5"),
            "sha256": digest(archive, "sha256"),
        },
        "extracted": extracted,
    }
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
