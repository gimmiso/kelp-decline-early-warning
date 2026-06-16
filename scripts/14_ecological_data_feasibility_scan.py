"""Scan feasibility for a V3 ecological transition case study.

This planning layer does not modify the V1/V2 modeling workflow and does not
download ecological datasets. It documents candidate sea urchin and kelp forest
monitoring data sources that could be joined to Kelpwatch canopy trajectories in
a future biologically informed Stage-2 case study.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path


REPORT_OUTPUT = Path("outputs/diagnostics/ecological_data_feasibility_report.md")
LOCAL_DATA_DIR = Path("data/external/ecological")


@dataclass(frozen=True)
class CandidateDataset:
    """Metadata for one candidate ecological monitoring dataset."""

    name: str
    spatial_coverage: str
    temporal_coverage: str
    likely_variables: str
    access_method: str
    coordinates_available: str
    kelpwatch_join: str
    transition_support: str
    limitations: str
    source_url: str
    expected_local_file: str


CANDIDATE_DATASETS = [
    CandidateDataset(
        name="BCO-DMO purple sea urchin density, California coast",
        spatial_coverage="37.8-39.3 N; Andrew Molera State Park to Manchester State Park",
        temporal_coverage="2005-2014",
        likely_variables="purple urchin density or counts, site, year, survey metadata",
        access_method="BCO-DMO dataset page; CSV or tabular download if needed",
        coordinates_available="Likely site coordinates or site locations; verify fields after download",
        kelpwatch_join="Join survey sites to nearest/intersecting 10 km Kelpwatch cells",
        transition_support="Useful pre-collapse and early collapse context; limited post-collapse coverage",
        limitations="Ends in 2014, so it does not fully cover post-collapse recovery or barren persistence",
        source_url="https://www.bco-dmo.org/dataset/541003",
        expected_local_file="data/external/ecological/bco_dmo_purple_urchin_density_2005_2014.csv",
    ),
    CandidateDataset(
        name="BCO-DMO / CDFW kelp forest monitoring, Sonoma-Mendocino Coast",
        spatial_coverage="Sonoma and Mendocino counties, northern California",
        temporal_coverage="1999-2023 in BCO-DMO subset; broader ongoing program is available via OPC/DataONE",
        likely_variables="organism counts, algal habitat cover, substrate cover, lengths, survey site, depth",
        access_method="BCO-DMO dataset family and OPC/DataONE repository",
        coordinates_available="Yes for monitoring locations or occurrence records; verify join-ready precision",
        kelpwatch_join="Strong candidate for spatial join to Kelpwatch cells or site-level canopy buffers",
        transition_support="Best candidate for pre-collapse, collapse, and post-collapse analysis",
        limitations="Survey cadence and site availability vary by year; requires harmonizing multiple tables",
        source_url="https://www.bco-dmo.org/dataset/927682",
        expected_local_file="data/external/ecological/bco_dmo_cdfw_sonoma_mendocino_kelp_monitoring_1999_2023.csv",
    ),
    CandidateDataset(
        name="California open data kelp forest transect surveys, Sonoma and Mendocino County",
        spatial_coverage="Sonoma and Mendocino counties; rocky reef habitats from 0-60 ft depth",
        temporal_coverage="Long-term northern California monitoring archive; verify year coverage in package",
        likely_variables="30 x 2 m transect observations, depth, site, species or habitat measurements",
        access_method="California open data / CNRA data package",
        coordinates_available="Expected for survey sites; verify after download",
        kelpwatch_join="Strong candidate; likely same regional monitoring lineage as CDFW/OPC products",
        transition_support="Likely supports local case-study analysis if years span 2014-2016 collapse window",
        limitations="Package structure may include multiple CSV/PDF/RTF files requiring schema harmonization",
        source_url=(
            "https://data.ca.gov/dataset/"
            "kelp-forest-transect-surveys-sonoma-and-mendocino-county-northern-california-coast"
        ),
        expected_local_file="data/external/ecological/ca_open_data_kelp_forest_transect_surveys.zip",
    ),
    CandidateDataset(
        name="Reef Check California Kelp Forest Monitoring Program",
        spatial_coverage="California coast, with expansion to Oregon and Washington in recent program summaries",
        temporal_coverage="Program began in 2006; 2024 data noted as available by request",
        likely_variables="fish, invertebrate, kelp, substrate, and site survey indicators",
        access_method="Reef Check data request form or program data access workflow",
        coordinates_available="Likely site coordinates; access terms and exact fields require data request",
        kelpwatch_join="Potentially strong for California-wide extension if site coordinates are provided",
        transition_support="Could support broader validation, but data access and harmonization are blockers",
        limitations="May require request/approval; volunteer protocol differs from CDFW/PISCO details",
        source_url="https://www.reefcheck.org/kelp-forest-program/kelp-forest-monitoring-and-mpas/",
        expected_local_file="data/external/ecological/reef_check_california_kelp_forest_monitoring.csv",
    ),
    CandidateDataset(
        name="PISCO kelp forest monitoring",
        spatial_coverage="California and Oregon nearshore rocky reef sites; central and southern California coverage",
        temporal_coverage="Continuous monitoring since 1999 at shallow rocky-bottom kelp forest sites",
        likely_variables="macroalgae, invertebrate, fish density/biomass, site, depth, survey protocol fields",
        access_method="PISCO data access and DataONE/OBIS-related products where available",
        coordinates_available="Likely site coordinates; confirm access and spatial precision",
        kelpwatch_join="Potentially strong, especially for central/southern California or MPA comparisons",
        transition_support="Good for broader ecological covariate design, less directly targeted to NorCal bull kelp collapse",
        limitations="Access and schema may vary by institution/product; may require more permissions or requests",
        source_url="https://piscoweb.org/kelp-forest-study",
        expected_local_file="data/external/ecological/pisco_kelp_forest_monitoring.csv",
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create a V3 ecological data feasibility report.")
    parser.add_argument("--output", type=Path, default=REPORT_OUTPUT)
    parser.add_argument("--local-data-dir", type=Path, default=LOCAL_DATA_DIR)
    return parser.parse_args()


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    """Render a simple Markdown table."""
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row.get(column, "") for column in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def local_availability_rows(local_data_dir: Path) -> list[dict[str, str]]:
    """Summarize whether expected local ecological input files are present."""
    rows = []
    for dataset in CANDIDATE_DATASETS:
        path = Path(dataset.expected_local_file)
        rows.append(
            {
                "Dataset": dataset.name,
                "Expected local path": str(path),
                "Present locally": "yes" if path.exists() else "no",
            }
        )
    if not local_data_dir.exists():
        rows.append(
            {
                "Dataset": "Local ecological data directory",
                "Expected local path": str(local_data_dir),
                "Present locally": "no",
            }
        )
    return rows


def source_inventory_rows() -> list[dict[str, str]]:
    """Return rows for the candidate ecological data inventory."""
    return [
        {
            "Dataset": dataset.name,
            "Spatial coverage": dataset.spatial_coverage,
            "Temporal coverage": dataset.temporal_coverage,
            "Likely variables": dataset.likely_variables,
            "Access": dataset.access_method,
            "Coordinates": dataset.coordinates_available,
            "Join to Kelpwatch 10 km cells": dataset.kelpwatch_join,
            "Transition analysis support": dataset.transition_support,
            "Limitations": dataset.limitations,
        }
        for dataset in CANDIDATE_DATASETS
    ]


def build_report(local_data_dir: Path) -> str:
    """Build the feasibility report body."""
    inventory_columns = [
        "Dataset",
        "Spatial coverage",
        "Temporal coverage",
        "Likely variables",
        "Access",
        "Coordinates",
        "Join to Kelpwatch 10 km cells",
        "Transition analysis support",
        "Limitations",
    ]
    local_columns = ["Dataset", "Expected local path", "Present locally"]
    source_rows = [
        {"Dataset": dataset.name, "Source URL": dataset.source_url}
        for dataset in CANDIDATE_DATASETS
    ]

    return f"""# Ecological Data Feasibility Report

Generated: {date.today().isoformat()}

## Purpose

This report assesses whether the repository can be extended from a climate-only
regional screening workflow into a Stage-2 ecological transition case study.
It does not download data, alter V1/V2 scripts, or claim improved model
performance.

## Short Answer

An urchin-integrated V3 analysis appears feasible as a focused northern
California case study, especially for the Sonoma-Mendocino Coast. It should not
start as a full California model. The strongest first candidate is the
BCO-DMO/CDFW Sonoma-Mendocino kelp forest monitoring dataset family because it
contains long-term local ecological monitoring across the known bull kelp
collapse period and can plausibly be joined to Kelpwatch canopy trajectories.

## Candidate Ecological Data Inventory

{markdown_table(source_inventory_rows(), inventory_columns)}

## Local Data Availability Check

{markdown_table(local_availability_rows(local_data_dir), local_columns)}

## Proposed V3 Modeling Design

Candidate targets:

- `abrupt_canopy_drop_next`: currently observable canopy followed by a sharp next-year relative canopy drop.
- `healthy_to_low_transition`: current canopy above a cell/site historical healthy threshold followed by low canopy.
- `post_heatwave_collapse_indicator`: transition into low canopy during or after a marine heatwave/event period.

Candidate features:

- OISST marine heatwave intensity or hot-day exposure.
- IDW-interpolated or buffer OISST heat stress exposure.
- Purple sea urchin density, count, or survey-derived grazing-pressure proxy.
- Sea star, predator, or community-structure proxy if available.
- Interaction term: heatwave intensity x urchin density.
- Year fixed effects, event-period indicators, or pre/post heatwave period flags.

Recommended unit of analysis:

- Start with monitored ecological sites or site-year observations.
- Join sites to Kelpwatch 10 km cells or local canopy buffers.
- Preserve both site-level ecological measurements and Kelpwatch-derived canopy trajectories.

## Feasibility Answers

**Is an urchin-integrated V3 analysis feasible with currently accessible data?**

Yes, as a planning direction and likely as an implementable case study after
downloading and harmonizing the monitoring tables. The evidence is strongest for
Sonoma-Mendocino, where long-term kelp forest monitoring and urchin/community
survey data overlap with the well-known northern California bull kelp collapse
period.

**Which dataset is the best first candidate?**

The best first candidate is the BCO-DMO/CDFW Sonoma-Mendocino kelp forest
monitoring dataset family, supplemented by the California open data transect
survey package if it provides easier table access or updated files. The older
BCO-DMO purple urchin density dataset is useful but shorter because it ends in
2014.

**What geographic scope should be used?**

Use a Northern California / Sonoma-Mendocino ecological transition case study.
This scope aligns with available ecological monitoring, documented bull kelp
loss, and the current Kelpwatch V1 spatial design.

**Should this be a full California model or a Northern California case study?**

Start with a Northern California / Sonoma-Mendocino case study. A full
California ecological model should wait until Reef Check, PISCO, and regional
survey schemas are harmonized and comparable across regions.

**What are the main blockers?**

- Local ecological datasets are not yet downloaded into this repository.
- Site coordinates, survey effort, and taxonomic fields must be harmonized.
- Annual aggregation rules must be defined before joining to Kelpwatch.
- Survey cadence may be uneven across years and sites.
- Predator or sea star wasting disease proxies may require additional data.
- Kelpwatch 10 km cells may be too coarse for site-scale ecological mechanisms;
  site buffers or nearest-cell joins should be compared.

## Recommended Conclusion

A climate-only model is better interpreted as a regional screening layer. A
biologically meaningful early-warning study should focus on abrupt transitions
in monitored kelp forest sites where urchin density and predator/community data
can be joined to Kelpwatch canopy trajectories.

## Source Pages Reviewed

{markdown_table(source_rows, ["Dataset", "Source URL"])}
"""


def main() -> None:
    """Write the V3 feasibility report."""
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(args.local_data_dir), encoding="utf-8")
    print(f"Wrote ecological feasibility report: {args.output}")


if __name__ == "__main__":
    main()
