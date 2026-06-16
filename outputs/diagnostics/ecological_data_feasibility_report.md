# Ecological Data Feasibility Report

Generated: 2026-06-16

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

| Dataset | Spatial coverage | Temporal coverage | Likely variables | Access | Coordinates | Join to Kelpwatch 10 km cells | Transition analysis support | Limitations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BCO-DMO purple sea urchin density, California coast | 37.8-39.3 N; Andrew Molera State Park to Manchester State Park | 2005-2014 | purple urchin density or counts, site, year, survey metadata | BCO-DMO dataset page; CSV or tabular download if needed | Likely site coordinates or site locations; verify fields after download | Join survey sites to nearest/intersecting 10 km Kelpwatch cells | Useful pre-collapse and early collapse context; limited post-collapse coverage | Ends in 2014, so it does not fully cover post-collapse recovery or barren persistence |
| BCO-DMO / CDFW kelp forest monitoring, Sonoma-Mendocino Coast | Sonoma and Mendocino counties, northern California | 1999-2023 in BCO-DMO subset; broader ongoing program is available via OPC/DataONE | organism counts, algal habitat cover, substrate cover, lengths, survey site, depth | BCO-DMO dataset family and OPC/DataONE repository | Yes for monitoring locations or occurrence records; verify join-ready precision | Strong candidate for spatial join to Kelpwatch cells or site-level canopy buffers | Best candidate for pre-collapse, collapse, and post-collapse analysis | Survey cadence and site availability vary by year; requires harmonizing multiple tables |
| California open data kelp forest transect surveys, Sonoma and Mendocino County | Sonoma and Mendocino counties; rocky reef habitats from 0-60 ft depth | Long-term northern California monitoring archive; verify year coverage in package | 30 x 2 m transect observations, depth, site, species or habitat measurements | California open data / CNRA data package | Expected for survey sites; verify after download | Strong candidate; likely same regional monitoring lineage as CDFW/OPC products | Likely supports local case-study analysis if years span 2014-2016 collapse window | Package structure may include multiple CSV/PDF/RTF files requiring schema harmonization |
| Reef Check California Kelp Forest Monitoring Program | California coast, with expansion to Oregon and Washington in recent program summaries | Program began in 2006; 2024 data noted as available by request | fish, invertebrate, kelp, substrate, and site survey indicators | Reef Check data request form or program data access workflow | Likely site coordinates; access terms and exact fields require data request | Potentially strong for California-wide extension if site coordinates are provided | Could support broader validation, but data access and harmonization are blockers | May require request/approval; volunteer protocol differs from CDFW/PISCO details |
| PISCO kelp forest monitoring | California and Oregon nearshore rocky reef sites; central and southern California coverage | Continuous monitoring since 1999 at shallow rocky-bottom kelp forest sites | macroalgae, invertebrate, fish density/biomass, site, depth, survey protocol fields | PISCO data access and DataONE/OBIS-related products where available | Likely site coordinates; confirm access and spatial precision | Potentially strong, especially for central/southern California or MPA comparisons | Good for broader ecological covariate design, less directly targeted to NorCal bull kelp collapse | Access and schema may vary by institution/product; may require more permissions or requests |

## Local Data Availability Check

| Dataset | Expected local path | Present locally |
| --- | --- | --- |
| BCO-DMO purple sea urchin density, California coast | data/external/ecological/bco_dmo_purple_urchin_density_2005_2014.csv | no |
| BCO-DMO / CDFW kelp forest monitoring, Sonoma-Mendocino Coast | data/external/ecological/bco_dmo_cdfw_sonoma_mendocino_kelp_monitoring_1999_2023.csv | no |
| California open data kelp forest transect surveys, Sonoma and Mendocino County | data/external/ecological/ca_open_data_kelp_forest_transect_surveys.zip | no |
| Reef Check California Kelp Forest Monitoring Program | data/external/ecological/reef_check_california_kelp_forest_monitoring.csv | no |
| PISCO kelp forest monitoring | data/external/ecological/pisco_kelp_forest_monitoring.csv | no |
| Local ecological data directory | data/external/ecological | no |

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

| Dataset | Source URL |
| --- | --- |
| BCO-DMO purple sea urchin density, California coast | https://www.bco-dmo.org/dataset/541003 |
| BCO-DMO / CDFW kelp forest monitoring, Sonoma-Mendocino Coast | https://www.bco-dmo.org/dataset/927682 |
| California open data kelp forest transect surveys, Sonoma and Mendocino County | https://data.ca.gov/dataset/kelp-forest-transect-surveys-sonoma-and-mendocino-county-northern-california-coast |
| Reef Check California Kelp Forest Monitoring Program | https://www.reefcheck.org/kelp-forest-program/kelp-forest-monitoring-and-mpas/ |
| PISCO kelp forest monitoring | https://piscoweb.org/kelp-forest-study |
