# Repository Cleanup Audit

Audit date: 2026-07-03

Repository: `gimmiso/kelp-decline-early-warning`

Latest commit before this audit: `4d791cb` (`Add reference-point state taxonomy`)

This audit is a recommendation document only. No files were deleted, moved, renamed, or rewritten during the audit. No heavy analyses, model training, or output regeneration were run.

## Executive Summary

The repository has matured from a broad class-project-style machine-learning comparison into a manuscript-ready decision-support study. The paper-facing story should now be narrowed around three claims:

1. Broad next-year low-canopy or decline-risk states can be screened with useful skill from satellite-derived canopy histories.
2. Much of the apparent early-warning skill is driven by canopy-state persistence, especially persistent low-canopy conditions.
3. Annual top-k alerts may help prioritise monitoring or intervention resources, but should not be interpreted as robust forecasts of new low-canopy transitions.

The current repository contains many useful analyses, but the README and tracked outputs expose too much of the exploratory work at the same level as the main paper claims. The next cleanup pass should not delete work immediately. Instead, it should separate the paper-facing pathway from supplementary diagnostics, future-work experiments, and local-only material.

Recommended direction:

- Keep the main paper narrative centered on the annual 10 km cell-year workflow, persistence diagnostics, reference-point transition taxonomy, and top-k prioritisation.
- Move CRW, bathymetry, trajectory, wave, rare-event, quarterly, multi-horizon, spatial-repair, and ecological V3 work to supplementary or future-work sections unless they directly support the three claims.
- Keep raw data, processed data, NOAA/CDIP caches, virtual environments, and local PDF references out of Git.
- Add a later paper-freeze manifest after the final figures/tables are selected.

## Current Git Status Summary

Safe inspection commands were run:

```bash
git status --short
git log --oneline -5
git ls-files
find . -maxdepth 3 -type f
find . -maxdepth 3 -type d
du -ah . | sort -h | tail -50
```

Status before creating audit documents:

```text
?? references/
```

Recent commits:

```text
4d791cb Add reference-point state taxonomy
eb467df Add alert-budget top-k evaluation
7dc4659 Document threshold trade-off analysis
d5aefd3 Add threshold trade-off curves
f1e1ed2 Polish study area grid map
```

Tracked files inspected: `560`

Untracked non-ignored files detected: `20`, all under `references/` and all PDF files.

Ignored local material detected includes:

- `.venv/`
- `.DS_Store`
- `data/raw/kelpwatch_aoi/`
- `data/processed/*.csv`
- `data/external/noaa/`
- `outputs/metadata/*` local rerun files not force-added
- `references.zip`
- `scripts/__pycache__/`

## Repository Size Notes

Approximate local repository size from `du`: `1.1G`.

Largest local storage drivers:

| Path | Approximate size | Status | Recommendation |
|---|---:|---|---|
| `.venv/` | 938 MB | ignored local | `ignore_or_do_not_commit` |
| `references/` | 40 MB | untracked non-ignored PDFs | `ignore_or_do_not_commit` |
| `references.zip` | 36 MB | ignored local zip | `ignore_or_do_not_commit` |
| `data/external/noaa/cache/` | 29 MB | ignored local cache | `ignore_or_do_not_commit` |
| `data/processed/` | local processed CSVs | ignored local | `ignore_or_do_not_commit` unless a paper-freeze release is intentionally created |

Largest tracked files are not extreme, but several are too detailed for the main paper:

| Path | Approximate size | Recommendation |
|---|---:|---|
| `results/tables/rare_event_topk_alert_evaluation.csv` | 6.01 MB | `supplementary_keep` or future archive |
| `results/tables/rare_event_threshold_tuning.csv` | 1.44 MB | `supplementary_keep` or future archive |
| `results/tables/integrated_model_comparison_master.csv` | 1.13 MB | `supplementary_keep` |
| `outputs/model_results/extended_threshold_tuning_results.csv` | 0.62 MB | `supplementary_keep` |
| `outputs/tables/reference_point_state_taxonomy.csv` | 0.55 MB | `main_paper_keep` |

## Major Folder Recommendations

| Path | Tracked? | Approx. role | Recommendation | Rationale |
|---|---|---|---|---|
| `README.md` | yes | project narrative | `main_paper_keep`, later slim | Contains the current full project story but overexposes exploratory analyses. Later rewrite should prioritize the three paper claims. |
| `.gitignore` | yes | repo hygiene | `main_paper_keep` | Already blocks raw/processed data, caches, virtual environments, and generated outputs. Add `references/` later after review. |
| `requirements.txt` | yes | reproducibility | `main_paper_keep` | Required for rerunning scripts. |
| `data/raw/README.md` | yes | raw data documentation | `main_paper_keep` | Explains why raw Kelpwatch CSVs are not committed and how they should be named. |
| `data/raw/`, `data/processed/`, `data/external/` | placeholders tracked; contents ignored | local data | `ignore_or_do_not_commit` | Correctly ignored except `.gitkeep` and raw README. |
| `geometries/regular_10km_fishnet/` | yes | AOI design | `main_paper_keep` | Core spatial design and historical-footprint filtering support the study area and sample definition. |
| `geometries/regular_10km_fishnet/single_cell_geojsons/` | yes | AOI reproducibility | `supplementary_keep` | Useful for reproducibility; not central to paper text. |
| `docs/kelpwatch_api_investigation.md` | yes | data-access notes | `supplementary_keep` | Useful audit trail, not main text. |
| `docs/maps/kelpwatch_regular_10km_fishnet_preview_map.html` | yes | preview map | `supplementary_keep` or archive later | Superseded by report-ready Figure 1 but still useful as an interactive preview. |
| `notebooks/` | yes | early exploratory notebooks | `archive_candidate` | Many notebooks are tiny placeholders or duplicate the script workflow. Keep until human review, then archive. |
| `src/` | yes | early utility skeletons | `needs_review` then likely `archive_candidate` | Not clearly imported by the current numbered scripts; may confuse readers about the canonical workflow. |
| `scripts/` | yes | reproducible analyses | mixed | Needs paper-facing subset plus supplementary/exploratory grouping. |
| `outputs/maps/` | yes | final maps | `main_paper_keep` | Figure 1 map is manuscript-relevant. |
| `outputs/figures/` | mostly force-tracked selected figures | visual outputs | mixed | Keep only a small main-text figure set in the README/paper; move detailed diagnostics to supplementary. |
| `outputs/tables/` | yes | decision-support tables | `main_paper_keep` / `supplementary_keep` | Threshold, top-k, and reference taxonomy tables are central or near-central. |
| `outputs/reports/` | yes | decision-support reports | `supplementary_keep` | Good reproducibility companions; not all need README exposure. |
| `outputs/metadata/` | yes for selected files, ignored by default | metadata/model outputs | mixed | Several files support core workflow; many diagnostics are supplementary. |
| `outputs/diagnostics/` | yes | robustness reports | mostly `supplementary_keep` | Important for transparency but too detailed for main text. |
| `outputs/model_results/` | yes for selected files, ignored by default | model result tables | mixed | Threshold and actionable tables are useful; extended results belong in supplement. |
| `results/tables/` | yes | later-stage experiment tables | mostly `supplementary_keep` or `exploratory_or_future_work` | Contains many non-main analyses that should not drive the paper claims. |
| `references/` | untracked | local PDFs | `ignore_or_do_not_commit` | Copyright/access-sensitive local library; should not be committed. |
| `references.zip` | ignored | local archive | `ignore_or_do_not_commit` | Large local archive; should remain ignored. |
| `.venv/` | ignored | local environment | `ignore_or_do_not_commit` | Correctly ignored. |

## File Classification Table

| Path or group | Status | Size/type | Recommendation | Required for paper claims? | Rationale |
|---|---|---|---|---|---|
| `README.md` | tracked | Markdown | `main_paper_keep` | yes | Current entry point, but should later be slimmed to the three-claim paper narrative. |
| `geometries/regular_10km_fishnet/*.csv`, `*.geojson`, validation files | tracked | AOI files | `main_paper_keep` / `supplementary_keep` | yes | Defines candidate, footprint-positive, and retained 10 km cells. |
| `outputs/maps/figure1_study_area_retained_10km_grid_cells.png/pdf` | tracked | figure | `main_paper_keep` | yes | Study area and retained grid cells, likely Figure 1. |
| `outputs/figures/model_performance_comparison.png` | tracked | figure | `main_paper_keep` | yes | Supports Claim 1 at a high level. |
| `outputs/figures/canopy_persistence_scatter.png` | tracked | figure | `main_paper_keep` | yes | Supports Claim 2. |
| `outputs/figures/canopy_quantile_decline_rate.png` | tracked | figure | `main_paper_keep` or supplement | yes | Supports Claim 2; may be combined with persistence scatter. |
| `outputs/figures/reference_point_transition_matrix_probabilities.png` | tracked | figure | `main_paper_keep` | yes | Directly supports Claim 2. |
| `outputs/figures/topk_transition_composition.png` | tracked | figure | `main_paper_keep` | yes | Directly supports Claim 3 by showing persistent-low concentration in top-k alerts. |
| `outputs/figures/topk_recall_vs_budget.png`, `topk_precision_vs_budget.png`, `topk_lift_vs_budget.png`, `topk_cumulative_gain.png`, `topk_hits_false_alerts_by_budget.png` | tracked | figures | `supplementary_keep` | partly | Useful details for Claim 3, but too many for main text. |
| `outputs/tables/reference_point_state_taxonomy.csv` | tracked | table | `main_paper_keep` | yes | Canonical state taxonomy table. |
| `outputs/tables/reference_point_transition_summary.csv` | tracked | table | `main_paper_keep` | yes | Summarizes persistent low, recovery, new transition, and stable non-low shares. |
| `outputs/tables/topk_transition_composition.csv` | tracked | table | `main_paper_keep` | yes | Best table for Claim 3. |
| `outputs/tables/alert_budget_topk_annual_summary.csv` | tracked | table | `main_paper_keep` or supplement | yes | Quantifies top-k precision/recall trade-offs. |
| `outputs/metadata/model_comparison_test_metrics.csv` | tracked | table | `main_paper_keep` | yes | Concise model comparison for Claim 1. |
| `outputs/metadata/model_comparison_test_predictions.csv` | tracked | table | `supplementary_keep` | yes for reproducibility | Required input for threshold/top-k/taxonomy risk linkage; not a main paper table. |
| `outputs/diagnostics/zero_persistence_transition_rates.csv` | tracked | table | `main_paper_keep` or supplement | yes | Directly supports Claim 2. |
| `outputs/diagnostics/naive_persistence_baseline_report.md` | tracked | report | `main_paper_keep` or supplement | yes | Important claim gate against overclaiming early-warning skill. |
| `outputs/reports/alert_budget_topk_summary.md` | tracked | report | `supplementary_keep` | yes | Reproducibility companion for Claim 3. |
| `outputs/reports/reference_point_state_taxonomy_summary.md` | tracked | report | `supplementary_keep` | yes | Reproducibility companion for Claims 2 and 3. |
| `outputs/reports/threshold_tradeoff_summary.md` | tracked | report | `supplementary_keep` | partly | Useful for decision support; not one of the three main claims unless threshold framing is retained. |
| `results/tables/integrated_model_comparison_master.csv` | tracked | table | `supplementary_keep` | no | Useful global index, but too broad for main paper. |
| `results/tables/claim_gate_summary.csv` | tracked | table | `supplementary_keep` | partly | Useful to justify cautious language. |
| `results/tables/rare_event_*.csv` | tracked | tables | `exploratory_or_future_work` | no | Detailed rare-event learning does not change main paper claims and is large. |
| `results/tables/quarterly_*`, `outputs/diagnostics/quarterly_*` | tracked | tables/report | `exploratory_or_future_work` | no | Current quarterly labels are seasonality-sensitive; keep out of main claims. |
| `results/tables/multihorizon_*`, `outputs/diagnostics/multihorizon_*` | tracked | tables/report | `exploratory_or_future_work` | no | Useful horizon sensitivity, but not the main annual one-year study. |
| `results/tables/crw5km_*`, `outputs/diagnostics/crw5km_*` | tracked | tables/report | `exploratory_or_future_work` | no | Environmental exposure sensitivity, but current claims are canopy/persistence/top-k focused. |
| `results/tables/bathymetry_*`, `outputs/diagnostics/bathymetry_*` | tracked | tables/report | `exploratory_or_future_work` | no | Habitat covariates are useful future extension, not core claims. |
| `results/tables/wave_*`, `outputs/diagnostics/wave_*` | tracked | tables/report | `exploratory_or_future_work` | no | CDIP wave layer is future ecological context, not core paper. |
| `results/tables/spatial_*`, `outputs/diagnostics/spatial_*` | tracked | tables/report | `supplementary_keep` | partly | Important robustness limitation; likely supplement, not main. |
| `outputs/figures/shap_*`, `outputs/metadata/shap_*` | tracked | figures/tables | `supplementary_keep` | no | Interpretability details, but not central to three claims. |
| `outputs/figures/multicollinearity_*`, `outputs/diagnostics/multicollinearity_*` | tracked | diagnostics | `supplementary_keep` | no | Important methodological caution, not main narrative. |
| `notebooks/*` | tracked | notebooks/placeholders | `archive_candidate` | no | Superseded by scripts; some duplicates. |
| `src/*` | tracked | utility skeletons | `needs_review` | unclear | Not clearly used by canonical numbered scripts. |
| `references/*` | untracked | PDFs | `ignore_or_do_not_commit` | no | Local personal PDF storage; keep untracked. |
| `data/processed/*`, `data/raw/kelpwatch_aoi/*`, `data/external/noaa/*` | ignored local | data/cache | `ignore_or_do_not_commit` | indirectly | Needed locally to rerun, but should not be committed unless a release artifact is explicitly chosen. |

## Main Paper Keep List

The main paper-facing set should remain small and reproducible:

1. `README.md` after a future slim paper-facing rewrite.
2. `.gitignore`, `requirements.txt`, and data-source documentation.
3. `geometries/regular_10km_fishnet/` retained-cell inventory and validation files.
4. `scripts/build_kelpwatch_panel.py`.
5. `scripts/construct_decline_labels.py`.
6. `scripts/train_model_comparison.py`.
7. `scripts/diagnose_zero_persistence.py`.
8. `scripts/11_naive_persistence_baseline_benchmark.py`.
9. `scripts/31_alert_budget_topk_evaluation.py`.
10. `scripts/32_reference_point_state_taxonomy.py`.
11. `scripts/plot_study_area_retained_grid_map.py`.
12. Selected main output figures and tables listed in `docs/main_paper_scope.md`.

## Supplementary Keep List

Useful but likely too detailed for the main text:

1. `scripts/tune_decision_thresholds.py`.
2. `scripts/30_threshold_tradeoff_curves.py`.
3. `scripts/diagnose_environmental_covariates.py`.
4. `scripts/diagnose_multicollinearity.py`.
5. `scripts/analyze_canopy_environment_context.py`.
6. `scripts/diagnose_model_results.py`.
7. `scripts/interpret_models_shap.py`.
8. `scripts/21_integrate_model_results.py`.
9. `scripts/22_apply_claim_gates.py`.
10. `scripts/26_spatial_validation_diagnostics.py`.
11. `outputs/reports/*.md` decision-support reports.
12. Full threshold, top-k, SHAP, spatial validation, and multicollinearity tables.

## Exploratory or Future Work

These analyses are valuable but should not support the current three main claims:

1. CRW 5 km SST daily/monthly-composite exposure workflows.
2. GEBCO bathymetry/habitat feature workflow.
3. Canopy trajectory feature extension beyond the main persistence diagnostics.
4. CDIP wave exposure feature workflow.
5. Rare-event alert learning.
6. Multi-horizon actionable warning experiment.
7. Quarterly actionable warning feasibility.
8. Spatial failure repair feature engineering.
9. V3 ecological data feasibility scan.
10. Multi-scale OISST exposure selection.

## Archive Candidates

Do not delete these yet. Review and archive only after the manuscript scope is frozen.

1. `notebooks/01_Kelpwatch_Panel_Construction` (extensionless placeholder-like file).
2. `notebooks/02_decline_label_construction.ipynb` and `notebooks/03_decline_label_construction.ipynb` duplicate/superseded pair.
3. `notebooks/04_modeling_xgboost_shap.ipynb`, superseded by scripted model comparison and SHAP scripts.
4. `src/data_loader.py`, `src/feature_engineering.py`, `src/labeling.py`, `src/modeling.py`, `src/visualization.py` if confirmed unused.
5. `docs/maps/kelpwatch_regular_10km_fishnet_preview_map.html` if the static Figure 1 map fully replaces it.
6. `outputs/metadata/final_repository_review.md`, superseded by this cleanup audit.
7. `scripts/16_build_crw_5km_sst_features.py` and `scripts/16a_download_crw5km_point_cache.py` if daily CRW remains a long-run path outside the paper.
8. `results/tables/crw5km_model_comparison.csv` and `outputs/diagnostics/crw5km_sst_feature_report.md` if they reflect dry-run feasibility rather than final results.
9. Large rare-event detail tables if a compact summary is enough: `results/tables/rare_event_topk_alert_evaluation.csv`, `results/tables/rare_event_threshold_tuning.csv`.
10. Old model-diagnostics figures if replaced by main paper figures: `outputs/figures/model_diagnostics_*`.

## Ignore or Do-Not-Commit Candidates

These should remain untracked or ignored:

- `references/`
- `references.zip`
- `.venv/`
- `.DS_Store`
- `data/raw/kelpwatch_aoi/`
- `data/processed/*.csv`
- `data/external/noaa/`
- `scripts/__pycache__/`
- any `.env` or API/token files
- raw NetCDF, GeoTIFF, shapefile, zip, and cache outputs

The current `.gitignore` already covers most of these, but it does not ignore `references/` PDFs. Add a `references/` or `references/**/*.pdf` rule later after human review.

## Needs Review

| Path | Reason |
|---|---|
| `src/` | Could become a clean library layer, but currently appears disconnected from the canonical scripts. |
| `notebooks/` | Some notebooks may be useful teaching artifacts; most look superseded by scripts. |
| `refs.bib` | Potentially useful citation metadata; decide whether it should replace local PDF tracking. |
| `docs/maps/kelpwatch_regular_10km_fishnet_preview_map.html` | Good exploratory map, but paper may only need static Figure 1. |
| `outputs/metadata/model_comparison_results.csv` vs `outputs/metadata/model_comparison_test_metrics.csv` | Decide one canonical main model table. |
| `results/tables/integrated_model_comparison_master.csv` | Useful index but may be too broad and generated from exploratory tasks. |
| `results/tables/claim_gate_summary.csv` | Useful caution layer; decide whether to cite in supplement. |
| `scripts/download_kelpwatch_cell_exports.py` | Automation status may depend on Kelpwatch interface stability. |
| `scripts/extract_regular_10km_fishnet_package.py` | One-time extraction helper; preserve only if package provenance is documented. |
| `outputs/diagnostics/ecological_data_feasibility_report.md` | Good future-work framing, not main paper evidence. |

## Script Classification

| Script | Recommended class | Rationale |
|---|---|---|
| `filter_kelpwatch_cells.py` | `main_pipeline` | Historical-footprint filtering defines final main cells. |
| `build_kelpwatch_panel.py` | `main_pipeline` | Builds annual canopy panel used by all main claims. |
| `construct_decline_labels.py` | `main_pipeline` | Creates p25 decline labels and next-year canopy fields. |
| `build_noaa_environmental_features.py` | `main_pipeline` / `supplementary_analysis` | Needed for current modeling dataset but NOAA is not central to the narrowed claims. |
| `train_model_comparison.py` | `main_pipeline` | Core model comparison for Claim 1. |
| `tune_decision_thresholds.py` | `supplementary_analysis` | Useful decision-threshold operating point analysis. |
| `plot_study_area_retained_grid_map.py` | `main_pipeline` | Produces paper Figure 1. |
| `diagnose_zero_persistence.py` | `main_decision_support` | Directly supports Claim 2. |
| `11_naive_persistence_baseline_benchmark.py` | `main_decision_support` | Directly gates early-warning claims. |
| `30_threshold_tradeoff_curves.py` | `supplementary_analysis` | Useful threshold diagnostic; not one of three main claims. |
| `31_alert_budget_topk_evaluation.py` | `main_decision_support` | Directly supports Claim 3. |
| `32_reference_point_state_taxonomy.py` | `main_decision_support` | Directly supports Claims 2 and 3. |
| `summarize_kelpwatch_cell_exports.py` | `supplementary_analysis` | Supports data QA. |
| `validate_kelpwatch_exports.py` | `supplementary_analysis` | Supports raw export QA. |
| `check_regular_fishnet_distribution.py` | `supplementary_analysis` | Spatial-design validation. |
| `verify_regular_10km_fishnet_design.py` | `supplementary_analysis` | Spatial-design validation. |
| `extract_regular_10km_fishnet_package.py` | `legacy_or_superseded` | One-time package extraction helper. |
| `download_kelpwatch_cell_exports.py` | `unclear` | Depends on current Kelpwatch interface; keep documented but not main. |
| `diagnose_model_results.py` | `supplementary_analysis` | Model-diagnostics detail. |
| `analyze_canopy_environment_context.py` | `supplementary_analysis` | Useful environmental context but not core claims. |
| `interpret_models_shap.py` | `supplementary_analysis` | SHAP interpretation is useful but not central. |
| `diagnose_environmental_covariates.py` | `supplementary_analysis` | Environmental QC/sensitivity. |
| `diagnose_multicollinearity.py` | `supplementary_analysis` | Method caution. |
| `run_recall_oriented_modeling_extensions.py` | `supplementary_analysis` | Useful but broad; keep out of main narrative. |
| `09_build_multiscale_environmental_features.py` | `exploratory` | V2 environmental scale sensitivity. |
| `10_multiscale_exposure_selection.py` | `exploratory` | V2 environmental scale selection. |
| `16_build_crw_5km_sst_features.py` | `exploratory` | Daily CRW dry-run/long-run path. |
| `16a_download_crw5km_point_cache.py` | `exploratory` | Optional long-run downloader. |
| `16b_build_crw5km_composite_features.py` | `exploratory` | CRW composite sensitivity. |
| `17_build_bathymetry_habitat_features.py` | `exploratory` | Static habitat covariate extension. |
| `19_build_canopy_trajectory_features.py` | `supplementary_analysis` / `exploratory` | Related to persistence, but beyond minimal claim set. |
| `21_integrate_model_results.py` | `supplementary_analysis` | Useful synthesis, too broad for main. |
| `22_apply_claim_gates.py` | `supplementary_analysis` | Useful caution/gate logic. |
| `24_rare_event_alert_learning.py` | `exploratory` | Rare-event learning should not drive current claims. |
| `25_build_wave_exposure_features.py` | `exploratory` | Wave covariate extension. |
| `26_spatial_validation_diagnostics.py` | `supplementary_analysis` | Important robustness limitation. |
| `27_spatial_failure_repair.py` | `exploratory` | Repair experiment, not current main evidence. |
| `28_multihorizon_actionable_warning.py` | `exploratory` | Horizon sensitivity only. |
| `29_quarterly_actionable_warning_feasibility.py` | `exploratory` | Feasibility only; current labels are season-sensitive. |
| `14_ecological_data_feasibility_scan.py` | `exploratory` | V3 planning only. |

## Main Paper Output Set

Proposed main-text figures, maximum five:

1. `outputs/maps/figure1_study_area_retained_10km_grid_cells.png`  
   Claim mapping: study design context for all claims.
2. `outputs/figures/model_performance_comparison.png`  
   Claim mapping: Claim 1, broad low-canopy/decline-risk screening.
3. `outputs/figures/canopy_persistence_scatter.png` or `outputs/figures/canopy_quantile_decline_rate.png`  
   Claim mapping: Claim 2, canopy-state persistence. If space is tight, combine or choose one.
4. `outputs/figures/reference_point_transition_matrix_probabilities.png`  
   Claim mapping: Claim 2, state-transition structure.
5. `outputs/figures/topk_transition_composition.png`  
   Claim mapping: Claim 3, top-k alerts and persistent-low concentration.

Proposed main-text tables, maximum four:

1. Spatial/data construction table from `geometries/regular_10km_fishnet/*filter*` and `outputs/metadata/kelpwatch_panel_ge500_summary.csv`: candidate cells, footprint-positive cells, retained cells, years.
2. Model comparison table from `outputs/metadata/model_comparison_test_metrics.csv`: compact test PR-AUC, recall, F1/F2, false negatives for main model families.
3. Persistence/transition table from `outputs/diagnostics/zero_persistence_transition_rates.csv` and `outputs/tables/reference_point_transition_summary.csv`: persistent low, recovery, new low transition, stable non-low shares.
4. Top-k prioritisation table from `outputs/tables/alert_budget_topk_annual_summary.csv` and `outputs/tables/topk_transition_composition.csv`: top 5/top 20 budgets, precision/recall, transition composition.

## Supplementary Output Set

Recommended supplementary outputs:

- Full `outputs/tables/threshold_tradeoff_metrics.csv`.
- Full `outputs/tables/alert_budget_topk_metrics.csv`.
- Full `outputs/tables/reference_point_state_taxonomy.csv`.
- `outputs/tables/model_risk_by_transition_type.csv`.
- SHAP figures and tables.
- Multicollinearity diagnostics.
- Spatial holdout diagnostics.
- Full model diagnostics and confusion matrices.
- Environmental covariate QC/sensitivity diagnostics.
- Integrated model comparison master table.
- Claim-gate summary and sensitivity tables.

## Outputs to Move Out of Main Text

These can stay in the repository but should not be foregrounded in the main paper narrative:

- CRW 5 km SST comparison.
- Bathymetry/habitat covariates.
- Wave exposure features.
- Multi-scale OISST exposure.
- Rare-event alert learning.
- Multi-horizon actionable warning.
- Quarterly actionable warning feasibility.
- Spatial failure repair features.
- Ecological V3 feasibility planning.

## Suggested `.gitignore` Review

Current `.gitignore` already covers:

- `__pycache__/`
- `.ipynb_checkpoints/`
- `.env`
- `venv/`, `.venv/`
- `data/raw/**`, `data/processed/**`, `data/external/**` with `.gitkeep`/README exceptions
- large geospatial/gridded files
- generated `outputs/metadata`, `outputs/figures`, `outputs/maps`, and `outputs/model_results` by default
- `.DS_Store`

Recommended later additions, after human review:

```gitignore
# Local literature PDFs and archives
references/
*.pdf
```

If `*.pdf` is too broad because selected paper-ready PDFs may be intentionally tracked later, prefer:

```gitignore
references/**/*.pdf
references.zip
```

Do not modify `.gitignore` in this audit pass.

## Wording and Claim Framing Audit

The repository generally uses cautious language. Searches found phrases such as "operational early warning" mostly in negative or cautionary statements, which is appropriate.

Maintain these preferred phrases:

- risk-state screening
- monitoring prioritisation
- decision-support diagnostic
- persistence-aware evaluation
- apparent early-warning skill
- strict new-transition warning remains limited

Avoid foregrounding these phrases as positive claims:

- operational early warning system
- successfully predicts kelp decline
- validated warning tool
- causal effect of SST/upwelling
- actionable warning without qualification

README issue:

- The README currently reads like a complete research log. That is good for transparency but broad for a manuscript-facing landing page.
- Later cleanup should split the README into a concise paper-facing README and a detailed `docs/full_workflow_log.md` or `docs/supplementary_workflow_index.md`.

## Markdown Issue Cleanup Plan

GitHub issues were not accessed in this audit. Use this markdown issue plan if issue cleanup is needed later.

Likely completed tasks:

- Kelpwatch 10 km fishnet design.
- Main annual Kelpwatch panel construction.
- NOAA V1 feature construction and validation.
- Five-model comparison.
- Threshold trade-off curves.
- Alert-budget top-k evaluation.
- Reference-point state taxonomy.
- Figure 1 study area map.

Future-work tasks:

- Paper-facing README rewrite.
- Freeze final main figures/tables.
- Move exploratory analyses into supplementary index.
- Decide whether to archive `notebooks/` and `src/`.
- Add `references/` ignore rule or replace local PDFs with citation metadata only.
- Decide whether large rare-event detail tables should remain tracked.
- Create a reproducibility manifest for the final paper run.

## Recommended Next Cleanup Steps

1. Create a paper-facing README outline that lists only the three claims and the main figures/tables.
2. Create `docs/supplementary_workflow_index.md` for all non-main analyses.
3. Choose one canonical model comparison table and one canonical prediction-probability table.
4. Decide whether `src/` should become a real package or be archived.
5. Decide whether notebooks are teaching artifacts or archive candidates.
6. Add `references/` to `.gitignore` after confirming local PDF policy.
7. Move exploratory output descriptions out of the README and into supplementary docs.
8. Create a final `paper_freeze_manifest.md` listing scripts, inputs, outputs, and commit hash for the manuscript.

## What Should Not Be Deleted Yet

- Do not delete `geometries/regular_10km_fishnet/`.
- Do not delete prediction probability outputs until all decision-support diagnostics are frozen.
- Do not delete detailed diagnostics until the final main/supplement split is approved.
- Do not delete notebooks or `src/` before confirming whether they are needed for coursework, teaching, or future packaging.
- Do not delete local raw/processed/cache data if they are needed to rerun analyses.
- Do not delete `references/`; just keep it untracked/local unless a citation-only structure replaces it.

