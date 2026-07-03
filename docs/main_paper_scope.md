# Main Paper Scope

This document proposes a narrow paper-facing scope for the current repository. It does not change any analysis outputs or scientific claims.

## Proposed Paper Frame

Working frame:

> Persistence-aware decision support for satellite-based kelp canopy decline screening.

The paper should be framed as a satellite monitoring and decision-support study, not as a fully operational ecological forecast system.

## Three Main Claims

### Claim 1: Broad Risk-State Screening

Broad next-year low-canopy or decline-risk states can be screened with useful skill from satellite-derived canopy histories.

Evidence base:

- Kelpwatch retained 10 km cell-year panel.
- Original next-year p25 low-canopy/decline label.
- Test-period model comparison.
- Canopy-only model performance.

Suggested wording:

> Satellite-derived canopy histories support broad next-year decline-risk state screening for retained 10 km kelp cells.

Avoid:

> The model successfully predicts kelp decline.

### Claim 2: Persistence-Aware Interpretation

Much of the apparent early-warning skill is driven by canopy-state persistence, especially persistent low-canopy conditions.

Evidence base:

- Current-to-next-year canopy persistence.
- Near-zero and at-risk diagnostics.
- Naive persistence baselines.
- Reference-point state taxonomy and transition matrix.
- Top-k transition composition showing concentration of persistent-low cases.

Suggested wording:

> Performance must be interpreted through canopy-state persistence: strong broad-label performance partly reflects persistent low-canopy states rather than new transition warning.

Avoid:

> High PR-AUC proves an early-warning signal before collapse.

### Claim 3: Monitoring Prioritisation Under Alert Budgets

Annual top-k alerts may help prioritise monitoring or intervention resources, but should not be interpreted as robust forecasts of new low-canopy transitions.

Evidence base:

- Alert-budget top-k analysis.
- Top-k transition composition.
- Reference-point taxonomy.

Suggested wording:

> Fixed-budget top-k evaluation shows how model risk scores behave as monitoring-prioritisation tools under limited inspection capacity.

Avoid:

> Top-k alerts are validated operational warnings.

## Recommended Main-Text Figures

Maximum five figures.

| Proposed figure | Candidate file | Main claim | Purpose |
|---|---|---|---|
| Figure 1. Study area and retained 10 km grid cells | `outputs/maps/figure1_study_area_retained_10km_grid_cells.png` | Study design context | Show candidate, footprint-positive, and retained cells. |
| Figure 2. Test-period model comparison | `outputs/figures/model_performance_comparison.png` | Claim 1 | Summarize broad-risk model skill without overclaiming operational warning. |
| Figure 3. Canopy persistence relationship | `outputs/figures/canopy_persistence_scatter.png` or `outputs/figures/canopy_quantile_decline_rate.png` | Claim 2 | Show current canopy as a strong short-term state signal. |
| Figure 4. Reference-state transition matrix | `outputs/figures/reference_point_transition_matrix_probabilities.png` | Claim 2 | Show persistence, recovery, new-low transitions, and non-low stability. |
| Figure 5. Top-k alert composition | `outputs/figures/topk_transition_composition.png` | Claim 3 | Show whether selected top-k alerts are mostly persistent low or include new transitions. |

Recommended figure logic:

- Use Figure 1 to define the study system.
- Use one model-performance figure for broad screening.
- Use one persistence figure and one state-transition matrix for persistence-aware interpretation.
- Use one top-k composition figure for monitoring-prioritisation.

Do not include every diagnostic figure in the main text.

## Recommended Main-Text Tables

Maximum four tables.

| Proposed table | Candidate source | Main claim | Purpose |
|---|---|---|---|
| Table 1. Dataset and retained-cell construction | `geometries/regular_10km_fishnet/*filters*.csv`, `outputs/metadata/kelpwatch_panel_ge500_summary.csv` | Study design context | Candidate cells, footprint-positive cells, retained cells, years, observations. |
| Table 2. Main model comparison | `outputs/metadata/model_comparison_test_metrics.csv` | Claim 1 | PR-AUC, recall, F1/F2, false negatives for selected feature sets/models. |
| Table 3. Persistence and transition taxonomy | `outputs/diagnostics/zero_persistence_transition_rates.csv`, `outputs/tables/reference_point_transition_summary.csv` | Claim 2 | Persistent low, recovery, new-low transition, stable non-low shares. |
| Table 4. Alert-budget prioritisation summary | `outputs/tables/alert_budget_topk_annual_summary.csv`, `outputs/tables/topk_transition_composition.csv` | Claim 3 | Top 5/top 20 budgets, precision/recall, false alerts, transition composition. |

## Recommended Supplementary Outputs

Use the supplement for:

- full model comparison tables;
- full threshold trade-off curves;
- full annual and pooled top-k metrics;
- full reference-point taxonomy table;
- model risk by transition type;
- SHAP feature importance and dependence plots;
- multicollinearity diagnostics;
- environmental covariate quality control;
- spatial holdout diagnostics;
- claim-gate tables;
- integrated model comparison master table;
- NOAA/CRW/wave/habitat/trajectory sensitivity results if mentioned.

## Analyses to Move Out of Main Text

These are useful but should not be part of the current main manuscript claims:

- NOAA OISST multi-scale exposure selection.
- CRW 5 km SST composite comparison.
- GEBCO bathymetry and habitat covariate analysis.
- CDIP wave exposure analysis.
- Rare-event alert learning.
- Multi-horizon actionable warning experiment.
- Quarterly actionable warning feasibility.
- Spatial failure repair features.
- Ecological V3 feasibility scan.

Suggested placement:

- Supplementary methods/results if already mature and compact.
- Future work if exploratory, underpowered, or not directly tied to the three claims.
- Archive branch/release if superseded or likely to distract from the manuscript.

## Suggested Future Work Items

Future work should be clear but not overpromised:

1. Test spatial generalization with larger multi-region Kelpwatch samples.
2. Replace broad low-state targets with seasonally and ecologically adjusted transition targets where possible.
3. Integrate biological monitoring data such as urchin density, predator/community indicators, substrate, and restoration history.
4. Compare OISST, CRW, and nearshore wave exposure in a smaller, ecologically grounded case study.
5. Use bootstrapping or grouped resampling to quantify uncertainty in top-k and transition-type metrics.
6. Develop a compact reproducibility package or release snapshot after paper acceptance.

## Cautious Language to Use

Use:

- broad risk-state screening;
- monitoring prioritisation;
- decision-support diagnostic;
- persistence-aware evaluation;
- satellite-derived canopy history;
- apparent early-warning skill;
- reference-point transition taxonomy;
- retained 10 km Kelpwatch cell-year panel;
- strict new-transition warning remains limited;
- top-k alerts prioritise follow-up monitoring under fixed budgets.

Examples:

> The canopy-only model performed well for broad next-year low-canopy risk-state screening, but reference-state diagnostics show that much of the highest-risk set corresponds to persistent low-canopy states.

> Annual top-k alerts are best interpreted as monitoring-prioritisation outputs rather than validated forecasts of new ecological transitions.

> Site-specific canopy reference points help distinguish persistent low states, recovery, stable non-low states, and new low-state transitions.

## Risky Language to Avoid

Avoid:

- operational early-warning system;
- successfully predicts kelp decline;
- validated warning tool;
- causal effect of SST/upwelling;
- carbon or ecosystem-service verification;
- actionable warning without qualification;
- true collapse prediction;
- robust forecast of new low-canopy transitions.

If "early warning" is used, qualify it:

> research-stage early-warning screening

or:

> apparent early-warning performance under a broad decline-state label

## Recommended Main Paper Outline

1. Introduction
   - Kelp canopy decline monitoring need.
   - Satellite time series as repeated evidence, not field replacement.
   - Problem of persistence-driven apparent predictability.

2. Data and Study Design
   - Kelpwatch 10 km retained-cell design.
   - Historical footprint filtering.
   - Annual growing-season maximum canopy panel.
   - Target/reference definitions.

3. Methods
   - Broad decline-risk model comparison.
   - Persistence and naive baseline diagnostics.
   - Reference-point state taxonomy.
   - Alert-budget top-k prioritisation.

4. Results
   - Study area/retained cells.
   - Broad model skill.
   - Persistence and transition-state results.
   - Top-k alert composition.

5. Discussion
   - What the workflow can support.
   - Why persistent low states matter.
   - Why new-transition warning remains harder.
   - Monitoring-prioritisation use case.
   - Future ecological covariates and spatial generalization.

## Paper-Freeze Checklist

Before a manuscript submission or portfolio freeze:

- Pick the five main figures and four main tables.
- Create a `paper_freeze_manifest.md` with commit hash, scripts, inputs, outputs, and Python environment.
- Move exploratory analyses from the README to a supplementary workflow index.
- Confirm that no raw data, processed data, cache files, local references, or virtual environments are staged.
- Confirm that `references/` remains untracked or is replaced by citation metadata only.
- Rerun only the final selected scripts in a clean environment if needed.

