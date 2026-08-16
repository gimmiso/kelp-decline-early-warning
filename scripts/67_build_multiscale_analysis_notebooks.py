"""Build reader-facing audit notebooks for the multiscale kelp analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat as nbf
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coast-run", type=Path, required=True)
    parser.add_argument("--california-run", type=Path, required=True)
    parser.add_argument("--coast-output", type=Path, default=Path("notebooks/10_coastwide_5km_primary_audit.ipynb"))
    parser.add_argument("--california-output", type=Path, default=Path("notebooks/11_california_local_ecology_case.ipynb"))
    return parser.parse_args()


def metadata() -> dict[str, object]:
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def build_coast(run: Path, output: Path) -> None:
    decision = json.loads((run / "decision.json").read_text(encoding="utf-8"))
    audit = json.loads((run / "input_audit.json").read_text(encoding="utf-8"))
    conclusions = decision["model_family_increment_conclusions"]
    current = decision["primary_logistic_current_macro_lift"]
    trajectory = conclusions["logistic"]["trajectory_minus_current"]
    oisst = conclusions["logistic"]["oisst_minus_trajectory"]
    run_ref = relative(run)
    notebook = nbf.v4.new_notebook(metadata=metadata())
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            f"""# Coastwide 5-km primary analysis audit

## tl;dr

- The 5-km panel contains {audit['cells']} stable coastal cells and {audit['eligible_forecast_rows']} eligible forecast cell-years.
- Current canopy alone produced a Logistic macro within-year AP lift of {current:.3f}.
- Trajectory added {trajectory['estimate']:+.3f} (95% year-block CI {trajectory['ci'][0]:+.3f} to {trajectory['ci'][1]:+.3f}); all three fixed model families supported a positive increment.
- OISST added {oisst['estimate']:+.3f} (95% CI {oisst['ci'][0]:+.3f} to {oisst['ci'][1]:+.3f}) and was not supported. No positive OISST point estimate appeared in the nine locked label/eligibility variants.
- All independent validation checks passed. This is a pilot-informed robustness analysis, not independent confirmation."""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

This analysis asks whether the original coastwide ranking result survives a finer 5-km support while retaining a deployment-aligned evaluation. Information blocks are added in a fixed order: current canopy, trajectory, OISST, and CUTI/BEUTI where supported.

### Key assumptions

- The prediction target is a next-year relative canopy decline, not a validated ecological-collapse threshold.
- Expanding-window predictions and macro within-year AP lift are primary because monitoring ranks cells within each deployment year.
- All feature-block comparisons use identical rows inside their stated support domain.
- OISST and CUTI/BEUTI are operational public proxies, not direct causal measurements.
- The 5-km grid reduces spatial aggregation but does not eliminate grid-origin sensitivity or overlapping ecological processes."""
        ),
        nbf.v4.new_code_cell(
            f"""from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
if not (ROOT / 'outputs').exists() and (ROOT.parent / 'outputs').exists():
    ROOT = ROOT.parent
RUN = ROOT / {run_ref!r}
assert (RUN / 'manifest.json').exists()

audit = json.loads((RUN / 'input_audit.json').read_text(encoding='utf-8'))
quality = pd.read_csv(RUN / 'data_quality_profile.csv')
protocol = pd.read_csv(RUN / 'protocol_comparison.csv')
summary = pd.read_csv(RUN / 'model_family_summary.csv')
increments = pd.read_csv(RUN / 'model_family_increment_summary.csv')
budget = pd.read_csv(RUN / 'budget_summary.csv')
sensitivity = pd.read_csv(RUN / 'sensitivity_increment_summary.csv')
validation = pd.read_csv(RUN / 'validation_recheck.csv')
assert validation['passed'].all()
print('Loaded', RUN.relative_to(ROOT))
print('Validation:', int(validation['passed'].sum()), '/', len(validation))"""
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell("pd.DataFrame([audit]).T.rename(columns={0: 'value'})"),
        nbf.v4.new_code_cell("quality"),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figure_00_5km_grid_map.png')))"),
        nbf.v4.new_markdown_cell(
            "The intended grain is one stable 5-km cell-year. OISST and upwelling comparisons use their own fixed supported domains, so a larger feature set never receives credit merely by changing its evaluation sample."
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### 1. Evaluation protocol"),
        nbf.v4.new_code_cell("protocol.round(4)"),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figure_01_protocol_comparison.png')))"),
        nbf.v4.new_markdown_cell("### 2. Information-block increments across model families"),
        nbf.v4.new_code_cell("increments[['domain', 'model_label', 'comparison', 'estimate', 'ci_low', 'ci_high']].round(4)"),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figure_02_model_family_increments.png')))"),
        nbf.v4.new_markdown_cell(
            "At 5 km, the trajectory increment is positive in Logistic, Random Forest, and XGBoost. Neither public environmental block has a confidence interval supporting a positive increment in any fixed family."
        ),
        nbf.v4.new_markdown_cell("### 3. Monitoring-budget trade-off"),
        nbf.v4.new_code_cell("budget[['feature_set', 'budget_fraction', 'micro_recall', 'recall_ci_low', 'recall_ci_high']].round(4)"),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figure_03_budget_recall_curve.png')))"),
        nbf.v4.new_markdown_cell("### 4. Outcome and eligibility sensitivity"),
        nbf.v4.new_code_cell("sensitivity.round(4)"),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figure_04_threshold_canopy_sensitivity.png')))"),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. The coastwide decision-support claim survives the finer 5-km support.
2. Recent canopy trajectory is a small, model-robust secondary block at 5 km.
3. OISST and CUTI/BEUTI do not add demonstrated ranking value after canopy history under the locked timing and support rules.
4. The negative proxy result must not be rewritten as evidence that temperature or upwelling are ecologically irrelevant.
5. Grid-origin sensitivity remains a warning against interpreting any single top-priority cell boundary as uniquely correct."""
        ),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)


def build_california(run: Path, output: Path) -> None:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    panel_manifest = json.loads((run / "panel_manifest.json").read_text(encoding="utf-8"))
    mur_manifest = json.loads((run / "mur_feature_manifest.json").read_text(encoding="utf-8"))
    increments = pd.read_csv(run / "increment_summary.csv")
    logistic_mur = increments.loc[(increments["model_family"].eq("logistic")) & increments["comparison"].eq("mur_minus_trajectory")].sort_values("support_m")
    mur_text = "; ".join(
        f"{int(row.support_m)} m {row.estimate:+.3f} (95% CI {row.ci_low:+.3f} to {row.ci_high:+.3f})"
        for row in logistic_mur.itertuples(index=False)
    )
    cover = pd.DataFrame(manifest["coverage"])
    coverage_text = "; ".join(f"{int(row.support_m)} m: {int(row.matched_sites)} sites, {int(row.matched_rows)} site-years" for row in cover.itertuples(index=False))
    run_ref = relative(run)
    notebook = nbf.v4.new_notebook(metadata=metadata())
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            f"""# California 300-m/1-km ecological error-diagnostic case study

## tl;dr

- Local KelpWatch canopy was rebuilt around Giraldo field sites at two prespecified supports rather than assigning the field data to coastwide 5-km cells.
- Field-matched coverage was {coverage_text}.
- Logistic MUR-minus-trajectory increments were {mur_text}.
- {manifest['stable_condition_count']} ecological axis–error associations passed the strict cross-model and cross-support stability rule.
- The result is a California partial-sample diagnostic of when the remote-data model fails; it is not a coastwide causal model and not a claim that 300 m is the true ecological process scale."""
        ),
        nbf.v4.new_markdown_cell(
            f"""## Context & Methods

The coastwide ranking model and the ecological explanation are separated because they answer different questions and operate at different spatial supports. Around each Giraldo site, pre-2005 positive KelpWatch pixels define a fixed footprint. Annual canopy requires at least three valid quarters including Q3 and at least 50% valid habitat pixels. Overlapping footprints are grouped before inference.

Daily NASA JPL MUR v4.1 SST is obtained from public ERDDAP replicas at 0.01° support, with a cross-endpoint value-parity gate. Features include annual mean and maximum SST, maximum 7-day anomaly, positive anomaly degree-days, days above an expanding 90th percentile, and maximum consecutive hot days. Climatologies use strictly prior days.

Complete MUR thermal features begin in 2004. Forecast evaluation begins in 2008 so every first fold has four complete training years; evaluation ends in 2021 with the field dataset.

### Key assumptions

- Giraldo sites are a non-probability California field sample ({panel_manifest['sites_with_coordinates']} sites with coordinates before local-support quality filters).
- MUR is surface SST and may differ from benthic temperature experienced by kelp and grazers.
- Field ecological axes describe associations with prediction errors, not identified causal effects.
- The field observation year is aligned to the forecast-year diagnostic; uneven survey timing and missing site-years constrain power.
- Multiple testing is controlled within each support, model family, and diagnostic outcome, followed by a stricter cross-support stability rule."""
        ),
        nbf.v4.new_code_cell(
            f"""from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
if not (ROOT / 'outputs').exists() and (ROOT.parent / 'outputs').exists():
    ROOT = ROOT.parent
RUN = ROOT / {run_ref!r}
assert (RUN / 'manifest.json').exists()

coverage = pd.read_csv(RUN / 'support_coverage.csv')
mur_quality = pd.read_csv(RUN / 'mur_quality_checks.csv')
mur_endpoint_parity = pd.read_csv(RUN / 'mur_endpoint_parity_checks.csv')
performance = pd.read_csv(RUN / 'performance_summary.csv')
increments = pd.read_csv(RUN / 'increment_summary.csv')
ecology_coverage = pd.read_csv(RUN / 'ecology_coverage.csv')
associations = pd.read_csv(RUN / 'ecological_axis_associations.csv')
stability = pd.read_csv(RUN / 'condition_stability_summary.csv')
leave_cluster = pd.read_csv(RUN / 'leave_overlap_cluster_out_summary.csv')
validation = pd.read_csv(RUN / 'validation_recheck.csv')
assert validation['passed'].all()
print('Loaded', RUN.relative_to(ROOT))
print('Validation:', int(validation['passed'].sum()), '/', len(validation))"""
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell("coverage"),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figure_00_california_sites_supports.png')))"),
        nbf.v4.new_code_cell("mur_quality"),
        nbf.v4.new_code_cell("mur_endpoint_parity"),
        nbf.v4.new_code_cell("ecology_coverage"),
        nbf.v4.new_markdown_cell(
            "The prediction panel can be much larger than the field-matched panel. Ecological claims therefore use only matched site-years and report both site and overlap-cluster counts."
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### 1. Local-support predictive increments"),
        nbf.v4.new_code_cell("increments[['support_m', 'model_family', 'comparison', 'estimate', 'ci_low', 'ci_high', 'paired_years']].round(4)"),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figure_01_local_support_increments.png')))"),
        nbf.v4.new_markdown_cell("### 2. Ecological associations with prediction errors"),
        nbf.v4.new_code_cell("stability.sort_values(['stable_condition_supported', 'outcome', 'axis'], ascending=[False, True, True]).round(4)"),
        nbf.v4.new_code_cell("leave_cluster.sort_values(['same_direction_fraction', 'maximum_absolute_change']).round(4)"),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figure_02_ecological_associations_by_support.png')))"),
        nbf.v4.new_markdown_cell(
            "A condition is highlighted only when the 300-m Logistic association passes BH control, at least two 300-m model families agree in direction, and the 1-km Logistic estimate has the same direction. This guards against selecting a favorable support after seeing results."
        ),
        nbf.v4.new_markdown_cell("### 3. Full performance context"),
        nbf.v4.new_code_cell("performance.round(4)"),
        nbf.v4.new_markdown_cell(
            """## Takeaways

1. This case study tests whether error patterns are systematically associated with local grazing, habitat, depth, or kelp-composition axes.
2. The two support sizes are a prespecified sensitivity analysis; concordance is more credible than either support alone.
3. MUR's incremental predictive value is evaluated separately from the ecological-axis error analysis.
4. Any supported association is hypothesis-generating because the field sites are partial, non-random, and observational.
5. Coastwide management performance should be reported from the 5-km analysis; ecological interpretation should be reported from this California case study with its own denominator and limitations."""
        ),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)


def main() -> None:
    args = parse_args()
    build_coast(args.coast_run, args.coast_output)
    build_california(args.california_run, args.california_output)
    print(args.coast_output)
    print(args.california_output)


if __name__ == "__main__":
    main()
