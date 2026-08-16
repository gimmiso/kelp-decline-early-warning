"""Build the reader-facing notebook for the Giraldo error case study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat as nbf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "outputs/experiments/20260816_giraldo_error_case_study_v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/09_giraldo_error_case_study.ipynb"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decision = json.loads((args.run_dir / "decision.json").read_text(encoding="utf-8"))
    matched = decision["matched_population"]
    logistic_oisst = decision["matched_subset_increments"]["logistic"][
        "oisst_minus_trajectory"
    ]
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            f"""# Giraldo field-data error-diagnostic case study

## tl;dr

- The Giraldo data overlap the operational prediction period in **{matched['cells']} cells and {matched['cell_years']} cell-years**, with {matched['events']} decline events from {matched['first_year']}–{matched['last_year']}.
- In the matched subset, Logistic OISST-minus-trajectory macro within-year AP lift is **{logistic_oisst['estimate']:+.4f}** with a year-bootstrap 95% interval of **[{logistic_oisst['ci_low']:+.4f}, {logistic_oisst['ci_high']:+.4f}]**. Random Forest and XGBoost intervals also include zero.
- None of the 20 prespecified Logistic combinations of four ecological axes and five prediction diagnostics met the full stability rule.
- The result is an identification limit of a California-only, non-random, cross-scale field subset. It is not evidence that urchins, habitat, or kelp identity are ecologically unimportant.

Classification: pilot-informed exploratory diagnostic; not independent confirmation."""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

The primary 165-cell study estimates the incremental decision value of current canopy, canopy trajectory, and public NOAA proxies. This secondary case study does not add Giraldo field variables as a fifth coastwide prediction block. Instead, it asks whether field-observed ecological conditions explain recurrent out-of-fold errors or identify conditions where OISST improved annual risk rankings.

### Key Assumptions

- Field observations are aggregated from 60 m² transects to site-year medians and then to 10 km cell-year medians.
- A site-year is assigned to its nearest locked cell center when the Haversine distance is at most 7.2 km; 5 km is a strict sensitivity analysis.
- Risk ranks and top-20% monitoring selections are calculated in each full annual prediction cohort before the matched field subset is selected.
- Four axes were fixed before execution: urchin grazing pressure, rock probability, depth, and bull-versus-giant kelp contrast.
- Association estimates include region and year fixed effects with two-way cell/year cluster-robust intervals and within-outcome Benjamini–Hochberg adjustment.
- Field coverage is not representative of the full West Coast domain and is not imputed to unobserved cells or years."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
if not (ROOT / 'outputs').exists() and (ROOT.parent / 'outputs').exists():
    ROOT = ROOT.parent
RUN = ROOT / 'outputs/experiments/20260816_giraldo_error_case_study_v1'
assert (RUN / 'manifest.json').exists()

quality = pd.read_csv(RUN / 'data_quality_profile.csv')
coverage = pd.read_csv(RUN / 'cohort_coverage_by_region.csv')
performance = pd.read_csv(RUN / 'performance_summary.csv')
increments = pd.read_csv(RUN / 'performance_increment_summary.csv')
associations = pd.read_csv(RUN / 'ecological_axis_associations.csv')
support = pd.read_csv(RUN / 'condition_support_summary.csv')
repeated = pd.read_csv(RUN / 'repeated_error_cells.csv')
validation = pd.read_csv(RUN / 'validation_recheck.csv')
decision = json.loads((RUN / 'decision.json').read_text(encoding='utf-8'))

assert validation['passed'].all()
assert decision['stable_condition_count'] == 0
print('Loaded', RUN)
print('Independent checks passed:', int(validation['passed'].sum()), '/', len(validation))"""
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell("quality"),
        nbf.v4.new_code_cell(
            """coverage[['region_group', 'full_cell_years', 'full_cells', 'matched_cell_years', 'matched_cells', 'matched_events', 'matched_cell_year_share']].round(3)"""
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(RUN / 'figure_01_field_coverage.png'), width=700))"
        ),
        nbf.v4.new_markdown_cell(
            """The raw table contains 9,728 transect rows, 191 sites, and 23 survey years, but the operational error analysis contains 424 matched cell-years in 52 California cells. Northern California contributes only 35 cell-years, and Oregon, Washington, and Baja have no matched field observations. The matched data therefore serve as a diagnostic case, not an external-validation population."""
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### 1. Full-domain and matched-cohort performance"),
        nbf.v4.new_code_cell(
            """performance.loc[performance['model_family'].eq('logistic'), ['scope', 'feature_set', 'estimable_years', 'macro_within_year_ap_lift', 'ci_low', 'ci_high']].round(4)"""
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(RUN / 'figure_04_matched_vs_full_performance.png'), width=760))"
        ),
        nbf.v4.new_code_cell(
            """increments.loc[increments['scope'].eq('giraldo_matched'), ['model_family', 'comparison', 'paired_years', 'estimate', 'ci_low', 'ci_high']].round(4)"""
        ),
        nbf.v4.new_markdown_cell(
            """The matched cohort shows higher absolute AP lift than the full domain, demonstrating selection rather than stronger generalizability. Within that matched cohort, OISST-minus-trajectory intervals include zero for all three model families. The subset therefore does not support a stable conditional OISST benefit."""
        ),
        nbf.v4.new_markdown_cell("### 2. Prespecified ecological-axis associations"),
        nbf.v4.new_code_cell(
            """primary = associations.loc[(associations['scope'].eq('primary_7p2km')) & (associations['model_family'].eq('logistic')), ['outcome_label', 'axis_label', 'n', 'estimate_per_sd', 'ci_low', 'ci_high', 'q_value', 'supported_bh']]
primary.round(4)"""
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(RUN / 'figure_02_primary_ecological_associations.png'), width=900))"
        ),
        nbf.v4.new_code_cell(
            """support[['outcome', 'axis', 'primary_estimate_per_sd', 'primary_ci_low', 'primary_ci_high', 'primary_q_value', 'same_direction_model_count', 'strict_same_direction', 'stable_condition_supported']].round(4)"""
        ),
        nbf.v4.new_markdown_cell(
            """No prespecified combination satisfies all four requirements: an interval excluding zero, q < 0.10, the same direction in at least two model families, and the same direction under the strict 5-km match. This does not establish zero ecological effect; it means the available partial and cross-scale observations do not identify a stable failure condition."""
        ),
        nbf.v4.new_markdown_cell("### 3. Recurrent error cells"),
        nbf.v4.new_code_cell(
            """repeated.loc[repeated['event_opportunities'].ge(3), ['cell_id', 'region_group', 'observed_years', 'event_opportunities', 'trajectory_false_negatives', 'trajectory_false_negative_rate', 'trajectory_false_positives', 'trajectory_false_positive_rate', 'mean_grazing_log1p']].sort_values(['trajectory_false_negative_rate', 'event_opportunities'], ascending=[False, False]).head(15).round(3)"""
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(RUN / 'figure_03_recurrent_miss_vs_grazing.png'), width=760))"
        ),
        nbf.v4.new_markdown_cell(
            """Several cells are repeatedly missed at a 20% annual monitoring budget, but those misses are not monotonically concentrated along the surveyed urchin-pressure axis after region and year are considered. Recurrent-error maps are still useful for designing new standardized field surveys, but not as evidence that the present Giraldo axes can repair the coastwide predictor."""
        ),
        nbf.v4.new_markdown_cell("## Takeaways"),
        nbf.v4.new_markdown_cell(
            """1. Keep Giraldo as a California-only ecological error diagnostic, not information block B5.
2. The matched field cohort is selected and has different apparent performance from the full domain.
3. The four fixed ecological axes do not reveal a robust condition where OISST becomes valuable or trajectory errors recur.
4. The defensible paper claim is that neither the average analysis nor this limited conditional diagnostic supports operational incremental value for the tested public proxy block.
5. A stronger test requires standardized biological sampling targeted to recurrent-error and comparison cells at a spatial support closer to the decision unit."""
        ),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
