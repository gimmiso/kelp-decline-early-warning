"""Build the reader-facing notebook for the journal-extension experiment."""

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
        default=Path("outputs/experiments/20260814_journal_extension_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/08_journal_extension_audit.ipynb"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decision = json.loads((args.run_dir / "decision.json").read_text(encoding="utf-8"))
    protocol = decision["protocol_current_only"]
    budgets = decision["budget_current_only"]
    sensitivity_range = decision["sensitivity_current_macro_lift_range"]
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
            f"""# Journal-extension robustness audit

## tl;dr

- The current-canopy signal survives fixed Logistic, Random Forest, and XGBoost expanding-window models.
- For current-only, random-CV versus expanding-window macro within-year AP lift is {protocol['random_macro_within_year_lift_mean']:.3f} versus {protocol['expanding_macro_within_year_lift']:.3f}; the pooled comparison is {protocol['random_pooled_lift_mean']:.3f} versus {protocol['expanding_pooled_lift']:.3f}.
- Trajectory adds a small supported increment for Logistic and Random Forest, but not XGBoost. OISST and CUTI/BEUTI show no supported positive increment in any family.
- Current-only event recall is {budgets['top_5pct']['recall']:.1%}, {budgets['top_10pct']['recall']:.1%}, {budgets['top_20pct']['recall']:.1%}, and {budgets['top_30pct']['recall']:.1%} at 5%, 10%, 20%, and 30% annual monitoring budgets.
- Across nine outcome/eligibility combinations, current-only macro within-year AP lift ranges from {sensitivity_range[0]:.3f} to {sensitivity_range[1]:.3f}; the OISST increment is non-positive in all nine.

This is a pilot-informed reviewer-robustness analysis, not preregistered or independent confirmation."""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

The notebook audits four additions to the locked 165-cell analysis: evaluation protocol/aggregation, model-family robustness, decision-budget curves, and label/eligibility sensitivity.

### Key Assumptions

- Operational estimates use 2005–2024 expanding-window predictions and macro within-year AP lift over annual prevalence.
- Repeated random cell-year CV intentionally permits future-year and repeated-cell information in training; it is an optimism diagnostic, not a deployment estimate.
- Logistic, Random Forest, and XGBoost use fixed prespecified specifications. This is a robustness check, not a hyperparameter competition.
- Decline thresholds are operational labels, not validated ecological-collapse thresholds.
- NOAA features are public environmental proxies; their predictive increment is not a causal test of environmental importance."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
if not (ROOT / 'outputs').exists() and (ROOT.parent / 'outputs').exists():
    ROOT = ROOT.parent
RUN = ROOT / 'outputs/experiments/20260814_journal_extension_v1'
assert (RUN / 'manifest.json').exists()

quality = pd.read_csv(RUN / 'data_quality_profile.csv')
protocol = pd.read_csv(RUN / 'protocol_comparison.csv')
model_summary = pd.read_csv(RUN / 'model_family_summary.csv')
increments = pd.read_csv(RUN / 'model_family_increment_summary.csv')
budget = pd.read_csv(RUN / 'budget_summary.csv')
sensitivity = pd.read_csv(RUN / 'sensitivity_model_metrics.csv')
sensitivity_increments = pd.read_csv(RUN / 'sensitivity_increment_summary.csv')
validation = pd.read_csv(RUN / 'validation_recheck.csv')
decision = json.loads((RUN / 'decision.json').read_text(encoding='utf-8'))

assert validation['passed'].all()
assert decision['classification'].startswith('pilot-informed')
print('Loaded', RUN)
print('Validation checks passed:', int(validation['passed'].sum()), '/', len(validation))"""
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell(
            """quality"""
        ),
        nbf.v4.new_markdown_cell(
            """The intended grain is one cell-year. The panel has 165 unique cells and 6,930 unique cell-year rows. The full OISST evaluation contains 2,614 eligible forecast rows; the supported CUTI/BEUTI domain contains 1,698."""
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### 1. Evaluation protocol and aggregation"),
        nbf.v4.new_code_cell(
            """protocol.loc[protocol['domain'].eq('full_oisst_domain'), [
    'feature_label', 'random_pooled_mean', 'expanding_pooled',
    'random_macro_within_year_mean', 'expanding_macro_within_year',
    'random_minus_expanding_pooled',
    'random_minus_expanding_macro_within_year'
]].round(4)"""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=str(RUN / 'figure_01_protocol_comparison.png')))"""
        ),
        nbf.v4.new_markdown_cell(
            """Random CV does not inflate every metric equally. The current-only macro within-year estimate is nearly unchanged, while pooled estimates and richer information blocks show larger optimism. The paper should therefore report the exact protocol/metric rather than claim a universal random-split inflation factor."""
        ),
        nbf.v4.new_markdown_cell("### 2. Fixed model-family robustness"),
        nbf.v4.new_code_cell(
            """increments.loc[
    (increments['domain'].eq('full_oisst_domain') & increments['comparison'].isin(['trajectory_minus_current', 'oisst_minus_trajectory']))
    | (increments['domain'].eq('upwelling_supported_domain') & increments['comparison'].eq('cuti_beuti_minus_oisst')),
    ['model_label', 'comparison', 'estimate', 'ci_low', 'ci_high']
].round(4)"""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=str(RUN / 'figure_02_model_family_increments.png')))"""
        ),
        nbf.v4.new_markdown_cell(
            """The defensible statement is asymmetric: the current-state ranking signal is model-robust; trajectory has small and partly model-dependent incremental value; neither OISST nor CUTI/BEUTI has supported positive incremental value in any tested model family."""
        ),
        nbf.v4.new_markdown_cell("### 3. Monitoring-budget trade-off"),
        nbf.v4.new_code_cell(
            """budget.loc[
    budget['domain'].eq('full_oisst_domain') & budget['model_family'].eq('logistic'),
    ['feature_set', 'budget_fraction', 'micro_recall', 'recall_ci_low', 'recall_ci_high', 'micro_precision']
].round(4)"""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=str(RUN / 'figure_03_budget_recall_curve.png')))"""
        ),
        nbf.v4.new_markdown_cell(
            """Trajectory changes current-only recall by less than one percentage point at 5%, 10%, and 20% budgets, and by about one point at 30%. OISST is lower than trajectory at every plotted budget. The statistically detectable trajectory AP increment therefore has limited operational magnitude."""
        ),
        nbf.v4.new_markdown_cell("### 4. Outcome and eligibility sensitivity"),
        nbf.v4.new_code_cell(
            """sensitivity.loc[
    sensitivity['domain'].eq('full_oisst_domain') & sensitivity['feature_set'].eq('current_only'),
    ['decline_threshold', 'canopy_cutoff', 'n', 'events', 'prevalence', 'macro_within_year_ap_lift']
].sort_values(['decline_threshold', 'canopy_cutoff']).round(4)"""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=str(RUN / 'figure_04_threshold_canopy_sensitivity.png')))"""
        ),
        nbf.v4.new_markdown_cell(
            """The current-only ranking conclusion persists across all nine operational definitions. Trajectory remains positive in point estimate across all nine, but its confidence interval crosses zero in two 40% variants. OISST is negative in point estimate in all nine and never has a confidence interval excluding zero on the positive side."""
        ),
        nbf.v4.new_markdown_cell("## Takeaways"),
        nbf.v4.new_markdown_cell(
            """1. The strongest journal claim is the robust value of low-cost current-canopy information under deployment-realistic ranking evaluation.
2. Recent trajectory is a small secondary information source, not a large operational improvement.
3. The selected coarse NOAA proxies do not justify claims of additional predictive information after canopy history; this does not imply that environment is ecologically unimportant.
4. The paper should foreground validation design and marginal information value, then present kelp ecology as the application and variable rationale.
5. All headline numbers are independently recomputed in `validation_recheck.csv`; the overall assessment is `Share with caveats`."""
        ),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
