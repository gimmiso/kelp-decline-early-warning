"""Audit whether the v3 positive direction is driven by one year or context."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_runner() -> object:
    path = Path(__file__).with_name("72_run_california_local_ecology_method_audit.py")
    spec = importlib.util.spec_from_file_location("method_audit_v3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runner at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ap_gain_by_year(frame: pd.DataFrame, left: str, right: str) -> pd.Series:
    values: dict[int, float] = {}
    for year, group in frame.groupby("year", sort=True):
        if group["event"].nunique() < 2:
            continue
        values[int(year)] = float(
            average_precision_score(group["event"], group[left])
            - average_precision_score(group["event"], group[right])
        )
    return pd.Series(values, name="ap_gain")


def main() -> None:
    args = parse_args()
    runner = load_runner()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    ranked = pd.read_csv(args.run_dir / "ranked_predictions.csv")
    annual = pd.read_csv(args.run_dir / "annual_metrics.csv")
    performance = pd.read_csv(args.run_dir / "performance_summary.csv")
    index = ["site_id", "support_m", "overlap_cluster", "context", "year", "event", "model_family"]
    wide = ranked.pivot(index=index, columns="feature_set", values="score").reset_index()
    left = "trajectory_plus_paper_full"
    right = "trajectory"
    primary = wide.loc[wide["support_m"].eq(300) & wide["model_family"].eq("logistic_spline")].copy()
    annual_gain = ap_gain_by_year(primary, left, right)
    loo_year = []
    for year in annual_gain.index:
        retained = annual_gain.drop(year)
        loo_year.append(
            {
                "excluded_year": int(year),
                "retained_years": len(retained),
                "macro_ap_gain": float(retained.mean()),
                "sign_positive": bool(retained.mean() > 0),
            }
        )
    loo_year_frame = pd.DataFrame(loo_year)

    context_rows = []
    contexts = sorted(primary["context"].unique())
    scenarios = [("all_contexts", primary)]
    scenarios.extend((f"exclude_{context}", primary.loc[~primary["context"].eq(context)]) for context in contexts)
    scenarios.extend((f"only_{context}", primary.loc[primary["context"].eq(context)]) for context in contexts)
    for scenario, frame in scenarios:
        gains = ap_gain_by_year(frame, left, right)
        context_rows.append(
            {
                "scenario": scenario,
                "estimable_years": len(gains),
                "macro_ap_gain": float(gains.mean()) if len(gains) else np.nan,
                "minimum_year_gain": float(gains.min()) if len(gains) else np.nan,
                "maximum_year_gain": float(gains.max()) if len(gains) else np.nan,
            }
        )
    context_frame = pd.DataFrame(context_rows)

    context_year_rows = []
    for (context, year), group in primary.groupby(["context", "year"], sort=True):
        if group["event"].nunique() < 2:
            continue
        context_year_rows.append(
            {
                "context": context,
                "year": int(year),
                "n": len(group),
                "events": int(group["event"].sum()),
                "within_context_ap_gain": float(
                    average_precision_score(group["event"], group[left])
                    - average_precision_score(group["event"], group[right])
                ),
            }
        )
    context_year = pd.DataFrame(context_year_rows)
    context_macro = (
        context_year.groupby("context", as_index=False)
        .agg(estimable_context_years=("year", "size"), macro_within_context_ap_gain=("within_context_ap_gain", "mean"))
    )

    excluding_2021 = primary.loc[primary["year"].ne(2021)].rename(columns={left: "score_left", right: "score_right"})
    ci_low, ci_high, successful = runner.hierarchical_increment_bootstrap(
        excluding_2021,
        int(config["forecast"]["bootstrap_replicates"]),
        int(config["forecast"]["seed"]) + 2021,
    )
    exclusion_rows = []
    for support_m in [300, 1000]:
        for family in runner.FAMILY_ORDER:
            group = annual.loc[annual["support_m"].eq(support_m) & annual["model_family"].eq(family)]
            pivot = group.pivot(index="year", columns="feature_set", values="ap_lift")
            differences = pivot[left] - pivot[right]
            exclusion_rows.append(
                {
                    "support_m": support_m,
                    "model_family": family,
                    "all_years_gain": float(differences.mean()),
                    "exclude_2021_gain": float(differences.drop(index=2021, errors="ignore").mean()),
                    "exclude_2021_positive": bool(differences.drop(index=2021, errors="ignore").mean() > 0),
                }
            )
    exclusion_frame = pd.DataFrame(exclusion_rows)

    decision_rows = []
    for support_m in [300, 1000]:
        for family in runner.FAMILY_ORDER:
            part = performance.loc[performance["support_m"].eq(support_m) & performance["model_family"].eq(family)].set_index("feature_set")
            decision_rows.append(
                {
                    "support_m": support_m,
                    "model_family": family,
                    "top20_recall_gain": float(part.loc[left, "micro_recall_top20"] - part.loc[right, "micro_recall_top20"]),
                    "brier_improvement": float(part.loc[right, "micro_brier"] - part.loc[left, "micro_brier"]),
                }
            )
    decision_frame = pd.DataFrame(decision_rows)

    annual_gain.rename_axis("year").reset_index().to_csv(args.run_dir / "primary_annual_gain.csv", index=False)
    loo_year_frame.to_csv(args.run_dir / "leave_one_year_out_sensitivity.csv", index=False)
    context_frame.to_csv(args.run_dir / "context_exclusion_sensitivity.csv", index=False)
    context_year.to_csv(args.run_dir / "within_context_year_sensitivity.csv", index=False)
    context_macro.to_csv(args.run_dir / "within_context_summary.csv", index=False)
    exclusion_frame.to_csv(args.run_dir / "exclude_2021_model_support_sensitivity.csv", index=False)
    decision_frame.to_csv(args.run_dir / "decision_metric_sensitivity.csv", index=False)
    sensitivity = {
        "primary_all_years_gain": float(annual_gain.mean()),
        "primary_excluding_2021_gain": float(annual_gain.drop(index=2021, errors="ignore").mean()),
        "primary_excluding_2021_hierarchical_ci": [ci_low, ci_high],
        "primary_excluding_2021_successful_bootstraps": successful,
        "leave_one_year_out_min": float(loo_year_frame["macro_ap_gain"].min()),
        "leave_one_year_out_max": float(loo_year_frame["macro_ap_gain"].max()),
        "leave_one_year_out_all_positive": bool(loo_year_frame["sign_positive"].all()),
        "primary_top20_recall_gain": float(
            decision_frame.loc[
                decision_frame["support_m"].eq(300) & decision_frame["model_family"].eq("logistic_spline"),
                "top20_recall_gain",
            ].iloc[0]
        ),
        "primary_brier_improvement": float(
            decision_frame.loc[
                decision_frame["support_m"].eq(300) & decision_frame["model_family"].eq("logistic_spline"),
                "brier_improvement",
            ].iloc[0]
        ),
    }
    (args.run_dir / "sensitivity_manifest.json").write_text(
        json.dumps(sensitivity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# California Giraldo v3 결과 민감도",
        "",
        f"전체연도 주 AP 증분은 {sensitivity['primary_all_years_gain']:+.4f}였으나 2021년 제외 시 {sensitivity['primary_excluding_2021_gain']:+.4f}, 계층 bootstrap 95% CI [{ci_low:+.4f}, {ci_high:+.4f}]였다.",
        f"한 해씩 제외한 추정범위는 {sensitivity['leave_one_year_out_min']:+.4f}~{sensitivity['leave_one_year_out_max']:+.4f}였다.",
        f"주모형의 top-20% recall 변화는 {sensitivity['primary_top20_recall_gain']:+.4f}, Brier 개선은 {sensitivity['primary_brier_improvement']:+.4f}였다.",
        "",
        "따라서 양의 AP 방향은 방법론 보완 후 나타났지만, 단일 연도 영향과 확률 calibration 악화를 통과하지 못하면 운영적 정보가치로 주장하지 않는다.",
    ]
    (args.run_dir / "sensitivity_summary_ko.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(sensitivity, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
