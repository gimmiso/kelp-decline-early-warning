"""Run the 300-m/1-km California ecological error-diagnostic case study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


TRAJECTORY = [
    "canopy_lag1",
    "canopy_lag2",
    "canopy_2yr_change",
    "canopy_3yr_change",
    "canopy_3yr_mean",
    "canopy_3yr_std",
    "canopy_3yr_slope",
    "canopy_drop_from_3yr_max",
]
MUR = [
    "annual_mean_sst_mur1km",
    "annual_max_sst_mur1km",
    "annual_max_7day_mean_anomaly_mur1km",
    "annual_positive_anomaly_degree_days_mur1km",
    "annual_days_above_expanding_p90_mur1km",
    "annual_max_consecutive_hot_days_mur1km",
]
FEATURES = {
    "current_only": ["relative_canopy"],
    "current_plus_trajectory": ["relative_canopy", *TRAJECTORY],
    "trajectory_plus_mur": ["relative_canopy", *TRAJECTORY, *MUR],
}
AXES = ["grazing_log1p", "rock_probability", "depth_m", "kelp_species_contrast"]
AXIS_LABELS = {
    "grazing_log1p": "Urchin grazing pressure",
    "rock_probability": "Rock probability",
    "depth_m": "Depth",
    "kelp_species_contrast": "Bull-vs-giant kelp contrast",
}
OUTCOMES = [
    "trajectory_rank_quality_pp",
    "trajectory_false_negative",
    "trajectory_false_positive",
    "mur_signed_rank_gain_pp",
    "mur_brier_gain",
]
OUTCOME_LABELS = {
    "trajectory_rank_quality_pp": "Trajectory rank quality (pp)",
    "trajectory_false_negative": "False-negative probability",
    "trajectory_false_positive": "False-positive probability",
    "mur_signed_rank_gain_pp": "MUR signed rank gain (pp)",
    "mur_brier_gain": "MUR Brier-score improvement",
}
FAMILY_LABELS = {
    "logistic": "Logistic",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--mur", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.SubprocessError:
        return "unknown"


def load_old_case_helpers() -> object:
    path = Path(__file__).with_name("60_run_giraldo_error_case_study.py")
    spec = importlib.util.spec_from_file_location("giraldo_case_v1_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper functions from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_pipeline(family: str, seed: int) -> Pipeline:
    if family == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=5000, random_state=seed)),
            ]
        )
    if family == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=400, max_depth=5, min_samples_leaf=10, max_features="sqrt", random_state=seed, n_jobs=1)),
            ]
        )
    if family == "xgboost":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, min_child_weight=5, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss", random_state=seed, n_jobs=1)),
            ]
        )
    raise KeyError(family)


def expanding_predictions(data: pd.DataFrame, config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = int(config["forecast"]["seed"])
    train_start = int(config["forecast"]["train_start_year"])
    first_year = int(config["forecast"]["first_forecast_year"])
    last_year = int(config["forecast"]["last_forecast_year"])
    rows: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for support_m, support in data.groupby("support_m", sort=True):
        for year in range(first_year, last_year + 1):
            train = support.loc[support["year"].between(train_start, year - 1)].copy()
            test = support.loc[support["year"].eq(year)].copy()
            if train["event"].nunique() < 2 or test.empty or test["event"].nunique() < 2:
                raise ValueError(f"Invalid local expanding fold support={support_m}, year={year}")
            audits.append(
                {
                    "support_m": int(support_m),
                    "year": year,
                    "train_start": int(train["year"].min()),
                    "train_end": int(train["year"].max()),
                    "n_train": len(train),
                    "events_train": int(train["event"].sum()),
                    "n_test": len(test),
                    "events_test": int(test["event"].sum()),
                    "test_sites": int(test["site_id"].nunique()),
                    "test_overlap_clusters": int(test["overlap_cluster"].nunique()),
                }
            )
            for family in FAMILY_LABELS:
                for feature_set, features in FEATURES.items():
                    estimator = model_pipeline(family, seed)
                    estimator.fit(train[features], train["event"])
                    output = test[["site_id", "support_m", "overlap_cluster", "region_group", "year", "event"]].copy()
                    output["score"] = estimator.predict_proba(test[features])[:, 1]
                    output["model_family"] = family
                    output["feature_set"] = feature_set
                    rows.append(output)
    prediction = pd.concat(rows, ignore_index=True)
    key = ["support_m", "site_id", "year", "model_family", "feature_set"]
    if prediction.duplicated(key).any():
        raise ValueError("Local prediction keys are not unique")
    return prediction, pd.DataFrame(audits)


def add_ranks(predictions: pd.DataFrame, budget_fraction: float) -> pd.DataFrame:
    rows = []
    keys = ["support_m", "model_family", "feature_set", "year"]
    for _, group in predictions.groupby(keys, sort=True):
        ranked = group.sort_values(["score", "site_id"], ascending=[False, True]).copy()
        selected_n = max(1, int(math.ceil(len(ranked) * budget_fraction)))
        ranked["selected_at_budget"] = False
        ranked.iloc[:selected_n, ranked.columns.get_loc("selected_at_budget")] = True
        ranked["risk_percentile"] = ranked["score"].rank(method="average", pct=True)
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True)


def safe_ap_lift(frame: pd.DataFrame) -> float:
    if frame.empty or frame["event"].nunique() < 2:
        return np.nan
    return float(average_precision_score(frame["event"], frame["score"]) - frame["event"].mean())


def bootstrap_mean(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(sample, [0.025, 0.975]))


def performance_tables(ranked: pd.DataFrame, config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    replicates = int(config["forecast"]["bootstrap_replicates"])
    seed = int(config["forecast"]["seed"])
    annual_rows = []
    for keys, group in ranked.groupby(["support_m", "model_family", "feature_set", "year"], sort=True):
        annual_rows.append(
            {
                "support_m": int(keys[0]),
                "model_family": keys[1],
                "feature_set": keys[2],
                "year": int(keys[3]),
                "n": len(group),
                "events": int(group["event"].sum()),
                "prevalence": float(group["event"].mean()),
                "ap_lift": safe_ap_lift(group),
                "selected": int(group["selected_at_budget"].sum()),
                "true_positives": int(group.loc[group["selected_at_budget"], "event"].sum()),
            }
        )
    annual = pd.DataFrame(annual_rows)
    summaries = []
    for keys, group in annual.groupby(["support_m", "model_family", "feature_set"], sort=True):
        values = group["ap_lift"].dropna().to_numpy()
        low, high = bootstrap_mean(values, replicates, seed + int(keys[0]) + sum(map(ord, keys[1] + keys[2])))
        summaries.append(
            {
                "support_m": int(keys[0]),
                "model_family": keys[1],
                "feature_set": keys[2],
                "estimable_years": len(values),
                "macro_within_year_ap_lift": float(values.mean()),
                "ci_low": low,
                "ci_high": high,
                "micro_recall_top20": float(group["true_positives"].sum() / group["events"].sum()),
            }
        )
    summary = pd.DataFrame(summaries)
    increments = []
    comparisons = [
        ("trajectory_minus_current", "current_plus_trajectory", "current_only"),
        ("mur_minus_trajectory", "trajectory_plus_mur", "current_plus_trajectory"),
    ]
    for (support_m, family), group in annual.groupby(["support_m", "model_family"], sort=True):
        pivot = group.pivot(index="year", columns="feature_set", values="ap_lift")
        for label, left, right in comparisons:
            values = (pivot[left] - pivot[right]).dropna().to_numpy()
            low, high = bootstrap_mean(values, replicates, seed + int(support_m) + sum(map(ord, family + label)))
            increments.append(
                {
                    "support_m": int(support_m),
                    "model_family": family,
                    "comparison": label,
                    "left_feature_set": left,
                    "right_feature_set": right,
                    "paired_years": len(values),
                    "estimate": float(values.mean()),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return annual, summary, pd.DataFrame(increments)


def build_diagnostics(ranked: pd.DataFrame) -> pd.DataFrame:
    wide = ranked.pivot(
        index=["site_id", "support_m", "overlap_cluster", "region_group", "year", "event", "model_family"],
        columns="feature_set",
        values=["score", "risk_percentile", "selected_at_budget"],
    ).reset_index()
    wide.columns = ["_".join(str(value) for value in column if str(value)) if isinstance(column, tuple) else str(column) for column in wide.columns]
    rename = {}
    short = {"current_only": "current", "current_plus_trajectory": "trajectory", "trajectory_plus_mur": "mur"}
    for full, short_name in short.items():
        for metric in ["score", "risk_percentile", "selected_at_budget"]:
            rename[f"{metric}_{full}"] = f"{metric}_{short_name}"
    wide = wide.rename(columns=rename)
    wide["trajectory_false_negative"] = (wide["event"].eq(1) & ~wide["selected_at_budget_trajectory"].astype(bool)).astype(int)
    wide["trajectory_false_positive"] = (wide["event"].eq(0) & wide["selected_at_budget_trajectory"].astype(bool)).astype(int)
    signed = 2 * wide["event"] - 1
    wide["trajectory_rank_quality_pp"] = signed * (wide["risk_percentile_trajectory"] - 0.5) * 100
    wide["mur_signed_rank_gain_pp"] = signed * (wide["risk_percentile_mur"] - wide["risk_percentile_trajectory"]) * 100
    wide["mur_brier_gain"] = (wide["event"] - wide["score_trajectory"]) ** 2 - (wide["event"] - wide["score_mur"]) ** 2
    return wide


def ecological_associations(analysis: pd.DataFrame, q_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    helpers = load_old_case_helpers()
    rows = []
    for (support_m, family), group in analysis.groupby(["support_m", "model_family"], sort=True):
        for outcome in OUTCOMES:
            outcome_group = group
            if outcome == "trajectory_false_negative":
                outcome_group = group.loc[group["event"].eq(1)]
            elif outcome == "trajectory_false_positive":
                outcome_group = group.loc[group["event"].eq(0)]
            for axis in AXES:
                model_data = outcome_group.rename(columns={"overlap_cluster": "cell_id"})
                estimate = helpers.two_way_cluster_ols(model_data, outcome, axis)
                rows.append(
                    {
                        "support_m": int(support_m),
                        "model_family": family,
                        "outcome": outcome,
                        "outcome_label": OUTCOME_LABELS[outcome],
                        "axis": axis,
                        "axis_label": AXIS_LABELS[axis],
                        **estimate,
                    }
                )
    results = pd.DataFrame(rows)
    results["q_value"] = np.nan
    for _, index in results.groupby(["support_m", "model_family", "outcome"]).groups.items():
        results.loc[index, "q_value"] = helpers.benjamini_hochberg(results.loc[index, "p_value"])
    results["ci_excludes_zero"] = results["ci_low"].gt(0) | results["ci_high"].lt(0)
    results["supported_bh"] = results["ci_excludes_zero"] & results["q_value"].lt(q_threshold)

    stability_rows = []
    primary = results.loc[(results["support_m"].eq(300)) & (results["model_family"].eq("logistic"))]
    for row in primary.itertuples(index=False):
        direction = np.sign(row.estimate_per_sd)
        same_support = results.loc[(results["support_m"].eq(300)) & results["outcome"].eq(row.outcome) & results["axis"].eq(row.axis)]
        other_support = results.loc[(results["support_m"].eq(1000)) & (results["model_family"].eq("logistic")) & results["outcome"].eq(row.outcome) & results["axis"].eq(row.axis)].iloc[0]
        same_family_count = int((np.sign(same_support["estimate_per_sd"]) == direction).sum())
        support_consistent = bool(np.sign(other_support.estimate_per_sd) == direction)
        stable = bool(row.supported_bh and same_family_count >= 2 and support_consistent)
        stability_rows.append(
            {
                "outcome": row.outcome,
                "axis": row.axis,
                "primary_300m_estimate": row.estimate_per_sd,
                "primary_300m_ci_low": row.ci_low,
                "primary_300m_ci_high": row.ci_high,
                "primary_300m_q_value": row.q_value,
                "same_direction_family_count_300m": same_family_count,
                "estimate_1000m_logistic": other_support.estimate_per_sd,
                "support_direction_consistent": support_consistent,
                "stable_condition_supported": stable,
            }
        )
    return results, pd.DataFrame(stability_rows)


def leave_overlap_cluster_out_audit(
    analysis: pd.DataFrame,
    associations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit primary association sensitivity without treating buffers as independent."""
    helpers = load_old_case_helpers()
    primary = analysis.loc[
        analysis["support_m"].eq(300)
        & analysis["model_family"].eq("logistic")
    ].copy()
    reference = associations.loc[
        associations["support_m"].eq(300)
        & associations["model_family"].eq("logistic")
    ].set_index(["outcome", "axis"])
    rows: list[dict[str, object]] = []
    for outcome in OUTCOMES:
        outcome_data = primary
        if outcome == "trajectory_false_negative":
            outcome_data = primary.loc[primary["event"].eq(1)]
        elif outcome == "trajectory_false_positive":
            outcome_data = primary.loc[primary["event"].eq(0)]
        for axis in AXES:
            reference_estimate = float(reference.loc[(outcome, axis), "estimate_per_sd"])
            for cluster in sorted(outcome_data["overlap_cluster"].dropna().unique()):
                reduced = outcome_data.loc[
                    ~outcome_data["overlap_cluster"].eq(cluster)
                ].rename(columns={"overlap_cluster": "cell_id"})
                estimate = helpers.two_way_cluster_ols(reduced, outcome, axis)
                rows.append(
                    {
                        "outcome": outcome,
                        "axis": axis,
                        "left_out_overlap_cluster": cluster,
                        "reference_estimate_per_sd": reference_estimate,
                        "estimate_per_sd": estimate["estimate_per_sd"],
                        "absolute_change": abs(
                            estimate["estimate_per_sd"] - reference_estimate
                        ),
                        "same_direction_as_reference": bool(
                            np.sign(estimate["estimate_per_sd"])
                            == np.sign(reference_estimate)
                        ),
                        "n": estimate["n"],
                        "remaining_overlap_clusters": estimate["cell_clusters"],
                        "years": estimate["years"],
                    }
                )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["outcome", "axis"], as_index=False)
        .agg(
            reference_estimate_per_sd=("reference_estimate_per_sd", "first"),
            clusters_left_out=("left_out_overlap_cluster", "nunique"),
            minimum_estimate=("estimate_per_sd", "min"),
            maximum_estimate=("estimate_per_sd", "max"),
            maximum_absolute_change=("absolute_change", "max"),
            same_direction_fraction=("same_direction_as_reference", "mean"),
        )
    )
    return detail, summary


def make_figures(
    output: Path,
    increments: pd.DataFrame,
    associations: pd.DataFrame,
    forecast_start: int,
    forecast_end: int,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    palette = {300: "#2F5D7C", 1000: "#D49A32"}
    markers = {300: "o", 1000: "s"}
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharey=True)
    for axis, comparison, title in [
        (axes[0], "trajectory_minus_current", "Trajectory minus current"),
        (axes[1], "mur_minus_trajectory", "MUR 1-km minus trajectory"),
    ]:
        part = increments.loc[increments["comparison"].eq(comparison)]
        for support_m in [300, 1000]:
            group = part.loc[part["support_m"].eq(support_m)].set_index("model_family").loc[list(FAMILY_LABELS)].reset_index()
            y = np.arange(len(group)) + (-0.08 if support_m == 300 else 0.08)
            estimate = group["estimate"].to_numpy()
            axis.errorbar(estimate, y, xerr=np.vstack([estimate - group["ci_low"], group["ci_high"] - estimate]), fmt=markers[support_m], color=palette[support_m], capsize=3, label=f"{support_m} m")
        axis.axvline(0, color="#33383C", linewidth=0.8, linestyle="--")
        axis.set_yticks(np.arange(len(FAMILY_LABELS)), [FAMILY_LABELS[value] for value in FAMILY_LABELS])
        axis.set_title(title, loc="left")
        axis.set_xlabel("Paired macro within-year AP-lift difference")
    axes[0].legend(frameon=False)
    figure.suptitle(
        f"Local-support information-block increments, {forecast_start}–{forecast_end}",
        x=0.02,
        ha="left",
    )
    figure.savefig(output / "figure_01_local_support_increments.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    primary = associations.loc[associations["model_family"].eq("logistic") & associations["outcome"].isin(["trajectory_rank_quality_pp", "trajectory_false_negative", "mur_signed_rank_gain_pp"])].copy()
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    for axis_plot, outcome in zip(axes, ["trajectory_rank_quality_pp", "trajectory_false_negative", "mur_signed_rank_gain_pp"], strict=True):
        for support_m in [300, 1000]:
            group = primary.loc[(primary["support_m"].eq(support_m)) & primary["outcome"].eq(outcome)].set_index("axis").loc[AXES].reset_index()
            scale = 100 if outcome == "trajectory_false_negative" else 1
            y = np.arange(len(group)) + (-0.08 if support_m == 300 else 0.08)
            estimate = group["estimate_per_sd"].to_numpy() * scale
            low = group["ci_low"].to_numpy() * scale
            high = group["ci_high"].to_numpy() * scale
            axis_plot.errorbar(estimate, y, xerr=np.vstack([estimate - low, high - estimate]), fmt=markers[support_m], color=palette[support_m], capsize=3, label=f"{support_m} m")
        axis_plot.axvline(0, color="#33383C", linewidth=0.8, linestyle="--")
        axis_plot.set_yticks(np.arange(len(AXES)), [AXIS_LABELS[value] for value in AXES])
        axis_plot.invert_yaxis()
        axis_plot.set_title(OUTCOME_LABELS[outcome], loc="left", fontsize=10)
        axis_plot.set_xlabel("Change per 1 SD" + (" (pp)" if outcome == "trajectory_false_negative" else ""))
    axes[0].legend(frameon=False)
    figure.suptitle("Ecological-axis associations across local supports", x=0.02, ha="left")
    figure.savefig(output / "figure_02_ecological_associations_by_support.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed run: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "command.txt").write_text(
        shlex.join(sys.argv) + "\n", encoding="utf-8"
    )
    started = time.time()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    panel = pd.read_csv(args.panel)
    field = pd.read_csv(args.field)
    mur = pd.read_csv(args.mur)
    if panel.duplicated(["support_m", "site_id", "year"]).any():
        raise ValueError("Local panel keys are not unique")
    if field.duplicated(["site_id", "year"]).any():
        raise ValueError("Field site-year keys are not unique")
    if mur.duplicated(["site_id", "year"]).any():
        raise ValueError("MUR site-year keys are not unique")
    merged = panel.merge(mur[["site_id", "year", *MUR]], on=["site_id", "year"], how="left", validate="many_to_one")
    feature_complete = merged[MUR].notna().all(axis=1)
    train_start = int(config["forecast"]["train_start_year"])
    analysis_rows = merged.loc[
        merged["eligible"]
        & merged["year"].between(train_start, int(config["forecast"]["last_forecast_year"]))
        & feature_complete
    ].copy()
    predictions, fold_audit = expanding_predictions(analysis_rows, config)
    ranked = add_ranks(predictions, float(config["forecast"]["monitoring_budget_fraction"]))
    annual, performance, increments = performance_tables(ranked, config)
    diagnostics = build_diagnostics(ranked)
    ecology = diagnostics.merge(field[["site_id", "year", *AXES, "n_transects"]], on=["site_id", "year"], how="inner", validate="many_to_one")
    associations, stability = ecological_associations(ecology, float(config["interpretation_rules"]["false_discovery_rate"]))
    leave_cluster_detail, leave_cluster_summary = leave_overlap_cluster_out_audit(
        ecology, associations
    )
    outputs = {
        "analysis_rows.csv": analysis_rows,
        "fold_audit.csv": fold_audit,
        "predictions.csv": predictions,
        "ranked_predictions.csv": ranked,
        "annual_metrics.csv": annual,
        "performance_summary.csv": performance,
        "increment_summary.csv": increments,
        "prediction_diagnostics.csv": diagnostics,
        "ecology_matched_diagnostics.csv": ecology,
        "ecological_axis_associations.csv": associations,
        "condition_stability_summary.csv": stability,
        "leave_overlap_cluster_out_associations.csv": leave_cluster_detail,
        "leave_overlap_cluster_out_summary.csv": leave_cluster_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    make_figures(
        args.output_dir,
        increments,
        associations,
        int(config["forecast"]["first_forecast_year"]),
        int(config["forecast"]["last_forecast_year"]),
    )
    coverage = []
    for support_m, group in ecology.groupby("support_m"):
        unique_site_years = group[["site_id", "year", "event"]].drop_duplicates()
        coverage.append({"support_m": int(support_m), "matched_rows": len(unique_site_years), "matched_sites": int(group["site_id"].nunique()), "overlap_clusters": int(group["overlap_cluster"].nunique()), "years": int(group["year"].nunique()), "events": int(unique_site_years["event"].sum())})
    coverage_frame = pd.DataFrame(coverage)
    coverage_frame.to_csv(args.output_dir / "ecology_coverage.csv", index=False)
    stable_count = int(stability["stable_condition_supported"].sum())
    logistic_mur = increments.loc[(increments["model_family"].eq("logistic")) & increments["comparison"].eq("mur_minus_trajectory")]
    lines = [
        "# California 300 m–1 km 생태적 오류진단 결과",
        "",
        "## 결론",
        "",
        f"300 m와 1 km에서 사전 고정된 안정성 기준을 통과한 생태조건은 **{stable_count}개**였다.",
        f"평가는 완전한 MUR 학습연도 4개를 확보한 {int(config['forecast']['first_forecast_year'])}–{int(config['forecast']['last_forecast_year'])}년에 수행했다.",
        "이 결과는 California 현장부분표본에서 원격자료 모형의 오류와 국지 생태축의 관계를 평가한 것이며 전 해안 인과기작으로 일반화하지 않는다.",
        "",
        "## MUR 1 km 증분가치",
        "",
    ]
    for row in logistic_mur.itertuples(index=False):
        lines.append(f"- {int(row.support_m)} m: Logistic MUR−trajectory AP-lift 증분 {row.estimate:+.4f}, 95% CI [{row.ci_low:+.4f}, {row.ci_high:+.4f}]")
    lines.extend([
        "",
        "## 해석 제한",
        "",
        "- MUR는 표층수온이며 현장 해저수온이 아니다.",
        "- 현장조사 위치는 비확률 표본이고 California에 한정된다.",
        "- 겹치는 site buffer는 overlap cluster로 군집화하여 추론했으며 독립 공간복제로 세지 않는다.",
        "- 음성결과는 성게·서식처·수온의 생태적 무관성을 의미하지 않는다.",
    ])
    (args.output_dir / "results_summary_ko.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    quality = pd.DataFrame([
        {"check": "analysis_keys_unique", "value": int(analysis_rows.duplicated(["support_m", "site_id", "year"]).sum()), "passed": not analysis_rows.duplicated(["support_m", "site_id", "year"]).any()},
        {"check": "forward_only_folds", "value": int((fold_audit["train_end"] >= fold_audit["year"]).sum()), "passed": bool((fold_audit["train_end"] < fold_audit["year"]).all())},
        {"check": "minimum_complete_training_years", "value": int((fold_audit["train_end"] - fold_audit["train_start"] + 1).min()), "passed": bool(((fold_audit["train_end"] - fold_audit["train_start"] + 1) >= int(config["forecast"]["minimum_complete_training_years"])).all())},
        {"check": "prediction_keys_unique", "value": int(predictions.duplicated(["support_m", "site_id", "year", "model_family", "feature_set"]).sum()), "passed": not predictions.duplicated(["support_m", "site_id", "year", "model_family", "feature_set"]).any()},
        {"check": "prediction_scores_bounded", "value": int((~predictions["score"].between(0, 1)).sum()), "passed": bool(predictions["score"].between(0, 1).all())},
        {"check": "both_supports_present", "value": int(ecology["support_m"].nunique()), "passed": int(ecology["support_m"].nunique()) == 2},
        {"check": "overlap_clusters_fewer_than_sites", "value": int((coverage_frame["overlap_clusters"] < coverage_frame["matched_sites"]).sum()), "passed": bool((coverage_frame["overlap_clusters"] < coverage_frame["matched_sites"]).all())},
        {"check": "leave_overlap_cluster_out_audit_complete", "value": int(len(leave_cluster_summary)), "passed": len(leave_cluster_summary) == len(OUTCOMES) * len(AXES)},
    ])
    quality.to_csv(args.output_dir / "quality_checks.csv", index=False)
    if not quality["passed"].all():
        raise AssertionError(quality.to_string(index=False))
    manifest = {
        "status": "complete",
        "protocol": config["protocol"],
        "forecast": config["forecast"],
        "runtime_seconds": round(time.time() - started, 2),
        "git_sha_before_run": git_sha(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": __import__("xgboost").__version__,
        "stable_condition_count": stable_count,
        "analysis_rows": int(len(analysis_rows)),
        "analysis_sites": int(analysis_rows["site_id"].nunique()),
        "effective_analysis_year_range": [
            int(analysis_rows["year"].min()),
            int(analysis_rows["year"].max()),
        ],
        "coverage": coverage,
        "inputs": {"config": {"path": str(args.config), "sha256": sha256(args.config)}, "panel": {"path": str(args.panel), "sha256": sha256(args.panel)}, "field": {"path": str(args.field), "sha256": sha256(args.field)}, "mur": {"path": str(args.mur), "sha256": sha256(args.mur)}},
        "output_sha256": {
            "command.txt": sha256(args.output_dir / "command.txt"),
            **{name: sha256(args.output_dir / name) for name in outputs},
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stable_condition_count": stable_count, "coverage": coverage}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
