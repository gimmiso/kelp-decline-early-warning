"""Run a pooled, penalized hierarchical-additive sensitivity on California v3."""

from __future__ import annotations

import argparse
import hashlib
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
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


TRAJECTORY = [
    "canopy_lag1", "canopy_lag2", "canopy_2yr_change", "canopy_3yr_change", "canopy_3yr_mean",
    "canopy_3yr_std", "canopy_3yr_slope", "canopy_drop_from_3yr_max",
]
BASE = ["relative_canopy", *TRAJECTORY]
FIELD = [
    "field_kelp_current_log1p", "purple_urchin_log1p", "purple_threshold_fraction",
    "field_prior_kelp_mean_log1p", "field_prior_kelp_cv", "depth_m", "rock_probability", "log_vrm",
    "active_grazing_proxy",
]
ENVIRONMENT = [
    "paper_temperature", "paper_nitrate", "paper_wave", "paper_orbital", "paper_npp", "depth_m", "rock_probability", "log_vrm",
]
REGIME = [
    "post_mhw", "grazer_era", "post_mhw_x_purple", "grazer_era_x_purple", "post_mhw_x_temperature",
    "grazer_era_x_temperature", "post_mhw_x_active_grazing", "grazer_era_x_active_grazing",
]
FULL = list(dict.fromkeys([*FIELD, *ENVIRONMENT, "paper_spores"]))
FEATURES = {
    "trajectory": BASE,
    "trajectory_plus_field_state": [*BASE, *FIELD],
    "trajectory_plus_paper_environment": [*BASE, *ENVIRONMENT],
    "trajectory_plus_paper_full": [*BASE, *FULL],
    "trajectory_plus_full_regime": [*BASE, *FULL, *REGIME],
}
SMOOTH = {
    "relative_canopy", "field_kelp_current_log1p", "purple_urchin_log1p", "active_grazing_proxy",
    "paper_temperature", "paper_nitrate", "paper_wave", "paper_orbital", "paper_npp", "paper_spores",
}
COMPARISONS = [
    ("field_state_minus_trajectory", "trajectory_plus_field_state", "trajectory"),
    ("paper_environment_minus_trajectory", "trajectory_plus_paper_environment", "trajectory"),
    ("paper_full_minus_trajectory", "trajectory_plus_paper_full", "trajectory"),
    ("paper_full_minus_environment", "trajectory_plus_paper_full", "trajectory_plus_paper_environment"),
    ("regime_minus_paper_full", "trajectory_plus_full_regime", "trajectory_plus_paper_full"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--analysis-rows", type=Path, required=True)
    parser.add_argument("--old-predictions", type=Path, required=True)
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


def contextual_standardize(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_out = train.copy()
    test_out = test.copy()
    for feature in features:
        train_out[feature] = pd.to_numeric(train_out[feature], errors="coerce").astype(float)
        test_out[feature] = pd.to_numeric(test_out[feature], errors="coerce").astype(float)
    rows = []
    for context in sorted(train["context"].unique()):
        train_mask = train["context"].eq(context)
        test_mask = test["context"].eq(context)
        for feature in features:
            values = pd.to_numeric(train.loc[train_mask, feature], errors="coerce")
            observed = values.dropna()
            structural_missing = observed.empty
            if structural_missing:
                median, mean, sd = 0.0, 0.0, 1.0
                train_out.loc[train_mask, feature] = 0.0
                test_out.loc[test_mask, feature] = 0.0
            else:
                median = float(observed.median())
                filled = values.fillna(median)
                mean = float(filled.mean())
                sd = float(filled.std(ddof=0))
                if not np.isfinite(sd) or sd < 1e-8:
                    sd = 1.0
                train_out.loc[train_mask, feature] = (values.fillna(median) - mean) / sd
                test_values = pd.to_numeric(test.loc[test_mask, feature], errors="coerce").fillna(median)
                test_out.loc[test_mask, feature] = (test_values - mean) / sd
            rows.append(
                {"context": context, "feature": feature, "median": median, "mean": mean, "sd": sd, "structural_missing": structural_missing, "training_rows": int(train_mask.sum())}
            )
    # A context absent from an early inner-training window is represented by the
    # neutral standardized value rather than borrowing another context's scale.
    train_out[features] = train_out[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    test_out[features] = test_out[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return train_out, test_out, pd.DataFrame(rows)


def model_pipeline(features: list[str], c_value: float, seed: int) -> Pipeline:
    smooth = [feature for feature in features if feature in SMOOTH]
    linear = [feature for feature in features if feature not in smooth]
    transformers = []
    if linear:
        transformers.append(("linear", "passthrough", linear))
    if smooth:
        transformers.append(
            ("smooth", Pipeline([("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False, extrapolation="linear")), ("scale", StandardScaler())]), smooth)
        )
    transformers.extend(
        [
            ("context", OneHotEncoder(handle_unknown="ignore"), ["context"]),
            ("region", OneHotEncoder(handle_unknown="ignore"), ["region_group"]),
            ("site", OneHotEncoder(handle_unknown="ignore"), ["site_id"]),
        ]
    )
    return Pipeline(
        [
            ("transform", ColumnTransformer(transformers, remainder="drop")),
            ("model", LogisticRegression(C=c_value, solver="lbfgs", max_iter=3000, random_state=seed)),
        ]
    )


def macro_ap_lift(frame: pd.DataFrame) -> float:
    values = []
    for _, group in frame.groupby("year", sort=True):
        if group["event"].nunique() < 2:
            continue
        values.append(average_precision_score(group["event"], group["score"]) - group["event"].mean())
    return float(np.mean(values)) if values else np.nan


def tune_and_fit(train: pd.DataFrame, test: pd.DataFrame, features: list[str], config: dict[str, object], seed: int) -> tuple[np.ndarray, float, pd.DataFrame, pd.DataFrame]:
    california = config["california"]
    years = np.array(sorted(train["year"].unique()))
    validation_years = years[-int(california["inner_validation_years"]):]
    inner_train = train.loc[~train["year"].isin(validation_years)].copy()
    inner_validation = train.loc[train["year"].isin(validation_years)].copy()
    scaled_train, scaled_validation, _ = contextual_standardize(inner_train, inner_validation, features)
    columns = [*features, "context", "region_group", "site_id"]
    rows = []
    for c_value in california["c_grid"]:
        model = model_pipeline(features, float(c_value), seed)
        model.fit(scaled_train[columns], scaled_train["event"])
        scored = scaled_validation[["year", "event"]].copy()
        scored["score"] = model.predict_proba(scaled_validation[columns])[:, 1]
        rows.append({"C": float(c_value), "inner_macro_ap_lift": macro_ap_lift(scored), "inner_rows": len(scored)})
    tuning = pd.DataFrame(rows).sort_values(["inner_macro_ap_lift", "C"], ascending=[False, True]).reset_index(drop=True)
    chosen = float(tuning.iloc[0]["C"])
    scaled_train, scaled_test, scaling = contextual_standardize(train, test, features)
    model = model_pipeline(features, chosen, seed)
    model.fit(scaled_train[columns], scaled_train["event"])
    return model.predict_proba(scaled_test[columns])[:, 1], chosen, tuning, scaling


def expanding_predictions(data: pd.DataFrame, config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    california = config["california"]
    seed = int(config["evaluation"]["seed"])
    predictions = []
    audits = []
    tuning_rows = []
    scaling_rows = []
    for support_m, support in data.groupby("support_m", sort=True):
        for year in range(int(california["first_forecast_year"]), int(california["last_forecast_year"]) + 1):
            train = support.loc[support["year"].between(int(california["train_start_year"]), year - 1)].copy()
            test = support.loc[support["year"].eq(year)].copy()
            if train["event"].nunique() < 2 or test.empty:
                raise ValueError(f"Invalid pooled California fold {support_m} {year}")
            for feature_set, features in FEATURES.items():
                score, chosen, tuning, scaling = tune_and_fit(train, test, features, config, seed + year + int(support_m))
                output = test[["site_id", "support_m", "overlap_cluster", "context", "region_group", "year", "event"]].copy()
                output["score"] = score
                output["feature_set"] = feature_set
                output["model_family"] = "pooled_hierarchical_additive"
                predictions.append(output)
                audits.append(
                    {"support_m": int(support_m), "year": year, "feature_set": feature_set, "train_start": int(train["year"].min()), "train_end": int(train["year"].max()),
                     "n_train": len(train), "events_train": int(train["event"].sum()), "n_test": len(test), "events_test": int(test["event"].sum()), "chosen_C": chosen}
                )
                tuning["support_m"] = int(support_m); tuning["year"] = year; tuning["feature_set"] = feature_set
                tuning_rows.append(tuning)
                scaling["support_m"] = int(support_m); scaling["year"] = year; scaling["feature_set"] = feature_set
                scaling_rows.append(scaling)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(audits), pd.concat(tuning_rows, ignore_index=True), pd.concat(scaling_rows, ignore_index=True)


def calibration_metrics(frame: pd.DataFrame) -> dict[str, float]:
    clipped = np.clip(frame["score"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000)
    model.fit(logits, frame["event"])
    return {"brier": float(brier_score_loss(frame["event"], frame["score"])), "log_loss": float(log_loss(frame["event"], frame["score"], labels=[0, 1])),
            "calibration_intercept": float(model.intercept_[0]), "calibration_slope": float(model.coef_[0, 0])}


def bootstrap_mean(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float); values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed); draws = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]).astype(float))


def summarize(predictions: pd.DataFrame, config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annual_rows = []
    for keys, group in predictions.groupby(["support_m", "feature_set", "year"], sort=True):
        ap = average_precision_score(group["event"], group["score"]) if group["event"].nunique() == 2 else np.nan
        annual_rows.append({"support_m": int(keys[0]), "feature_set": keys[1], "year": int(keys[2]), "n": len(group), "events": int(group["event"].sum()),
                            "prevalence": float(group["event"].mean()), "ap": ap, "ap_lift": ap - group["event"].mean() if np.isfinite(ap) else np.nan})
    annual = pd.DataFrame(annual_rows)
    reps = int(config["evaluation"]["bootstrap_replicates"]); seed = int(config["evaluation"]["seed"])
    summary_rows = []
    for keys, group in predictions.groupby(["support_m", "feature_set"], sort=True):
        scoped = annual.loc[annual["support_m"].eq(keys[0]) & annual["feature_set"].eq(keys[1])]
        values = scoped["ap_lift"].dropna().to_numpy(); low, high = bootstrap_mean(values, reps, seed + int(keys[0]) + sum(map(ord, keys[1])))
        summary_rows.append({"support_m": int(keys[0]), "feature_set": keys[1], "n": len(group), "events": int(group["event"].sum()),
                             "macro_within_year_ap_lift": float(values.mean()), "ci_low": low, "ci_high": high, **calibration_metrics(group)})
    increments = []
    sensitivity = []
    for support_m, group in annual.groupby("support_m", sort=True):
        pivot = group.pivot(index="year", columns="feature_set", values="ap_lift")
        for label, left, right in COMPARISONS:
            difference = (pivot[left] - pivot[right]).dropna(); low, high = bootstrap_mean(difference.to_numpy(), reps, seed + int(support_m) + sum(map(ord, label)))
            increments.append({"support_m": int(support_m), "comparison": label, "left_feature_set": left, "right_feature_set": right,
                               "estimate": float(difference.mean()), "ci_low": low, "ci_high": high, "years": len(difference)})
            loo = [float(difference.drop(year).mean()) for year in difference.index]
            sensitivity.append({"support_m": int(support_m), "comparison": label, "exclude_2021": float(difference.drop(2021, errors="ignore").mean()),
                                "leave_one_year_min": min(loo), "leave_one_year_max": max(loo), "all_leave_one_year_positive": all(value > 0 for value in loo)})
    return annual, pd.DataFrame(summary_rows), pd.DataFrame(increments), pd.DataFrame(sensitivity)


def budget_summary(predictions: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["support_m", "feature_set"], sort=True):
        for fraction in config["evaluation"]["budget_fractions"]:
            selected = []
            for _, year_data in group.groupby("year", sort=True):
                count = max(1, int(math.ceil(len(year_data) * float(fraction))))
                selected.append(year_data.sort_values(["score", "site_id"], ascending=[False, True]).head(count))
            chosen = pd.concat(selected, ignore_index=True)
            rows.append({"support_m": int(keys[0]), "feature_set": keys[1], "budget_fraction": float(fraction), "selected": len(chosen),
                         "true_positives": int(chosen["event"].sum()), "total_events": int(group["event"].sum()),
                         "micro_recall": float(chosen["event"].sum() / group["event"].sum()), "micro_precision": float(chosen["event"].mean())})
    return pd.DataFrame(rows)


def compare_old_new(new: pd.DataFrame, old_path: Path) -> pd.DataFrame:
    old = pd.read_csv(old_path)
    old = old.loc[old["model_family"].eq("logistic_spline")]
    keys = ["site_id", "support_m", "context", "year", "event", "feature_set"]
    joined = new[keys + ["score"]].merge(old[keys + ["score"]], on=keys, suffixes=("_new", "_old"), validate="one_to_one")
    rows = []
    for (support_m, feature_set, year), group in joined.groupby(["support_m", "feature_set", "year"], sort=True):
        if group["event"].nunique() < 2:
            continue
        rows.append({"support_m": int(support_m), "feature_set": feature_set, "year": int(year), "matched_rows": len(group),
                     "new_ap_lift": float(average_precision_score(group["event"], group["score_new"]) - group["event"].mean()),
                     "old_ap_lift": float(average_precision_score(group["event"], group["score_old"]) - group["event"].mean())})
    return pd.DataFrame(rows)


def make_figure(output_dir: Path, increments: pd.DataFrame) -> None:
    part = increments.loc[increments["comparison"].isin(["field_state_minus_trajectory", "paper_environment_minus_trajectory", "paper_full_minus_trajectory"])]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for support_m, group in part.groupby("support_m", sort=True):
        group = group.sort_values("comparison"); y = np.arange(len(group)) + (-0.08 if support_m == 300 else 0.08)
        estimate = group["estimate"].to_numpy()
        axis.errorbar(estimate, y, xerr=np.vstack([estimate - group["ci_low"], group["ci_high"] - estimate]), fmt="o", capsize=3, label=f"{support_m} m")
    axis.axvline(0, linestyle="--", linewidth=0.8, color="#33383C")
    labels = part.loc[part["support_m"].eq(300)].sort_values("comparison")["comparison"]
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xlabel("Macro within-year AP-lift increment")
    axis.set_title("Pooled hierarchical-additive California sensitivity", loc="left")
    axis.legend(frameon=False)
    figure.savefig(output_dir / "figure_01_california_pooled_hierarchical.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")
    started = time.time(); config = json.loads(args.config.read_text(encoding="utf-8")); data = pd.read_csv(args.analysis_rows)
    predictions, fold_audit, tuning, scaling = expanding_predictions(data, config)
    annual, summary, increments, sensitivity = summarize(predictions, config)
    budgets = budget_summary(predictions, config)
    matched = compare_old_new(predictions, args.old_predictions)
    outputs = {"predictions.csv": predictions, "fold_audit.csv": fold_audit, "inner_tuning_audit.csv": tuning, "context_scaling_audit.csv": scaling,
               "annual_metrics.csv": annual, "model_summary.csv": summary, "increment_summary.csv": increments, "year_sensitivity.csv": sensitivity,
               "budget_summary.csv": budgets, "matched_old_new_annual.csv": matched}
    for name, frame in outputs.items(): frame.to_csv(args.output_dir / name, index=False)
    make_figure(args.output_dir, increments)
    primary = increments.loc[increments["support_m"].eq(300) & increments["comparison"].eq("paper_full_minus_trajectory")].iloc[0]
    lines = ["# California pooled hierarchical-additive 결과", "", f"300 m paper-full−trajectory 증분은 {primary.estimate:+.4f}, 95% CI [{primary.ci_low:+.4f}, {primary.ci_high:+.4f}]였다.",
             "", "이 분석은 사후 구조 민감도이며 Giraldo의 동시점 GAM 재현이나 전 해안 운영모형이 아니다."]
    (args.output_dir / "results_summary_ko.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    quality = pd.DataFrame([
        {"check": "prediction_keys_unique", "value": int(predictions.duplicated(["support_m", "site_id", "year", "feature_set"]).sum()), "passed": not predictions.duplicated(["support_m", "site_id", "year", "feature_set"]).any()},
        {"check": "forward_only", "value": int(fold_audit["train_end"].ge(fold_audit["year"]).sum()), "passed": fold_audit["train_end"].lt(fold_audit["year"]).all()},
        {"check": "scores_bounded", "value": int((~predictions["score"].between(0, 1)).sum()), "passed": predictions["score"].between(0, 1).all()},
        {"check": "both_supports", "value": int(predictions["support_m"].nunique()), "passed": predictions["support_m"].nunique() == 2},
        {"check": "matched_old_new_nonempty", "value": len(matched), "passed": len(matched) > 0},
    ])
    quality.to_csv(args.output_dir / "quality_checks.csv", index=False)
    if not quality["passed"].all(): raise AssertionError(quality.to_string(index=False))
    manifest = {"status": "complete", "protocol": config["protocol"], "runtime_seconds": round(time.time() - started, 2), "git_sha_before_run": git_sha(),
                "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                "inputs": {"config": {"path": str(args.config), "sha256": sha256(args.config)}, "analysis_rows": {"path": str(args.analysis_rows), "sha256": sha256(args.analysis_rows)},
                           "old_predictions": {"path": str(args.old_predictions), "sha256": sha256(args.old_predictions)}},
                "primary": primary.to_dict(), "outputs": {name: sha256(args.output_dir / name) for name in [*outputs, "quality_checks.csv", "results_summary_ko.md", "figure_01_california_pooled_hierarchical.png"]}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runtime_seconds": manifest["runtime_seconds"], "primary": primary.to_dict()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
