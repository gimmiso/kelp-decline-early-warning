"""Run post-hoc structure-aware model robustness on the locked 5-km panel."""

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
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRanker
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

warnings.filterwarnings("ignore", message="X does not have valid feature names")


MODEL_ORDER = ["elastic_net", "hierarchical_additive", "lambdamart_ranker"]
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
OISST = [
    "annual_mean_sst_anomaly",
    "annual_max_sst",
    "hot_weeks_p90",
    "winter_mean_sst_anomaly",
    "summer_mean_sst_anomaly",
    "lag1_annual_mean_sst_anomaly",
]
UPWELLING = [
    "winter_cuti_anomaly",
    "spring_cuti_anomaly",
    "winter_beuti_anomaly",
    "spring_beuti_anomaly",
    "upwelling_season_beuti_anomaly",
]
FEATURES = {
    "full_oisst_domain": {
        "current_only": ["relative_canopy"],
        "current_plus_trajectory": ["relative_canopy", *TRAJECTORY],
        "trajectory_plus_oisst": ["relative_canopy", *TRAJECTORY, *OISST],
    },
    "upwelling_supported_domain": {
        "current_only": ["relative_canopy"],
        "current_plus_trajectory": ["relative_canopy", *TRAJECTORY],
        "trajectory_plus_oisst": ["relative_canopy", *TRAJECTORY, *OISST],
        "trajectory_plus_oisst_cuti_beuti": ["relative_canopy", *TRAJECTORY, *OISST, *UPWELLING],
    },
}
COMPARISONS = {
    "full_oisst_domain": [
        ("trajectory_minus_current", "current_plus_trajectory", "current_only"),
        ("oisst_minus_trajectory", "trajectory_plus_oisst", "current_plus_trajectory"),
    ],
    "upwelling_supported_domain": [
        ("trajectory_minus_current", "current_plus_trajectory", "current_only"),
        ("oisst_minus_trajectory", "trajectory_plus_oisst", "current_plus_trajectory"),
        ("cuti_beuti_minus_oisst", "trajectory_plus_oisst_cuti_beuti", "trajectory_plus_oisst"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
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


def support_flag(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().eq("true")


def prepare_data(panel_path: Path, environment_path: Path, config: dict[str, object]) -> dict[str, pd.DataFrame]:
    panel = pd.read_csv(panel_path)
    environment = pd.read_csv(environment_path)
    if panel.duplicated(["cell_id", "year"]).any() or environment.duplicated(["cell_id", "year"]).any():
        raise ValueError("Input cell-year keys must be unique")
    merged = panel.merge(environment, on=["cell_id", "year"], how="left", validate="one_to_one")
    coast = config["coastwide"]
    base = merged.loc[
        merged["year"].between(int(coast["train_start_year"]), int(coast["last_forecast_year"]))
        & merged["relative_canopy"].gt(float(coast["canopy_cutoff"]))
        & merged["relative_drop_next"].notna()
    ].copy()
    base["event"] = base["relative_drop_next"].ge(float(coast["decline_threshold"])).astype(int)
    full = base.dropna(subset=OISST).copy()
    supported = base.loc[support_flag(base["upwelling_supported"])].dropna(subset=[*OISST, *UPWELLING]).copy()
    return {"full_oisst_domain": full, "upwelling_supported_domain": supported}


def elastic_pipeline(features: list[str], c_value: float, l1_ratio: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="saga",
                    C=c_value,
                    l1_ratio=l1_ratio,
                    max_iter=5000,
                    tol=1e-4,
                    random_state=seed,
                ),
            ),
        ]
    )


def hierarchical_pipeline(features: list[str], c_value: float, seed: int, include_cell: bool = True) -> Pipeline:
    smooth = [column for column in features if column == "relative_canopy" or column in OISST or column in UPWELLING]
    linear = [column for column in features if column not in smooth]
    transformers = []
    if linear:
        transformers.append(
            ("linear", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), linear)
        )
    if smooth:
        transformers.append(
            (
                "smooth",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False, extrapolation="linear")),
                        ("scale", StandardScaler()),
                    ]
                ),
                smooth,
            )
        )
    transformers.append(("region", OneHotEncoder(handle_unknown="ignore"), ["region_group"]))
    if include_cell:
        transformers.append(("cell", OneHotEncoder(handle_unknown="ignore"), ["cell_id"]))
    return Pipeline(
        [
            ("transform", ColumnTransformer(transformers, remainder="drop")),
            ("model", LogisticRegression(C=c_value, solver="lbfgs", max_iter=3000, random_state=seed)),
        ]
    )


def ranker_preprocessor(features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), features),
            ("region", OneHotEncoder(handle_unknown="ignore"), ["region_group"]),
        ],
        remainder="drop",
    )


def new_ranker(seed: int) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank",
        n_estimators=160,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )


def rank_percentile(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)


def within_year_rank_percentile(years: pd.Series, values: np.ndarray) -> np.ndarray:
    """Convert raw ranking scores to percentiles within each deployment year."""
    frame = pd.DataFrame({"year": years.to_numpy(), "value": values})
    return frame.groupby("year", sort=False)["value"].rank(method="average", pct=True).to_numpy(dtype=float)


def fit_ranker(train: pd.DataFrame, test: pd.DataFrame, features: list[str], calibration_years: int, seed: int) -> tuple[np.ndarray, dict[str, object]]:
    years = np.array(sorted(train["year"].unique()))
    calibration = years[-calibration_years:]
    calibration_rows = []
    for year in calibration:
        inner_train = train.loc[train["year"].lt(year)].sort_values(["year", "cell_id"]).copy()
        inner_test = train.loc[train["year"].eq(year)].copy()
        if inner_train["event"].nunique() < 2 or inner_test.empty:
            continue
        prep = ranker_preprocessor(features)
        x_train = prep.fit_transform(inner_train[[*features, "region_group"]])
        x_test = prep.transform(inner_test[[*features, "region_group"]])
        groups = inner_train.groupby("year", sort=True).size().to_numpy()
        ranker = new_ranker(seed + int(year))
        ranker.fit(x_train, inner_train["event"], group=groups)
        part = inner_test[["event"]].copy()
        part["rank_percentile"] = rank_percentile(ranker.predict(x_test))
        calibration_rows.append(part)
    calibration_frame = pd.concat(calibration_rows, ignore_index=True)
    if calibration_frame["event"].nunique() < 2:
        raise ValueError("Ranker calibration window has one class")
    platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
    platt.fit(calibration_frame[["rank_percentile"]], calibration_frame["event"])

    ordered = train.sort_values(["year", "cell_id"]).copy()
    prep = ranker_preprocessor(features)
    x_train = prep.fit_transform(ordered[[*features, "region_group"]])
    x_test = prep.transform(test[[*features, "region_group"]])
    groups = ordered.groupby("year", sort=True).size().to_numpy()
    ranker = new_ranker(seed)
    ranker.fit(x_train, ordered["event"], group=groups)
    test_percentile = within_year_rank_percentile(test["year"], ranker.predict(x_test))
    probability = platt.predict_proba(test_percentile.reshape(-1, 1))[:, 1]
    audit = {
        "calibration_start": int(calibration.min()),
        "calibration_end": int(calibration.max()),
        "calibration_rows": int(len(calibration_frame)),
        "platt_intercept": float(platt.intercept_[0]),
        "platt_slope": float(platt.coef_[0, 0]),
    }
    return probability, audit


def macro_ap_lift(frame: pd.DataFrame, score_column: str = "score") -> float:
    values = []
    for _, group in frame.groupby("year", sort=True):
        if group["event"].nunique() < 2:
            continue
        values.append(average_precision_score(group["event"], group[score_column]) - group["event"].mean())
    return float(np.mean(values)) if values else np.nan


def inner_split(train: pd.DataFrame, n_years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = np.array(sorted(train["year"].unique()))
    validation_years = years[-n_years:]
    return train.loc[~train["year"].isin(validation_years)].copy(), train.loc[train["year"].isin(validation_years)].copy()


def tune_regularized(
    train: pd.DataFrame,
    features: list[str],
    model_family: str,
    config: dict[str, object],
    seed: int,
    include_cell: bool = True,
) -> tuple[dict[str, float], pd.DataFrame]:
    coast = config["coastwide"]
    inner_train, inner_validation = inner_split(train, int(coast["inner_validation_years"]))
    candidates = []
    if model_family == "elastic_net":
        for c_value in coast["c_grid"]:
            for l1_ratio in coast["elastic_l1_ratio_grid"]:
                candidates.append({"C": float(c_value), "l1_ratio": float(l1_ratio)})
    else:
        candidates = [{"C": float(value)} for value in coast["c_grid"]]
    rows = []
    for candidate in candidates:
        if model_family == "elastic_net":
            model = elastic_pipeline(features, candidate["C"], candidate["l1_ratio"], seed)
            model.fit(inner_train[features], inner_train["event"])
            score = model.predict_proba(inner_validation[features])[:, 1]
        else:
            model = hierarchical_pipeline(features, candidate["C"], seed, include_cell=include_cell)
            columns = [*features, "region_group", "cell_id"]
            model.fit(inner_train[columns], inner_train["event"])
            score = model.predict_proba(inner_validation[columns])[:, 1]
        scored = inner_validation[["year", "event"]].copy()
        scored["score"] = score
        rows.append({**candidate, "inner_macro_ap_lift": macro_ap_lift(scored), "inner_rows": len(scored)})
    audit = pd.DataFrame(rows).sort_values(["inner_macro_ap_lift", "C"], ascending=[False, True]).reset_index(drop=True)
    chosen = audit.iloc[0].to_dict()
    return {key: float(chosen[key]) for key in candidate}, audit


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_family: str,
    config: dict[str, object],
    seed: int,
    include_cell: bool = True,
) -> tuple[np.ndarray, dict[str, object], pd.DataFrame]:
    if model_family == "lambdamart_ranker":
        score, audit = fit_ranker(train, test, features, int(config["coastwide"]["inner_validation_years"]), seed)
        return score, audit, pd.DataFrame()
    chosen, tuning = tune_regularized(train, features, model_family, config, seed, include_cell=include_cell)
    if model_family == "elastic_net":
        model = elastic_pipeline(features, chosen["C"], chosen["l1_ratio"], seed)
        model.fit(train[features], train["event"])
        score = model.predict_proba(test[features])[:, 1]
    else:
        model = hierarchical_pipeline(features, chosen["C"], seed, include_cell=include_cell)
        columns = [*features, "region_group", "cell_id"]
        model.fit(train[columns], train["event"])
        score = model.predict_proba(test[columns])[:, 1]
    return score, chosen, tuning


def expanding_predictions(domains: dict[str, pd.DataFrame], config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coast = config["coastwide"]
    seed = int(config["evaluation"]["seed"])
    predictions = []
    audits = []
    tuning_rows = []
    for domain, data in domains.items():
        for year in range(int(coast["first_forecast_year"]), int(coast["last_forecast_year"]) + 1):
            train = data.loc[data["year"].between(int(coast["train_start_year"]), year - 1)].copy()
            test = data.loc[data["year"].eq(year)].copy()
            if train["event"].nunique() < 2 or test.empty:
                raise ValueError(f"Invalid fold {domain} {year}")
            for model_family in MODEL_ORDER:
                for feature_set, features in FEATURES[domain].items():
                    score, chosen, tuning = fit_predict(train, test, features, model_family, config, seed + year)
                    output = test[["cell_id", "region_group", "year", "event", "relative_drop_next"]].copy()
                    output["score"] = score
                    output["domain"] = domain
                    output["model_family"] = model_family
                    output["feature_set"] = feature_set
                    predictions.append(output)
                    audits.append(
                        {
                            "domain": domain,
                            "year": year,
                            "model_family": model_family,
                            "feature_set": feature_set,
                            "train_start": int(train["year"].min()),
                            "train_end": int(train["year"].max()),
                            "n_train": len(train),
                            "events_train": int(train["event"].sum()),
                            "n_test": len(test),
                            "events_test": int(test["event"].sum()),
                            "chosen": json.dumps(chosen, sort_keys=True),
                        }
                    )
                    if not tuning.empty:
                        tuning = tuning.copy()
                        tuning["domain"] = domain
                        tuning["year"] = year
                        tuning["model_family"] = model_family
                        tuning["feature_set"] = feature_set
                        tuning_rows.append(tuning)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(audits), pd.concat(tuning_rows, ignore_index=True)


def calibration_metrics(frame: pd.DataFrame) -> dict[str, float]:
    clipped = np.clip(frame["score"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibration = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000)
    calibration.fit(logits, frame["event"])
    bins = pd.qcut(frame["score"], q=min(10, frame["score"].nunique()), duplicates="drop")
    grouped = frame.assign(_bin=bins).groupby("_bin", observed=True).agg(n=("event", "size"), observed=("event", "mean"), predicted=("score", "mean"))
    ece = float(np.average((grouped["observed"] - grouped["predicted"]).abs(), weights=grouped["n"]))
    return {
        "brier": float(brier_score_loss(frame["event"], frame["score"])),
        "log_loss": float(log_loss(frame["event"], frame["score"], labels=[0, 1])),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
        "ece": ece,
    }


def bootstrap_mean(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]).astype(float))


def summarize(predictions: pd.DataFrame, config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annual_rows = []
    for keys, group in predictions.groupby(["domain", "model_family", "feature_set", "year"], sort=True):
        ap = average_precision_score(group["event"], group["score"]) if group["event"].nunique() == 2 else np.nan
        annual_rows.append(
            {
                "domain": keys[0], "model_family": keys[1], "feature_set": keys[2], "year": int(keys[3]),
                "n": len(group), "events": int(group["event"].sum()), "prevalence": float(group["event"].mean()),
                "ap": ap, "ap_lift": ap - group["event"].mean() if np.isfinite(ap) else np.nan,
            }
        )
    annual = pd.DataFrame(annual_rows)
    summary_rows = []
    reps = int(config["evaluation"]["bootstrap_replicates"])
    seed = int(config["evaluation"]["seed"])
    for keys, group in predictions.groupby(["domain", "model_family", "feature_set"], sort=True):
        scoped_annual = annual.loc[annual["domain"].eq(keys[0]) & annual["model_family"].eq(keys[1]) & annual["feature_set"].eq(keys[2])]
        values = scoped_annual["ap_lift"].dropna().to_numpy()
        low, high = bootstrap_mean(values, reps, seed + sum(map(ord, "".join(keys))))
        summary_rows.append(
            {"domain": keys[0], "model_family": keys[1], "feature_set": keys[2], "n": len(group), "events": int(group["event"].sum()),
             "estimable_years": len(values), "macro_within_year_ap_lift": float(values.mean()), "ci_low": low, "ci_high": high,
             **calibration_metrics(group)}
        )
    summary = pd.DataFrame(summary_rows)
    increment_rows = []
    sensitivity_rows = []
    for (domain, family), group in annual.groupby(["domain", "model_family"], sort=True):
        pivot = group.pivot(index="year", columns="feature_set", values="ap_lift")
        for label, left, right in COMPARISONS[domain]:
            difference = (pivot[left] - pivot[right]).dropna()
            low, high = bootstrap_mean(difference.to_numpy(), reps, seed + sum(map(ord, domain + family + label)))
            increment_rows.append(
                {"domain": domain, "model_family": family, "comparison": label, "left_feature_set": left, "right_feature_set": right,
                 "estimate": float(difference.mean()), "ci_low": low, "ci_high": high, "estimable_years": len(difference)}
            )
            loo = {int(year): float(difference.drop(year).mean()) for year in difference.index}
            sensitivity_rows.append(
                {"domain": domain, "model_family": family, "comparison": label,
                 "exclude_2021": float(difference.drop(2021, errors="ignore").mean()),
                 "exclude_2022": float(difference.drop(2022, errors="ignore").mean()),
                 "leave_one_year_min": min(loo.values()), "leave_one_year_max": max(loo.values()),
                 "all_leave_one_year_positive": all(value > 0 for value in loo.values())}
            )
    increments = pd.DataFrame(increment_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)

    budget_rows = []
    fractions = [float(value) for value in config["evaluation"]["budget_fractions"]]
    for keys, group in predictions.groupby(["domain", "model_family", "feature_set"], sort=True):
        for fraction in fractions:
            selected = []
            for _, year_data in group.groupby("year", sort=True):
                count = max(1, int(math.ceil(len(year_data) * fraction)))
                selected.append(year_data.sort_values(["score", "cell_id"], ascending=[False, True]).head(count))
            chosen = pd.concat(selected, ignore_index=True)
            budget_rows.append(
                {"domain": keys[0], "model_family": keys[1], "feature_set": keys[2], "budget_fraction": fraction,
                 "selected": len(chosen), "true_positives": int(chosen["event"].sum()), "total_events": int(group["event"].sum()),
                 "micro_recall": float(chosen["event"].sum() / group["event"].sum()), "micro_precision": float(chosen["event"].mean())}
            )
    return annual, summary, increments, sensitivity, pd.DataFrame(budget_rows)


def spatiotemporal_region_holdout(domains: dict[str, pd.DataFrame], config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    coast = config["coastwide"]
    data = domains["full_oisst_domain"]
    train_end = int(coast["spatiotemporal_holdout_train_end"])
    test_start = int(coast["spatiotemporal_holdout_test_start"])
    test_end = int(coast["spatiotemporal_holdout_test_end"])
    seed = int(config["evaluation"]["seed"])
    rows = []
    audits = []
    for heldout_region in sorted(data["region_group"].unique()):
        train = data.loc[data["year"].le(train_end) & ~data["region_group"].eq(heldout_region)].copy()
        test = data.loc[data["year"].between(test_start, test_end) & data["region_group"].eq(heldout_region)].copy()
        if test.empty or train["event"].nunique() < 2:
            continue
        for model_family in MODEL_ORDER:
            for feature_set, features in FEATURES["full_oisst_domain"].items():
                score, chosen, _ = fit_predict(train, test, features, model_family, config, seed + sum(map(ord, heldout_region)), include_cell=False)
                output = test[["cell_id", "region_group", "year", "event"]].copy()
                output["score"] = score
                output["heldout_region"] = heldout_region
                output["model_family"] = model_family
                output["feature_set"] = feature_set
                rows.append(output)
                audits.append(
                    {
                        "heldout_region": heldout_region,
                        "model_family": model_family,
                        "feature_set": feature_set,
                        "n_train": len(train),
                        "n_test": len(test),
                        "train_year_max": int(train["year"].max()),
                        "test_year_min": int(test["year"].min()),
                        "test_year_max": int(test["year"].max()),
                        "train_regions": json.dumps(sorted(train["region_group"].unique().tolist())),
                        "chosen": json.dumps(chosen, sort_keys=True),
                    }
                )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audits)


def summarize_holdout(predictions: pd.DataFrame, config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    annual_rows = []
    for keys, group in predictions.groupby(["model_family", "feature_set", "year"], sort=True):
        if group["event"].nunique() < 2:
            continue
        ap = average_precision_score(group["event"], group["score"])
        annual_rows.append({"model_family": keys[0], "feature_set": keys[1], "year": int(keys[2]), "ap_lift": float(ap - group["event"].mean())})
    annual = pd.DataFrame(annual_rows)
    reps = int(config["evaluation"]["bootstrap_replicates"])
    seed = int(config["evaluation"]["seed"])
    for family, group in annual.groupby("model_family", sort=True):
        pivot = group.pivot(index="year", columns="feature_set", values="ap_lift")
        for label, left, right in COMPARISONS["full_oisst_domain"]:
            difference = (pivot[left] - pivot[right]).dropna()
            low, high = bootstrap_mean(difference.to_numpy(), reps, seed + sum(map(ord, family + label + "holdout")))
            rows.append({"model_family": family, "comparison": label, "estimate": float(difference.mean()), "ci_low": low, "ci_high": high, "years": len(difference)})
    return annual, pd.DataFrame(rows)


def make_figure(output_dir: Path, increments: pd.DataFrame, budgets: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    primary = increments.loc[increments["domain"].eq("full_oisst_domain")].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, comparison, title in [
        (axes[0], "trajectory_minus_current", "Trajectory minus current"),
        (axes[1], "oisst_minus_trajectory", "OISST minus trajectory"),
    ]:
        part = primary.loc[primary["comparison"].eq(comparison)].set_index("model_family").loc[MODEL_ORDER].reset_index()
        y = np.arange(len(part))
        estimate = part["estimate"].to_numpy()
        axis.errorbar(estimate, y, xerr=np.vstack([estimate - part["ci_low"], part["ci_high"] - estimate]), fmt="o", capsize=3, color="#2F5D7C")
        axis.axvline(0, linestyle="--", linewidth=0.8, color="#33383C")
        axis.set_yticks(y, part["model_family"])
        axis.set_title(title, loc="left")
        axis.set_xlabel("Macro within-year AP-lift increment")
    figure.suptitle("Post-hoc 5-km model-architecture robustness", x=0.02, ha="left")
    figure.savefig(output_dir / "figure_01_model_architecture_increments.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    part = budgets.loc[budgets["domain"].eq("full_oisst_domain") & budgets["feature_set"].eq("trajectory_plus_oisst")]
    figure, axis = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for family, group in part.groupby("model_family", sort=True):
        axis.plot(group["budget_fraction"] * 100, group["micro_recall"] * 100, marker="o", label=family)
    axis.set_xlabel("Monitoring budget (%)")
    axis.set_ylabel("Event recall (%)")
    axis.set_title("Budget–recall after adding OISST", loc="left")
    axis.legend(frameon=False)
    figure.savefig(output_dir / "figure_02_budget_recall.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")
    started = time.time()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    domains = prepare_data(args.panel, args.environment, config)
    predictions, fold_audit, tuning = expanding_predictions(domains, config)
    annual, summary, increments, sensitivity, budgets = summarize(predictions, config)
    holdout_predictions, holdout_audit = spatiotemporal_region_holdout(domains, config)
    holdout_annual, holdout_increments = summarize_holdout(holdout_predictions, config)
    outputs = {
        "predictions.csv": predictions,
        "fold_audit.csv": fold_audit,
        "inner_tuning_audit.csv": tuning,
        "annual_metrics.csv": annual,
        "model_summary.csv": summary,
        "increment_summary.csv": increments,
        "year_exclusion_sensitivity.csv": sensitivity,
        "budget_summary.csv": budgets,
        "spatiotemporal_region_holdout_predictions.csv": holdout_predictions,
        "spatiotemporal_region_holdout_audit.csv": holdout_audit,
        "spatiotemporal_region_holdout_annual.csv": holdout_annual,
        "spatiotemporal_region_holdout_increments.csv": holdout_increments,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    make_figure(args.output_dir, increments, budgets)
    oisst = increments.loc[increments["domain"].eq("full_oisst_domain") & increments["comparison"].eq("oisst_minus_trajectory")]
    lines = ["# 전 해안 5 km 모형구조 확장 결과", "", "## OISST−trajectory AP-lift 증분", ""]
    for row in oisst.itertuples(index=False):
        lines.append(f"- {row.model_family}: {row.estimate:+.4f}, 95% CI [{row.ci_low:+.4f}, {row.ci_high:+.4f}]")
    lines.extend(["", "이 분석은 사후 강건성 분석이며 기존 잠금 주분석을 교체하지 않는다."])
    (args.output_dir / "results_summary_ko.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    quality = pd.DataFrame(
        [
            {"check": "prediction_keys_unique", "value": int(predictions.duplicated(["domain", "cell_id", "year", "model_family", "feature_set"]).sum()), "passed": not predictions.duplicated(["domain", "cell_id", "year", "model_family", "feature_set"]).any()},
            {"check": "forward_only", "value": int(fold_audit["train_end"].ge(fold_audit["year"]).sum()), "passed": fold_audit["train_end"].lt(fold_audit["year"]).all()},
            {"check": "scores_bounded", "value": int((~predictions["score"].between(0, 1)).sum()), "passed": predictions["score"].between(0, 1).all()},
            {"check": "all_models", "value": int(predictions["model_family"].nunique()), "passed": predictions["model_family"].nunique() == len(MODEL_ORDER)},
            {"check": "region_holdout_no_region_overlap", "value": int(len(holdout_audit)), "passed": len(holdout_audit) > 0},
        ]
    )
    quality.to_csv(args.output_dir / "quality_checks.csv", index=False)
    if not quality["passed"].all():
        raise AssertionError(quality.to_string(index=False))
    manifest = {
        "status": "complete", "protocol": config["protocol"], "runtime_seconds": round(time.time() - started, 2), "git_sha_before_run": git_sha(),
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
        "inputs": {"config": {"path": str(args.config), "sha256": sha256(args.config)}, "panel": {"path": str(args.panel), "sha256": sha256(args.panel)}, "environment": {"path": str(args.environment), "sha256": sha256(args.environment)}},
        "domains": {name: {"rows": len(frame), "cells": frame["cell_id"].nunique(), "events": int(frame["event"].sum())} for name, frame in domains.items()},
        "outputs": {name: sha256(args.output_dir / name) for name in [*outputs, "quality_checks.csv", "results_summary_ko.md", "figure_01_model_architecture_increments.png", "figure_02_budget_recall.png"]},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runtime_seconds": manifest["runtime_seconds"], "oisst_increments": oisst[["model_family", "estimate", "ci_low", "ci_high"]].to_dict(orient="records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
