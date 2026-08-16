"""Run a paper-aligned nonlinear audit of the California ecological case study."""

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
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
BASE = ["relative_canopy", *TRAJECTORY]
FIELD_COMMON = [
    "field_kelp_current_log1p",
    "purple_urchin_log1p",
    "purple_threshold_fraction",
    "field_prior_kelp_mean_log1p",
    "field_prior_kelp_cv",
    "depth_m",
    "rock_probability",
    "log_vrm",
]
ENV_COMMON = ["paper_temperature", "paper_nitrate", "paper_wave", "paper_npp", "depth_m", "rock_probability", "log_vrm"]
REGIME = [
    "post_mhw",
    "grazer_era",
    "post_mhw_x_purple",
    "grazer_era_x_purple",
    "post_mhw_x_temperature",
    "grazer_era_x_temperature",
    "post_mhw_x_active_grazing",
    "grazer_era_x_active_grazing",
]
FEATURE_SET_ORDER = [
    "trajectory",
    "trajectory_plus_field_state",
    "trajectory_plus_paper_environment",
    "trajectory_plus_paper_full",
    "trajectory_plus_full_regime",
]
FAMILY_ORDER = ["logistic_linear", "logistic_spline", "random_forest", "xgboost"]
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
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--code-release", type=Path, required=True)
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


def context_sites(code_release: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = pd.read_csv(code_release / "No_survey_years_per_site.csv")
    bull = pd.read_csv(code_release / "bull_kelp_site_region_metadata.csv")
    giant = pd.read_csv(code_release / "giant_kelp_site_region_metadata.csv")
    valid = years.loc[years["preMHW"].ge(3)].copy()
    valid_sites = set(valid["site_campus_unique_ID"])
    north = set(valid.loc[valid["mlpa_region"].eq("NORTH"), "site_campus_unique_ID"]) & set(bull["site_campus_unique_ID"])
    csw = valid_sites & set(giant.loc[giant["region"].eq("central-southwest"), "site_campus_unique_ID"])
    southeast = valid_sites & set(giant.loc[giant["region"].eq("southeast"), "site_campus_unique_ID"])
    rows = []
    for context, sites in [
        ("bull_north", north),
        ("giant_central_southwest", csw),
        ("giant_southeast", southeast),
    ]:
        rows.extend({"site_id": site, "context": context} for site in sorted(sites))
    assignment = pd.DataFrame(rows)
    if assignment["site_id"].duplicated().any():
        raise ValueError("Context assignment must be non-overlapping")
    audit = pd.DataFrame(
        [
            {"stage": "official_year_metadata", "context": "all", "sites": years["site_campus_unique_ID"].nunique()},
            {"stage": "pre_mhw_at_least_3", "context": "all", "sites": len(valid_sites)},
            *[
                {"stage": "remote_label_assignment", "context": context, "sites": int(group["site_id"].nunique())}
                for context, group in assignment.groupby("context", sort=True)
            ],
        ]
    )
    return assignment, audit


def collapse_field(field_path: Path, assignment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(field_path, low_memory=False).rename(
        columns={"site_campus_unique_ID": "site_id", "survey_year": "year"}
    )
    raw = raw.merge(assignment, on="site_id", how="inner", validate="many_to_one")
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    raw["transect_key"] = raw["transect"].astype(str)
    duplicate_mask = raw.duplicated(["site_id", "year", "zone", "transect_key"], keep=False)
    duplicate_audit = (
        raw.loc[duplicate_mask]
        .groupby(["context", "site_id", "year", "zone", "transect_key"], as_index=False)
        .agg(raw_rows=("site_id", "size"), spores_min=("prev_year_spores", "min"), spores_max=("prev_year_spores", "max"))
    )
    duplicate_audit["spores_disagree"] = duplicate_audit["spores_min"].ne(duplicate_audit["spores_max"])

    numeric = raw.select_dtypes(include=[np.number]).columns.tolist()
    keep_numeric = [column for column in numeric if column != "year"]
    transects = (
        raw.groupby(["site_id", "context", "year", "zone", "transect_key"], as_index=False)[keep_numeric]
        .median(numeric_only=True)
    )
    transects["kelp_target"] = np.where(
        transects["context"].eq("bull_north"), transects["den_NERLUE"], transects["den_MACSTIPES"]
    )
    transects["purple_threshold"] = np.where(
        transects["context"].eq("bull_north"), np.expm1(2.5), np.expm1(5.0)
    )
    transects["purple_above_threshold"] = transects["den_STRPURAD"].gt(transects["purple_threshold"]).astype(float)
    giant_per_m2 = transects["den_MACSTIPES"].clip(lower=0) / 60.0
    transects["active_grazing_proxy"] = (
        -0.6487 * (1.0 - np.exp(-5.1438 * giant_per_m2)) + 0.7169
    ).clip(0, 1)
    transects.loc[transects["context"].eq("bull_north"), "active_grazing_proxy"] = np.nan
    observed_depth = transects["depth_mean"]
    mapped_depth = -transects["mean_depth"]
    transects["depth_mapped_fallback"] = observed_depth.isna().astype(int)
    transects["depth_m"] = observed_depth.fillna(mapped_depth)

    context_map = {
        "bull_north": {
            "temperature": "Mean_Monthly_Upwelling_Temp",
            "nitrate": "Min_Monthly_Nitrate",
            "wave": "wh_mean",
            "orbital": "UBR_Max",
        },
        "giant_central_southwest": {
            "temperature": "Days_21C",
            "nitrate": "Days_4N",
            "wave": "wh_mean",
            "orbital": "UBR_Max",
        },
        "giant_southeast": {
            "temperature": "Days_21C",
            "nitrate": "Max_Monthly_Anomaly_Summer_Nitrate",
            "wave": "wh_mean",
            "orbital": None,
        },
    }
    mapped = []
    for context, group in transects.groupby("context", sort=True):
        columns = context_map[context]
        part = group.copy()
        part["paper_temperature"] = part[columns["temperature"]]
        part["paper_nitrate"] = part[columns["nitrate"]]
        part["paper_wave"] = part[columns["wave"]]
        part["paper_orbital"] = part[columns["orbital"]] if columns["orbital"] else np.nan
        part["paper_npp"] = part["Mean_Monthly_NPP"]
        part["paper_spores"] = np.where(part["context"].eq("bull_north"), np.nan, part["prev_year_spores"])
        mapped.append(part)
    transects = pd.concat(mapped, ignore_index=True)
    aggregation = {
        "latitude": "median",
        "longitude": "median",
        "kelp_target": "mean",
        "den_STRPURAD": "median",
        "purple_above_threshold": "mean",
        "active_grazing_proxy": "mean",
        "depth_m": "median",
        "depth_mapped_fallback": "max",
        "mean_prob_of_rock": "median",
        "mean_vrm": "median",
        "paper_temperature": "median",
        "paper_nitrate": "median",
        "paper_wave": "median",
        "paper_orbital": "median",
        "paper_npp": "median",
        "paper_spores": "median",
        "transect_key": "size",
    }
    site_year = transects.groupby(["site_id", "context", "year"], as_index=False).agg(aggregation).rename(
        columns={
            "den_STRPURAD": "purple_urchin_median",
            "purple_above_threshold": "purple_threshold_fraction",
            "mean_prob_of_rock": "rock_probability",
            "transect_key": "n_unique_transects",
        }
    )
    site_year["field_kelp_current_log1p"] = np.log1p(site_year["kelp_target"].clip(lower=0))
    site_year["purple_urchin_log1p"] = np.log1p(site_year["purple_urchin_median"].clip(lower=0))
    site_year["log_vrm"] = np.log1p(site_year["mean_vrm"].clip(lower=0))
    for column in ["paper_wave", "paper_orbital", "paper_npp", "paper_spores"]:
        site_year[column] = np.log1p(site_year[column].clip(lower=0))
    for column in ["paper_temperature", "paper_nitrate"]:
        if site_year[column].min(skipna=True) >= 0:
            site_year[column] = np.log1p(site_year[column])

    site_year = site_year.sort_values(["site_id", "year"]).reset_index(drop=True)
    prior_mean = []
    prior_cv = []
    for _, group in site_year.groupby("site_id", sort=False):
        values = group["kelp_target"].astype(float)
        prior = values.shift(1)
        mean = prior.expanding(min_periods=1).mean()
        sd = prior.expanding(min_periods=2).std(ddof=1)
        prior_mean.extend(np.log1p(mean.clip(lower=0)).tolist())
        prior_cv.extend((sd / mean.replace(0, np.nan)).tolist())
    site_year["field_prior_kelp_mean_log1p"] = prior_mean
    site_year["field_prior_kelp_cv"] = prior_cv
    return site_year, duplicate_audit


def add_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["post_mhw"] = output["year"].ge(2014).astype(int)
    output["grazer_era"] = output["year"].ge(2019).astype(int)
    output["post_mhw_x_purple"] = output["post_mhw"] * output["purple_urchin_log1p"]
    output["grazer_era_x_purple"] = output["grazer_era"] * output["purple_urchin_log1p"]
    output["post_mhw_x_temperature"] = output["post_mhw"] * output["paper_temperature"]
    output["grazer_era_x_temperature"] = output["grazer_era"] * output["paper_temperature"]
    active = output["active_grazing_proxy"].fillna(0)
    output["post_mhw_x_active_grazing"] = output["post_mhw"] * active
    output["grazer_era_x_active_grazing"] = output["grazer_era"] * active
    return output


def features_for(context: str, feature_set: str) -> list[str]:
    field = list(FIELD_COMMON)
    if context != "bull_north":
        field.append("active_grazing_proxy")
    environment = list(ENV_COMMON)
    if context != "giant_southeast":
        environment.append("paper_orbital")
    full = list(dict.fromkeys([*field, *environment]))
    if context != "bull_north":
        full.append("paper_spores")
    mapping = {
        "trajectory": BASE,
        "trajectory_plus_field_state": [*BASE, *field],
        "trajectory_plus_paper_environment": [*BASE, *environment],
        "trajectory_plus_paper_full": [*BASE, *full],
        "trajectory_plus_full_regime": [*BASE, *full, *REGIME],
    }
    return mapping[feature_set]


def estimator(family: str, features: list[str], seed: int) -> Pipeline:
    if family == "logistic_linear":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=5000, random_state=seed)),
            ]
        )
    if family == "logistic_spline":
        base = [column for column in features if column in BASE or column in ["post_mhw", "grazer_era"]]
        nonlinear = [column for column in features if column not in base]
        transformers = []
        if base:
            transformers.append(
                (
                    "linear",
                    Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]),
                    base,
                )
            )
        if nonlinear:
            transformers.append(
                (
                    "spline",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                            ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False, extrapolation="linear")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    nonlinear,
                )
            )
        return Pipeline(
            [
                ("transform", ColumnTransformer(transformers, remainder="drop")),
                ("model", LogisticRegression(C=0.5, max_iter=5000, random_state=seed)),
            ]
        )
    if family == "random_forest":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=160,
                        max_depth=5,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    if family == "xgboost":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=120,
                        max_depth=2,
                        learning_rate=0.03,
                        min_child_weight=5,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=2.0,
                        eval_metric="logloss",
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    raise KeyError(family)


def expanding_predictions(data: pd.DataFrame, config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast = config["forecast"]
    seed = int(forecast["seed"])
    rows = []
    audits = []
    for support_m, support in data.groupby("support_m", sort=True):
        for year in range(int(forecast["first_forecast_year"]), int(forecast["last_forecast_year"]) + 1):
            for context, context_data in support.groupby("context", sort=True):
                train = context_data.loc[context_data["year"].between(int(forecast["train_start_year"]), year - 1)].copy()
                test = context_data.loc[context_data["year"].eq(year)].copy()
                status = "fit"
                if test.empty:
                    status = "no_test_rows"
                elif train["event"].nunique() < 2:
                    status = "one_class_train"
                audits.append(
                    {
                        "support_m": int(support_m),
                        "year": year,
                        "context": context,
                        "status": status,
                        "train_start": int(train["year"].min()) if not train.empty else np.nan,
                        "train_end": int(train["year"].max()) if not train.empty else np.nan,
                        "n_train": len(train),
                        "events_train": int(train["event"].sum()),
                        "n_test": len(test),
                        "events_test": int(test["event"].sum()),
                        "test_sites": int(test["site_id"].nunique()),
                    }
                )
                if status != "fit":
                    continue
                for family in FAMILY_ORDER:
                    for feature_set in FEATURE_SET_ORDER:
                        features = features_for(context, feature_set)
                        model = estimator(family, features, seed + year)
                        model.fit(train[features], train["event"])
                        output = test[
                            ["site_id", "support_m", "overlap_cluster", "region_group", "context", "year", "event"]
                        ].copy()
                        output["score"] = model.predict_proba(test[features])[:, 1]
                        output["model_family"] = family
                        output["feature_set"] = feature_set
                        rows.append(output)
    predictions = pd.concat(rows, ignore_index=True)
    key = ["support_m", "site_id", "context", "year", "model_family", "feature_set"]
    if predictions.duplicated(key).any():
        raise ValueError("Prediction keys are not unique")
    return predictions, pd.DataFrame(audits)


def add_decision_ranks(predictions: pd.DataFrame, budget_fraction: float) -> pd.DataFrame:
    rows = []
    keys = ["support_m", "model_family", "feature_set", "year"]
    for _, group in predictions.groupby(keys, sort=True):
        ranked = group.sort_values(["score", "site_id", "context"], ascending=[False, True, True]).copy()
        selected_n = max(1, int(math.ceil(len(ranked) * budget_fraction)))
        ranked["selected_at_budget"] = False
        ranked.iloc[:selected_n, ranked.columns.get_loc("selected_at_budget")] = True
        ranked["risk_percentile"] = ranked["score"].rank(method="average", pct=True)
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True)


def safe_ap(group: pd.DataFrame) -> float:
    if group.empty or group["event"].nunique() < 2:
        return np.nan
    return float(average_precision_score(group["event"], group["score"]))


def annual_metrics(ranked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in ranked.groupby(["support_m", "model_family", "feature_set", "year"], sort=True):
        ap = safe_ap(group)
        rows.append(
            {
                "support_m": int(keys[0]),
                "model_family": keys[1],
                "feature_set": keys[2],
                "year": int(keys[3]),
                "n": len(group),
                "events": int(group["event"].sum()),
                "prevalence": float(group["event"].mean()),
                "ap": ap,
                "ap_lift": ap - group["event"].mean() if np.isfinite(ap) else np.nan,
                "brier": float(brier_score_loss(group["event"], group["score"])),
                "log_loss": float(log_loss(group["event"], group["score"], labels=[0, 1])),
                "selected": int(group["selected_at_budget"].sum()),
                "true_positives": int(group.loc[group["selected_at_budget"], "event"].sum()),
            }
        )
    return pd.DataFrame(rows)


def year_bootstrap(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).astype(float))


def hierarchical_increment_bootstrap(
    pair: pd.DataFrame, replicates: int, seed: int
) -> tuple[float, float, int]:
    def array_ap(events: np.ndarray, scores: np.ndarray) -> float:
        positives = int(events.sum())
        if positives == 0 or positives == len(events):
            return np.nan
        order = np.argsort(-scores, kind="mergesort")
        ordered_events = events[order]
        precision = np.cumsum(ordered_events) / np.arange(1, len(ordered_events) + 1)
        return float(np.sum(precision * ordered_events) / positives)

    years = np.array(sorted(pair["year"].unique()))
    cached: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for year, year_data in pair.groupby("year", sort=True):
        clusters = []
        for _, cluster_data in year_data.groupby("overlap_cluster", sort=False):
            clusters.append(
                (
                    cluster_data["event"].to_numpy(dtype=np.int8),
                    cluster_data["score_left"].to_numpy(dtype=float),
                    cluster_data["score_right"].to_numpy(dtype=float),
                )
            )
        cached[int(year)] = clusters
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(replicates):
        sampled_years = rng.choice(years, size=len(years), replace=True)
        differences = []
        for year in sampled_years:
            clusters = cached[int(year)]
            draws = np.arange(len(clusters)) if len(clusters) < 2 else rng.integers(0, len(clusters), size=len(clusters))
            events = np.concatenate([clusters[index][0] for index in draws])
            left_scores = np.concatenate([clusters[index][1] for index in draws])
            right_scores = np.concatenate([clusters[index][2] for index in draws])
            if events.min() == events.max():
                continue
            left_ap = array_ap(events, left_scores)
            right_ap = array_ap(events, right_scores)
            differences.append(left_ap - right_ap)
        if differences:
            boot.append(float(np.mean(differences)))
    if not boot:
        return np.nan, np.nan, 0
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high), len(boot)


def summarize_performance(
    ranked: pd.DataFrame, annual: pd.DataFrame, config: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast = config["forecast"]
    reps = int(forecast["bootstrap_replicates"])
    seed = int(forecast["seed"])
    summaries = []
    for keys, group in annual.groupby(["support_m", "model_family", "feature_set"], sort=True):
        values = group["ap_lift"].dropna().to_numpy()
        low, high = year_bootstrap(values, reps, seed + int(keys[0]) + sum(map(ord, keys[1] + keys[2])))
        summaries.append(
            {
                "support_m": int(keys[0]),
                "model_family": keys[1],
                "feature_set": keys[2],
                "estimable_years": len(values),
                "macro_within_year_ap_lift": float(values.mean()),
                "year_bootstrap_ci_low": low,
                "year_bootstrap_ci_high": high,
                "micro_brier": float(np.average(group["brier"], weights=group["n"])),
                "micro_recall_top20": float(group["true_positives"].sum() / group["events"].sum()),
            }
        )

    increments = []
    index = ["site_id", "support_m", "overlap_cluster", "context", "year", "event", "model_family"]
    wide = ranked.pivot(index=index, columns="feature_set", values="score").reset_index()
    for (support_m, family), group in wide.groupby(["support_m", "model_family"], sort=True):
        annual_family = annual.loc[annual["support_m"].eq(support_m) & annual["model_family"].eq(family)]
        for label, left, right in COMPARISONS:
            pair = group[index + [left, right]].rename(columns={left: "score_left", right: "score_right"}).dropna()
            per_year = annual_family.pivot(index="year", columns="feature_set", values="ap_lift")
            differences = (per_year[left] - per_year[right]).dropna()
            low, high, successful = hierarchical_increment_bootstrap(
                pair, reps, seed + int(support_m) + sum(map(ord, family + label))
            )
            brier_gain = np.mean((pair["event"] - pair["score_right"]) ** 2 - (pair["event"] - pair["score_left"]) ** 2)
            increments.append(
                {
                    "support_m": int(support_m),
                    "model_family": family,
                    "comparison": label,
                    "left_feature_set": left,
                    "right_feature_set": right,
                    "paired_years": len(differences),
                    "estimate": float(differences.mean()),
                    "hierarchical_ci_low": low,
                    "hierarchical_ci_high": high,
                    "successful_bootstrap_replicates": successful,
                    "paired_rows": len(pair),
                    "brier_improvement": float(brier_gain),
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(increments)


def condition_summary(ranked: pd.DataFrame, analysis: pd.DataFrame) -> pd.DataFrame:
    index = ["site_id", "support_m", "overlap_cluster", "context", "year", "event", "model_family"]
    wide = ranked.pivot(index=index, columns="feature_set", values="score").reset_index()
    fields = analysis[
        ["site_id", "support_m", "context", "year", "purple_urchin_log1p", "active_grazing_proxy"]
    ].drop_duplicates()
    wide = wide.merge(fields, on=["site_id", "support_m", "context", "year"], how="left", validate="many_to_one")
    wide["disturbance_regime"] = pd.cut(
        wide["year"], bins=[-np.inf, 2013, 2018, np.inf], labels=["pre_mhw", "transition", "grazer_era"]
    )
    wide["grazing_level"] = "low"
    for (_, context), indices in wide.groupby(["support_m", "context"]).groups.items():
        threshold = wide.loc[indices, "purple_urchin_log1p"].median()
        wide.loc[indices, "grazing_level"] = np.where(
            wide.loc[indices, "purple_urchin_log1p"].ge(threshold), "high", "low"
        )
    rows = []
    for keys, group in wide.groupby(
        ["support_m", "model_family", "context", "disturbance_regime", "grazing_level"], observed=True, sort=True
    ):
        if len(group) < 10 or group["event"].nunique() < 2:
            continue
        full_ap = average_precision_score(group["event"], group["trajectory_plus_paper_full"])
        base_ap = average_precision_score(group["event"], group["trajectory"])
        brier_gain = np.mean(
            (group["event"] - group["trajectory"]) ** 2
            - (group["event"] - group["trajectory_plus_paper_full"]) ** 2
        )
        rows.append(
            {
                "support_m": int(keys[0]),
                "model_family": keys[1],
                "context": keys[2],
                "disturbance_regime": str(keys[3]),
                "grazing_level": keys[4],
                "n": len(group),
                "events": int(group["event"].sum()),
                "paper_full_ap_gain": float(full_ap - base_ap),
                "paper_full_brier_improvement": float(brier_gain),
            }
        )
    return pd.DataFrame(rows)


def decision_table(increments: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    interpretation = config["interpretation"]
    rows = []
    for comparison in [value[0] for value in COMPARISONS]:
        primary = increments.loc[
            increments["support_m"].eq(int(interpretation["primary_support_m"]))
            & increments["model_family"].eq(interpretation["primary_family"])
            & increments["comparison"].eq(comparison)
        ].iloc[0]
        direction = np.sign(primary["estimate"])
        same_support = increments.loc[
            increments["support_m"].eq(int(interpretation["primary_support_m"]))
            & increments["comparison"].eq(comparison)
        ]
        same_direction_families = int((np.sign(same_support["estimate"]) == direction).sum())
        other_support = increments.loc[
            increments["support_m"].eq(1000)
            & increments["model_family"].eq(interpretation["primary_family"])
            & increments["comparison"].eq(comparison)
        ].iloc[0]
        ci_excludes_zero = bool(primary["hierarchical_ci_low"] > 0 or primary["hierarchical_ci_high"] < 0)
        stable_positive = bool(
            primary["estimate"] > 0
            and primary["hierarchical_ci_low"] > 0
            and same_direction_families >= int(interpretation["requires_same_direction_model_families"])
            and other_support["estimate"] > 0
        )
        rows.append(
            {
                "comparison": comparison,
                "primary_estimate": primary["estimate"],
                "primary_ci_low": primary["hierarchical_ci_low"],
                "primary_ci_high": primary["hierarchical_ci_high"],
                "primary_ci_excludes_zero": ci_excludes_zero,
                "same_direction_families_300m": same_direction_families,
                "logistic_spline_estimate_1000m": other_support["estimate"],
                "stable_positive_increment": stable_positive,
            }
        )
    return pd.DataFrame(rows)


def make_figure(output_dir: Path, increments: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    part = increments.loc[increments["comparison"].isin(["field_state_minus_trajectory", "paper_environment_minus_trajectory", "paper_full_minus_trajectory"])]
    labels = {
        "field_state_minus_trajectory": "Field state",
        "paper_environment_minus_trajectory": "Paper environment",
        "paper_full_minus_trajectory": "Paper full",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False, constrained_layout=False)
    for axis, support_m in zip(axes, [300, 1000], strict=True):
        panel = part.loc[part["support_m"].eq(support_m)].copy()
        panel["row"] = panel["comparison"].map(labels) + " / " + panel["model_family"]
        panel = panel.sort_values(["comparison", "model_family"])
        y = np.arange(len(panel))
        estimate = panel["estimate"].to_numpy()
        axis.errorbar(
            estimate,
            y,
            xerr=np.vstack([estimate - panel["hierarchical_ci_low"], panel["hierarchical_ci_high"] - estimate]),
            fmt="o",
            color="#2F5D7C" if support_m == 300 else "#D49A32",
            capsize=3,
        )
        axis.axvline(0, color="#33383C", linewidth=0.8, linestyle="--")
        axis.set_yticks(y, panel["row"] if support_m == 300 else [])
        axis.set_title(f"{support_m} m support", loc="left")
        axis.set_xlabel("Macro within-year AP-lift increment")
    figure.suptitle(
        "Paper-aligned California information-block audit",
        x=0.01,
        y=0.995,
        ha="left",
        va="top",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_dir / "figure_01_paper_aligned_increments.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed run: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")
    started = time.time()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    assignment, cohort_audit = context_sites(args.code_release)
    field, duplicate_audit = collapse_field(args.field, assignment)
    panel = pd.read_csv(args.panel)
    if panel.duplicated(["support_m", "site_id", "year"]).any():
        raise ValueError("Local support panel keys are not unique")
    merged = panel.merge(field, on=["site_id", "year"], how="inner", validate="many_to_one", suffixes=("", "_field"))
    merged = add_regime_features(merged)
    forecast = config["forecast"]
    analysis = merged.loc[
        merged["eligible"]
        & merged["year"].between(int(forecast["train_start_year"]), int(forecast["last_forecast_year"]))
    ].copy()
    analysis["context"] = pd.Categorical(
        analysis["context"], ["bull_north", "giant_central_southwest", "giant_southeast"], ordered=True
    )
    predictions, fold_audit = expanding_predictions(analysis, config)
    ranked = add_decision_ranks(predictions, float(forecast["monitoring_budget_fraction"]))
    annual = annual_metrics(ranked)
    performance, increments = summarize_performance(ranked, annual, config)
    conditions = condition_summary(ranked, analysis)
    decisions = decision_table(increments, config)
    coverage = (
        analysis.groupby(["support_m", "context"], observed=True, as_index=False)
        .agg(rows=("site_id", "size"), sites=("site_id", "nunique"), years=("year", "nunique"), events=("event", "sum"), overlap_clusters=("overlap_cluster", "nunique"))
    )
    missingness_rows = []
    all_features = sorted(set(column for context in assignment["context"].unique() for feature_set in FEATURE_SET_ORDER for column in features_for(context, feature_set)))
    for (support_m, context), group in analysis.groupby(["support_m", "context"], observed=True, sort=True):
        for feature in all_features:
            missingness_rows.append(
                {"support_m": int(support_m), "context": context, "feature": feature, "missing_fraction": float(group[feature].isna().mean())}
            )
    missingness = pd.DataFrame(missingness_rows)
    outputs = {
        "context_site_assignment.csv": assignment,
        "cohort_audit.csv": cohort_audit,
        "duplicate_transect_key_audit.csv": duplicate_audit,
        "field_site_year_features.csv": field,
        "analysis_rows.csv": analysis,
        "fold_audit.csv": fold_audit,
        "predictions.csv": predictions,
        "ranked_predictions.csv": ranked,
        "annual_metrics.csv": annual,
        "performance_summary.csv": performance,
        "increment_summary.csv": increments,
        "condition_summary.csv": conditions,
        "decision_table.csv": decisions,
        "coverage.csv": coverage,
        "feature_missingness.csv": missingness,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    make_figure(args.output_dir, increments)

    primary = decisions.loc[decisions["comparison"].eq(config["interpretation"]["primary_comparison"])].iloc[0]
    conclusion = (
        "방법론 보완 후 생태블록의 안정적 추가가치가 확인되었다."
        if primary["stable_positive_increment"]
        else "논문 정렬·비선형·교란체제 보완 후에도 안정적인 추가가치는 확인되지 않았다."
    )
    summary_lines = [
        "# California Giraldo 방법론 감사 v3 결과",
        "",
        "## 판정",
        "",
        conclusion,
        "",
        f"주 비교(300 m logistic spline, paper full−trajectory)는 {primary['primary_estimate']:+.4f}, 계층 bootstrap 95% CI [{primary['primary_ci_low']:+.4f}, {primary['primary_ci_high']:+.4f}]였다.",
        f"300 m에서 같은 방향인 모형은 {int(primary['same_direction_families_300m'])}/4개였고, 1 km logistic spline 추정치는 {primary['logistic_spline_estimate_1000m']:+.4f}였다.",
        "",
        "## 해석",
        "",
        "- 이 실험은 Giraldo의 동시점 켈프 밀도 GAM을 재현한 것이 아니라, 그 논문에서 확인한 코호트·종별 성게 임계·환경변수·비선형성을 다음 해 위성 캐노피 급감 문제에 이식한 방법론 감사다.",
        "- 북부 Reef Check site의 현장 수심은 공개 CSV에서 전부 결측이므로 지도형 수심으로 대체했다. 따라서 북부 문맥은 별도 민감도 해석이 필요하다.",
        "- 현장부분표본은 비확률 표본이며 California 보조 사례연구 밖으로 일반화하지 않는다.",
        "- 안정성 규칙을 통과하지 못한 결과는 성게·수온·영양염이 생태적으로 중요하지 않다는 뜻이 아니다.",
    ]
    (args.output_dir / "results_summary_ko.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    quality = pd.DataFrame(
        [
            {"check": "analysis_keys_unique", "value": int(analysis.duplicated(["support_m", "site_id", "year"]).sum()), "passed": not analysis.duplicated(["support_m", "site_id", "year"]).any()},
            {"check": "prediction_keys_unique", "value": int(predictions.duplicated(["support_m", "site_id", "context", "year", "model_family", "feature_set"]).sum()), "passed": not predictions.duplicated(["support_m", "site_id", "context", "year", "model_family", "feature_set"]).any()},
            {"check": "prediction_scores_bounded", "value": int((~predictions["score"].between(0, 1)).sum()), "passed": bool(predictions["score"].between(0, 1).all())},
            {"check": "forward_only_folds", "value": int((fold_audit.loc[fold_audit["status"].eq("fit"), "train_end"] >= fold_audit.loc[fold_audit["status"].eq("fit"), "year"]).sum()), "passed": bool((fold_audit.loc[fold_audit["status"].eq("fit"), "train_end"] < fold_audit.loc[fold_audit["status"].eq("fit"), "year"]).all())},
            {"check": "both_supports_present", "value": int(analysis["support_m"].nunique()), "passed": analysis["support_m"].nunique() == 2},
            {"check": "all_model_families_present", "value": int(predictions["model_family"].nunique()), "passed": predictions["model_family"].nunique() == len(FAMILY_ORDER)},
            {"check": "all_feature_sets_present", "value": int(predictions["feature_set"].nunique()), "passed": predictions["feature_set"].nunique() == len(FEATURE_SET_ORDER)},
            {"check": "primary_decision_present", "value": int(len(primary)), "passed": len(primary) > 0},
        ]
    )
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
        "primary_decision": primary.to_dict(),
        "coverage": coverage.to_dict(orient="records"),
        "inputs": {
            "config": {"path": str(args.config), "sha256": sha256(args.config)},
            "panel": {"path": str(args.panel), "sha256": sha256(args.panel)},
            "field": {"path": str(args.field), "sha256": sha256(args.field)},
            "code_release_manifest": {"path": str(args.code_release / "source_manifest.json"), "sha256": sha256(args.code_release / "source_manifest.json")},
        },
        "outputs": {name: sha256(args.output_dir / name) for name in [*outputs, "quality_checks.csv", "results_summary_ko.md", "figure_01_paper_aligned_increments.png"]},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"conclusion": conclusion, "primary": primary.to_dict(), "runtime_seconds": manifest["runtime_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
