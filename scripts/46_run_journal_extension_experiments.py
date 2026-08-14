"""Run journal-oriented robustness experiments on the locked 165-cell panel.

The run adds four reviewer-facing analyses without changing the original
forecast target or selecting a preferred result after inspection:

1. repeated random cell-year CV / pooled metrics versus expanding-window /
   within-year metrics;
2. fixed logistic, random-forest, and XGBoost model-family robustness;
3. 5%, 10%, 20%, and 30% monitoring-budget recall curves;
4. 20%, 30%, and 40% decline thresholds crossed with 0.02, 0.05, and 0.10
   current-canopy eligibility cutoffs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
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
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


SEED = 20260814
TRAIN_START = 1989
FORECAST_START = 2005
FORECAST_END = 2024
BOOTSTRAP_REPLICATES = 2000
RANDOM_REPEATS = 20
RANDOM_FOLDS = 5

MODEL_LABELS = {
    "logistic": "Logistic",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}
FEATURE_LABELS = {
    "current_only": "Current",
    "current_plus_trajectory": "Current + trajectory",
    "trajectory_plus_oisst": "Current + trajectory + OISST",
    "trajectory_plus_oisst_cuti_beuti": "Current + trajectory + OISST + CUTI/BEUTI",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/processed/kelpwatch_west_coast_panel_165.csv"),
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=Path(
            "outputs/experiments/20260814_minimal_paper_extension_v2/environment_features.csv"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/journal_extension_v1.json"),
    )
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
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.SubprocessError:
        return "unknown"


def read_config(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def feature_definitions(config: dict[str, object]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    blocks = config["information_blocks"]
    trajectory = list(blocks["trajectory"])
    oisst = list(blocks["oisst"])
    upwelling = list(blocks["cuti_beuti"])
    full = {
        "current_only": ["relative_canopy"],
        "current_plus_trajectory": ["relative_canopy", *trajectory],
        "trajectory_plus_oisst": ["relative_canopy", *trajectory, *oisst],
    }
    supported = {
        **full,
        "trajectory_plus_oisst_cuti_beuti": [
            "relative_canopy",
            *trajectory,
            *oisst,
            *upwelling,
        ],
    }
    return full, supported


def model_pipeline(family: str) -> Pipeline:
    if family == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(C=1.0, max_iter=5000, random_state=SEED),
                ),
            ]
        )
    if family == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=5,
                        min_samples_leaf=10,
                        max_features="sqrt",
                        random_state=SEED,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    if family == "xgboost":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.03,
                        min_child_weight=5,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        reg_alpha=0.0,
                        eval_metric="logloss",
                        random_state=SEED,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    raise KeyError(family)


def support_flag(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().eq("true")


def prepare_merged(panel: pd.DataFrame, environment: pd.DataFrame) -> pd.DataFrame:
    if panel["cell_id"].nunique() != 165:
        raise ValueError("Expected the locked 165-cell panel")
    if panel.duplicated(["cell_id", "year"]).any():
        raise ValueError("Panel contains duplicate cell-year keys")
    if environment.duplicated(["cell_id", "year"]).any():
        raise ValueError("Environment table contains duplicate cell-year keys")
    merged = panel.merge(
        environment,
        on=["cell_id", "year"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_environment"),
    )
    if len(merged) != len(panel):
        raise ValueError("Environment merge changed the panel grain")
    return merged


def prepare_domains(
    merged: pd.DataFrame,
    full_features: dict[str, list[str]],
    supported_features: dict[str, list[str]],
    decline_threshold: float = 0.30,
    canopy_cutoff: float = 0.05,
) -> dict[str, pd.DataFrame]:
    oisst = sorted(
        set(full_features["trajectory_plus_oisst"])
        - set(full_features["current_plus_trajectory"])
    )
    upwelling = sorted(
        set(supported_features["trajectory_plus_oisst_cuti_beuti"])
        - set(full_features["trajectory_plus_oisst"])
    )
    base = merged.loc[
        merged["year"].between(TRAIN_START, FORECAST_END)
        & merged["relative_canopy"].gt(canopy_cutoff)
        & merged["relative_drop_next"].notna()
    ].copy()
    base["event_variant"] = base["relative_drop_next"].ge(decline_threshold).astype(int)
    full = base.dropna(subset=oisst).copy()
    supported = base.loc[support_flag(base["upwelling_supported"])].dropna(
        subset=[*oisst, *upwelling]
    ).copy()
    domains = {
        "full_oisst_domain": full,
        "upwelling_supported_domain": supported,
    }
    expected = set(range(FORECAST_START, FORECAST_END + 1))
    for domain, frame in domains.items():
        observed = set(frame.loc[frame["year"].between(FORECAST_START, FORECAST_END), "year"])
        if observed != expected:
            raise ValueError(f"{domain} is missing forecast years: {sorted(expected - observed)}")
    return domains


def score_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    family: str,
    feature_set: str,
    domain: str,
    protocol: str,
    repeat: int | None = None,
    fold: int | None = None,
) -> pd.DataFrame:
    estimator = model_pipeline(family)
    estimator.fit(train[features], train["event_variant"])
    output = test[
        ["cell_id", "region_group", "year", "event_variant", "relative_drop_next"]
    ].copy()
    output = output.rename(columns={"event_variant": "event"})
    output["score"] = estimator.predict_proba(test[features])[:, 1]
    output["model_family"] = family
    output["feature_set"] = feature_set
    output["domain"] = domain
    output["protocol"] = protocol
    output["repeat"] = repeat
    output["fold"] = fold
    return output


def expanding_predictions(
    domains: dict[str, pd.DataFrame],
    feature_sets: dict[str, dict[str, list[str]]],
    families: tuple[str, ...] = ("logistic", "random_forest", "xgboost"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for domain, data in domains.items():
        for year in range(FORECAST_START, FORECAST_END + 1):
            train = data.loc[data["year"].between(TRAIN_START, year - 1)].copy()
            test = data.loc[data["year"].eq(year)].copy()
            if train["event_variant"].nunique() < 2 or test.empty:
                raise ValueError(f"Invalid expanding fold: {domain}, {year}")
            audits.append(
                {
                    "protocol": "expanding_window",
                    "domain": domain,
                    "year": year,
                    "train_start": int(train["year"].min()),
                    "train_end": int(train["year"].max()),
                    "n_train": len(train),
                    "events_train": int(train["event_variant"].sum()),
                    "n_test": len(test),
                    "events_test": int(test["event_variant"].sum()),
                    "test_cells": int(test["cell_id"].nunique()),
                }
            )
            for family in families:
                for feature_set, features in feature_sets[domain].items():
                    predictions.append(
                        score_model(
                            train,
                            test,
                            features,
                            family,
                            feature_set,
                            domain,
                            "expanding_window",
                        )
                    )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(audits)


def random_oof_predictions(
    domains: dict[str, pd.DataFrame],
    feature_sets: dict[str, dict[str, list[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_metrics: list[dict[str, object]] = []
    saved_repeat_zero: list[pd.DataFrame] = []
    for domain, data in domains.items():
        history = data.loc[data["year"].lt(FORECAST_START)].copy()
        evaluation = data.loc[data["year"].between(FORECAST_START, FORECAST_END)].copy()
        evaluation = evaluation.reset_index(drop=True)
        for repeat in range(RANDOM_REPEATS):
            splitter = StratifiedKFold(
                n_splits=RANDOM_FOLDS,
                shuffle=True,
                random_state=SEED + repeat,
            )
            split_indices = list(
                splitter.split(evaluation, evaluation["event_variant"])
            )
            for feature_set, features in feature_sets[domain].items():
                pieces: list[pd.DataFrame] = []
                for fold, (train_index, test_index) in enumerate(split_indices):
                    train = pd.concat(
                        [history, evaluation.iloc[train_index]], ignore_index=True
                    )
                    test = evaluation.iloc[test_index]
                    pieces.append(
                        score_model(
                            train,
                            test,
                            features,
                            "logistic",
                            feature_set,
                            domain,
                            "random_cell_year_cv",
                            repeat=repeat,
                            fold=fold,
                        )
                    )
                prediction = pd.concat(pieces, ignore_index=True)
                metric = prediction_metrics(prediction)
                all_metrics.append(
                    {
                        "protocol": "random_cell_year_cv",
                        "repeat": repeat,
                        "domain": domain,
                        "model_family": "logistic",
                        "feature_set": feature_set,
                        **metric,
                    }
                )
                if repeat == 0:
                    saved_repeat_zero.append(prediction)
    return pd.DataFrame(all_metrics), pd.concat(saved_repeat_zero, ignore_index=True)


def safe_ap(frame: pd.DataFrame) -> float:
    if frame.empty or frame["event"].nunique() < 2:
        return np.nan
    return float(average_precision_score(frame["event"], frame["score"]))


def annual_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["domain", "model_family", "feature_set", "protocol", "year"]
    for values, group in predictions.groupby(keys, sort=True, dropna=False):
        prevalence = float(group["event"].mean())
        ap = safe_ap(group)
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "n": len(group),
                "events": int(group["event"].sum()),
                "prevalence": prevalence,
                "average_precision": ap,
                "ap_lift": ap - prevalence if np.isfinite(ap) else np.nan,
                "estimable": bool(group["event"].nunique() == 2),
            }
        )
    return pd.DataFrame(rows)


def prediction_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    annual = []
    for _, group in frame.groupby("year"):
        if group["event"].nunique() < 2:
            continue
        prevalence = float(group["event"].mean())
        annual.append(safe_ap(group) - prevalence)
    pooled_prevalence = float(frame["event"].mean())
    pooled_ap = safe_ap(frame)
    return {
        "n": len(frame),
        "events": int(frame["event"].sum()),
        "prevalence": pooled_prevalence,
        "pooled_ap": pooled_ap,
        "pooled_ap_lift": pooled_ap - pooled_prevalence,
        "macro_within_year_ap_lift": float(np.mean(annual)),
        "estimable_years": len(annual),
        "brier": float(brier_score_loss(frame["event"], frame["score"])),
        "log_loss": float(log_loss(frame["event"], frame["score"], labels=[0, 1])),
    }


def bootstrap_mean(values: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.full(BOOTSTRAP_REPLICATES, np.nan)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    return values[indices].mean(axis=1)


def summarize_expanding(
    predictions: pd.DataFrame, annual: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    for (domain, family, feature_set), group in predictions.groupby(
        ["domain", "model_family", "feature_set"], sort=True
    ):
        metrics = prediction_metrics(group)
        yearly = annual.loc[
            annual["domain"].eq(domain)
            & annual["model_family"].eq(family)
            & annual["feature_set"].eq(feature_set)
            & annual["estimable"],
            "ap_lift",
        ].to_numpy()
        draws = bootstrap_mean(
            yearly,
            SEED + sum(ord(character) for character in domain + family + feature_set),
        )
        summaries.append(
            {
                "domain": domain,
                "model_family": family,
                "model_label": MODEL_LABELS[family],
                "feature_set": feature_set,
                "feature_label": FEATURE_LABELS[feature_set],
                **metrics,
                "macro_ci_low": float(np.nanquantile(draws, 0.025)),
                "macro_ci_high": float(np.nanquantile(draws, 0.975)),
            }
        )
    summary = pd.DataFrame(summaries)

    comparisons = {
        "full_oisst_domain": [
            ("current_plus_trajectory", "current_only", "trajectory_minus_current"),
            ("trajectory_plus_oisst", "current_plus_trajectory", "oisst_minus_trajectory"),
        ],
        "upwelling_supported_domain": [
            ("current_plus_trajectory", "current_only", "trajectory_minus_current"),
            ("trajectory_plus_oisst", "current_plus_trajectory", "oisst_minus_trajectory"),
            (
                "trajectory_plus_oisst_cuti_beuti",
                "trajectory_plus_oisst",
                "cuti_beuti_minus_oisst",
            ),
        ],
    }
    increment_rows: list[dict[str, object]] = []
    for domain, pairs in comparisons.items():
        for family in MODEL_LABELS:
            source = annual.loc[
                annual["domain"].eq(domain)
                & annual["model_family"].eq(family)
                & annual["estimable"]
            ]
            for left, right, label in pairs:
                left_table = source.loc[
                    source["feature_set"].eq(left), ["year", "ap_lift"]
                ].rename(columns={"ap_lift": "left"})
                right_table = source.loc[
                    source["feature_set"].eq(right), ["year", "ap_lift"]
                ].rename(columns={"ap_lift": "right"})
                joined = left_table.merge(right_table, on="year", validate="one_to_one")
                differences = (joined["left"] - joined["right"]).to_numpy()
                draws = bootstrap_mean(
                    differences,
                    SEED + sum(ord(character) for character in domain + family + label),
                )
                increment_rows.append(
                    {
                        "domain": domain,
                        "model_family": family,
                        "model_label": MODEL_LABELS[family],
                        "comparison": label,
                        "left_feature_set": left,
                        "right_feature_set": right,
                        "estimate": float(np.mean(differences)),
                        "ci_low": float(np.nanquantile(draws, 0.025)),
                        "ci_high": float(np.nanquantile(draws, 0.975)),
                        "estimable_years": len(differences),
                    }
                )
    return summary, pd.DataFrame(increment_rows)


def protocol_comparison(
    random_metrics: pd.DataFrame, expanding_summary: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expanding = expanding_summary.loc[expanding_summary["model_family"].eq("logistic")]
    for (domain, feature_set), group in random_metrics.groupby(
        ["domain", "feature_set"], sort=True
    ):
        operational = expanding.loc[
            expanding["domain"].eq(domain)
            & expanding["feature_set"].eq(feature_set)
        ].iloc[0]
        row: dict[str, object] = {
            "domain": domain,
            "feature_set": feature_set,
            "feature_label": FEATURE_LABELS[feature_set],
            "random_repeats": len(group),
        }
        for metric in ["pooled_ap_lift", "macro_within_year_ap_lift"]:
            random_values = group[metric].to_numpy(dtype=float)
            expanding_value = float(operational[metric])
            prefix = "pooled" if metric == "pooled_ap_lift" else "macro_within_year"
            row[f"random_{prefix}_mean"] = float(np.mean(random_values))
            row[f"random_{prefix}_p025"] = float(np.quantile(random_values, 0.025))
            row[f"random_{prefix}_p975"] = float(np.quantile(random_values, 0.975))
            row[f"expanding_{prefix}"] = expanding_value
            row[f"random_minus_expanding_{prefix}"] = float(
                np.mean(random_values) - expanding_value
            )
        rows.append(row)
    return pd.DataFrame(rows)


def budget_metrics(
    predictions: pd.DataFrame, budgets: tuple[float, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, object]] = []
    group_keys = ["domain", "model_family", "feature_set"]
    for keys, frame in predictions.groupby(group_keys, sort=True):
        for budget in budgets:
            for year, group in frame.groupby("year"):
                ranked = group.sort_values(
                    ["score", "cell_id"], ascending=[False, True]
                )
                selected_n = max(1, int(math.ceil(len(ranked) * budget)))
                selected = ranked.head(selected_n)
                events = int(ranked["event"].sum())
                true_positives = int(selected["event"].sum())
                detail_rows.append(
                    {
                        **dict(zip(group_keys, keys, strict=True)),
                        "budget_fraction": budget,
                        "year": int(year),
                        "n_candidates": len(ranked),
                        "selected_n": selected_n,
                        "events": events,
                        "true_positives": true_positives,
                        "precision": true_positives / selected_n,
                        "recall": true_positives / events if events else np.nan,
                    }
                )
    detail = pd.DataFrame(detail_rows)
    summary_rows: list[dict[str, object]] = []
    for keys, group in detail.groupby([*group_keys, "budget_fraction"], sort=True):
        observed_recall = float(group["true_positives"].sum() / group["events"].sum())
        observed_precision = float(
            group["true_positives"].sum() / group["selected_n"].sum()
        )
        values = group[["true_positives", "events", "selected_n"]].to_numpy(dtype=float)
        rng = np.random.default_rng(SEED + sum(ord(str(value)[0]) for value in keys))
        sampled = rng.integers(
            0, len(values), size=(BOOTSTRAP_REPLICATES, len(values))
        )
        sampled_values = values[sampled].sum(axis=1)
        recall_draws = np.divide(
            sampled_values[:, 0],
            sampled_values[:, 1],
            out=np.full(BOOTSTRAP_REPLICATES, np.nan),
            where=sampled_values[:, 1] > 0,
        )
        precision_draws = np.divide(
            sampled_values[:, 0],
            sampled_values[:, 2],
            out=np.full(BOOTSTRAP_REPLICATES, np.nan),
            where=sampled_values[:, 2] > 0,
        )
        summary_rows.append(
            {
                **dict(zip([*group_keys, "budget_fraction"], keys, strict=True)),
                "years": len(group),
                "total_selected": int(group["selected_n"].sum()),
                "total_events": int(group["events"].sum()),
                "total_true_positives": int(group["true_positives"].sum()),
                "micro_recall": observed_recall,
                "recall_ci_low": float(np.nanquantile(recall_draws, 0.025)),
                "recall_ci_high": float(np.nanquantile(recall_draws, 0.975)),
                "micro_precision": observed_precision,
                "precision_ci_low": float(np.nanquantile(precision_draws, 0.025)),
                "precision_ci_high": float(np.nanquantile(precision_draws, 0.975)),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def run_sensitivity(
    merged: pd.DataFrame,
    full_features: dict[str, list[str]],
    supported_features: dict[str, list[str]],
    decline_thresholds: tuple[float, ...],
    canopy_cutoffs: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_sets = {
        "full_oisst_domain": full_features,
        "upwelling_supported_domain": supported_features,
    }
    metric_rows: list[dict[str, object]] = []
    annual_rows: list[pd.DataFrame] = []
    increment_rows: list[dict[str, object]] = []
    for decline_threshold in decline_thresholds:
        for canopy_cutoff in canopy_cutoffs:
            domains = prepare_domains(
                merged,
                full_features,
                supported_features,
                decline_threshold=decline_threshold,
                canopy_cutoff=canopy_cutoff,
            )
            prediction, _ = expanding_predictions(
                domains, feature_sets, families=("logistic",)
            )
            annual = annual_metrics(prediction)
            annual["decline_threshold"] = decline_threshold
            annual["canopy_cutoff"] = canopy_cutoff
            annual_rows.append(annual)
            for (domain, feature_set), group in prediction.groupby(
                ["domain", "feature_set"], sort=True
            ):
                metrics = prediction_metrics(group)
                metric_rows.append(
                    {
                        "decline_threshold": decline_threshold,
                        "canopy_cutoff": canopy_cutoff,
                        "domain": domain,
                        "feature_set": feature_set,
                        **metrics,
                    }
                )
            comparison_map = {
                "full_oisst_domain": [
                    ("current_plus_trajectory", "current_only", "trajectory_minus_current"),
                    ("trajectory_plus_oisst", "current_plus_trajectory", "oisst_minus_trajectory"),
                ],
                "upwelling_supported_domain": [
                    ("current_plus_trajectory", "current_only", "trajectory_minus_current"),
                    ("trajectory_plus_oisst", "current_plus_trajectory", "oisst_minus_trajectory"),
                    (
                        "trajectory_plus_oisst_cuti_beuti",
                        "trajectory_plus_oisst",
                        "cuti_beuti_minus_oisst",
                    ),
                ],
            }
            for domain, pairs in comparison_map.items():
                source = annual.loc[
                    annual["domain"].eq(domain) & annual["estimable"]
                ]
                for left, right, label in pairs:
                    left_table = source.loc[
                        source["feature_set"].eq(left), ["year", "ap_lift"]
                    ].rename(columns={"ap_lift": "left"})
                    right_table = source.loc[
                        source["feature_set"].eq(right), ["year", "ap_lift"]
                    ].rename(columns={"ap_lift": "right"})
                    joined = left_table.merge(
                        right_table, on="year", validate="one_to_one"
                    )
                    differences = (joined["left"] - joined["right"]).to_numpy()
                    draws = bootstrap_mean(
                        differences,
                        SEED
                        + int(decline_threshold * 1000)
                        + int(canopy_cutoff * 10000)
                        + sum(ord(character) for character in domain + label),
                    )
                    increment_rows.append(
                        {
                            "decline_threshold": decline_threshold,
                            "canopy_cutoff": canopy_cutoff,
                            "domain": domain,
                            "comparison": label,
                            "estimate": float(np.mean(differences)),
                            "ci_low": float(np.nanquantile(draws, 0.025)),
                            "ci_high": float(np.nanquantile(draws, 0.975)),
                            "estimable_years": len(differences),
                        }
                    )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(annual_rows, ignore_index=True),
        pd.DataFrame(increment_rows),
    )


def data_quality_profile(
    panel: pd.DataFrame, environment: pd.DataFrame, merged: pd.DataFrame
) -> pd.DataFrame:
    eligible = (
        merged["year"].between(TRAIN_START, FORECAST_END)
        & merged["relative_canopy"].gt(0.05)
        & merged["relative_drop_next"].notna()
    )
    checks = [
        ("panel_rows", len(panel)),
        ("panel_cells", panel["cell_id"].nunique()),
        ("panel_year_min", int(panel["year"].min())),
        ("panel_year_max", int(panel["year"].max())),
        ("panel_duplicate_keys", int(panel.duplicated(["cell_id", "year"]).sum())),
        ("environment_rows", len(environment)),
        ("environment_duplicate_keys", int(environment.duplicated(["cell_id", "year"]).sum())),
        ("merged_rows", len(merged)),
        ("primary_eligible_rows", int(eligible.sum())),
        ("primary_eligible_cells", int(merged.loc[eligible, "cell_id"].nunique())),
        ("primary_missing_relative_drop", int(merged.loc[eligible, "relative_drop_next"].isna().sum())),
        ("supported_upwelling_cells", int(merged.loc[support_flag(merged["upwelling_supported"]), "cell_id"].nunique())),
    ]
    return pd.DataFrame(checks, columns=["check", "value"])


def make_figures(
    output: Path,
    protocol: pd.DataFrame,
    increments: pd.DataFrame,
    budget_summary: pd.DataFrame,
    sensitivity_increments: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    primary_protocol = protocol.loc[protocol["domain"].eq("full_oisst_domain")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for axis, metric, title in [
        (axes[0], "pooled", "Pooled AP lift"),
        (axes[1], "macro_within_year", "Macro within-year AP lift"),
    ]:
        y = np.arange(len(primary_protocol))
        random_mean = primary_protocol[f"random_{metric}_mean"].to_numpy()
        random_low = primary_protocol[f"random_{metric}_p025"].to_numpy()
        random_high = primary_protocol[f"random_{metric}_p975"].to_numpy()
        expanding = primary_protocol[f"expanding_{metric}"].to_numpy()
        axis.errorbar(
            random_mean,
            y - 0.12,
            xerr=[random_mean - random_low, random_high - random_mean],
            fmt="o",
            capsize=3,
            label="Random cell-year CV",
        )
        axis.scatter(expanding, y + 0.12, marker="s", label="Expanding window")
        axis.axvline(0, color="black", linewidth=1)
        axis.set_yticks(y, primary_protocol["feature_label"])
        axis.set_xlabel(title)
        axis.legend()
    fig.suptitle("Evaluation protocol and aggregation change the apparent performance")
    fig.savefig(output / "figure_01_protocol_comparison.png", dpi=180)
    plt.close(fig)

    selected_increments = increments.loc[
        (increments["domain"].eq("full_oisst_domain") & increments["comparison"].isin(["trajectory_minus_current", "oisst_minus_trajectory"]))
        | (
            increments["domain"].eq("upwelling_supported_domain")
            & increments["comparison"].eq("cuti_beuti_minus_oisst")
        )
    ].copy()
    selected_increments["label"] = (
        selected_increments["model_label"]
        + " | "
        + selected_increments["comparison"].replace(
            {
                "trajectory_minus_current": "trajectory − current",
                "oisst_minus_trajectory": "OISST − trajectory",
                "cuti_beuti_minus_oisst": "CUTI/BEUTI − OISST",
            }
        )
    )
    selected_increments = selected_increments.sort_values(
        ["comparison", "model_family"]
    ).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    y = np.arange(len(selected_increments))
    ax.errorbar(
        selected_increments["estimate"],
        y,
        xerr=[
            selected_increments["estimate"] - selected_increments["ci_low"],
            selected_increments["ci_high"] - selected_increments["estimate"],
        ],
        fmt="o",
        capsize=3,
        color="#ef6c00",
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y, selected_increments["label"])
    ax.set_xlabel("Paired difference in macro within-year AP lift (95% year-block CI)")
    ax.set_title("Incremental information value across fixed model families")
    fig.savefig(output / "figure_02_model_family_increments.png", dpi=180)
    plt.close(fig)

    budget = budget_summary.loc[
        budget_summary["domain"].eq("full_oisst_domain")
        & budget_summary["model_family"].eq("logistic")
    ]
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    colors = ["#455a64", "#1976d2", "#ef6c00"]
    for color, (feature_set, group) in zip(colors, budget.groupby("feature_set")):
        group = group.sort_values("budget_fraction")
        ax.plot(
            group["budget_fraction"] * 100,
            group["micro_recall"] * 100,
            marker="o",
            label=FEATURE_LABELS[feature_set],
            color=color,
        )
        ax.fill_between(
            group["budget_fraction"] * 100,
            group["recall_ci_low"] * 100,
            group["recall_ci_high"] * 100,
            alpha=0.15,
            color=color,
        )
    ax.set_xlabel("Cells inspected per year (%)")
    ax.set_ylabel("Decline events recovered (%)")
    ax.set_title("Monitoring-budget recall, expanding-window logistic model")
    ax.legend()
    ax.set_xlim(4, 31)
    ax.set_ylim(bottom=0)
    fig.savefig(output / "figure_03_budget_recall_curve.png", dpi=180)
    plt.close(fig)

    sensitivity = sensitivity_increments.loc[
        sensitivity_increments["domain"].eq("full_oisst_domain")
        & sensitivity_increments["comparison"].isin(
            ["trajectory_minus_current", "oisst_minus_trajectory"]
        )
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, comparison, title in [
        (axes[0], "trajectory_minus_current", "Trajectory − current"),
        (axes[1], "oisst_minus_trajectory", "OISST − trajectory"),
    ]:
        pivot = sensitivity.loc[sensitivity["comparison"].eq(comparison)].pivot(
            index="decline_threshold", columns="canopy_cutoff", values="estimate"
        )
        image = axis.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-0.08, vmax=0.08)
        axis.set_xticks(np.arange(len(pivot.columns)), [f">{x:.2f}" for x in pivot.columns])
        axis.set_yticks(np.arange(len(pivot.index)), [f"{x:.0%}" for x in pivot.index])
        axis.set_xlabel("Minimum current relative canopy")
        axis.set_ylabel("Decline threshold")
        axis.set_title(title)
        for row in range(pivot.shape[0]):
            for column in range(pivot.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{pivot.iloc[row, column]:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
    fig.colorbar(image, ax=axes, label="Δ macro within-year AP lift")
    fig.suptitle("Outcome and eligibility sensitivity, expanding-window logistic model")
    fig.savefig(output / "figure_04_threshold_canopy_sensitivity.png", dpi=180)
    plt.close(fig)


def write_decision_and_report(
    output: Path,
    protocol: pd.DataFrame,
    model_summary: pd.DataFrame,
    increments: pd.DataFrame,
    budget_summary: pd.DataFrame,
    sensitivity_metrics: pd.DataFrame,
    sensitivity_increments: pd.DataFrame,
) -> dict[str, object]:
    protocol_current = protocol.loc[
        protocol["domain"].eq("full_oisst_domain")
        & protocol["feature_set"].eq("current_only")
    ].iloc[0]
    logistic_current = model_summary.loc[
        model_summary["domain"].eq("full_oisst_domain")
        & model_summary["model_family"].eq("logistic")
        & model_summary["feature_set"].eq("current_only")
    ].iloc[0]
    primary_increments = increments.loc[
        (
            increments["domain"].eq("full_oisst_domain")
            & increments["comparison"].isin(
                ["trajectory_minus_current", "oisst_minus_trajectory"]
            )
        )
        | (
            increments["domain"].eq("upwelling_supported_domain")
            & increments["comparison"].eq("cuti_beuti_minus_oisst")
        )
    ]
    model_family_conclusions: dict[str, dict[str, object]] = {}
    for family in MODEL_LABELS:
        family_rows = primary_increments.loc[
            primary_increments["model_family"].eq(family)
        ]
        model_family_conclusions[family] = {
            row.comparison: {
                "estimate": float(row.estimate),
                "ci": [float(row.ci_low), float(row.ci_high)],
                "supported_positive": bool(row.ci_low > 0),
            }
            for row in family_rows.itertuples()
        }
    primary_budget = budget_summary.loc[
        budget_summary["domain"].eq("full_oisst_domain")
        & budget_summary["model_family"].eq("logistic")
        & budget_summary["feature_set"].eq("current_only")
    ].sort_values("budget_fraction")
    primary_sensitivity = sensitivity_metrics.loc[
        sensitivity_metrics["domain"].eq("full_oisst_domain")
        & sensitivity_metrics["feature_set"].eq("current_only")
    ]
    oisst_sensitivity = sensitivity_increments.loc[
        sensitivity_increments["domain"].eq("full_oisst_domain")
        & sensitivity_increments["comparison"].eq("oisst_minus_trajectory")
    ]
    decision = {
        "classification": "pilot-informed reviewer robustness; not independent confirmation",
        "protocol_current_only": {
            "random_pooled_lift_mean": float(protocol_current.random_pooled_mean),
            "random_macro_within_year_lift_mean": float(
                protocol_current.random_macro_within_year_mean
            ),
            "expanding_pooled_lift": float(protocol_current.expanding_pooled),
            "expanding_macro_within_year_lift": float(
                protocol_current.expanding_macro_within_year
            ),
        },
        "primary_logistic_current_macro_lift": float(
            logistic_current.macro_within_year_ap_lift
        ),
        "model_family_increment_conclusions": model_family_conclusions,
        "budget_current_only": {
            f"top_{int(row.budget_fraction * 100)}pct": {
                "recall": float(row.micro_recall),
                "ci": [float(row.recall_ci_low), float(row.recall_ci_high)],
            }
            for row in primary_budget.itertuples()
        },
        "sensitivity_current_macro_lift_range": [
            float(primary_sensitivity.macro_within_year_ap_lift.min()),
            float(primary_sensitivity.macro_within_year_ap_lift.max()),
        ],
        "sensitivity_oisst_increment_positive_combinations": int(
            oisst_sensitivity["estimate"].gt(0).sum()
        ),
        "sensitivity_oisst_increment_combinations": len(oisst_sensitivity),
    }
    (output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    random_pooled = float(protocol_current.random_pooled_mean)
    expanding_pooled = float(protocol_current.expanding_pooled)
    random_macro = float(protocol_current.random_macro_within_year_mean)
    expanding_macro = float(protocol_current.expanding_macro_within_year)
    lines = [
        "# 저널 보완 실험 결과",
        "",
        "> 이 분석은 기존 결과를 확인한 뒤 수행한 reviewer robustness 분석이다. 사전등록 또는 독립 확증으로 표현하지 않는다.",
        "",
        "## 1. 검증 프로토콜과 집계 방식",
        "",
        f"- 현재 상태 모델의 random-CV pooled AP lift는 {random_pooled:.3f}, expanding-window pooled AP lift는 {expanding_pooled:.3f}였다.",
        f"- 같은 모델의 random-CV macro within-year AP lift는 {random_macro:.3f}, expanding-window 값은 {expanding_macro:.3f}였다.",
        "- random-CV는 미래 연도와 동일 셀의 다른 연도를 학습에 허용하는 의도적 비운영 비교이므로 최종 성능으로 사용하지 않는다.",
        "",
        "## 2. 모델 계열 강건성",
        "",
    ]
    for family in MODEL_LABELS:
        conclusions = model_family_conclusions[family]
        parts = []
        for comparison in [
            "trajectory_minus_current",
            "oisst_minus_trajectory",
            "cuti_beuti_minus_oisst",
        ]:
            if comparison in conclusions:
                result = conclusions[comparison]
                parts.append(
                    f"{comparison} {result['estimate']:+.3f} "
                    f"(CI {result['ci'][0]:+.3f}–{result['ci'][1]:+.3f})"
                )
        lines.append(f"- {MODEL_LABELS[family]}: " + "; ".join(parts) + ".")
    lines.extend(
        [
            "",
            "## 3. 조사예산 곡선",
            "",
        ]
    )
    for row in primary_budget.itertuples():
        lines.append(
            f"- 현재 상태 모델, 상위 {row.budget_fraction:.0%} 조사: 사건 회수율 "
            f"{row.micro_recall:.1%} (95% year-block CI {row.recall_ci_low:.1%}–{row.recall_ci_high:.1%})."
        )
    lines.extend(
        [
            "",
            "## 4. 라벨·최소 캐노피 민감도",
            "",
            f"- 20/30/40% 급감 × 최소 캐노피 0.02/0.05/0.10의 9개 조합에서 현재 상태 macro within-year AP lift 범위는 {primary_sensitivity.macro_within_year_ap_lift.min():.3f}–{primary_sensitivity.macro_within_year_ap_lift.max():.3f}였다.",
            f"- OISST 증분 추정치가 양수였던 조합은 {int(oisst_sensitivity['estimate'].gt(0).sum())}/{len(oisst_sensitivity)}개였다. 개별 추정치와 신뢰구간은 sensitivity_increment_summary.csv에 기록했다.",
            "",
            "## 해석 가드레일",
            "",
            "- pooled와 within-year, random과 expanding 결과를 섞어 단일 성능처럼 보고하지 않는다.",
            "- 모델 계열 비교는 고정된 합리적 사양의 강건성 분석이며 최고 알고리즘 선발전이 아니다.",
            "- 30%는 생태 붕괴 임계값이 아니라 운영 라벨이다.",
            "- OISST/CUTI/BEUTI 결과는 선택한 공개 proxy의 증분 예측정보에 관한 것이며 환경의 생태적 중요성을 부정하지 않는다.",
        ]
    )
    (output / "results_summary_ko.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return decision


def main() -> None:
    args = parse_args()
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite completed run directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config = read_config(args.config)
    full_features, supported_features = feature_definitions(config)
    feature_sets = {
        "full_oisst_domain": full_features,
        "upwelling_supported_domain": supported_features,
    }
    panel = pd.read_csv(args.panel)
    environment = pd.read_csv(args.environment)
    merged = prepare_merged(panel, environment)
    domains = prepare_domains(merged, full_features, supported_features)

    expanding, fold_audit = expanding_predictions(domains, feature_sets)
    annual = annual_metrics(expanding)
    model_summary, increment_summary = summarize_expanding(expanding, annual)

    random_metrics, random_repeat_zero = random_oof_predictions(
        domains, feature_sets
    )
    protocol = protocol_comparison(random_metrics, model_summary)

    budgets = tuple(float(value) for value in config["decision_budget"]["fractions"])
    budget_detail, budget_summary = budget_metrics(expanding, budgets)

    decline_thresholds = tuple(
        float(value) for value in config["sensitivity"]["decline_fractions"]
    )
    canopy_cutoffs = tuple(
        float(value)
        for value in config["sensitivity"]["eligibility_relative_canopy_gt"]
    )
    sensitivity_metrics, sensitivity_annual, sensitivity_increments = run_sensitivity(
        merged,
        full_features,
        supported_features,
        decline_thresholds,
        canopy_cutoffs,
    )
    quality_profile = data_quality_profile(panel, environment, merged)

    outputs: dict[str, pd.DataFrame] = {
        "data_quality_profile.csv": quality_profile,
        "fold_audit.csv": fold_audit,
        "expanding_predictions.csv": expanding,
        "expanding_year_metrics.csv": annual,
        "model_family_summary.csv": model_summary,
        "model_family_increment_summary.csv": increment_summary,
        "random_protocol_replicate_metrics.csv": random_metrics,
        "random_protocol_predictions_repeat0.csv": random_repeat_zero,
        "protocol_comparison.csv": protocol,
        "budget_year_detail.csv": budget_detail,
        "budget_summary.csv": budget_summary,
        "sensitivity_model_metrics.csv": sensitivity_metrics,
        "sensitivity_year_metrics.csv": sensitivity_annual,
        "sensitivity_increment_summary.csv": sensitivity_increments,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)

    make_figures(
        args.output_dir,
        protocol,
        increment_summary,
        budget_summary,
        sensitivity_increments,
    )
    decision = write_decision_and_report(
        args.output_dir,
        protocol,
        model_summary,
        increment_summary,
        budget_summary,
        sensitivity_metrics,
        sensitivity_increments,
    )

    all_output_files = [
        *outputs,
        "decision.json",
        "results_summary_ko.md",
        "figure_01_protocol_comparison.png",
        "figure_02_model_family_increments.png",
        "figure_03_budget_recall_curve.png",
        "figure_04_threshold_canopy_sensitivity.png",
    ]
    manifest = {
        "status": "complete",
        "classification": "pilot-informed reviewer robustness; not independent confirmation",
        "started_at_utc_epoch": started,
        "finished_at_utc_epoch": time.time(),
        "runtime_seconds": round(time.time() - started, 2),
        "command": " ".join(sys.argv),
        "git_sha_before_run": git_sha(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": __import__("xgboost").__version__,
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "random_cv_repeats": RANDOM_REPEATS,
        "random_cv_folds": RANDOM_FOLDS,
        "panel": str(args.panel),
        "panel_sha256": sha256(args.panel),
        "environment": str(args.environment),
        "environment_sha256": sha256(args.environment),
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "forecast_years": [FORECAST_START, FORECAST_END],
        "model_families": list(MODEL_LABELS),
        "budgets": list(budgets),
        "decline_thresholds": list(decline_thresholds),
        "canopy_cutoffs": list(canopy_cutoffs),
        "decision": decision,
        "output_sha256": {
            name: sha256(args.output_dir / name) for name in all_output_files
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "command.txt").write_text(
        " ".join(sys.argv) + "\n", encoding="utf-8"
    )
    print(model_summary.to_string(index=False))
    print("\nINCREMENTS\n", increment_summary.to_string(index=False))
    print("\nPROTOCOL\n", protocol.to_string(index=False))
    print("\nDECISION\n", json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
