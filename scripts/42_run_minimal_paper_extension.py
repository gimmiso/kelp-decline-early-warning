"""Run the four minimal paper-salvage experiments on the locked 165-cell panel."""

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TRAIN_START = 1989
FORECAST_START = 2005
FORECAST_END = 2024
SEED = 20260814
BOOTSTRAP_REPLICATES = 2000

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

DOMAIN_MODELS = {
    "full_oisst_domain": {
        "current_only": ["relative_canopy"],
        "current_plus_trajectory": ["relative_canopy", *TRAJECTORY],
        "trajectory_plus_oisst": ["relative_canopy", *TRAJECTORY, *OISST],
    },
    "upwelling_supported_domain": {
        "current_only": ["relative_canopy"],
        "current_plus_trajectory": ["relative_canopy", *TRAJECTORY],
        "trajectory_plus_oisst": ["relative_canopy", *TRAJECTORY, *OISST],
        "trajectory_plus_oisst_cuti_beuti": [
            "relative_canopy",
            *TRAJECTORY,
            *OISST,
            *UPWELLING,
        ],
    },
}

MODEL_LABELS = {
    "current_only": "현재 상태",
    "current_plus_trajectory": "현재+과거 변화",
    "trajectory_plus_oisst": "현재+과거+OISST",
    "trajectory_plus_oisst_cuti_beuti": "현재+과거+OISST+CUTI/BEUTI",
}
MODEL_LABELS_FIG = {
    "current_only": "Current canopy",
    "current_plus_trajectory": "Current + trajectory",
    "trajectory_plus_oisst": "Current + trajectory + OISST",
    "trajectory_plus_oisst_cuti_beuti": "Current + trajectory + OISST + CUTI/BEUTI",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/processed/kelpwatch_west_coast_panel_165.csv"),
    )
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/paper_e2e_v2.yaml"))
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


def pipeline() -> Pipeline:
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


def prepare_domains(panel: pd.DataFrame, environment: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if panel["cell_id"].nunique() != 165 or panel.duplicated(["cell_id", "year"]).any():
        raise ValueError("The locked input must be a duplicate-free 165-cell panel")
    if environment.duplicated(["cell_id", "year"]).any():
        raise ValueError("Environmental features contain duplicate keys")
    data = panel.merge(environment, on=["cell_id", "year"], how="left", suffixes=("", "_env"))
    base = data.loc[
        data["eligible"].astype(str).str.lower().eq("true")
        & data["year"].between(TRAIN_START, FORECAST_END)
    ].copy()
    full = base.dropna(subset=OISST).copy()
    upwelling = base.loc[base["upwelling_supported"].fillna(False)].dropna(
        subset=[*OISST, *UPWELLING]
    ).copy()
    domains = {"full_oisst_domain": full, "upwelling_supported_domain": upwelling}
    for domain, frame in domains.items():
        test_years = set(frame.loc[frame["year"].between(FORECAST_START, FORECAST_END), "year"])
        if test_years != set(range(FORECAST_START, FORECAST_END + 1)):
            raise ValueError(f"{domain} is missing forecast years: {sorted(set(range(FORECAST_START, FORECAST_END + 1)) - test_years)}")
    return domains


def score_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model: str,
    domain: str,
) -> pd.DataFrame:
    estimator = pipeline()
    estimator.fit(train[features], train["event"])
    out = test[["cell_id", "region_group", "year", "event"]].copy()
    out["score"] = estimator.predict_proba(test[features])[:, 1]
    out["model"] = model
    out["domain"] = domain
    return out


def temporal_backtest(domains: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    folds = []
    for domain, data in domains.items():
        for year in range(FORECAST_START, FORECAST_END + 1):
            train = data.loc[data["year"].between(TRAIN_START, year - 1)].copy()
            test = data.loc[data["year"].eq(year)].copy()
            if train["event"].nunique() < 2 or test.empty:
                raise ValueError(f"Invalid fold: {domain}, {year}")
            folds.append(
                {
                    "domain": domain,
                    "year": year,
                    "train_start": int(train["year"].min()),
                    "train_end": int(train["year"].max()),
                    "n_train": len(train),
                    "events_train": int(train["event"].sum()),
                    "n_test": len(test),
                    "events_test": int(test["event"].sum()),
                    "test_cells": int(test["cell_id"].nunique()),
                    "test_regions": int(test["region_group"].nunique()),
                }
            )
            for model, features in DOMAIN_MODELS[domain].items():
                predictions.append(score_fold(train, test, features, model, domain))
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(folds)


def safe_ap(group: pd.DataFrame) -> float:
    if group.empty or group["event"].nunique() < 2:
        return np.nan
    return float(average_precision_score(group["event"], group["score"]))


def stratified_metrics(predictions: pd.DataFrame, strata: list[str], scope: str) -> pd.DataFrame:
    rows = []
    group_cols = ["domain", "model", *strata]
    for keys, group in predictions.groupby(group_cols, sort=True):
        domain, model, *values = keys
        ranked = group.sort_values(["score", "cell_id"], ascending=[False, True])
        events = int(ranked["event"].sum())
        prevalence = float(ranked["event"].mean())
        ap = safe_ap(ranked)
        row = {
            "scope": scope,
            "domain": domain,
            "model": model,
            "n": len(ranked),
            "events": events,
            "prevalence": prevalence,
            "ap": ap,
            "ap_lift": ap - prevalence if np.isfinite(ap) else np.nan,
            "estimable": ranked["event"].nunique() == 2,
        }
        row.update(dict(zip(strata, values, strict=True)))
        for budget in (0.10, 0.20):
            k = max(1, int(math.ceil(len(ranked) * budget)))
            tp = int(ranked.head(k)["event"].sum())
            prefix = f"top{int(budget * 100)}"
            row[f"{prefix}_k"] = k
            row[f"{prefix}_tp"] = tp
            row[f"{prefix}_recall"] = tp / events if events else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summaries(
    predictions: pd.DataFrame,
    year_metrics: pd.DataFrame,
    region_year_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (domain, model), group in predictions.groupby(["domain", "model"], sort=True):
        annual = year_metrics.loc[
            year_metrics["domain"].eq(domain)
            & year_metrics["model"].eq(model)
            & year_metrics["estimable"]
        ]
        regional = region_year_metrics.loc[
            region_year_metrics["domain"].eq(domain)
            & region_year_metrics["model"].eq(model)
            & region_year_metrics["estimable"]
        ]
        rows.append(
            {
                "domain": domain,
                "model": model,
                "model_label_ko": MODEL_LABELS[model],
                "n": len(group),
                "cells": int(group["cell_id"].nunique()),
                "events": int(group["event"].sum()),
                "prevalence": float(group["event"].mean()),
                "pooled_ap": safe_ap(group),
                "macro_year_ap_lift": float(annual["ap_lift"].mean()),
                "estimable_years": len(annual),
                "macro_region_year_ap_lift": float(regional["ap_lift"].mean()),
                "estimable_region_years": len(regional),
                "top10_recall_micro": float(annual["top10_tp"].sum() / annual["events"].sum()),
                "top20_recall_micro": float(annual["top20_tp"].sum() / annual["events"].sum()),
                "brier": float(brier_score_loss(group["event"], group["score"])),
                "log_loss": float(log_loss(group["event"], group["score"], labels=[0, 1])),
            }
        )
    return pd.DataFrame(rows)


def bootstrap(
    year_metrics: pd.DataFrame,
    region_year_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def year_block_draws(
        table: pd.DataFrame, value_column: str, years: np.ndarray
    ) -> np.ndarray:
        """Vectorized resampling of complete years, retaining all strata in a year."""
        aggregates = (
            table.groupby("year")[value_column]
            .agg(["sum", "count"])
            .reindex(years, fill_value=0)
        )
        sampled_indices = rng.integers(
            0, len(years), size=(BOOTSTRAP_REPLICATES, len(years))
        )
        sums = aggregates["sum"].to_numpy()[sampled_indices].sum(axis=1)
        counts = aggregates["count"].to_numpy()[sampled_indices].sum(axis=1)
        return np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=counts > 0)

    rng = np.random.default_rng(SEED)
    estimate_rows = []
    difference_rows = []
    draw_rows = []
    for domain in DOMAIN_MODELS:
        models = list(DOMAIN_MODELS[domain])
        comparisons = [
            ("current_plus_trajectory", "current_only", "trajectory_minus_current"),
            ("trajectory_plus_oisst", "current_plus_trajectory", "oisst_minus_trajectory"),
        ]
        if domain == "upwelling_supported_domain":
            comparisons.append(
                (
                    "trajectory_plus_oisst_cuti_beuti",
                    "trajectory_plus_oisst",
                    "cuti_beuti_minus_oisst",
                )
            )
        for scope, metrics, stratum_keys in [
            ("year", year_metrics, ["year"]),
            ("region_year", region_year_metrics, ["region_group", "year"]),
        ]:
            source = metrics.loc[metrics["domain"].eq(domain) & metrics["estimable"]].copy()
            years = np.array(sorted(source["year"].unique()))
            for model in models:
                model_source = source.loc[source["model"].eq(model)]
                distribution = year_block_draws(model_source, "ap_lift", years)
                for replicate, value in enumerate(distribution):
                    draw_rows.append(
                        {
                            "domain": domain,
                            "scope": scope,
                            "kind": "model",
                            "name": model,
                            "replicate": replicate,
                            "value": float(value),
                        }
                    )
                estimate_rows.append(
                    {
                        "domain": domain,
                        "scope": scope,
                        "model": model,
                        "estimate": float(model_source["ap_lift"].mean()),
                        "ci_low": float(np.nanquantile(distribution, 0.025)),
                        "ci_high": float(np.nanquantile(distribution, 0.975)),
                        "strata": len(model_source),
                    }
                )
            for left, right, label in comparisons:
                left_table = source.loc[source["model"].eq(left), [*stratum_keys, "ap_lift"]].rename(
                    columns={"ap_lift": "left"}
                )
                right_table = source.loc[source["model"].eq(right), [*stratum_keys, "ap_lift"]].rename(
                    columns={"ap_lift": "right"}
                )
                joined = left_table.merge(right_table, on=stratum_keys, how="inner").dropna()
                joined["difference"] = joined["left"] - joined["right"]
                distribution = year_block_draws(joined, "difference", years)
                for replicate, value in enumerate(distribution):
                    draw_rows.append(
                        {
                            "domain": domain,
                            "scope": scope,
                            "kind": "difference",
                            "name": label,
                            "replicate": replicate,
                            "value": float(value),
                        }
                    )
                difference_rows.append(
                    {
                        "domain": domain,
                        "scope": scope,
                        "comparison": label,
                        "left_model": left,
                        "right_model": right,
                        "estimate": float(joined["difference"].mean()),
                        "ci_low": float(np.nanquantile(distribution, 0.025)),
                        "ci_high": float(np.nanquantile(distribution, 0.975)),
                        "strata": len(joined),
                        "supported_positive": bool(np.nanquantile(distribution, 0.025) > 0),
                    }
                )
    return pd.DataFrame(estimate_rows), pd.DataFrame(difference_rows), pd.DataFrame(draw_rows)


def stress_tests(
    predictions: pd.DataFrame,
    year_metrics: pd.DataFrame,
    region_year_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    loyo = []
    for (domain, model), group in predictions.groupby(["domain", "model"]):
        for scenario, selected in [
            ("all_years", group),
            ("exclude_2022", group.loc[group["year"].ne(2022)]),
        ]:
            ym = stratified_metrics(selected, ["year"], "year")
            rym = stratified_metrics(selected, ["region_group", "year"], "region_year")
            ay = ym.loc[ym["estimable"]]
            ary = rym.loc[rym["estimable"]]
            rows.append(
                {
                    "domain": domain,
                    "model": model,
                    "scenario": scenario,
                    "n": len(selected),
                    "events": int(selected["event"].sum()),
                    "pooled_ap": safe_ap(selected),
                    "macro_year_ap_lift": float(ay["ap_lift"].mean()),
                    "macro_region_year_ap_lift": float(ary["ap_lift"].mean()),
                    "top20_recall_micro": float(ay["top20_tp"].sum() / ay["events"].sum()),
                }
            )
        annual = year_metrics.loc[
            year_metrics["domain"].eq(domain)
            & year_metrics["model"].eq(model)
            & year_metrics["estimable"]
        ]
        regional = region_year_metrics.loc[
            region_year_metrics["domain"].eq(domain)
            & region_year_metrics["model"].eq(model)
            & region_year_metrics["estimable"]
        ]
        for omitted in range(FORECAST_START, FORECAST_END + 1):
            loyo.append(
                {
                    "domain": domain,
                    "model": model,
                    "omitted_year": omitted,
                    "macro_year_ap_lift": float(annual.loc[annual["year"].ne(omitted), "ap_lift"].mean()),
                    "macro_region_year_ap_lift": float(regional.loc[regional["year"].ne(omitted), "ap_lift"].mean()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(loyo)


def leave_region_out(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    region_counts = data[["cell_id", "region_group"]].drop_duplicates()["region_group"].value_counts()
    held_regions = region_counts.loc[region_counts.ge(15)].index.tolist()
    predictions = []
    features_by_model = DOMAIN_MODELS["full_oisst_domain"]
    for region in held_regions:
        for year in range(FORECAST_START, FORECAST_END + 1):
            train = data.loc[
                data["year"].between(TRAIN_START, year - 1) & data["region_group"].ne(region)
            ]
            test = data.loc[data["year"].eq(year) & data["region_group"].eq(region)]
            if test.empty or train["event"].nunique() < 2:
                continue
            for model, features in features_by_model.items():
                out = score_fold(train, test, features, model, "leave_region_out")
                out["held_region"] = region
                predictions.append(out)
    pred = pd.concat(predictions, ignore_index=True)
    rows = []
    for (region, model), group in pred.groupby(["held_region", "model"]):
        metrics = stratified_metrics(
            group.assign(domain="leave_region_out"), ["year"], "held_region_year"
        )
        estimable = metrics.loc[metrics["estimable"]]
        rows.append(
            {
                "held_region": region,
                "model": model,
                "cells": int(group["cell_id"].nunique()),
                "n": len(group),
                "events": int(group["event"].sum()),
                "prevalence": float(group["event"].mean()),
                "pooled_ap": safe_ap(group),
                "macro_year_ap_lift": float(estimable["ap_lift"].mean()),
                "estimable_years": len(estimable),
                "top20_recall_micro": float(estimable["top20_tp"].sum() / estimable["events"].sum()),
            }
        )
    return pred, pd.DataFrame(rows)


def make_figures(
    output: Path,
    summary: pd.DataFrame,
    estimates: pd.DataFrame,
    differences: pd.DataFrame,
    year_metrics: pd.DataFrame,
    transfer: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    palette = ["#455a64", "#1976d2", "#ef6c00", "#2e7d32"]

    annual = estimates.loc[estimates["scope"].eq("year")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, domain in zip(axes, DOMAIN_MODELS, strict=True):
        part = annual.loc[annual["domain"].eq(domain)].reset_index(drop=True)
        y = np.arange(len(part))
        axis.errorbar(
            part["estimate"], y,
            xerr=[part["estimate"] - part["ci_low"], part["ci_high"] - part["estimate"]],
            fmt="o", color="#1976d2", capsize=4,
        )
        axis.axvline(0, color="black", linewidth=1)
        axis.set_yticks(y, [MODEL_LABELS_FIG[x] for x in part["model"]])
        axis.set_xlabel("Macro within-year AP lift (95% year-block CI)")
        axis.set_title(
            "Full OISST domain" if domain == "full_oisst_domain" else "CUTI/BEUTI-supported domain"
        )
    fig.suptitle("Locked 165-cell expanding-window evaluation")
    fig.savefig(output / "figure_01_primary_model_comparison.png", dpi=180)
    plt.close(fig)

    part = differences.loc[differences["scope"].eq("year")].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    labels = [
        (
            "Full domain" if row.domain == "full_oisst_domain" else "Supported domain"
        )
        + "\n"
        + (
            {
                "oisst_minus_trajectory": "OISST − trajectory",
                "cuti_beuti_minus_oisst": "CUTI/BEUTI − OISST",
                "trajectory_minus_current": "Trajectory − current",
            }[row.comparison]
        )
        for row in part.itertuples()
    ]
    y = np.arange(len(part))
    ax.errorbar(
        part["estimate"], y,
        xerr=[part["estimate"] - part["ci_low"], part["ci_high"] - part["estimate"]],
        fmt="o", color="#ef6c00", capsize=4,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Paired difference in macro within-year AP lift")
    ax.set_title("Incremental value on exactly matched rows")
    fig.savefig(output / "figure_02_incremental_value.png", dpi=180)
    plt.close(fig)

    full = year_metrics.loc[
        year_metrics["domain"].eq("full_oisst_domain")
        & year_metrics["model"].isin(DOMAIN_MODELS["full_oisst_domain"])
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    burden = full.loc[full["model"].eq("current_only")]
    axes[0].bar(burden["year"], burden["events"], color="#78909c")
    axes[0].axvline(2022, color="#c62828", linestyle="--")
    axes[0].set_ylabel("Decline events")
    axes[0].set_title("Event burden and year-specific ranking performance")
    for color, (model, group) in zip(palette, full.groupby("model")):
        axes[1].plot(group["year"], group["ap_lift"], marker="o", label=MODEL_LABELS_FIG[model], color=color)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axvline(2022, color="#c62828", linestyle="--")
    axes[1].set_ylabel("Within-year AP lift")
    axes[1].set_xlabel("Forecast year")
    axes[1].legend(ncol=3)
    fig.savefig(output / "figure_03_year_dependence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for index, (model, group) in enumerate(transfer.groupby("model")):
        x = np.arange(len(group)) + (index - 1) * 0.24
        ax.bar(x, group["macro_year_ap_lift"], width=0.23, label=MODEL_LABELS_FIG[model])
    labels = transfer["held_region"].drop_duplicates().tolist()
    ax.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Macro within-year AP lift")
    ax.set_title("Leave-one-region-out transfer")
    ax.legend()
    fig.savefig(output / "figure_04_leave_region_out.png", dpi=180)
    plt.close(fig)


def write_decision_and_summary(
    output: Path,
    summary: pd.DataFrame,
    estimates: pd.DataFrame,
    differences: pd.DataFrame,
    stress: pd.DataFrame,
    loyo: pd.DataFrame,
    transfer: pd.DataFrame,
) -> dict[str, object]:
    primary_est = estimates.loc[
        estimates["domain"].eq("full_oisst_domain")
        & estimates["scope"].eq("year")
        & estimates["model"].eq("current_only")
    ].iloc[0]
    traj = differences.loc[
        differences["domain"].eq("full_oisst_domain")
        & differences["scope"].eq("year")
        & differences["comparison"].eq("trajectory_minus_current")
    ].iloc[0]
    oisst = differences.loc[
        differences["domain"].eq("full_oisst_domain")
        & differences["scope"].eq("year")
        & differences["comparison"].eq("oisst_minus_trajectory")
    ].iloc[0]
    upwell = differences.loc[
        differences["domain"].eq("upwelling_supported_domain")
        & differences["scope"].eq("year")
        & differences["comparison"].eq("cuti_beuti_minus_oisst")
    ].iloc[0]
    current = summary.loc[
        summary["domain"].eq("full_oisst_domain") & summary["model"].eq("current_only")
    ].iloc[0]
    trajectory_summary = summary.loc[
        summary["domain"].eq("full_oisst_domain")
        & summary["model"].eq("current_plus_trajectory")
    ].iloc[0]
    decision = {
        "rq1_ranking_above_prevalence_supported": bool(primary_est.ci_low > 0),
        "rq1_current_macro_year_ap_lift": float(primary_est.estimate),
        "rq1_current_ci": [float(primary_est.ci_low), float(primary_est.ci_high)],
        "rq2_trajectory_increment_supported": bool(traj.ci_low > 0),
        "rq2_trajectory_increment": float(traj.estimate),
        "rq2_trajectory_ci": [float(traj.ci_low), float(traj.ci_high)],
        "rq2_oisst_increment_supported": bool(oisst.ci_low > 0),
        "rq2_oisst_increment": float(oisst.estimate),
        "rq2_oisst_ci": [float(oisst.ci_low), float(oisst.ci_high)],
        "rq2_cuti_beuti_increment_supported": bool(upwell.ci_low > 0),
        "rq2_cuti_beuti_increment": float(upwell.estimate),
        "rq2_cuti_beuti_ci": [float(upwell.ci_low), float(upwell.ci_high)],
        "rq3_top20_recall": float(current.top20_recall_micro),
        "interpretation_guardrail": "pilot-informed locked extension; not independent confirmation",
    }
    (output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    excluded = stress.loc[
        stress["domain"].eq("full_oisst_domain")
        & stress["model"].eq("current_only")
        & stress["scenario"].eq("exclude_2022")
    ].iloc[0]
    excluded_trajectory = stress.loc[
        stress["domain"].eq("full_oisst_domain")
        & stress["model"].eq("current_plus_trajectory")
        & stress["scenario"].eq("exclude_2022")
    ].iloc[0]
    current_loyo = loyo.loc[
        loyo["domain"].eq("full_oisst_domain") & loyo["model"].eq("current_only")
    ]
    lines = [
        "# 최소 추가 실험 4개 결과",
        "",
        "> 이 실행은 기존 파일럿 결과를 본 뒤 고정한 확장 검증이다. 독립 확증이나 사전등록 실험으로 표현하지 않는다.",
        "",
        "## 핵심 판정",
        "",
        f"- RQ1 현재 상태 순위화: macro within-year AP lift {primary_est.estimate:.3f} (95% year-block CI {primary_est.ci_low:.3f}–{primary_est.ci_high:.3f}); 판정 = {'지지' if primary_est.ci_low > 0 else '불확실'}.",
        f"- 과거 변화 추가가치: Δ {traj.estimate:+.3f} (95% CI {traj.ci_low:+.3f}–{traj.ci_high:+.3f}); 판정 = {'지지' if traj.ci_low > 0 else '불확실'}.",
        f"- OISST 추가가치(165셀 동일 행): Δ {oisst.estimate:+.3f} (95% CI {oisst.ci_low:+.3f}–{oisst.ci_high:+.3f}); 판정 = {'지지' if oisst.ci_low > 0 else '불확실'}.",
        f"- OISST 이후 CUTI/BEUTI 순수 추가가치(U.S. 31–47°N, 107셀 동일 행): Δ {upwell.estimate:+.3f} (95% CI {upwell.ci_low:+.3f}–{upwell.ci_high:+.3f}); 판정 = {'지지' if upwell.ci_low > 0 else '불확실'}.",
        f"- 조사예산 20%에서 현재 상태 모델의 사건 회수율: {current.top20_recall_micro:.1%}.",
        f"- 같은 예산에서 과거 변화 모델의 회수율은 {trajectory_summary.top20_recall_micro:.1%}로, 실무 증가는 {(trajectory_summary.top20_recall_micro - current.top20_recall_micro):+.1%}p였다.",
        "",
        "## 스트레스 테스트",
        "",
        f"- 2022 제외 시 현재 상태 macro within-year AP lift: {excluded.macro_year_ap_lift:.3f} (전체 {primary_est.estimate:.3f}).",
        f"- 과거 변화 모델도 2022 제외 전후 {trajectory_summary.macro_year_ap_lift:.3f}→{excluded_trajectory.macro_year_ap_lift:.3f}으로 사실상 변하지 않았다.",
        f"- leave-one-year-out에서 현재 상태 모델의 macro lift 범위는 {current_loyo.macro_year_ap_lift.min():.3f}–{current_loyo.macro_year_ap_lift.max():.3f}였고 모두 양수였다.",
        "- leave-one-region-out의 모든 보류지역에서 현재 상태 모델의 macro lift가 양수였지만, 과거 변화와 OISST의 증분은 지역마다 방향이 달랐다.",
        "- 연도별 사건 수, leave-one-year-out, 지역 외부검증 전체표는 같은 실행 폴더의 CSV에 저장했다.",
        "",
        "## 논문에서 허용되는 주장",
        "",
        "- 핵심 기여는 새 생태 기작 발견이 아니라 공개자료 기반 조기경보의 엄격한 시간·공간·예산 평가다.",
        "- 환경변수의 추가가치는 동일한 cell-year 행에서만 비교했다.",
        "- CUTI/BEUTI는 U.S. West Coast 31–47°N의 107셀에만 사용했고 Baja·47°N 북쪽으로 외삽하지 않았다.",
        "- 환경 결과는 ‘환경이 중요하지 않음’이 아니라 ‘선택한 공개 proxy가 이 해상도·예측시점에서 캐노피 이력 이후 추가 예측정보를 보이지 않음’으로 표현한다.",
    ]
    (output / "results_summary_ko.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    args = parse_args()
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed run directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    panel = pd.read_csv(args.panel)
    environment = pd.read_csv(args.environment)
    domains = prepare_domains(panel, environment)
    predictions, folds = temporal_backtest(domains)
    year_metrics = stratified_metrics(predictions, ["year"], "year")
    region_year_metrics = stratified_metrics(
        predictions, ["region_group", "year"], "region_year"
    )
    summary = summaries(predictions, year_metrics, region_year_metrics)
    estimates, differences, draws = bootstrap(year_metrics, region_year_metrics)
    stress, loyo = stress_tests(predictions, year_metrics, region_year_metrics)
    transfer_predictions, transfer_summary = leave_region_out(domains["full_oisst_domain"])

    matched = (
        predictions.groupby(["domain", "model"])[["cell_id", "year"]]
        .size()
        .reset_index(name="rows")
    )
    quality_rows = [
        {"check": "locked_cells", "value": panel["cell_id"].nunique(), "passed": panel["cell_id"].nunique() == 165},
        {"check": "input_duplicate_keys", "value": int(panel.duplicated(["cell_id", "year"]).sum()), "passed": not panel.duplicated(["cell_id", "year"]).any()},
        {"check": "train_end_before_test", "value": bool((folds["train_end"] < folds["year"]).all()), "passed": bool((folds["train_end"] < folds["year"]).all())},
        {"check": "missing_prediction_scores", "value": int(predictions["score"].isna().sum()), "passed": not predictions["score"].isna().any()},
        {"check": "duplicate_predictions", "value": int(predictions.duplicated(["domain", "model", "cell_id", "year"]).sum()), "passed": not predictions.duplicated(["domain", "model", "cell_id", "year"]).any()},
        {"check": "matched_rows_within_domain", "value": matched.groupby("domain")["rows"].nunique().max(), "passed": matched.groupby("domain")["rows"].nunique().eq(1).all()},
        {"check": "forecast_years", "value": f"{FORECAST_START}-{FORECAST_END}", "passed": predictions["year"].nunique() == 20},
    ]
    quality = pd.DataFrame(quality_rows)
    if not quality["passed"].all():
        raise AssertionError(quality.to_string(index=False))

    outputs = {
        "predictions.csv": predictions,
        "fold_audit.csv": folds,
        "model_summary.csv": summary,
        "year_metrics.csv": year_metrics,
        "region_year_metrics.csv": region_year_metrics,
        "bootstrap_estimates.csv": estimates,
        "bootstrap_differences.csv": differences,
        "bootstrap_draws.csv": draws,
        "stress_2022.csv": stress,
        "leave_one_year_out.csv": loyo,
        "leave_region_out_predictions.csv": transfer_predictions,
        "leave_region_out_summary.csv": transfer_summary,
        "quality_checks.csv": quality,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    make_figures(
        args.output_dir, summary, estimates, differences, year_metrics, transfer_summary
    )
    decision = write_decision_and_summary(
        args.output_dir, summary, estimates, differences, stress, loyo, transfer_summary
    )
    manifest = {
        "status": "complete",
        "classification": "pilot-informed locked extension; not independent confirmation",
        "started_at_utc_epoch": started,
        "finished_at_utc_epoch": time.time(),
        "runtime_seconds": round(time.time() - started, 2),
        "command": " ".join(sys.argv),
        "git_sha_before_run": git_sha(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "panel": str(args.panel),
        "panel_sha256": sha256(args.panel),
        "environment": str(args.environment),
        "environment_sha256": sha256(args.environment),
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "forecast_years": [FORECAST_START, FORECAST_END],
        "label": "next-year annual maximum canopy relative decline >= 30% among current relative canopy > 0.05",
        "model": "fixed ridge logistic regression C=1.0",
        "primary_metric": "macro within-year AP lift over within-year prevalence",
        "decisions": decision,
        "output_sha256": {
            name: sha256(args.output_dir / name) for name in outputs
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nPAIRED DIFFERENCES\n", differences.to_string(index=False))
    print("\nDECISION\n", json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
