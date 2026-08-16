"""Compare year-end and March kelp-risk forecast clocks on matched 165-cell rows.

The primary March outcome is a 30% year-over-year decline in the maximum valid
Q2-Q3 canopy. Predictors are restricted to prior-calendar-year canopy history
plus, for March models, January-March SST of the outcome year. The script keeps
all information-block comparisons on identical cell-year rows and uses an
expanding-window backtest with within-year ranking metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TRAIN_OUTCOME_START = 1990
FORECAST_START = 2005
FORECAST_END = 2024
SEED = 20260816
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

FEATURE_SETS = {
    "prior_current": ["relative_canopy"],
    "prior_trajectory": ["relative_canopy", *TRAJECTORY],
    "december_plus_prior_winter_oisst": [
        "relative_canopy",
        *TRAJECTORY,
        "winter_oisst_yminus1",
    ],
    "march_plus_current_winter_oisst": [
        "relative_canopy",
        *TRAJECTORY,
        "winter_oisst_y",
    ],
    "march_plus_current_crw5km": [
        "relative_canopy",
        *TRAJECTORY,
        "q1_mean_sst_anomaly_crw5km",
        "q1_max_monthly_sst_anomaly_crw5km",
    ],
    "march_plus_current_crw5km_oisst": [
        "relative_canopy",
        *TRAJECTORY,
        "winter_oisst_y",
        "q1_mean_sst_anomaly_crw5km",
        "q1_max_monthly_sst_anomaly_crw5km",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annual-panel", type=Path, required=True)
    parser.add_argument("--quarterly", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--crw", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/paper_e2e_v2.yaml"))
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


def model_pipeline() -> Pipeline:
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


def q23_outcomes(quarterly: pd.DataFrame, cell_ids: set[str]) -> pd.DataFrame:
    required = {
        "cell_id",
        "region_group",
        "year",
        "quarter",
        "kelp_area_m2",
        "valid_fraction",
    }
    missing = required - set(quarterly.columns)
    if missing:
        raise ValueError(f"Missing quarterly columns: {sorted(missing)}")
    q = quarterly.loc[
        quarterly["cell_id"].isin(cell_ids)
        & quarterly["year"].between(1984, 2025)
        & quarterly["quarter"].isin([2, 3])
    ].copy()
    q["quarter_valid"] = q["valid_fraction"].ge(0.50)
    keys = ["cell_id", "region_group", "year"]
    annual = (
        q.groupby(keys, as_index=False)["quarter_valid"]
        .sum()
        .rename(columns={"quarter_valid": "valid_q23_quarters"})
    )
    maxima = (
        q.loc[q["quarter_valid"]]
        .groupby(keys)["kelp_area_m2"]
        .max()
        .rename("q23_max_kelp_area_m2")
        .reset_index()
    )
    annual = annual.merge(maxima, on=keys, how="left")
    annual.loc[annual["valid_q23_quarters"].lt(2), "q23_max_kelp_area_m2"] = np.nan
    reference = (
        annual.loc[annual["year"].between(1984, 2004)]
        .groupby("cell_id")["q23_max_kelp_area_m2"]
        .quantile(0.95)
        .rename("pre2005_q23_p95_area_m2")
    )
    annual = annual.merge(reference, on="cell_id", how="left").sort_values(
        ["cell_id", "year"]
    )
    grouped = annual.groupby("cell_id")
    annual["prior_q23_area_m2"] = grouped["q23_max_kelp_area_m2"].shift(1)
    annual["prior_q23_relative_canopy"] = (
        annual["prior_q23_area_m2"] / annual["pre2005_q23_p95_area_m2"]
    )
    annual["q23_relative_drop"] = (
        annual["prior_q23_area_m2"] - annual["q23_max_kelp_area_m2"]
    ) / annual["prior_q23_area_m2"]
    annual["q23_eligible"] = (
        annual["prior_q23_relative_canopy"].gt(0.05)
        & annual["prior_q23_area_m2"].notna()
        & annual["q23_max_kelp_area_m2"].notna()
        & annual["pre2005_q23_p95_area_m2"].gt(0)
    )
    annual["q23_event"] = annual["q23_relative_drop"].ge(0.30).astype(int)
    return annual


def prepare_analysis(
    annual_panel: pd.DataFrame,
    quarterly: pd.DataFrame,
    environment: pd.DataFrame,
    crw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if annual_panel["cell_id"].nunique() != 165:
        raise ValueError("Annual panel is not the locked 165-cell cohort")
    for name, frame in {
        "annual": annual_panel,
        "environment": environment,
        "crw": crw,
    }.items():
        if frame.duplicated(["cell_id", "year"]).any():
            raise ValueError(f"Duplicate cell-year keys in {name}")

    q23 = q23_outcomes(quarterly, set(annual_panel["cell_id"].unique()))
    prior = annual_panel[
        [
            "cell_id",
            "region_group",
            "year",
            "relative_canopy",
            "eligible",
            "event",
            *TRAJECTORY,
        ]
    ].copy()
    prior["outcome_year"] = prior["year"] + 1
    oisst_prior = environment[
        ["cell_id", "year", "winter_mean_sst_anomaly"]
    ].rename(columns={"winter_mean_sst_anomaly": "winter_oisst_yminus1"})
    oisst_current = environment[
        ["cell_id", "year", "winter_mean_sst_anomaly"]
    ].rename(
        columns={
            "year": "outcome_year",
            "winter_mean_sst_anomaly": "winter_oisst_y",
        }
    )
    crw_current = crw.rename(columns={"year": "outcome_year"})
    q23_current = q23.rename(columns={"year": "outcome_year"})[
        [
            "cell_id",
            "outcome_year",
            "q23_eligible",
            "q23_event",
            "q23_relative_drop",
            "valid_q23_quarters",
        ]
    ]
    data = (
        prior.merge(oisst_prior, on=["cell_id", "year"], how="left", validate="one_to_one")
        .merge(oisst_current, on=["cell_id", "outcome_year"], how="left", validate="one_to_one")
        .merge(crw_current, on=["cell_id", "outcome_year"], how="left", validate="one_to_one")
        .merge(q23_current, on=["cell_id", "outcome_year"], how="left", validate="one_to_one")
    )
    complete_features = sorted({feature for features in FEATURE_SETS.values() for feature in features})
    common = data.loc[
        data["outcome_year"].between(TRAIN_OUTCOME_START, FORECAST_END)
    ].dropna(subset=complete_features)

    q23_analysis = common.loc[common["q23_eligible"].eq(True)].copy()
    q23_analysis["target"] = q23_analysis["q23_event"].astype(int)
    q23_analysis["outcome_definition"] = "q23_year_over_year_decline"

    annual_analysis = common.loc[
        common["eligible"].astype(str).str.lower().eq("true")
    ].copy()
    annual_analysis["target"] = annual_analysis["event"].astype(int)
    annual_analysis["outcome_definition"] = "annual_max_decline_sensitivity"
    return q23_analysis, annual_analysis


def backtest(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    outcome_definition = str(data["outcome_definition"].iloc[0])
    for year in range(FORECAST_START, FORECAST_END + 1):
        train = data.loc[data["outcome_year"].between(TRAIN_OUTCOME_START, year - 1)]
        test = data.loc[data["outcome_year"].eq(year)]
        if train["target"].nunique() < 2 or test["target"].nunique() < 2:
            raise ValueError(f"Non-estimable fold {outcome_definition}, {year}")
        fold_rows.append(
            {
                "outcome_definition": outcome_definition,
                "forecast_year": year,
                "train_start": int(train["outcome_year"].min()),
                "train_end": int(train["outcome_year"].max()),
                "n_train": len(train),
                "events_train": int(train["target"].sum()),
                "n_test": len(test),
                "events_test": int(test["target"].sum()),
                "test_cells": int(test["cell_id"].nunique()),
            }
        )
        for model_name, features in FEATURE_SETS.items():
            estimator = model_pipeline()
            estimator.fit(train[features], train["target"])
            scored = test[
                ["cell_id", "region_group", "outcome_year", "target"]
            ].copy()
            scored["score"] = estimator.predict_proba(test[features])[:, 1]
            scored["model"] = model_name
            scored["outcome_definition"] = outcome_definition
            predictions.append(scored)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(fold_rows)


def year_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (outcome, model, year), group in predictions.groupby(
        ["outcome_definition", "model", "outcome_year"], sort=True
    ):
        prevalence = float(group["target"].mean())
        ap = float(average_precision_score(group["target"], group["score"]))
        ranked = group.sort_values(["score", "cell_id"], ascending=[False, True])
        row: dict[str, object] = {
            "outcome_definition": outcome,
            "model": model,
            "forecast_year": int(year),
            "n": len(group),
            "events": int(group["target"].sum()),
            "prevalence": prevalence,
            "average_precision": ap,
            "ap_lift": ap - prevalence,
        }
        for budget in [0.05, 0.10, 0.20, 0.30]:
            selected = ranked.head(max(1, int(math.ceil(budget * len(ranked)))))
            row[f"recall_top_{int(budget * 100)}pct"] = float(
                selected["target"].sum() / group["target"].sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def moving_block_bootstrap(values: pd.Series) -> np.ndarray:
    values = values.sort_index()
    years = values.index.to_numpy()
    starts = years[:-1]
    rng = np.random.default_rng(SEED)
    draws = []
    while len(draws) < BOOTSTRAP_REPLICATES:
        sampled: list[float] = []
        while len(sampled) < len(years):
            start = int(rng.choice(starts))
            sampled.extend([float(values.loc[start]), float(values.loc[start + 1])])
        draws.append(float(np.mean(sampled[: len(years)])))
    return np.asarray(draws)


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        metrics.groupby(["outcome_definition", "model"])
        .agg(
            forecast_years=("forecast_year", "nunique"),
            mean_ap=("average_precision", "mean"),
            macro_within_year_ap_lift=("ap_lift", "mean"),
            mean_recall_top_5pct=("recall_top_5pct", "mean"),
            mean_recall_top_10pct=("recall_top_10pct", "mean"),
            mean_recall_top_20pct=("recall_top_20pct", "mean"),
            mean_recall_top_30pct=("recall_top_30pct", "mean"),
        )
        .reset_index()
    )
    comparisons = [
        ("march_plus_current_crw5km", "prior_trajectory", "primary_crw_increment"),
        ("march_plus_current_winter_oisst", "prior_trajectory", "march_oisst_increment"),
        ("december_plus_prior_winter_oisst", "prior_trajectory", "december_oisst_increment"),
        ("march_plus_current_crw5km", "march_plus_current_winter_oisst", "crw_minus_oisst"),
        ("march_plus_current_crw5km_oisst", "march_plus_current_crw5km", "oisst_after_crw"),
    ]
    increment_rows: list[dict[str, object]] = []
    draw_rows: list[dict[str, object]] = []
    for outcome, outcome_metrics in metrics.groupby("outcome_definition"):
        wide = outcome_metrics.pivot(index="forecast_year", columns="model", values="ap_lift")
        recall_wide = outcome_metrics.pivot(index="forecast_year", columns="model", values="recall_top_20pct")
        for augmented, baseline, label in comparisons:
            difference = wide[augmented] - wide[baseline]
            recall_difference = recall_wide[augmented] - recall_wide[baseline]
            draws = moving_block_bootstrap(difference)
            increment_rows.append(
                {
                    "outcome_definition": outcome,
                    "comparison": label,
                    "augmented_model": augmented,
                    "baseline_model": baseline,
                    "mean_ap_lift_difference": float(difference.mean()),
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                    "positive_years": int(difference.gt(0).sum()),
                    "forecast_years": int(len(difference)),
                    "mean_top20_recall_difference": float(recall_difference.mean()),
                }
            )
            draw_rows.extend(
                {
                    "outcome_definition": outcome,
                    "comparison": label,
                    "replicate": replicate,
                    "mean_ap_lift_difference": float(value),
                }
                for replicate, value in enumerate(draws)
            )
    return summary, pd.DataFrame(increment_rows), pd.DataFrame(draw_rows)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}"
        )
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def write_summary(
    path: Path,
    model_summary: pd.DataFrame,
    increments: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    primary = increments.loc[
        (increments["outcome_definition"] == "q23_year_over_year_decline")
        & (increments["comparison"] == "primary_crw_increment")
    ].iloc[0]
    oisst = increments.loc[
        (increments["outcome_definition"] == "q23_year_over_year_decline")
        & (increments["comparison"] == "march_oisst_increment")
    ].iloc[0]
    lines = [
        "# 예측시점·CRW 5 km 실험 결과",
        "",
        "## 핵심 판정",
        "",
        (
            f"- March CRW 5 km의 과거 캐노피 이후 AP 증분: "
            f"`{primary['mean_ap_lift_difference']:.4f}` "
            f"(2-year block 95% CI `{primary['ci_low']:.4f}`–`{primary['ci_high']:.4f}`)."
        ),
        (
            f"- March OISST의 동일 증분: `{oisst['mean_ap_lift_difference']:.4f}` "
            f"(95% CI `{oisst['ci_low']:.4f}`–`{oisst['ci_high']:.4f}`)."
        ),
        "- 주 결과는 같은 165셀 후보군, 같은 forecast year, 같은 적격 행에서 비교했다.",
        "- March 결과는 Q2-Q3 canopy만 사용하므로 3월 31일 이후의 결과창과 겹치지 않는다.",
        "",
        "## 모델 요약",
        "",
        markdown_table(model_summary),
        "",
        "## 증분 비교",
        "",
        markdown_table(increments),
        "",
        "## 데이터 품질",
        "",
        markdown_table(quality),
        "",
        "## 해석 제한",
        "",
        "- CRW와 OISST는 근해 표층수온 프록시이며 현장 수온이 아니다.",
        "- CRW 특징은 월평균 3개만 사용하므로 일별 극값·지속시간 효과를 검정하지 않는다.",
        "- 이 실험은 예측시점과 제품 support의 민감도이며 인과효과 분석이 아니다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = {
        "annual_panel": args.annual_panel,
        "quarterly": args.quarterly,
        "environment": args.environment,
        "crw": args.crw,
        "config": args.config,
    }
    for name, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    annual_panel = pd.read_csv(args.annual_panel)
    quarterly = pd.read_csv(args.quarterly)
    environment = pd.read_csv(args.environment)
    crw = pd.read_csv(args.crw)
    q23_data, annual_data = prepare_analysis(annual_panel, quarterly, environment, crw)
    q23_data.to_csv(args.output_dir / "analysis_rows_q23.csv", index=False)
    annual_data.to_csv(args.output_dir / "analysis_rows_annual_sensitivity.csv", index=False)

    prediction_frames = []
    fold_frames = []
    for data in [q23_data, annual_data]:
        predictions, folds = backtest(data)
        prediction_frames.append(predictions)
        fold_frames.append(folds)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    folds = pd.concat(fold_frames, ignore_index=True)
    metrics = year_metrics(predictions)
    model_summary, increments, bootstrap_draws = summarize(metrics)

    q23_eval = q23_data.loc[q23_data["outcome_year"].between(FORECAST_START, FORECAST_END)]
    mapping = crw[["cell_id", "distance_to_crw_grid_km"]].drop_duplicates()
    quality = pd.DataFrame(
        [
            {"check": "locked_cells", "value": q23_data["cell_id"].nunique(), "expected": 165, "passed": q23_data["cell_id"].nunique() == 165},
            {"check": "forecast_years", "value": q23_eval["outcome_year"].nunique(), "expected": 20, "passed": q23_eval["outcome_year"].nunique() == 20},
            {"check": "q23_evaluation_rows", "value": len(q23_eval), "expected": ">=2000", "passed": len(q23_eval) >= 2000},
            {"check": "q23_events", "value": int(q23_eval["target"].sum()), "expected": ">=200", "passed": q23_eval["target"].sum() >= 200},
            {"check": "all_years_two_classes", "value": int(q23_eval.groupby("outcome_year")["target"].nunique().eq(2).sum()), "expected": 20, "passed": q23_eval.groupby("outcome_year")["target"].nunique().eq(2).all()},
            {"check": "crw_mapping_max_km", "value": float(mapping["distance_to_crw_grid_km"].max()), "expected": "<=15", "passed": mapping["distance_to_crw_grid_km"].le(15).all()},
            {
                "check": "same_rows_all_models_within_outcome",
                "value": int(
                    predictions.groupby(["outcome_definition", "model"])
                    .size()
                    .groupby(level=0)
                    .nunique()
                    .max()
                ),
                "expected": 1,
                "passed": bool(
                    predictions.groupby(["outcome_definition", "model"])
                    .size()
                    .groupby(level=0)
                    .nunique()
                    .eq(1)
                    .all()
                ),
            },
        ]
    )
    if not quality["passed"].all():
        raise AssertionError(quality.to_string(index=False))

    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    folds.to_csv(args.output_dir / "fold_audit.csv", index=False)
    metrics.to_csv(args.output_dir / "year_metrics.csv", index=False)
    model_summary.to_csv(args.output_dir / "model_summary.csv", index=False)
    increments.to_csv(args.output_dir / "increment_summary.csv", index=False)
    bootstrap_draws.to_csv(args.output_dir / "bootstrap_draws.csv", index=False)
    quality.to_csv(args.output_dir / "quality_checks.csv", index=False)

    manifest = {
        "experiment": "forecast_clock_crw5km_v1",
        "classification": "exploratory_monthly_composite_sensitivity",
        "git_sha": git_sha(),
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "seed": SEED,
        "forecast_years": [FORECAST_START, FORECAST_END],
        "train_outcome_start": TRAIN_OUTCOME_START,
        "bootstrap": {"type": "two_year_moving_block", "replicates": BOOTSTRAP_REPLICATES},
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in inputs.items()
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    write_summary(
        args.output_dir / "results_summary_ko.md", model_summary, increments, quality
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
