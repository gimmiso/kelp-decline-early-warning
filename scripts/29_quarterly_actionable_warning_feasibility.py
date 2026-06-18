"""Assess quarterly actionable kelp warning feasibility.

This feasibility-first workflow checks whether existing Kelpwatch raw exports
contain usable quarterly observations, builds a retained-cell quarterly panel,
creates non-leaky short-horizon actionable drop labels, and runs a compact
model comparison only when train/validation/test event counts are sufficient.
"""

from __future__ import annotations

import argparse
import csv
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
FILTERED_CELLS = ROOT / "geometries" / "regular_10km_fishnet" / "filtered_cells_historic_footprint_ge500.csv"
RAW_DIR = ROOT / "data" / "raw" / "kelpwatch_aoi"
PANEL_OUTPUT = ROOT / "data" / "processed" / "quarterly_kelpwatch_panel_ge500.csv"

RESULTS_DIR = ROOT / "results" / "tables"
DIAGNOSTICS_DIR = ROOT / "outputs" / "diagnostics"
FEASIBILITY_OUTPUT = RESULTS_DIR / "quarterly_kelpwatch_feasibility_summary.csv"
LABEL_SUMMARY_OUTPUT = RESULTS_DIR / "quarterly_actionable_label_summary.csv"
FEATURE_DIAGNOSTICS_OUTPUT = RESULTS_DIR / "quarterly_actionable_feature_diagnostics.csv"
MODEL_COMPARISON_OUTPUT = RESULTS_DIR / "quarterly_actionable_model_comparison.csv"
REPORT_OUTPUT = DIAGNOSTICS_DIR / "quarterly_actionable_warning_feasibility_report.md"

REQUIRED_RAW_COLUMNS = {
    "year",
    "quarter",
    "kelp_area_m2",
    "count_cells_kelp",
    "count_cells_no_clouds",
    "count_cells_historic_footprint",
}
QUARTERS = [1, 2, 3, 4]
MODEL_START_YEAR = 1989
TRAIN_END_YEAR = 2016
VALIDATION_START_YEAR = 2017
VALIDATION_END_YEAR = 2020
TEST_START_YEAR = 2021
EPSILON = 1e-6
MIN_TRAIN_POSITIVES = 20
MIN_VALIDATION_POSITIVES = 5
MIN_TEST_POSITIVES = 5
RANDOM_STATE = 42

CSV_WRITE_KWARGS = {
    "index": False,
    "lineterminator": "\n",
    "na_rep": "",
    "float_format": "%.6f",
}

CURRENT_FEATURES = ["relative_canopy"]
TRAJECTORY_FEATURES = [
    "lag1_quarter_relative_canopy",
    "lag2_quarter_relative_canopy",
    "lag4_quarter_relative_canopy",
    "change_2quarters",
    "change_4quarters",
    "rolling_4quarter_mean",
    "rolling_4quarter_slope",
    "rolling_4quarter_cv",
    "drop_from_rolling_4quarter_max",
]


@dataclass(frozen=True)
class HorizonSpec:
    """Quarterly actionable warning horizon."""

    name: str
    target: str
    valid_column: str
    quarters_ahead: int


HORIZONS = [
    HorizonSpec("next_1quarter", "actionable_drop_next_1quarter", "label_valid_next_1quarter", 1),
    HorizonSpec("within_2quarters", "actionable_drop_next_2quarters", "label_valid_next_2quarters", 2),
    HorizonSpec("within_4quarters", "actionable_drop_next_4quarters", "label_valid_next_4quarters", 4),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Assess quarterly Kelpwatch actionable warning feasibility.")
    parser.add_argument("--filtered-cells", type=Path, default=FILTERED_CELLS)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--panel-output", type=Path, default=PANEL_OUTPUT)
    parser.add_argument("--feasibility-output", type=Path, default=FEASIBILITY_OUTPUT)
    parser.add_argument("--label-summary-output", type=Path, default=LABEL_SUMMARY_OUTPUT)
    parser.add_argument("--feature-diagnostics-output", type=Path, default=FEATURE_DIAGNOSTICS_OUTPUT)
    parser.add_argument("--model-comparison-output", type=Path, default=MODEL_COMPARISON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT)
    parser.add_argument("--current-year", type=int, default=date.today().year)
    return parser.parse_args()


def clean_csv_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten embedded line breaks for GitHub-friendly CSV display."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = (
            out[col]
            .astype("string")
            .str.replace("\r\n", " ", regex=False)
            .str.replace("\n", " ", regex=False)
            .str.replace("\r", " ", regex=False)
        )
        out[col] = out[col].where(out[col].notna(), "")
    return out


def write_portable_csv(df: pd.DataFrame, path: Path) -> None:
    """Write stable LF CSV with final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = clean_csv_cells(df).to_csv(**CSV_WRITE_KWARGS)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def expected_csv_name(cell_id: str) -> str:
    """Return raw Kelpwatch CSV name for a cell."""
    return f"kelpwatch_cell_{cell_id.replace('cell_', '')}.csv"


def read_filtered_cells(path: Path) -> pd.DataFrame:
    """Load retained GE500 cell inventory."""
    if not path.exists():
        raise FileNotFoundError(path)
    cells = pd.read_csv(path)
    if "cell_id" not in cells.columns:
        raise ValueError(f"{path} is missing cell_id.")
    return cells


def read_raw_cell(cell: pd.Series, raw_dir: Path, current_year: int) -> tuple[pd.DataFrame | None, dict[str, object]]:
    """Read one raw Kelpwatch CSV and return quarterly rows plus status."""
    cell_id = str(cell["cell_id"])
    csv_name = str(cell.get("kelpwatch_csv_file") or expected_csv_name(cell_id))
    path = raw_dir / csv_name
    status: dict[str, object] = {
        "cell_id": cell_id,
        "region_group": cell.get("region_group", ""),
        "raw_file": csv_name,
        "raw_file_exists": path.exists(),
        "has_required_columns": False,
        "raw_rows": 0,
        "quarterly_rows": 0,
        "quarter_values": "",
        "first_year": np.nan,
        "last_year": np.nan,
        "missing_quarter_rows_before_current_year": np.nan,
        "status": "missing_raw_file",
    }
    if not path.exists():
        return None, status
    raw = pd.read_csv(path)
    status["raw_rows"] = len(raw)
    missing = sorted(REQUIRED_RAW_COLUMNS - set(raw.columns))
    if missing:
        status["status"] = f"missing_columns:{','.join(missing)}"
        return None, status

    status["has_required_columns"] = True
    raw = raw.copy()
    raw["quarter_raw"] = raw["quarter"].astype(str).str.strip().str.lower()
    status["quarter_values"] = ",".join(sorted(raw["quarter_raw"].dropna().unique()))
    quarterly = raw.loc[raw["quarter_raw"].isin(["1", "2", "3", "4"])].copy()
    quarterly["quarter"] = quarterly["quarter_raw"].astype(int)
    quarterly["year"] = quarterly["year"].astype(int)
    quarterly = quarterly.loc[quarterly["year"] < current_year].copy()
    if quarterly.empty:
        status["status"] = "no_quarterly_rows_before_current_year"
        return None, status

    years = range(int(quarterly["year"].min()), int(quarterly["year"].max()) + 1)
    expected = pd.MultiIndex.from_product([years, QUARTERS], names=["year", "quarter"]).to_frame(index=False)
    observed = quarterly[["year", "quarter"]].drop_duplicates()
    missing_quarters = expected.merge(observed, on=["year", "quarter"], how="left", indicator=True)
    missing_count = int((missing_quarters["_merge"] == "left_only").sum())

    footprint_cells = pd.to_numeric(quarterly["count_cells_historic_footprint"], errors="coerce")
    kelp_area = pd.to_numeric(quarterly["kelp_area_m2"], errors="coerce")
    footprint_area = footprint_cells * 900
    quarterly["cell_id"] = cell_id
    quarterly["region_group"] = cell.get("region_group", "")
    quarterly["center_lat"] = cell.get("center_lat", np.nan)
    quarterly["center_lon"] = cell.get("center_lon", np.nan)
    quarterly["historical_footprint_area_m2"] = footprint_area
    quarterly["relative_canopy"] = kelp_area / footprint_area.replace(0, np.nan)
    quarterly["source_csv_file"] = csv_name
    quarterly = quarterly[
        [
            "cell_id",
            "region_group",
            "center_lat",
            "center_lon",
            "year",
            "quarter",
            "kelp_area_m2",
            "count_cells_kelp",
            "count_cells_no_clouds",
            "count_cells_historic_footprint",
            "historical_footprint_area_m2",
            "relative_canopy",
            "source_csv_file",
        ]
    ].sort_values(["year", "quarter"])

    status.update(
        {
            "quarterly_rows": len(quarterly),
            "first_year": int(quarterly["year"].min()),
            "last_year": int(quarterly["year"].max()),
            "missing_quarter_rows_before_current_year": missing_count,
            "status": "usable_quarterly",
        }
    )
    return quarterly, status


def build_quarterly_panel(cells: pd.DataFrame, raw_dir: Path, current_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build quarterly retained-cell panel and feasibility status rows."""
    panels: list[pd.DataFrame] = []
    statuses: list[dict[str, object]] = []
    for _, cell in cells.iterrows():
        panel, status = read_raw_cell(cell, raw_dir, current_year)
        statuses.append(status)
        if panel is not None:
            panels.append(panel)
    if not panels:
        return pd.DataFrame(), pd.DataFrame(statuses)

    out = pd.concat(panels, ignore_index=True).sort_values(["cell_id", "year", "quarter"]).reset_index(drop=True)
    min_year = int(out["year"].min())
    out["time_index"] = (out["year"].astype(int) - min_year) * 4 + out["quarter"].astype(int)
    columns = [
        "cell_id",
        "region_group",
        "center_lat",
        "center_lon",
        "year",
        "quarter",
        "time_index",
        "kelp_area_m2",
        "count_cells_kelp",
        "count_cells_no_clouds",
        "count_cells_historic_footprint",
        "relative_canopy",
    ]
    return out[columns], pd.DataFrame(statuses)


def feasibility_summary(panel: pd.DataFrame, statuses: pd.DataFrame, retained_cells: int) -> pd.DataFrame:
    """Create compact feasibility summary table."""
    rows: list[dict[str, object]] = []
    usable_statuses = statuses.loc[statuses["status"].eq("usable_quarterly")]
    rows.extend(
        [
            {"section": "overall", "metric": "retained_ge500_cells", "value": retained_cells, "notes": ""},
            {"section": "overall", "metric": "usable_quarterly_cells", "value": int(len(usable_statuses)), "notes": ""},
            {"section": "overall", "metric": "raw_files_missing", "value": int((~statuses["raw_file_exists"]).sum()), "notes": ""},
            {
                "section": "overall",
                "metric": "cells_missing_required_columns",
                "value": int((statuses["raw_file_exists"] & ~statuses["has_required_columns"]).sum()),
                "notes": "",
            },
        ]
    )
    if panel.empty:
        rows.append({"section": "overall", "metric": "quarterly_panel_rows", "value": 0, "notes": "No usable quarterly panel."})
        return pd.DataFrame(rows)

    expected_rows = panel["cell_id"].nunique() * panel["year"].nunique() * 4
    rows.extend(
        [
            {"section": "overall", "metric": "quarterly_panel_rows", "value": len(panel), "notes": ""},
            {"section": "overall", "metric": "cells_in_panel", "value": int(panel["cell_id"].nunique()), "notes": ""},
            {"section": "overall", "metric": "first_year", "value": int(panel["year"].min()), "notes": ""},
            {"section": "overall", "metric": "last_year", "value": int(panel["year"].max()), "notes": ""},
            {
                "section": "overall",
                "metric": "expected_complete_cell_year_quarter_rows",
                "value": int(expected_rows),
                "notes": "Based on cells x years x 4 quarters after excluding current year.",
            },
            {
                "section": "overall",
                "metric": "panel_completeness_rate",
                "value": len(panel) / expected_rows if expected_rows else np.nan,
                "notes": "",
            },
        ]
    )
    for quarter, group in panel.groupby("quarter"):
        rows.append(
            {
                "section": "quarter_counts",
                "metric": f"quarter_{quarter}_rows",
                "value": int(len(group)),
                "notes": f"unique_cells={group['cell_id'].nunique()}, years={group['year'].nunique()}",
            }
        )
    for status, group in statuses.groupby("status"):
        rows.append({"section": "cell_status", "metric": status, "value": int(len(group)), "notes": ""})
    missing_by_cell = statuses["missing_quarter_rows_before_current_year"].fillna(0)
    rows.append(
        {
            "section": "missingness",
            "metric": "max_missing_quarter_rows_per_cell",
            "value": int(missing_by_cell.max()) if len(missing_by_cell) else np.nan,
            "notes": "Counts expected q1-q4 rows missing between each cell's first and last raw year.",
        }
    )
    rows.append(
        {
            "section": "missingness",
            "metric": "cells_with_any_missing_quarter_rows",
            "value": int((missing_by_cell > 0).sum()),
            "notes": "",
        }
    )
    return pd.DataFrame(rows)


def slope(values: np.ndarray) -> float:
    """Return linear slope for a complete rolling window."""
    if len(values) < 2 or np.isnan(values).any():
        return np.nan
    return float(np.polyfit(np.arange(len(values)), values, 1)[0])


def add_quarterly_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add current/past-only quarterly trajectory features."""
    out = panel.sort_values(["cell_id", "time_index"]).copy()
    grouped = out.groupby("cell_id", group_keys=False)
    canopy = "relative_canopy"
    out["lag1_quarter_relative_canopy"] = grouped[canopy].shift(1)
    out["lag2_quarter_relative_canopy"] = grouped[canopy].shift(2)
    out["lag4_quarter_relative_canopy"] = grouped[canopy].shift(4)
    out["change_2quarters"] = out[canopy] - out["lag2_quarter_relative_canopy"]
    out["change_4quarters"] = out[canopy] - out["lag4_quarter_relative_canopy"]
    out["rolling_4quarter_mean"] = grouped[canopy].rolling(4, min_periods=4).mean().reset_index(level=0, drop=True)
    out["rolling_4quarter_slope"] = grouped[canopy].rolling(4, min_periods=4).apply(slope, raw=True).reset_index(level=0, drop=True)
    rolling_std = grouped[canopy].rolling(4, min_periods=4).std().reset_index(level=0, drop=True)
    out["rolling_4quarter_cv"] = rolling_std / np.maximum(out["rolling_4quarter_mean"], EPSILON)
    rolling_max = grouped[canopy].rolling(4, min_periods=1).max().reset_index(level=0, drop=True)
    out["drop_from_rolling_4quarter_max"] = (rolling_max - out[canopy]) / np.maximum(rolling_max, EPSILON)
    return out


def add_quarterly_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Add actionable labels for 1Q, 2Q, and 4Q horizons."""
    out = data.sort_values(["cell_id", "time_index"]).copy()
    grouped = out.groupby("cell_id", group_keys=False)
    canopy = "relative_canopy"
    for k in range(1, 5):
        out[f"future_canopy_q{k}"] = grouped[canopy].shift(-k)
        out[f"future_time_index_q{k}"] = grouped["time_index"].shift(-k)
        out[f"future_q{k}_continuous"] = out[f"future_time_index_q{k}"].eq(out["time_index"] + k)

    for horizon in HORIZONS:
        future_cols = [f"future_canopy_q{k}" for k in range(1, horizon.quarters_ahead + 1)]
        continuous_cols = [f"future_q{k}_continuous" for k in range(1, horizon.quarters_ahead + 1)]
        out[horizon.valid_column] = out[future_cols].notna().all(axis=1) & out[continuous_cols].all(axis=1)
        future_min = out[future_cols].min(axis=1)
        out[f"relative_drop_{horizon.name}"] = (out[canopy] - future_min) / np.maximum(out[canopy], EPSILON)
        out[horizon.target] = np.where(
            out[horizon.valid_column],
            ((out[canopy] > 0.05) & (out[f"relative_drop_{horizon.name}"] >= 0.30)).astype(int),
            np.nan,
        )
    return out


def add_splits(data: pd.DataFrame) -> pd.DataFrame:
    """Add temporal split labels."""
    out = data.copy()
    out["split"] = np.select(
        [
            out["year"].between(MODEL_START_YEAR, TRAIN_END_YEAR),
            out["year"].between(VALIDATION_START_YEAR, VALIDATION_END_YEAR),
            out["year"] >= TEST_START_YEAR,
        ],
        ["train", "validation", "test"],
        default="pre_model",
    )
    return out


def label_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize quarterly labels by horizon and split."""
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for split in ["all", "train", "validation", "test"]:
            split_df = data if split == "all" else data.loc[data["split"].eq(split)]
            valid_df = split_df.loc[split_df[horizon.valid_column]]
            labels = valid_df[horizon.target].dropna().astype(int)
            rows.append(
                {
                    "horizon": horizon.name,
                    "target": horizon.target,
                    "quarters_ahead": horizon.quarters_ahead,
                    "split": split,
                    "valid_rows": int(len(labels)),
                    "positive_count": int(labels.sum()) if len(labels) else 0,
                    "event_rate": float(labels.mean()) if len(labels) else np.nan,
                    "missing_label_rows": int((~split_df[horizon.valid_column]).sum()),
                    "year_min": int(valid_df["year"].min()) if len(valid_df) else np.nan,
                    "year_max": int(valid_df["year"].max()) if len(valid_df) else np.nan,
                    "quarter_min": int(valid_df["quarter"].min()) if len(valid_df) else np.nan,
                    "quarter_max": int(valid_df["quarter"].max()) if len(valid_df) else np.nan,
                }
            )
        test_df = data.loc[data["split"].eq("test")]
        for quarter in QUARTERS:
            quarter_df = test_df.loc[test_df["quarter"].eq(quarter)]
            valid_df = quarter_df.loc[quarter_df[horizon.valid_column]]
            labels = valid_df[horizon.target].dropna().astype(int)
            rows.append(
                {
                    "horizon": horizon.name,
                    "target": horizon.target,
                    "quarters_ahead": horizon.quarters_ahead,
                    "split": f"test_quarter_{quarter}",
                    "valid_rows": int(len(labels)),
                    "positive_count": int(labels.sum()) if len(labels) else 0,
                    "event_rate": float(labels.mean()) if len(labels) else np.nan,
                    "missing_label_rows": int((~quarter_df[horizon.valid_column]).sum()),
                    "year_min": int(valid_df["year"].min()) if len(valid_df) else np.nan,
                    "year_max": int(valid_df["year"].max()) if len(valid_df) else np.nan,
                    "quarter_min": quarter,
                    "quarter_max": quarter,
                }
            )
    return pd.DataFrame(rows)


def feature_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize quarterly feature coverage."""
    rows: list[dict[str, object]] = []
    for feature in CURRENT_FEATURES + TRAJECTORY_FEATURES:
        values = pd.to_numeric(data[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "feature_family": "quarterly_current_only" if feature in CURRENT_FEATURES else "quarterly_trajectory",
                "missingness": float(values.isna().mean()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def enough_events(label_counts: pd.DataFrame) -> dict[str, bool]:
    """Determine whether each horizon has enough events for compact modeling."""
    ok: dict[str, bool] = {}
    for horizon in HORIZONS:
        subset = label_counts.loc[label_counts["horizon"].eq(horizon.name)]
        counts = {row["split"]: int(row["positive_count"]) for _, row in subset.iterrows()}
        ok[horizon.name] = (
            counts.get("train", 0) >= MIN_TRAIN_POSITIVES
            and counts.get("validation", 0) >= MIN_VALIDATION_POSITIVES
            and counts.get("test", 0) >= MIN_TEST_POSITIVES
        )
    return ok


def feature_sets() -> dict[str, list[str]]:
    """Return compact quarterly feature families."""
    return {
        "quarterly_current_only": CURRENT_FEATURES,
        "quarterly_current_plus_trajectory": CURRENT_FEATURES + TRAJECTORY_FEATURES,
    }


def preprocessor(features: list[str], scale: bool) -> ColumnTransformer:
    """Create numeric feature preprocessing."""
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([("num", Pipeline(steps), features)], remainder="drop")


def positive_class_weight(y: pd.Series) -> float:
    """Return negative-to-positive ratio for boosted models."""
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    return negatives / positives if positives else 1.0


def model_specs(y_train: pd.Series) -> dict[str, tuple[object, bool]]:
    """Return available compact classifiers."""
    weight = positive_class_weight(y_train)
    specs: dict[str, tuple[object, bool]] = {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
            True,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=250,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                min_samples_leaf=5,
                n_jobs=1,
            ),
            False,
        ),
    }
    if XGBClassifier is not None:
        specs["XGBoost"] = (
            XGBClassifier(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                scale_pos_weight=weight,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            False,
        )
    if LGBMClassifier is not None:
        specs["LightGBM"] = (
            LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=15,
                scale_pos_weight=weight,
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=1,
            ),
            False,
        )
    return specs


def select_f2_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
    """Select validation threshold by F2."""
    precision, recall, thresholds = precision_recall_curve(y_true.astype(int), scores)
    if len(thresholds) == 0:
        return 0.5
    candidates = pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1], "threshold": thresholds})
    candidates["f2"] = (5 * candidates["precision"] * candidates["recall"]) / (
        4 * candidates["precision"] + candidates["recall"]
    ).replace(0, np.nan)
    candidates["f2"] = candidates["f2"].fillna(0)
    return float(candidates.sort_values(["f2", "recall", "precision"], ascending=[False, False, False]).iloc[0]["threshold"])


def score_metrics(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, object]:
    """Score predictions with validation-selected threshold."""
    y = y_true.astype(int)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    classes = set(y.unique())
    return {
        "pr_auc": float(average_precision_score(y, scores)) if len(classes) > 1 else np.nan,
        "roc_auc": float(roc_auc_score(y, scores)) if len(classes) > 1 else np.nan,
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "positive_count": int(y.sum()),
        "total_evaluated_rows": int(len(y)),
        "event_rate": float(y.mean()) if len(y) else np.nan,
    }


def run_models(data: pd.DataFrame, label_counts: pd.DataFrame) -> pd.DataFrame:
    """Run compact model comparison when horizon events are sufficient."""
    horizon_ok = enough_events(label_counts)
    rows: list[dict[str, object]] = []
    families = feature_sets()
    for horizon in HORIZONS:
        if not horizon_ok[horizon.name]:
            rows.append(
                {
                    "horizon": horizon.name,
                    "target": horizon.target,
                    "feature_family": "not_run",
                    "model": "not_run",
                    "status": "insufficient_events",
                }
            )
            continue
        working = data.loc[data[horizon.valid_column] & data["split"].isin(["train", "validation", "test"])].copy()
        splits = {split: working.loc[working["split"].eq(split)].copy() for split in ["train", "validation", "test"]}
        y_train = splits["train"][horizon.target].astype(int)
        for family, features in families.items():
            for model_name, (estimator, scale) in model_specs(y_train).items():
                pipeline = Pipeline([("preprocess", preprocessor(features, scale)), ("model", estimator)])
                try:
                    pipeline.fit(splits["train"][features], y_train)
                    validation_scores = pipeline.predict_proba(splits["validation"][features])[:, 1]
                    threshold = select_f2_threshold(splits["validation"][horizon.target].astype(int), validation_scores)
                    test_scores = pipeline.predict_proba(splits["test"][features])[:, 1]
                    metrics = score_metrics(splits["test"][horizon.target].astype(int), test_scores, threshold)
                    rows.append(
                        {
                            "horizon": horizon.name,
                            "target": horizon.target,
                            "quarters_ahead": horizon.quarters_ahead,
                            "feature_family": family,
                            "model": model_name,
                            "decision_threshold": threshold,
                            "train_rows": len(splits["train"]),
                            "validation_rows": len(splits["validation"]),
                            "test_year_min": int(splits["test"]["year"].min()),
                            "test_year_max": int(splits["test"]["year"].max()),
                            "status": "computed",
                            **metrics,
                        }
                    )
                except Exception as exc:  # pragma: no cover
                    rows.append(
                        {
                            "horizon": horizon.name,
                            "target": horizon.target,
                            "quarters_ahead": horizon.quarters_ahead,
                            "feature_family": family,
                            "model": model_name,
                            "status": "failed",
                            "notes": str(exc),
                        }
                    )
    return pd.DataFrame(rows)


def compact_table(df: pd.DataFrame) -> str:
    """Render Markdown table."""
    if df.empty:
        return "No rows."
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
        else:
            out[col] = out[col].astype(str)
    header = "| " + " | ".join(out.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join(row[col] for col in out.columns) + " |" for _, row in out.iterrows()]
    return "\n".join([header, divider, *rows])


def write_report(
    output: Path,
    feasibility: pd.DataFrame,
    label_counts: pd.DataFrame,
    feature_diag: pd.DataFrame,
    model_results: pd.DataFrame,
) -> None:
    """Write quarterly feasibility report."""
    usability = feasibility.loc[feasibility["metric"].isin(["usable_quarterly_cells", "retained_ge500_cells", "panel_completeness_rate"])]
    test_labels = label_counts.loc[label_counts["split"].eq("test")][
        ["horizon", "valid_rows", "positive_count", "event_rate", "missing_label_rows", "year_min", "year_max"]
    ]
    test_quarters = label_counts.loc[label_counts["split"].str.startswith("test_quarter_")][
        ["horizon", "split", "valid_rows", "positive_count", "event_rate", "year_min", "year_max"]
    ]
    computed = model_results.loc[model_results["status"].eq("computed")].copy()
    best = pd.DataFrame()
    if not computed.empty:
        best = computed.sort_values(["horizon", "pr_auc", "f2", "recall"], ascending=[True, False, False, False]).groupby("horizon").head(1)
        best = best[
            [
                "horizon",
                "feature_family",
                "model",
                "pr_auc",
                "recall",
                "precision",
                "f2",
                "false_negatives",
                "positive_count",
                "event_rate",
            ]
        ]

    lines = [
        "# Quarterly Actionable Warning Feasibility Report",
        "",
        "## Purpose",
        "",
        "The annual within-two-year experiment is a broader-horizon risk-screening check, not the desired short warning direction for the course/report framing.",
        "This workflow therefore tests whether existing Kelpwatch quarterly exports can support shorter actionable warning horizons: next quarter, within two quarters, and within four quarters.",
        "",
        "## Quarterly Data Usability",
        "",
        compact_table(usability),
        "",
        "The raw Kelpwatch files contain quarterly values `1`, `2`, `3`, `4`, plus `max`. The quarterly panel excludes the current year to avoid incomplete observations.",
        "",
        "## Label Definitions",
        "",
        "- `actionable_drop_next_1quarter`: current relative canopy > 0.05 and q to q+1 drop >= 30%.",
        "- `actionable_drop_next_2quarters`: current relative canopy > 0.05 and q to min(q+1, q+2) drop >= 30%.",
        "- `actionable_drop_next_4quarters`: current relative canopy > 0.05 and q to min(q+1, q+2, q+3, q+4) drop >= 30%.",
        "- Missing future quarters are not filled; rows lacking the required future window are excluded for that horizon.",
        "- Future canopy values are used only for labels, never predictors.",
        "",
        "## Label Counts",
        "",
        compact_table(test_labels),
        "",
        "Test event rates by current quarter:",
        "",
        compact_table(test_quarters),
        "",
        "## Quarterly Feature Diagnostics",
        "",
        compact_table(feature_diag[["feature", "feature_family", "missingness", "mean", "std"]]),
        "",
        "## Compact Model Results",
        "",
        compact_table(best),
        "",
        "## Interpretation",
        "",
        "Quarterly modeling is feasible if all retained cells have complete quarterly coverage and each horizon has enough positive events in train, validation, and test splits.",
        "Shorter quarterly horizons are closer to actionable early-warning framing than the annual two-year horizon, but they remain risk-screening diagnostics rather than operational warning claims.",
        "The test event rates vary strongly by current quarter, especially for high-canopy summer/fall quarters, so these labels likely capture seasonal canopy drawdown as well as true deterioration.",
        "A stronger quarterly early-warning design should add seasonal baselines or same-quarter year-over-year decline labels before making strong warning claims.",
        "The quarterly labels use sharper temporal resolution, so they are useful for course/report framing if performance remains meaningful without relying only on annual persistence.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run quarterly feasibility workflow."""
    args = parse_args()
    cells = read_filtered_cells(args.filtered_cells)
    panel, statuses = build_quarterly_panel(cells, args.raw_dir, args.current_year)
    feasibility = feasibility_summary(panel, statuses, len(cells))
    if panel.empty:
        label_counts = pd.DataFrame()
        feature_diag = pd.DataFrame()
        model_results = pd.DataFrame()
    else:
        panel = add_splits(add_quarterly_labels(add_quarterly_features(panel)))
        label_counts = label_summary(panel)
        feature_diag = feature_diagnostics(panel)
        model_results = run_models(panel, label_counts)
        write_portable_csv(panel, args.panel_output)

    write_portable_csv(feasibility, args.feasibility_output)
    write_portable_csv(label_counts, args.label_summary_output)
    write_portable_csv(feature_diag, args.feature_diagnostics_output)
    write_portable_csv(model_results, args.model_comparison_output)
    write_report(args.report_output, feasibility, label_counts, feature_diag, model_results)

    print("Quarterly actionable warning feasibility complete.")
    print("Feasibility:")
    print(feasibility.to_string(index=False))
    print("Label test summary:")
    if not label_counts.empty:
        print(label_counts.loc[label_counts["split"].eq("test")][["horizon", "valid_rows", "positive_count", "event_rate"]].to_string(index=False))
    print("Best computed model rows:")
    computed = model_results.loc[model_results["status"].eq("computed")].copy() if not model_results.empty else pd.DataFrame()
    if not computed.empty:
        print(
            computed.sort_values(["horizon", "pr_auc", "f2"], ascending=[True, False, False])
            .groupby("horizon")
            .head(1)[["horizon", "feature_family", "model", "pr_auc", "recall", "precision", "f2"]]
            .to_string(index=False)
        )
    print(f"Wrote feasibility summary: {args.feasibility_output}")
    print(f"Wrote label summary: {args.label_summary_output}")
    print(f"Wrote feature diagnostics: {args.feature_diagnostics_output}")
    print(f"Wrote model comparison: {args.model_comparison_output}")
    print(f"Wrote report: {args.report_output}")


if __name__ == "__main__":
    main()
