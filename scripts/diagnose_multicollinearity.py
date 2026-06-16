"""Diagnose multicollinearity among Version 1 modeling features.

This diagnostic is intended for interpretation quality control. It evaluates
pairwise Pearson correlations, variance inflation factors (VIF), and condition
numbers for the feature sets used in the main model comparison. Raw and
processed data are read locally but are not modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from train_model_comparison import INPUT_DATASET, feature_sets, load_dataset, main_subset


CORRELATION_OUTPUT = Path("outputs/diagnostics/multicollinearity_correlation_matrix.csv")
HIGH_CORRELATION_OUTPUT = Path("outputs/diagnostics/multicollinearity_high_correlation_pairs.csv")
VIF_OUTPUT = Path("outputs/diagnostics/multicollinearity_vif_table.csv")
REPORT_OUTPUT = Path("outputs/diagnostics/multicollinearity_report.md")
HEATMAP_OUTPUT = Path("outputs/figures/multicollinearity_correlation_heatmap.png")

HIGH_CORRELATION_THRESHOLD = 0.80
MODERATE_VIF_THRESHOLD = 5.0
HIGH_VIF_THRESHOLD = 10.0
MAX_VIF_R2 = 0.999999


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Diagnose multicollinearity in V1 model features.")
    parser.add_argument("--input", type=Path, default=INPUT_DATASET)
    parser.add_argument("--correlation-output", type=Path, default=CORRELATION_OUTPUT)
    parser.add_argument("--high-correlation-output", type=Path, default=HIGH_CORRELATION_OUTPUT)
    parser.add_argument("--vif-output", type=Path, default=VIF_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT)
    parser.add_argument("--heatmap-output", type=Path, default=HEATMAP_OUTPUT)
    parser.add_argument("--correlation-threshold", type=float, default=HIGH_CORRELATION_THRESHOLD)
    return parser.parse_args()


def numeric_features(data: pd.DataFrame, features: list[str]) -> list[str]:
    """Return numeric features that have nonzero variation."""
    usable = []
    for feature in features:
        if feature not in data.columns or not is_numeric_dtype(data[feature]):
            continue
        values = pd.to_numeric(data[feature], errors="coerce")
        if values.notna().sum() < 3:
            continue
        if values.nunique(dropna=True) <= 1:
            continue
        usable.append(feature)
    return usable


def feature_memberships(sets: dict[str, list[str]]) -> dict[str, str]:
    """Map each feature to the model feature sets where it appears."""
    memberships: dict[str, list[str]] = {}
    for set_name, features in sets.items():
        for feature in features:
            memberships.setdefault(feature, []).append(set_name)
    return {feature: ",".join(names) for feature, names in memberships.items()}


def correlation_matrix(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compute Pearson correlation matrix for numeric model features."""
    return data[features].apply(pd.to_numeric, errors="coerce").corr(method="pearson")


def high_correlation_pairs(
    corr: pd.DataFrame,
    memberships: dict[str, str],
    threshold: float,
) -> pd.DataFrame:
    """Return feature pairs with absolute correlation above the threshold."""
    rows = []
    features = list(corr.columns)
    for i, left in enumerate(features):
        for right in features[i + 1 :]:
            value = corr.loc[left, right]
            if pd.isna(value) or abs(value) < threshold:
                continue
            rows.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "correlation": float(value),
                    "abs_correlation": float(abs(value)),
                    "feature_1_sets": memberships.get(left, ""),
                    "feature_2_sets": memberships.get(right, ""),
                }
            )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def prepared_matrix(data: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Impute and standardize a numeric feature matrix."""
    values = data[features].apply(pd.to_numeric, errors="coerce")
    imputed = SimpleImputer(strategy="median").fit_transform(values)
    return StandardScaler().fit_transform(imputed)


def condition_number(matrix: np.ndarray) -> float:
    """Compute condition number from singular values."""
    if matrix.shape[1] < 2:
        return np.nan
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    smallest = singular_values[-1]
    if smallest <= 1e-12:
        return float("inf")
    return float(singular_values[0] / smallest)


def vif_table(data: pd.DataFrame, sets: dict[str, list[str]]) -> pd.DataFrame:
    """Compute VIF values for each numeric feature set."""
    rows = []
    for set_name, features in sets.items():
        numeric = numeric_features(data, features)
        if len(numeric) < 2:
            continue
        matrix = prepared_matrix(data, numeric)
        cond = condition_number(matrix)
        for index, feature in enumerate(numeric):
            y = matrix[:, index]
            x = np.delete(matrix, index, axis=1)
            model = LinearRegression()
            model.fit(x, y)
            r_squared = float(model.score(x, y))
            vif = float("inf") if r_squared >= MAX_VIF_R2 else float(1.0 / (1.0 - r_squared))
            rows.append(
                {
                    "feature_set": set_name,
                    "feature": feature,
                    "vif": vif,
                    "r_squared_with_other_features": r_squared,
                    "condition_number_feature_set": cond,
                    "n_features_in_vif_model": len(numeric),
                    "n_observations": len(data),
                    "missing_values": int(data[feature].isna().sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature_set", "vif"], ascending=[True, False]).reset_index(drop=True)


def plot_correlation_heatmap(corr: pd.DataFrame, output: Path) -> None:
    """Plot the model-feature correlation heatmap."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig_width = max(10, len(corr.columns) * 0.45)
    fig_height = max(8, len(corr.columns) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("Pairwise correlation among numeric model features")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def format_table(frame: pd.DataFrame, columns: list[str], n: int) -> str:
    """Render a compact markdown table without requiring optional tabulate."""
    subset = frame.loc[:, columns].head(n).copy()
    if subset.empty:
        return "_No rows._"
    for column in subset.columns:
        if pd.api.types.is_numeric_dtype(subset[column]):
            subset[column] = subset[column].map(lambda value: "inf" if np.isinf(value) else f"{value:.3f}")
    header = "| " + " | ".join(subset.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(subset.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in subset.to_numpy()]
    return "\n".join([header, divider, *rows])


def write_report(
    output: Path,
    data: pd.DataFrame,
    numeric: list[str],
    high_pairs: pd.DataFrame,
    vif: pd.DataFrame,
    heatmap_output: Path,
) -> None:
    """Write a concise multicollinearity interpretation report."""
    high_vif = vif.loc[vif["vif"] >= HIGH_VIF_THRESHOLD]
    moderate_vif = vif.loc[(vif["vif"] >= MODERATE_VIF_THRESHOLD) & (vif["vif"] < HIGH_VIF_THRESHOLD)]
    condition_summary = (
        vif[["feature_set", "condition_number_feature_set"]]
        .drop_duplicates()
        .sort_values("condition_number_feature_set", ascending=False)
    )
    lines = [
        "# Multicollinearity Diagnostic Report",
        "",
        "## Purpose",
        "",
        "This diagnostic checks whether the Version 1 model features contain strong pairwise or multivariate redundancy. It is intended to support cautious interpretation of linear coefficients, SHAP summaries, and feature-importance narratives. It does not change the modeling dataset or retrain the main models.",
        "",
        "## Data and Feature Scope",
        "",
        f"- Modeling rows evaluated: `{len(data)}`.",
        f"- Numeric model features evaluated in the combined correlation matrix: `{len(numeric)}`.",
        f"- High-correlation threshold: `abs(r) >= {HIGH_CORRELATION_THRESHOLD:.2f}`.",
        f"- VIF caution thresholds: moderate `>= {MODERATE_VIF_THRESHOLD:.1f}`, high `>= {HIGH_VIF_THRESHOLD:.1f}`.",
        f"- Heatmap: `{heatmap_output}`.",
        "",
        "Categorical region variables are not included in the numeric correlation matrix. The VIF table is calculated separately for each model feature set using numeric predictors only.",
        "",
        "## Main Findings",
        "",
        f"- High-correlation feature pairs: `{len(high_pairs)}`.",
        f"- High-VIF rows: `{len(high_vif)}`.",
        f"- Moderate-VIF rows: `{len(moderate_vif)}`.",
        "",
        "### Top High-Correlation Pairs",
        "",
        format_table(high_pairs, ["feature_1", "feature_2", "correlation", "abs_correlation"], 12),
        "",
        "### Top VIF Values",
        "",
        format_table(vif, ["feature_set", "feature", "vif", "r_squared_with_other_features"], 15),
        "",
        "### Condition Numbers by Feature Set",
        "",
        format_table(condition_summary, ["feature_set", "condition_number_feature_set"], 10),
        "",
        "## Interpretation",
        "",
        "Several canopy-size variables are structurally related, especially `kelp_area_m2`, `count_cells_kelp`, and `relative_canopy`. Several NOAA thermal variables are also expected to be correlated because annual mean, maximum, anomaly, and hot-day metrics are derived from the same OISST time series. CUTI and BEUTI seasonal and anomaly summaries can likewise be redundant within a small spatial domain.",
        "",
        "This does not invalidate the tree-based screening models, but it means coefficient-level interpretation from Logistic Regression and feature-level importance narratives should be treated cautiously. For paper framing, the safer interpretation is feature-group-level evidence: canopy state, OISST thermal exposure, CUTI upwelling proxy, and BEUTI nitrate-flux proxy, rather than isolated claims about one highly correlated predictor.",
        "",
        "## Recommended Use",
        "",
        "- Keep Random Forest, XGBoost, and LightGBM as prediction benchmarks because tree models can tolerate correlated predictors, while still sharing importance across redundant variables.",
        "- Use Logistic Regression mainly as a transparent baseline, not as definitive evidence about individual variable effects when VIF is high.",
        "- Prefer grouped SHAP or grouped feature-set ablation over single-feature causal wording.",
        "- Consider a reduced predictor set in sensitivity analysis, selecting one representative from each highly correlated group.",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the multicollinearity diagnostic."""
    args = parse_args()
    data = main_subset(load_dataset(args.input))
    sets = feature_sets(data)
    memberships = feature_memberships(sets)
    combined_features = sorted({feature for features in sets.values() for feature in features})
    combined_numeric = numeric_features(data, combined_features)

    corr = correlation_matrix(data, combined_numeric)
    pairs = high_correlation_pairs(corr, memberships, args.correlation_threshold)
    vif = vif_table(data, sets)

    args.correlation_output.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(args.correlation_output)
    pairs.to_csv(args.high_correlation_output, index=False)
    vif.to_csv(args.vif_output, index=False)
    plot_correlation_heatmap(corr, args.heatmap_output)
    write_report(args.report_output, data, combined_numeric, pairs, vif, args.heatmap_output)

    print(f"Wrote correlation matrix: {args.correlation_output}")
    print(f"Wrote high-correlation pairs: {args.high_correlation_output}")
    print(f"Wrote VIF table: {args.vif_output}")
    print(f"Wrote report: {args.report_output}")
    print(f"Wrote heatmap: {args.heatmap_output}")


if __name__ == "__main__":
    main()
