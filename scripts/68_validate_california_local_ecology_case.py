"""Independently validate the California 300-m/1-km ecological case study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


KEY = ["support_m", "site_id", "year", "model_family", "feature_set"]


def load_mur_feature_builder() -> object:
    path = Path(__file__).with_name("65_build_mur1km_site_features.py")
    spec = importlib.util.spec_from_file_location("mur_feature_builder_for_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import MUR feature builder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--mur", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_check(rows: list[dict[str, object]], check: str, passed: bool, evidence: object, severity: str = "blocker") -> None:
    rows.append(
        {
            "check": check,
            "passed": bool(passed),
            "severity_if_failed": severity,
            "evidence": str(evidence),
        }
    )


def main() -> None:
    args = parse_args()
    panel = pd.read_csv(args.panel)
    field = pd.read_csv(args.field)
    mur = pd.read_csv(args.mur)
    predictions = pd.read_csv(args.run_dir / "predictions.csv")
    ranked = pd.read_csv(args.run_dir / "ranked_predictions.csv")
    annual = pd.read_csv(args.run_dir / "annual_metrics.csv")
    increments = pd.read_csv(args.run_dir / "increment_summary.csv")
    diagnostics = pd.read_csv(args.run_dir / "prediction_diagnostics.csv")
    ecology = pd.read_csv(args.run_dir / "ecology_matched_diagnostics.csv")
    leave_cluster = pd.read_csv(args.run_dir / "leave_overlap_cluster_out_summary.csv")
    folds = pd.read_csv(args.run_dir / "fold_audit.csv")
    quality = pd.read_csv(args.run_dir / "quality_checks.csv")
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    mur_manifest = json.loads((args.run_dir / "mur_feature_manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []

    add_check(rows, "panel_keys_unique", not panel.duplicated(["support_m", "site_id", "year"]).any(), int(panel.duplicated(["support_m", "site_id", "year"]).sum()))
    add_check(rows, "field_keys_unique", not field.duplicated(["site_id", "year"]).any(), int(field.duplicated(["site_id", "year"]).sum()))
    add_check(rows, "mur_keys_unique", not mur.duplicated(["site_id", "year"]).any(), int(mur.duplicated(["site_id", "year"]).sum()))
    add_check(rows, "prediction_keys_unique", not predictions.duplicated(KEY).any(), int(predictions.duplicated(KEY).sum()))
    add_check(rows, "forward_only_expanding_folds", bool((folds["train_end"] < folds["year"]).all()), f"max_train_end_minus_test={(folds['train_end'] - folds['year']).max()}")
    add_check(rows, "at_least_four_complete_training_years", bool(((folds["train_end"] - folds["train_start"] + 1) >= 4).all()), f"minimum={(folds['train_end'] - folds['train_start'] + 1).min()}")
    add_check(rows, "scores_finite_and_bounded", bool(np.isfinite(predictions["score"]).all() and predictions["score"].between(0, 1).all()), f"missing={predictions['score'].isna().sum()}; min={predictions['score'].min():.6f}; max={predictions['score'].max():.6f}")

    counts = predictions.groupby(["support_m", "year", "model_family", "feature_set"]).size().unstack(["model_family", "feature_set"])
    add_check(rows, "exact_rows_across_models_within_support_year", bool(counts.nunique(axis=1).eq(1).all()), f"groups={len(counts)}")
    event_hashes = predictions.groupby(["support_m", "year", "model_family", "feature_set"], sort=True).apply(
        lambda frame: hashlib.sha256(frame.sort_values("site_id")[["site_id", "event"]].to_csv(index=False).encode()).hexdigest(),
        include_groups=False,
    ).unstack(["model_family", "feature_set"])
    add_check(rows, "exact_outcomes_across_models_within_support_year", bool(event_hashes.nunique(axis=1).eq(1).all()), "ordered site-event vectors match")

    rank_errors = []
    budget_errors = []
    for _, group in ranked.groupby(["support_m", "year", "model_family", "feature_set"], sort=True):
        expected_rank = group["score"].rank(method="average", pct=True)
        rank_errors.append(float((expected_rank - group["risk_percentile"]).abs().max()))
        selected_n = max(1, int(math.ceil(len(group) * 0.20)))
        expected_selected = set(group.sort_values(["score", "site_id"], ascending=[False, True]).head(selected_n)["site_id"])
        actual_selected = set(group.loc[group["selected_at_budget"].astype(bool), "site_id"])
        budget_errors.append(len(expected_selected.symmetric_difference(actual_selected)))
    add_check(rows, "risk_percentiles_recomputed", max(rank_errors, default=0) < 1e-12, f"max_error={max(rank_errors, default=0):.3g}")
    add_check(rows, "top20_selection_recomputed", max(budget_errors, default=0) == 0, f"max_symmetric_difference={max(budget_errors, default=0)}")

    recomputed = []
    for keys, group in predictions.groupby(["support_m", "model_family", "feature_set", "year"], sort=True):
        ap_lift = average_precision_score(group["event"], group["score"]) - group["event"].mean()
        recomputed.append({"support_m": keys[0], "model_family": keys[1], "feature_set": keys[2], "year": keys[3], "ap_lift_recheck": ap_lift})
    annual_check = annual.merge(pd.DataFrame(recomputed), on=["support_m", "model_family", "feature_set", "year"], validate="one_to_one")
    annual_error = float((annual_check["ap_lift"] - annual_check["ap_lift_recheck"]).abs().max())
    add_check(rows, "annual_ap_lifts_recomputed", annual_error < 1e-12, f"max_error={annual_error:.3g}")

    comparison_map = {
        "trajectory_minus_current": ("current_plus_trajectory", "current_only"),
        "mur_minus_trajectory": ("trajectory_plus_mur", "current_plus_trajectory"),
    }
    increment_errors = []
    for row in increments.itertuples(index=False):
        left, right = comparison_map[row.comparison]
        scoped = annual.loc[(annual["support_m"].eq(row.support_m)) & annual["model_family"].eq(row.model_family)]
        pivot = scoped.pivot(index="year", columns="feature_set", values="ap_lift")
        estimate = float((pivot[left] - pivot[right]).mean())
        increment_errors.append(abs(estimate - row.estimate))
    add_check(rows, "paired_increments_recomputed", max(increment_errors, default=0) < 1e-12, f"max_error={max(increment_errors, default=0):.3g}")

    diagnostic_keys = ["support_m", "site_id", "year", "model_family"]
    add_check(rows, "diagnostic_keys_unique", not diagnostics.duplicated(diagnostic_keys).any(), int(diagnostics.duplicated(diagnostic_keys).sum()))
    field_keys = field[["site_id", "year"]].drop_duplicates()
    expected_ecology = diagnostics.merge(field_keys, on=["site_id", "year"], how="inner")
    actual_keys = ecology[diagnostic_keys].sort_values(diagnostic_keys).reset_index(drop=True)
    expected_keys = expected_ecology[diagnostic_keys].sort_values(diagnostic_keys).reset_index(drop=True)
    add_check(rows, "ecology_join_keys_recomputed", actual_keys.equals(expected_keys), f"expected={len(expected_keys)}; actual={len(actual_keys)}")
    add_check(rows, "both_spatial_supports_present", sorted(ecology["support_m"].unique().tolist()) == [300, 1000], sorted(ecology["support_m"].unique().tolist()))
    cluster_table = ecology.groupby("support_m").agg(sites=("site_id", "nunique"), clusters=("overlap_cluster", "nunique"))
    add_check(rows, "overlap_not_counted_as_independent_sites", bool((cluster_table["clusters"] < cluster_table["sites"]).all()), cluster_table.to_dict(orient="index"), "claim")
    add_check(rows, "leave_overlap_cluster_out_audit_complete", len(leave_cluster) == 20 and leave_cluster["same_direction_fraction"].between(0, 1).all(), f"rows={len(leave_cluster)}")
    add_check(rows, "source_quality_checks_pass", bool(quality["passed"].all()), quality.loc[~quality["passed"], "check"].tolist())
    add_check(rows, "mur_quality_gate_passed", bool(mur_manifest["quality_passed"]), mur_manifest.get("forecast_feature_completeness"))
    add_check(rows, "mur_climatology_rule_recorded_as_past_only", bool(mur["climatology_rule"].eq("strictly prior MUR days at the same source pixel").all()), mur["climatology_rule"].drop_duplicates().tolist(), "claim")
    mur_builder = load_mur_feature_builder()
    synthetic_dates = pd.date_range("2002-06-01", "2006-12-31", freq="D")
    synthetic = pd.DataFrame(
        {
            "date": synthetic_dates,
            "sst_c": 15 + np.sin(np.arange(len(synthetic_dates)) * 2 * np.pi / 365.25),
        }
    )
    original_features = mur_builder.annual_features(synthetic, 34.0, -120.0)
    altered = synthetic.copy()
    altered.loc[altered["date"].ge("2005-01-01"), "sst_c"] += 50
    altered_features = mur_builder.annual_features(altered, 34.0, -120.0)
    thermal_columns = [column for column in mur.columns if column.endswith("_mur1km")]
    original_2004 = original_features.loc[original_features["year"].eq(2004), thermal_columns].to_numpy(dtype=float)
    altered_2004 = altered_features.loc[altered_features["year"].eq(2004), thermal_columns].to_numpy(dtype=float)
    add_check(
        rows,
        "future_sst_cannot_change_prior_year_features",
        bool(np.allclose(original_2004, altered_2004, equal_nan=True)),
        f"max_error={float(np.nanmax(np.abs(original_2004 - altered_2004))):.3g}",
    )

    input_paths = {"panel": args.panel, "field": args.field, "mur": args.mur}
    for name, path in input_paths.items():
        expected = manifest["inputs"][name]["sha256"]
        actual = sha256(path)
        add_check(rows, f"{name}_hash_matches_manifest", actual == expected, actual)
    mismatches = []
    for name, expected in manifest["output_sha256"].items():
        path = args.run_dir / name
        if not path.exists() or sha256(path) != expected:
            mismatches.append(name)
    add_check(rows, "saved_output_hashes_match_manifest", not mismatches, mismatches)

    checks = pd.DataFrame(rows)
    checks.to_csv(args.run_dir / "validation_recheck.csv", index=False)
    blockers = checks.loc[~checks["passed"] & checks["severity_if_failed"].eq("blocker")]
    assessment = "Share with caveats" if blockers.empty else "Do not share"
    report = [
        "# California 국지 사례연구 독립 검증",
        "",
        f"- 판정: **{assessment}**",
        f"- 통과: {int(checks['passed'].sum())}/{len(checks)}",
        f"- 차단 실패: {len(blockers)}",
        "",
        "이 검증은 입력 grain, 누수 없는 미래분할, 동일표본, 연도별 AP, 순위·예산 선택, 현장자료 결합, 해시를 독립 재계산한다.",
        "MUR climatology의 과거일 전용 규칙은 소스 코드와 저장된 provenance 필드로 확인했으며 원격 일자료 전체를 재다운로드하지는 않았다.",
        "",
        "## 점검표",
        "",
        checks.to_markdown(index=False),
    ]
    (args.run_dir / "validation_report_ko.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    validation_manifest = {
        "assessment": assessment,
        "checks": len(checks),
        "passed": int(checks["passed"].sum()),
        "blockers": len(blockers),
        "validation_csv_sha256": sha256(args.run_dir / "validation_recheck.csv"),
    }
    (args.run_dir / "validation_manifest.json").write_text(json.dumps(validation_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(checks.to_string(index=False))
    print(json.dumps(validation_manifest, ensure_ascii=False, indent=2))
    if not blockers.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
