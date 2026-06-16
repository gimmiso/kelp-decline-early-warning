# Naive Persistence Baseline Benchmark Report

## Purpose

This diagnostic treats simple canopy-persistence rules as official benchmark models. The goal is to test whether ML/GeoAI results provide early-warning signal beyond current or recent canopy-state persistence.

## Claim-Gating Rule

- If ML clearly outperforms naive baselines on high-canopy and transition/actionable targets, this supports early-warning signal beyond persistence.
- If ML performs well only on the full-sample original decline target, the repository should not claim operational early warning.
- If naive baselines perform similarly to or better than ML, report persistence bias clearly.

## Summary

- Naive/logistic baseline rows: `96`.
- High-canopy/subgroup performance rows: `400`.
- ML-vs-naive comparison rows: `16`.
- Any high-canopy transition/actionable claim gate passed: `False`.

### Full-Sample Original Decline

- Best naive baseline: `B_fixed_low_canopy_005_rule` with PR-AUC `0.893`.
- Best ML model: `canopy_only / Random Forest` with PR-AUC `0.897`.
- ML minus naive PR-AUC: `0.004`.

### Transition and Actionable Targets

- `C_new_decline_transition` / `current_canopy_ge_historical_p50`: best naive PR-AUC `0.610`, best ML PR-AUC `0.531`, gap `-0.078` (naive_matches_or_exceeds_ml).
- `C_new_decline_transition` / `current_canopy_ge_historical_p75`: best naive PR-AUC `0.640`, best ML PR-AUC `0.639`, gap `-0.001` (ml_similar_to_naive).
- `C_new_decline_transition` / `current_canopy_gt_0_05`: best naive PR-AUC `0.522`, best ML PR-AUC `0.496`, gap `-0.026` (naive_matches_or_exceeds_ml).
- `C_new_decline_transition` / `full_sample`: best naive PR-AUC `0.430`, best ML PR-AUC `0.401`, gap `-0.029` (naive_matches_or_exceeds_ml).
- `D_actionable_decline_drop` / `current_canopy_ge_historical_p50`: best naive PR-AUC `0.650`, best ML PR-AUC `0.691`, gap `0.041` (ml_similar_to_naive).
- `D_actionable_decline_drop` / `current_canopy_ge_historical_p75`: best naive PR-AUC `0.640`, best ML PR-AUC `0.631`, gap `-0.009` (ml_similar_to_naive).
- `D_actionable_decline_drop` / `current_canopy_gt_0_05`: best naive PR-AUC `0.563`, best ML PR-AUC `0.527`, gap `-0.037` (naive_matches_or_exceeds_ml).
- `D_actionable_decline_drop` / `full_sample`: best naive PR-AUC `0.563`, best ML PR-AUC `0.551`, gap `-0.012` (naive_matches_or_exceeds_ml).

### High-Canopy Subgroups

- High-canopy comparisons evaluated: `8`.
- High-canopy transition/actionable comparisons passing the claim gate: `0`.

## Interpretation

High full-sample performance should not be interpreted as operational early-warning skill unless the model also outperforms simple canopy-persistence baselines under at-risk, high-canopy, and transition-oriented evaluation settings.

This repository provides a persistence-aware diagnostic framework for separating apparent decline predictability from true early-warning signal. The current benchmark should be used to gate claims: strong performance on already-low or full-sample decline-state rows is useful for diagnostic screening, but it is not by itself evidence of operational early warning.
