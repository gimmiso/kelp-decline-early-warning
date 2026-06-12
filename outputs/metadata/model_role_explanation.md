# Model Role Explanation

## Why Multiple Models Were Compared

Multiple models were compared not as a leaderboard exercise, but to test whether kelp decline prediction was better explained by linear relationships, nonlinear decision boundaries, tree-based interactions, or environmental feature augmentation.

## Logistic Regression

Role: Transparent linear baseline.

Purpose: Tests whether canopy and environmental predictors separate decline and non-decline rows through mostly linear additive effects.

Interpretation: If Logistic Regression performs competitively, the decline signal may be relatively simple and linearly separable. If nonlinear models outperform it, this suggests nonlinearities or interactions.

## SVM

Role: Nonlinear margin-based classifier.

Purpose: Tests whether a flexible decision boundary improves early-warning recall in a relatively small cell-year dataset.

Interpretation: Useful for checking whether NOAA environmental variables help separate difficult decline cases, especially under recall and false-negative metrics.

## Random Forest

Role: Robust nonlinear tree-based benchmark.

Purpose: Captures nonlinear feature interactions without requiring strong parametric assumptions.

Interpretation: Useful for tabular ecological data and compatible with SHAP TreeExplainer. In this project, the canopy-only Random Forest achieved the best aggregate test PR-AUC.

## XGBoost

Role: Gradient-boosted tree model.

Purpose: Tests whether sequential boosting improves predictive performance beyond Random Forest by correcting earlier errors and capturing complex interactions.

Interpretation: Useful as a strong tabular-data benchmark, but should be compared against simpler baselines to avoid overclaiming complexity.

## LightGBM

Role: Efficient gradient-boosted tree model.

Purpose: Tests whether a faster boosting implementation improves performance on the same feature sets.

Interpretation: Provides an additional boosted-tree benchmark and helps check whether results are robust across tree-ensemble methods.

## Feature-Set Comparison Logic

The main comparison is not only across algorithms, but also within each algorithm:

- `canopy_only` shows the predictive value of direct biological state monitoring.
- `oisst_only` shows the predictive value of thermal exposure variables alone.
- `canopy_noaa` tests whether adding NOAA environmental context improves or changes model behavior.

Within-model comparisons are important because they isolate the effect of feature-set changes while holding the algorithm fixed.
