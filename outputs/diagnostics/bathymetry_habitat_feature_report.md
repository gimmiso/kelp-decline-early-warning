# Bathymetry and Habitat Feature Report

## Purpose

This report adds static GEBCO-derived bathymetry and habitat-context covariates to the retained 10 km Kelpwatch cells.
These features are intended to reduce over-reliance on canopy persistence and SST-only exposure by representing habitat suitability and exposure context.

## Data Access

- Source: GEBCO 2026 gridded bathymetry/elevation subset via `https://download.gebco.net`.
- GEBCO product page: https://www.gebco.net/data-products/gridded-bathymetry-data
- Raw GEBCO zip and NetCDF files are temporary by default and are not committed.
- Basket id: `GEBCO_16_Jun_2026_0afb3ae1b829`
- NetCDF file: `gebco_2026_n39.5393_s34.3648_w-123.8902_e-120.1676.nc`

## Feature Definitions

- GEBCO elevation is in meters relative to mean sea level.
- Ocean pixels are identified as `elevation_m < 0`.
- Positive depth is computed as `depth_m = -elevation_m` for ocean pixels.
- Coastal cells with land are summarized using valid ocean pixels only.
- `ocean_pixel_share` and `bathymetry_missing_rate` are retained for coverage diagnostics.

## Feature Summary

- Retained cells with habitat features: `50` / `50`
- Mean ocean pixel share: `0.526`
- Mean bathymetry missing rate: `0.000`
- Mean depth: `76.80` m
- Mean shallow 0-30 m share: `0.311`
- Mean shallow 0-50 m share: `0.511`

## Model Comparison

- Computed model-comparison rows: `68`

Best result per target:

| target_definition | feature_family | model | pr_auc | recall | precision | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_decline_drop | canopy_plus_oisst_plus_habitat | Logistic Regression L2 | 0.619 | 0.794 | 0.435 | 0.562 | 0.857 |
| at_risk_original_gt005 | habitat_only | Logistic Regression L2 | 0.764 | 0.556 | 0.781 | 0.649 | 0.714 |
| new_decline_transition | canopy_plus_oisst_plus_habitat | Logistic Regression L2 | 0.408 | 0.346 | 0.529 | 0.419 | 0.757 |
| original_decline | naive_persistence_baseline | current_low_canopy_score | 0.893 | 0.858 | 0.732 | 0.790 | 0.767 |

Best row by target and feature family:

| target_definition | feature_family | model | pr_auc | recall | precision | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_decline_drop | canopy_only | Logistic Regression L2 | 0.579 | 0.765 | 0.531 | 0.627 | 0.874 |
| actionable_decline_drop | canopy_plus_crw_plus_habitat | Logistic Regression L2 | 0.473 | 0.471 | 0.381 | 0.421 | 0.768 |
| actionable_decline_drop | canopy_plus_oisst_plus_habitat | Logistic Regression L2 | 0.619 | 0.794 | 0.435 | 0.562 | 0.857 |
| actionable_decline_drop | crw_composite_only | Logistic Regression L2 | 0.184 | 0.676 | 0.240 | 0.354 | 0.569 |
| actionable_decline_drop | crw_plus_habitat | Logistic Regression L2 | 0.175 | 0.735 | 0.223 | 0.342 | 0.541 |
| actionable_decline_drop | habitat_only | Random Forest | 0.171 | 1.000 | 0.170 | 0.291 | 0.471 |
| actionable_decline_drop | naive_persistence_baseline | current_low_canopy_score | 0.099 | 0.971 | 0.167 | 0.284 | 0.107 |
| actionable_decline_drop | oisst_only | Logistic Regression L2 | 0.543 | 1.000 | 0.182 | 0.308 | 0.870 |
| actionable_decline_drop | oisst_plus_habitat | Logistic Regression L2 | 0.477 | 1.000 | 0.173 | 0.294 | 0.847 |
| at_risk_original_gt005 | canopy_only | Random Forest | 0.602 | 0.578 | 0.565 | 0.571 | 0.610 |
| at_risk_original_gt005 | canopy_plus_crw_plus_habitat | Logistic Regression L2 | 0.612 | 0.311 | 0.538 | 0.394 | 0.565 |
| at_risk_original_gt005 | canopy_plus_oisst_plus_habitat | Logistic Regression L2 | 0.606 | 0.467 | 0.600 | 0.525 | 0.584 |
| at_risk_original_gt005 | crw_composite_only | Logistic Regression L2 | 0.549 | 0.333 | 0.517 | 0.405 | 0.513 |
| at_risk_original_gt005 | crw_plus_habitat | Logistic Regression L2 | 0.629 | 0.311 | 0.583 | 0.406 | 0.581 |
| at_risk_original_gt005 | habitat_only | Logistic Regression L2 | 0.764 | 0.556 | 0.781 | 0.649 | 0.714 |
| at_risk_original_gt005 | naive_persistence_baseline | current_low_canopy_score | 0.573 | 0.978 | 0.484 | 0.647 | 0.538 |
| at_risk_original_gt005 | oisst_only | Logistic Regression L2 | 0.548 | 0.311 | 0.500 | 0.384 | 0.563 |
| at_risk_original_gt005 | oisst_plus_habitat | Logistic Regression L2 | 0.632 | 0.378 | 0.708 | 0.493 | 0.632 |
| new_decline_transition | canopy_only | Logistic Regression L2 | 0.407 | 0.308 | 0.471 | 0.372 | 0.775 |
| new_decline_transition | canopy_plus_crw_plus_habitat | Logistic Regression L2 | 0.405 | 0.115 | 0.750 | 0.200 | 0.716 |
| new_decline_transition | canopy_plus_oisst_plus_habitat | Logistic Regression L2 | 0.408 | 0.346 | 0.529 | 0.419 | 0.757 |
| new_decline_transition | crw_composite_only | Logistic Regression L2 | 0.195 | 0.308 | 0.083 | 0.131 | 0.422 |
| new_decline_transition | crw_plus_habitat | Logistic Regression L2 | 0.123 | 0.308 | 0.084 | 0.132 | 0.398 |
| new_decline_transition | habitat_only | Random Forest | 0.141 | 0.538 | 0.125 | 0.203 | 0.489 |
| new_decline_transition | naive_persistence_baseline | current_low_canopy_score | 0.082 | 0.962 | 0.126 | 0.223 | 0.223 |
| new_decline_transition | oisst_only | Logistic Regression L2 | 0.148 | 0.692 | 0.159 | 0.259 | 0.556 |
| new_decline_transition | oisst_plus_habitat | Logistic Regression L2 | 0.138 | 0.385 | 0.139 | 0.204 | 0.547 |
| original_decline | canopy_only | Random Forest | 0.879 | 0.739 | 0.818 | 0.776 | 0.769 |
| original_decline | canopy_plus_crw_plus_habitat | Logistic Regression L2 | 0.880 | 0.769 | 0.811 | 0.789 | 0.777 |
| original_decline | canopy_plus_oisst_plus_habitat | Random Forest | 0.831 | 0.410 | 0.859 | 0.556 | 0.651 |
| original_decline | crw_composite_only | Logistic Regression L2 | 0.852 | 0.619 | 0.783 | 0.692 | 0.741 |
| original_decline | crw_plus_habitat | Logistic Regression L2 | 0.870 | 0.843 | 0.790 | 0.816 | 0.764 |
| original_decline | habitat_only | Random Forest | 0.889 | 0.821 | 0.809 | 0.815 | 0.800 |
| original_decline | naive_persistence_baseline | current_low_canopy_score | 0.893 | 0.858 | 0.732 | 0.790 | 0.767 |
| original_decline | oisst_only | Logistic Regression L2 | 0.714 | 0.291 | 0.684 | 0.408 | 0.520 |
| original_decline | oisst_plus_habitat | Logistic Regression L2 | 0.754 | 0.306 | 0.759 | 0.436 | 0.589 |

## Interpretation

Bathymetry and habitat features are static covariates. They are not direct biological drivers and should not be interpreted as operational early-warning signals by themselves.
Habitat-only performance indicates spatial risk-screening structure, not detection of future ecological transition. Improvement on the original broad decline target does not necessarily imply at-risk, new-transition, or actionable early-warning skill.
No future target information is used in these static features.