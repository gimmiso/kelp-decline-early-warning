# Quarterly Actionable Warning Feasibility Report

## Purpose

The annual within-two-year experiment is a broader-horizon risk-screening check, not the desired short warning direction for the course/report framing.
This workflow therefore tests whether existing Kelpwatch quarterly exports can support shorter actionable warning horizons: next quarter, within two quarters, and within four quarters.

## Quarterly Data Usability

| section | metric | value | notes |
| --- | --- | --- | --- |
| overall | retained_ge500_cells | 50.000 |  |
| overall | usable_quarterly_cells | 50.000 |  |
| overall | panel_completeness_rate | 1.000 |  |

The raw Kelpwatch files contain quarterly values `1`, `2`, `3`, `4`, plus `max`. The quarterly panel excludes the current year to avoid incomplete observations.

## Label Definitions

- `actionable_drop_next_1quarter`: current relative canopy > 0.05 and q to q+1 drop >= 30%.
- `actionable_drop_next_2quarters`: current relative canopy > 0.05 and q to min(q+1, q+2) drop >= 30%.
- `actionable_drop_next_4quarters`: current relative canopy > 0.05 and q to min(q+1, q+2, q+3, q+4) drop >= 30%.
- Missing future quarters are not filled; rows lacking the required future window are excluded for that horizon.
- Future canopy values are used only for labels, never predictors.

## Label Counts

| horizon | valid_rows | positive_count | event_rate | missing_label_rows | year_min | year_max |
| --- | --- | --- | --- | --- | --- | --- |
| next_1quarter | 950 | 138 | 0.145 | 50 | 2021 | 2025 |
| within_2quarters | 900 | 149 | 0.166 | 100 | 2021 | 2025 |
| within_4quarters | 800 | 147 | 0.184 | 200 | 2021 | 2024 |

Test event rates by current quarter:

| horizon | split | valid_rows | positive_count | event_rate | year_min | year_max |
| --- | --- | --- | --- | --- | --- | --- |
| next_1quarter | test_quarter_1 | 250 | 0 | 0.000 | 2021 | 2025 |
| next_1quarter | test_quarter_2 | 250 | 9 | 0.036 | 2021 | 2025 |
| next_1quarter | test_quarter_3 | 250 | 107 | 0.428 | 2021 | 2025 |
| next_1quarter | test_quarter_4 | 200 | 22 | 0.110 | 2021 | 2024 |
| within_2quarters | test_quarter_1 | 250 | 1 | 0.004 | 2021 | 2025 |
| within_2quarters | test_quarter_2 | 250 | 38 | 0.152 | 2021 | 2025 |
| within_2quarters | test_quarter_3 | 200 | 88 | 0.440 | 2021 | 2024 |
| within_2quarters | test_quarter_4 | 200 | 22 | 0.110 | 2021 | 2024 |
| within_4quarters | test_quarter_1 | 200 | 1 | 0.005 | 2021 | 2024 |
| within_4quarters | test_quarter_2 | 200 | 36 | 0.180 | 2021 | 2024 |
| within_4quarters | test_quarter_3 | 200 | 88 | 0.440 | 2021 | 2024 |
| within_4quarters | test_quarter_4 | 200 | 22 | 0.110 | 2021 | 2024 |

## Quarterly Feature Diagnostics

| feature | feature_family | missingness | mean | std |
| --- | --- | --- | --- | --- |
| relative_canopy | quarterly_current_only | 0.000 | 0.055 | 0.093 |
| lag1_quarter_relative_canopy | quarterly_trajectory | 0.006 | 0.055 | 0.093 |
| lag2_quarter_relative_canopy | quarterly_trajectory | 0.012 | 0.055 | 0.093 |
| lag4_quarter_relative_canopy | quarterly_trajectory | 0.024 | 0.056 | 0.094 |
| change_2quarters | quarterly_trajectory | 0.012 | 0.000 | 0.140 |
| change_4quarters | quarterly_trajectory | 0.024 | 0.000 | 0.081 |
| rolling_4quarter_mean | quarterly_trajectory | 0.018 | 0.056 | 0.057 |
| rolling_4quarter_slope | quarterly_trajectory | 0.018 | 0.000 | 0.043 |
| rolling_4quarter_cv | quarterly_trajectory | 0.018 | 1.205 | 0.338 |
| drop_from_rolling_4quarter_max | quarterly_trajectory | 0.000 | 0.577 | 0.415 |

## Compact Model Results

| horizon | feature_family | model | pr_auc | recall | precision | f2 | false_negatives | positive_count | event_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next_1quarter | quarterly_current_plus_trajectory | LightGBM | 0.975 | 1.000 | 0.868 | 0.970 | 0 | 138 | 0.145 |
| within_2quarters | quarterly_current_plus_trajectory | Random Forest | 0.999 | 0.993 | 0.961 | 0.987 | 1 | 149 | 0.166 |
| within_4quarters | quarterly_current_only | Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 147 | 0.184 |

## Interpretation

Quarterly modeling is feasible if all retained cells have complete quarterly coverage and each horizon has enough positive events in train, validation, and test splits.
Shorter quarterly horizons are closer to actionable early-warning framing than the annual two-year horizon, but they remain risk-screening diagnostics rather than operational warning claims.
The test event rates vary strongly by current quarter, especially for high-canopy summer/fall quarters, so these labels likely capture seasonal canopy drawdown as well as true deterioration.
A stronger quarterly early-warning design should add seasonal baselines or same-quarter year-over-year decline labels before making strong warning claims.
The quarterly labels use sharper temporal resolution, so they are useful for course/report framing if performance remains meaningful without relying only on annual persistence.
