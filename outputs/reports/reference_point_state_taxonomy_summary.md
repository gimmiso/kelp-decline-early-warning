# Reference-Point State Taxonomy Summary

## Purpose

This analysis classifies retained 10 km Kelpwatch cell-years into site-specific canopy reference states and transition types. It is an ecological state-assessment and decision-support diagnostic, not proof of fully operational early-warning skill.

## Inputs

- Modeling panel: `data/processed/modeling_dataset_ge500_noaa_v1.csv`
- Prediction file: `outputs/metadata/model_comparison_test_predictions.csv`
- Reference period: `1984-2013`
- Number of cells: `50`
- Grid-year observations: `2,050`
- Available splits: `pre_modeling, test, train, validation`

## Reference-State Definitions

- `low_below_p25`: relative canopy < cell-specific q25.
- `lower_mid_p25_p50`: q25 <= relative canopy < q50.
- `upper_mid_p50_p75`: q50 <= relative canopy < q75.
- `high_above_p75`: relative canopy >= q75.

## Transition-Type Definitions

- `persistent_low`: current and next-year state are both below p25.
- `recovery_from_low`: current state is below p25 and next-year state is not below p25.
- `new_low_transition`: current state is not below p25 and next-year state is below p25.
- `stable_non_low`: neither current nor next-year state is below p25.
- `missing_transition`: current or next-year state could not be assigned.

## Transition Counts

| transition_type | n_observations | share_of_observations |
| --- | --- | --- |
| persistent_low | 354 | 0.173 |
| recovery_from_low | 354 | 0.173 |
| new_low_transition | 342 | 0.167 |
| stable_non_low | 1000 | 0.488 |

## State-Transition Matrix Counts

| current_state | low_below_p25 | lower_mid_p25_p50 | upper_mid_p50_p75 | high_above_p75 | missing_reference_state |
| --- | --- | --- | --- | --- | --- |
| low_below_p25 | 354 | 161 | 112 | 81 | 0 |
| lower_mid_p25_p50 | 121 | 95 | 110 | 102 | 0 |
| upper_mid_p50_p75 | 120 | 80 | 90 | 130 | 0 |
| high_above_p75 | 101 | 94 | 114 | 185 | 0 |
| missing_reference_state | 0 | 0 | 0 | 0 | 0 |

## Model-Risk Linkage

For the main group `decline_event_next / canopy_only / Random Forest / test`, the highest mean predicted probability was in `persistent_low` (`0.597`).

## Annual Top-k Transition Composition

Main group: `decline_event_next / canopy_only / Random Forest / test`.

| budget_k | transition_type | selected_count | selected_share | true_decline_count | false_alert_count | actionable_low_entry_count | new_low_transition_count | persistent_low_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | persistent_low | 19 | 0.950 | 19 | 0 | 0 | 0 | 19 |
| 5 | new_low_transition | 1 | 0.050 | 1 | 0 | 0 | 1 | 0 |
| 20 | persistent_low | 65 | 0.812 | 65 | 0 | 7 | 0 | 65 |
| 20 | new_low_transition | 7 | 0.087 | 7 | 0 | 4 | 7 | 0 |
| 20 | recovery_from_low | 5 | 0.062 | 0 | 5 | 0 | 0 | 0 |
| 20 | stable_non_low | 3 | 0.037 | 0 | 3 | 0 | 0 | 0 |

## Assumptions and Warnings

- No split column was present in the modeling panel; split was inferred as pre_modeling=1984-1988, train=1989-2016, validation=2017-2020, test=2021-2024.
- q25 was reused from `baseline_p25_relative_canopy_1984_2013`; q50 and q75 were computed from 1984-2013.
- No label target column found in predictions; assigned `decline_event_next`.

## Limitations

- Reference states are retrospective labels used for evaluation and interpretation; they are not model inputs in the original workflow.
- q25/q50/q75 depend on the selected baseline period.
- Persistent low states should not be interpreted as new early-warning success.
- Sharp drops and low-state entry are related but not identical outcomes.
- The taxonomy supports cautious decision-support interpretation, not causal attribution.

## Outputs

- `outputs/tables/reference_point_state_taxonomy.csv`
- `outputs/tables/reference_point_transition_matrix.csv`
- `outputs/tables/reference_point_transition_matrix_by_year.csv`
- `outputs/tables/reference_point_transition_summary.csv`
- `outputs/tables/reference_point_transition_summary_by_year.csv`
- `outputs/tables/model_risk_by_transition_type.csv`
- `outputs/tables/topk_transition_composition.csv`
- `outputs/figures/reference_point_transition_matrix_counts.png`
- `outputs/figures/reference_point_transition_matrix_probabilities.png`
- `outputs/figures/reference_point_transition_types_by_year.png`
- `outputs/figures/reference_point_current_vs_next_canopy.png`
- `outputs/figures/model_risk_by_transition_type.png`
- `outputs/figures/topk_transition_composition.png`
