"""Decline label construction for kelp canopy time series."""

import pandas as pd


def label_decline_events(
    df,
    group_col,
    canopy_col,
    baseline_col,
    decline_threshold=0.5,
    output_col="decline_event",
):
    """Label observations where canopy falls below a fraction of baseline canopy."""
    df = df.copy()
    df[output_col] = df[canopy_col] <= (df[baseline_col] * decline_threshold)
    return df


def add_future_decline_label(
    df,
    group_col,
    decline_col="decline_event",
    horizon=1,
    output_col="future_decline",
):
    """Create a prediction target indicating decline within a future horizon."""
    df = df.copy()
    future = (
        df.groupby(group_col)[decline_col]
        .transform(lambda values: values.shift(-horizon).rolling(horizon, min_periods=1).max())
    )
    df[output_col] = future.fillna(False).astype(bool)
    return df
