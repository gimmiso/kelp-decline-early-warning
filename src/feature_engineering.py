"""Feature engineering for kelp canopy and SST early-warning indicators."""

import pandas as pd


def add_rolling_features(df, group_col, value_col, windows=(3, 6, 12)):
    """Add rolling mean features within each site or spatial unit."""
    df = df.copy()
    for window in windows:
        feature_name = f"{value_col}_rolling_mean_{window}"
        df[feature_name] = (
            df.groupby(group_col)[value_col]
            .transform(lambda values: values.rolling(window, min_periods=1).mean())
        )
    return df


def add_anomaly(df, value_col, baseline_col=None, output_col=None):
    """Add anomaly values relative to a supplied baseline or overall mean."""
    df = df.copy()
    output_col = output_col or f"{value_col}_anomaly"
    baseline = df[baseline_col] if baseline_col else df[value_col].mean()
    df[output_col] = df[value_col] - baseline
    return df


def add_lag_features(df, group_col, value_col, lags=(1, 2, 3)):
    """Add lagged versions of a time-series variable within each group."""
    df = df.copy()
    for lag in lags:
        df[f"{value_col}_lag_{lag}"] = df.groupby(group_col)[value_col].shift(lag)
    return df
