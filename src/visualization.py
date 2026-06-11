"""Visualization helpers for kelp canopy, SST, and model outputs."""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_canopy_time_series(df, date_col, canopy_col, site_col=None, ax=None):
    """Plot kelp canopy through time."""
    ax = ax or plt.subplots(figsize=(10, 4))[1]
    if site_col:
        sns.lineplot(data=df, x=date_col, y=canopy_col, hue=site_col, ax=ax)
    else:
        sns.lineplot(data=df, x=date_col, y=canopy_col, ax=ax)
    ax.set_title("Kelp canopy time series")
    ax.set_xlabel("Date")
    ax.set_ylabel(canopy_col)
    return ax


def plot_feature_importance(importance_df, ax=None, top_n=20):
    """Plot model feature importances."""
    ax = ax or plt.subplots(figsize=(8, 6))[1]
    data = importance_df.head(top_n)
    sns.barplot(data=data, x="importance", y="feature", ax=ax)
    ax.set_title("Feature importance")
    return ax
