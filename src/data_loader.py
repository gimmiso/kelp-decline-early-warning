"""Data loading utilities for kelp decline early-warning analysis."""

from pathlib import Path

import pandas as pd
import xarray as xr


def load_kelpwatch_table(path):
    """Load a Kelpwatch canopy table from CSV or Parquet."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_oisst_dataset(path):
    """Load a NOAA OISST NetCDF dataset with xarray."""
    return xr.open_dataset(path)


def standardize_date_column(df, date_col="date"):
    """Return a copy with a parsed datetime column."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df
