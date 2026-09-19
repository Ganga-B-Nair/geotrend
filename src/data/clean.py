"""Data quality profiling + cleaning of the REAL dataset.

Report every decision here in the 'Data Understanding & Quality' section --
this is graded DS work, not plumbing.
"""
from __future__ import annotations
import pandas as pd


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missingness, dtype, cardinality, outlier share."""
    raise NotImplementedError


def flag_unit_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Detect area values likely entered in sq-metres (ratio ~10.764)."""
    raise NotImplementedError


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame: ...
def handle_missing(df: pd.DataFrame) -> pd.DataFrame: ...
def winsorize_price(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame: ...
def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame: ...


# Melbourne-specific notes:
#   - BuildingArea and YearBuilt are missing for a large share of rows. Decide
#     between dropping, imputing, or modelling without them -- and justify it.
#     Dropping every row missing BuildingArea can discard ~half the dataset.
#   - Price itself is missing in Melbourne_housing_FULL.csv for unsold
#     listings. Those rows cannot be used as target observations.
#   - Rows missing coordinates cannot be clustered; drop them and report how
#     many, since this may bias coverage geographically.
