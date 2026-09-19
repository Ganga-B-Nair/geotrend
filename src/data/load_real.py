"""
Load and normalise the REAL housing dataset.

PRIMARY SOURCE: Melbourne Housing Market (Kaggle)
    https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot
    (or the fuller `Melbourne_housing_FULL.csv` from the same author)

Why this dataset: it is one of the few public housing datasets carrying
real coordinates AND a real sale date AND price. Most alternatives give
you space or time, not both -- and this project needs both.

KNOWN QUIRK: the coordinate columns in the Kaggle CSV are misspelled as
`Lattitude` and `Longtitude`. This loader handles both spellings, so do
not "fix" the CSV by hand -- keeping the raw file untouched is what makes
your pipeline reproducible for anyone who re-downloads it.

The loader normalises whatever it is given to the project's internal
schema, so every downstream notebook reads the same column names:

    listing_id, transaction_date, month_index, locality,
    latitude, longitude, property_type, area_sqft, bedrooms,
    bathrooms, parking, property_age_years, dist_cbd_km, price

Usage:
    python -m src.data.load_real --raw data/raw/Melbourne_housing_FULL.csv
    python -m src.data.load_real --fixture     # offline stand-in, see below
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SQM_TO_SQFT = 10.7639

# Canonical internal schema. Downstream code imports this, never raw names.
SCHEMA = [
    "listing_id",
    "transaction_date",
    "month_index",
    "locality",
    "latitude",
    "longitude",
    "property_type",
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "parking",
    "property_age_years",
    "dist_cbd_km",
    "price",
]

# Maps Melbourne's column names -> internal schema.
# Extend this dict if you switch datasets; nothing else should need changing.
MELBOURNE_MAP = {
    "Suburb": "locality",
    "Rooms": "bedrooms",
    "Type": "property_type",
    "Price": "price",
    "Date": "transaction_date",
    "Distance": "dist_cbd_km",
    "Bathroom": "bathrooms",
    "Car": "parking",
    "BuildingArea": "area_sqm",
    "YearBuilt": "year_built",
    "Lattitude": "latitude",  # sic -- misspelled in the source file
    "Longtitude": "longitude",  # sic
    "Latitude": "latitude",  # in case a cleaned version is used
    "Longitude": "longitude",
}

TYPE_MAP = {"h": "house", "u": "unit", "t": "townhouse"}


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the CSV exactly as downloaded. No edits to the source file."""
    return pd.read_csv(path)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Melbourne schema -> internal schema. Report anything dropped."""
    out = df.rename(columns=MELBOURNE_MAP).copy()

    out["transaction_date"] = pd.to_datetime(
        out["transaction_date"], dayfirst=True, errors="coerce"
    )

    # Building area arrives in square metres; the project works in sqft.
    if "area_sqm" in out.columns:
        out["area_sqft"] = pd.to_numeric(out["area_sqm"], errors="coerce") * SQM_TO_SQFT

    # Age at time of sale, not age today -- the distinction matters for a
    # dataset spanning several years.
    if "year_built" in out.columns:
        yb = pd.to_numeric(out["year_built"], errors="coerce")
        out["property_age_years"] = out["transaction_date"].dt.year - yb
        out.loc[out["property_age_years"] < 0, "property_age_years"] = np.nan

    if "property_type" in out.columns:
        out["property_type"] = out["property_type"].map(TYPE_MAP).fillna(
            out["property_type"]
        )

    out = add_month_index(out)

    if "listing_id" not in out.columns:
        out["listing_id"] = [f"MEL{i:06d}" for i in range(len(out))]

    missing = [c for c in SCHEMA if c not in out.columns]
    for c in missing:
        out[c] = np.nan
    if missing:
        print(f"[normalise] columns absent from source, filled with NaN: {missing}")

    return out[SCHEMA]


def add_month_index(df: pd.DataFrame) -> pd.DataFrame:
    """Months elapsed since the first sale. The time axis for everything."""
    d = df["transaction_date"]
    base = d.min()
    df["month_index"] = ((d.dt.year - base.year) * 12 + (d.dt.month - base.month))
    return df


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """What you actually received. Put this table in Notebook 02."""
    rep = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_missing": df.isna().sum(),
            "pct_missing": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(),
        }
    )
    return rep.sort_values("pct_missing", ascending=False)


def temporal_coverage(df: pd.DataFrame) -> dict:
    """Does the date range support time-series work at all?

    Decomposition needs >= 24 months to separate trend from seasonality.
    DiD needs enough months either side of your chosen event dates.
    Check this BEFORE committing to the event months in config.yaml.
    """
    d = df["transaction_date"].dropna()
    n_months = int(df["month_index"].max()) + 1 if len(d) else 0
    return {
        "first_sale": str(d.min().date()) if len(d) else None,
        "last_sale": str(d.max().date()) if len(d) else None,
        "n_months": n_months,
        "sales_per_month_median": float(
            df.groupby("month_index").size().median() if len(d) else 0
        ),
        "supports_seasonal_decomposition": n_months >= 24,
    }


# --------------------------------------------------------------------------
# Offline fixture
# --------------------------------------------------------------------------

def make_fixture(n: int = 8000, seed: int = 42) -> pd.DataFrame:
    """A small stand-in with the SAME schema as the real data.

    This exists so the pipeline and tests run before you have downloaded
    anything, and so CI has something deterministic to work against.

    It is NOT the project's dataset and must never appear in your results.
    Every notebook reads data/raw/housing_real.csv; the fixture writes to
    data/raw/_fixture.csv precisely so the two can never be confused.
    """
    rng = np.random.default_rng(seed)
    subs = ["Northcote", "Brunswick", "Richmond", "Footscray", "Preston",
            "Coburg", "Yarraville", "Reservoir", "Thornbury", "Fitzroy",
            "Carlton", "Hawthorn"]
    # spread over a 4x3 grid so the stand-in has separable spatial structure
    # and enough clusters for the default 3-event / 2-control design
    centres = {
        s: (-37.86 + 0.05 * (i % 4), 144.88 + 0.06 * (i // 4))
        for i, s in enumerate(subs)
    }

    loc = rng.choice(subs, size=n)
    lat = np.array([centres[s][0] for s in loc]) + rng.normal(0, 0.008, n)
    lon = np.array([centres[s][1] for s in loc]) + rng.normal(0, 0.008, n)
    month = rng.integers(0, 36, n)
    area = np.exp(rng.normal(np.log(1400), 0.35, n)).clip(400, 6000)
    beds = np.clip(rng.poisson(2.2, n) + 1, 1, 6)
    age = np.clip(rng.gamma(2.5, 12, n), 0, 130).round(0)
    dist = np.abs(rng.normal(8, 4, n)).clip(0.5, 30)

    log_price = (
        12.05
        + 0.55 * np.log(area)
        + 0.05 * beds
        - 0.0015 * age
        - 0.02 * dist
        + 0.004 * month
        + rng.normal(0, 0.28, n)
    )

    return pd.DataFrame({
        "listing_id": [f"FIX{i:06d}" for i in range(n)],
        "transaction_date": pd.Timestamp("2016-02-01") + pd.to_timedelta(month * 30, "D"),
        "month_index": month,
        "locality": loc,
        "latitude": lat.round(6),
        "longitude": lon.round(6),
        "property_type": rng.choice(["house", "unit", "townhouse"], n, p=[.6, .28, .12]),
        "area_sqft": area.round(0),
        "bedrooms": beds,
        "bathrooms": np.clip(beds - rng.binomial(1, .5, n), 1, 5),
        "parking": rng.integers(0, 3, n),
        "property_age_years": age,
        "dist_cbd_km": dist.round(2),
        "price": np.exp(log_price).round(-3),
    })[SCHEMA]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", help="path to the downloaded Kaggle CSV")
    ap.add_argument("--out", default="data/raw/housing_real.csv")
    ap.add_argument("--fixture", action="store_true",
                    help="write an offline stand-in instead (testing only)")
    args = ap.parse_args()

    if args.fixture:
        df = make_fixture()
        out = Path("data/raw/_fixture.csv")
    else:
        if not args.raw:
            ap.error("--raw is required unless --fixture is passed")
        df = normalise(load_raw(args.raw))
        out = Path(args.out)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"wrote {len(df):,} rows -> {out}")
    print("\ntemporal coverage:")
    for k, v in temporal_coverage(df).items():
        print(f"  {k}: {v}")
    print("\nmissingness (top 6):")
    print(coverage_report(df).head(6))


if __name__ == "__main__":
    main()
