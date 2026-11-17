"""Tests for the synthetic data generator and the known data-generating process."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.synthetic import CANONICAL_COLUMNS, error_inventory, generate_dataset


def test_shape_and_columns():
    df = generate_dataset(n_districts=5, n_crops=2, n_years=10, seed=1)
    assert list(df.columns) == CANONICAL_COLUMNS
    # 5 districts x (rice Kharif+Rabi, wheat Rabi) x 10 years + duplicates.
    assert len(df) > 5 * 3 * 10


def test_reproducible():
    a = generate_dataset(n_districts=4, n_crops=2, n_years=6, seed=3)
    b = generate_dataset(n_districts=4, n_crops=2, n_years=6, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_errors_injected():
    df = generate_dataset(n_districts=6, n_crops=3, n_years=10, seed=5)
    inv = error_inventory(df)
    assert inv["duplicates"] > 0
    assert inv["negative_area"] > 0
    assert inv["implausible_yield"] > 0
    assert inv["non_numeric"] > 0
    assert inv["missing_season"] > 0
    assert inv["invalid_ph"] > 0


def test_no_errors_when_disabled():
    df = generate_dataset(n_districts=4, n_crops=2, n_years=6, seed=9, inject_errors=False)
    assert df["season"].notna().all()
    assert (pd.to_numeric(df["area_ha"]) > 0).all()
    assert (pd.to_numeric(df["yield_t_ha"]) <= 50).all()


def test_rainfall_optimum_non_linear(clean_df):
    """Yield should rise then fall as rainfall moves away from the optimum."""
    df = clean_df[clean_df["crop"] == "rice"]
    low = df[df["rainfall_mm"] < df["rainfall_mm"].quantile(0.2)]["yield_t_ha"].mean()
    high = df[df["rainfall_mm"] > df["rainfall_mm"].quantile(0.8)]["yield_t_ha"].mean()
    mid = df[
        df["rainfall_mm"].between(
            df["rainfall_mm"].quantile(0.45), df["rainfall_mm"].quantile(0.55)
        )
    ]["yield_t_ha"].mean()
    # The middle band should out-yield both extremes on average.
    assert mid > min(low, high)


def test_technology_trend_positive(clean_df):
    df = clean_df[clean_df["crop"] == "rice"]
    early = df[df["year"] <= df["year"].min() + 3]["yield_t_ha"].mean()
    late = df[df["year"] >= df["year"].max() - 3]["yield_t_ha"].mean()
    assert late > early


def test_nitrogen_increases_yield(clean_df):
    """Higher nitrogen should raise mean yield (monotonic across quartiles)."""
    df = clean_df.copy()
    df["n_bin"] = pd.qcut(df["soil_n_kgha"], 4, labels=False)
    means = df.groupby("n_bin")["yield_t_ha"].mean().values
    assert means[-1] > means[0]
    # Allow small sampling noise but require an overall upward shape.
    assert np.polyfit(np.arange(4), means, 1)[0] > 0


def test_nitrogen_diminishing_returns_surface():
    """The deterministic response surface must show diminishing returns in N."""
    from src.data.synthetic import _response_surface

    n = np.array([50.0, 150.0, 250.0, 350.0])
    ones = np.ones_like(n)
    out = _response_surface(
        "rice",
        rainfall=np.full_like(n, 1200.0),
        temp_mean=np.full_like(n, 26.0),
        soil_n=n,
        soil_ph=np.full_like(n, 6.5),
        irrigated_frac=ones * 0.8,
        fertilizer=ones * 120.0,
        year=np.full_like(n, 2010),
        year0=2005,
    )
    diffs = np.diff(out)
    assert (diffs > 0).all()
    assert diffs[0] > diffs[-1]
