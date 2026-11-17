"""Tests for trend analytics (spec section 6.7)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.trends import (
    climate_yield_correlation,
    district_crop_trends,
    mann_kendall,
    top_bottom_regions,
    trend_for_series,
    yield_time_series,
)
from src.data.synthetic import generate_dataset


def test_mann_kendall_detects_increasing():
    years = np.arange(2000, 2020)
    values = 2.0 + 0.1 * np.arange(20) + np.linspace(-0.05, 0.05, 20)
    out = mann_kendall(years, values)
    assert out["trend"] == "increasing"
    assert out["p_value"] < 0.05
    assert 0.05 < out["sen_slope_t_ha_per_year"] < 0.15


def test_mann_kendall_detects_decreasing():
    years = np.arange(2000, 2020)
    values = 5.0 - 0.08 * np.arange(20)
    out = mann_kendall(years, values)
    assert out["trend"] == "decreasing"


def test_mann_kendall_flat():
    years = np.arange(2000, 2020)
    values = np.full(20, 3.0)
    out = mann_kendall(years, values)
    assert out["trend"] == "no trend"


def test_sen_slope_matches_linear():
    years = np.arange(2000, 2010)
    values = 1.0 + 0.2 * (years - 2000)
    out = mann_kendall(years, values)
    assert abs(out["sen_slope_t_ha_per_year"] - 0.2) < 1e-9


def test_cv_yield():
    df = pd.DataFrame({"year": [2000, 2001, 2002, 2003], "yield_t_ha": [2.0, 4.0, 2.0, 4.0]})
    out = trend_for_series(df)
    # std(ddof=1)=1.1547, mean=3 -> CV ~0.385
    assert abs(out["cv_yield"] - 0.385) < 0.01


def test_district_crop_trends_columns():
    df = generate_dataset(n_districts=4, n_crops=2, n_years=12, seed=2, inject_errors=False)
    trends = district_crop_trends(df)
    for col in ["state", "district", "crop", "trend", "sen_slope_t_ha_per_year", "cv_yield"]:
        assert col in trends.columns


def test_synthetic_technology_trend_detected():
    """Mann-Kendall should detect the injected upward technology trend."""
    df = generate_dataset(
        n_districts=8, n_crops=3, n_years=20, seed=11, inject_errors=False, start_year=2000
    )
    trends = district_crop_trends(df)
    increasing = (trends["trend"] == "increasing").mean()
    assert increasing > 0.5, f"Only {increasing:.0%} of series detected as increasing"


def test_sen_slope_matches_technology_trend():
    """Sen's slope should recover the ~2 %/yr multiplicative technology trend."""
    df = generate_dataset(
        n_districts=8, n_crops=3, n_years=20, seed=11, inject_errors=False, start_year=2000
    )
    trends = district_crop_trends(df)
    # The DGP multiplies yield by (1 + 0.02 * years_elapsed), so the absolute
    # per-year increment is ~2 % of the level. Compare within a generous band.
    truth = 0.02 * df["yield_t_ha"].mean()
    median_slope = trends["sen_slope_t_ha_per_year"].median()
    assert abs(median_slope - truth) / truth < 0.6, (
        f"median Sen slope {median_slope:.4f} far from analytic truth {truth:.4f}"
    )
    assert median_slope > 0


def test_yield_time_series_filters():
    df = generate_dataset(n_districts=4, n_crops=2, n_years=10, seed=3, inject_errors=False)
    series = yield_time_series(df, crop="rice", start=2005, end=2010)
    assert series["year"].min() >= 2005
    assert series["year"].max() <= 2010
    assert "trend_line" in series.columns


def test_climate_correlation():
    df = generate_dataset(n_districts=6, n_crops=2, n_years=15, seed=4, inject_errors=False)
    out = climate_yield_correlation(df, variable="rainfall_mm", crop="rice")
    assert out["n"] > 0
    assert -1 <= out["spearman_r"] <= 1


def test_top_bottom_regions():
    df = generate_dataset(n_districts=6, n_crops=2, n_years=10, seed=5, inject_errors=False)
    out = top_bottom_regions(df, n=3)
    assert len(out["top"]) == 3
    assert out["top"][0]["mean_yield"] >= out["top"][-1]["mean_yield"]
