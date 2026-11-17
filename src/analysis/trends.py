"""Trend and stability analytics (spec section 6.7).

* Mann-Kendall trend test and Sen's slope of yield (t/ha per year) per
  district-crop.
* Coefficient of variation (yield stability).
* Climate-yield Spearman correlations with significance.
* Top / bottom regions by mean yield.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

try:  # pymannkendall is optional at import time but listed in requirements
    import pymannkendall as mk
except Exception:  # pragma: no cover
    mk = None

CLIMATE_VARIABLES = [
    "rainfall_mm",
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "humidity_pct",
]


def _sen_slope(years: np.ndarray, values: np.ndarray) -> float:
    """Sen's slope estimator (median of pairwise slopes)."""
    n = len(values)
    if n < 2:
        return float("nan")
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = years[j] - years[i]
            if dx != 0:
                slopes.append((values[j] - values[i]) / dx)
    return float(np.median(slopes)) if slopes else float("nan")


def mann_kendall(years: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    """Return trend, p-value, Kendall tau and Sen slope for a series."""
    mask = ~(np.isnan(values) | np.isnan(years))
    years = np.asarray(years)[mask]
    values = np.asarray(values)[mask]
    if len(values) < 4:
        return {
            "trend": "insufficient_data",
            "p_value": None,
            "tau": None,
            "sen_slope_t_ha_per_year": None,
        }
    if mk is not None:
        result = mk.original_test(values)
        trend = result.trend
        p_value = float(result.p)
        tau = float(result.Tau)
    else:  # pragma: no cover - fallback if pymannkendall is unavailable
        tau_stat, p_value = stats.kendalltau(years, values)
        tau = float(tau_stat)
        if p_value < 0.05:
            trend = "increasing" if tau > 0 else "decreasing"
        else:
            trend = "no trend"
    return {
        "trend": trend,
        "p_value": p_value,
        "tau": tau,
        "sen_slope_t_ha_per_year": _sen_slope(years, values),
    }


def trend_for_series(df: pd.DataFrame) -> dict[str, Any]:
    """Trend statistics for one district-crop time series."""
    ordered = df.sort_values("year")
    years = ordered["year"].to_numpy(dtype=float)
    values = ordered["yield_t_ha"].to_numpy(dtype=float)
    out = mann_kendall(years, values)
    mean = np.nanmean(values) if len(values) else float("nan")
    std = np.nanstd(values, ddof=1) if len(values) > 1 else float("nan")
    out["cv_yield"] = float(std / mean) if mean and not np.isnan(mean) else None
    out["mean_yield"] = float(mean) if not np.isnan(mean) else None
    out["n_years"] = int(len(values))
    return out


def district_crop_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trend statistics for every district-crop pair."""
    rows = []
    for (state, district, crop), group in df.groupby(
        ["state", "district", "crop"], dropna=False
    ):
        stats_row = trend_for_series(group)
        rows.append(
            {
                "state": state,
                "district": district,
                "crop": crop,
                **stats_row,
            }
        )
    return pd.DataFrame(rows)


def yield_time_series(
    df: pd.DataFrame,
    crop: str | None = None,
    district: str | None = None,
    start: int | None = None,
    end: int | None = None,
) -> pd.DataFrame:
    """Aggregate mean yield by year with filters applied."""
    sub = df.copy()
    if crop:
        sub = sub[sub["crop"] == crop]
    if district:
        sub = sub[sub["district"] == district]
    if start is not None:
        sub = sub[sub["year"] >= start]
    if end is not None:
        sub = sub[sub["year"] <= end]
    grouped = sub.groupby("year", as_index=False).agg(
        mean_yield=("yield_t_ha", "mean"),
        n=("yield_t_ha", "size"),
    )
    grouped["trend_line"] = np.nan
    if len(grouped) >= 2:
        slope, intercept = np.polyfit(grouped["year"], grouped["mean_yield"], 1)
        grouped["trend_line"] = intercept + slope * grouped["year"]
    return grouped


def climate_yield_correlation(
    df: pd.DataFrame, variable: str = "rainfall_mm", crop: str | None = None
) -> dict[str, Any]:
    """Spearman correlation between a climate variable and yield."""
    sub = df.copy()
    if crop:
        sub = sub[sub["crop"] == crop]
    if variable not in sub.columns:
        return {"variable": variable, "spearman_r": None, "p_value": None, "n": 0}
    sub = sub.dropna(subset=[variable, "yield_t_ha"])
    if len(sub) < 3:
        return {"variable": variable, "spearman_r": None, "p_value": None, "n": len(sub)}
    r, p = stats.spearmanr(sub[variable], sub["yield_t_ha"])
    return {
        "variable": variable,
        "spearman_r": float(r),
        "p_value": float(p),
        "n": int(len(sub)),
        "significant": bool(p < 0.05),
    }


def climate_yield_scatter(
    df: pd.DataFrame, variable: str = "rainfall_mm", crop: str | None = None
) -> pd.DataFrame:
    sub = df.copy()
    if crop:
        sub = sub[sub["crop"] == crop]
    sub = sub.dropna(subset=[variable, "yield_t_ha"])
    return sub[[variable, "yield_t_ha", "district", "crop", "year"]]


def top_bottom_regions(
    df: pd.DataFrame, crop: str | None = None, n: int = 5
) -> dict[str, list[dict[str, Any]]]:
    """Regions with the highest and lowest mean yield."""
    sub = df.copy()
    if crop:
        sub = sub[sub["crop"] == crop]
    grouped = (
        sub.groupby(["state", "district"], as_index=False)
        .agg(mean_yield=("yield_t_ha", "mean"), n=("yield_t_ha", "size"))
        .sort_values("mean_yield", ascending=False)
    )
    records = grouped.to_dict(orient="records")
    return {"top": records[:n], "bottom": records[-n:][::-1] if records else []}


def crop_comparison(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("crop", as_index=False)
        .agg(
            mean_yield=("yield_t_ha", "mean"),
            median_yield=("yield_t_ha", "median"),
            std_yield=("yield_t_ha", "std"),
            n=("yield_t_ha", "size"),
        )
        .sort_values("mean_yield", ascending=False)
    )


def persist_trends(session, trends: pd.DataFrame, region_lookup: dict, crop_lookup: dict) -> int:
    """Persist trend results into the ``trend_results`` table."""
    from src.data.db import TrendResult

    count = 0
    for _, row in trends.iterrows():
        region_id = region_lookup.get((row["state"], row["district"]))
        crop_id = crop_lookup.get(row["crop"])
        session.add(
            TrendResult(
                region_id=region_id,
                crop_id=crop_id,
                trend=row.get("trend"),
                p_value=_none_if_nan(row.get("p_value")),
                tau=_none_if_nan(row.get("tau")),
                sen_slope_t_ha_per_year=_none_if_nan(row.get("sen_slope_t_ha_per_year")),
                cv_yield=_none_if_nan(row.get("cv_yield")),
            )
        )
        count += 1
    session.flush()
    return count


def _none_if_nan(value):
    if value is None:
        return None
    try:
        return None if np.isnan(value) else float(value)
    except (TypeError, ValueError):
        return None
