"""Pre-processing: unit corrections, duplicate resolution and imputation flags.

Runs after the user has reviewed validation issues and before records are
committed to the database.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.data.validate import ACRE_TO_HA, MANDATORY_KEY

DuplicatePolicy = Literal["keep_first", "keep_last", "average"]


def fix_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["area_ha", "production_t", "rainfall_mm"]:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            df[col] = numeric.where(numeric >= 0, numeric.abs())
    return df


def fix_temp_swap(df: pd.DataFrame) -> pd.DataFrame:
    if not {"temp_min_c", "temp_max_c"}.issubset(df.columns):
        return df
    df = df.copy()
    tmin = pd.to_numeric(df["temp_min_c"], errors="coerce")
    tmax = pd.to_numeric(df["temp_max_c"], errors="coerce")
    swap = tmin.notna() & tmax.notna() & (tmin > tmax)
    df.loc[swap, "temp_min_c"] = tmax[swap]
    df.loc[swap, "temp_max_c"] = tmin[swap]
    return df


def convert_area_units(df: pd.DataFrame, rows: list[int] | None = None) -> pd.DataFrame:
    """Convert area values from acres to hectares for the given rows (or all)."""
    if "area_ha" not in df.columns:
        return df
    df = df.copy()
    if rows is None:
        rows = list(df.index)
    df.loc[rows, "area_ha"] = pd.to_numeric(df.loc[rows, "area_ha"], errors="coerce") / ACRE_TO_HA
    return df


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that fail hard (row-level error) rules."""
    df = df.copy()
    if "year" in df.columns:
        year = pd.to_numeric(df["year"], errors="coerce")
        df = df[year.between(1950, pd.Timestamp.now("UTC").year) | year.isna()]
    for col in ["area_ha", "production_t", "rainfall_mm"]:
        if col in df.columns:
            df = df[~(pd.to_numeric(df[col], errors="coerce") < 0)]
    if "soil_ph" in df.columns:
        ph = pd.to_numeric(df["soil_ph"], errors="coerce")
        df = df[~(ph.notna() & ((ph < 3) | (ph > 10)))]
    return df.reset_index(drop=True)


def resolve_duplicates(df: pd.DataFrame, policy: DuplicatePolicy = "keep_first") -> pd.DataFrame:
    keys = [k for k in MANDATORY_KEY if k in df.columns]
    if len(keys) < 4:
        return df
    df = df.copy()
    if policy == "keep_first":
        return df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)
    if policy == "keep_last":
        return df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    if policy == "average":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        first_cols = [c for c in df.columns if c not in numeric_cols]
        agg = {c: "first" for c in first_cols}
        agg.update({c: "mean" for c in numeric_cols})
        return df.groupby(keys, as_index=False).agg(agg)
    raise ValueError(f"Unknown duplicate policy: {policy}")


def add_missing_indicators(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Add ``<col>_missing`` 0/1 flags for feature columns with missing values."""
    df = df.copy()
    columns = columns or [c for c in df.columns if df[c].isna().any()]
    for col in columns:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def drop_high_missingness(df: pd.DataFrame, threshold: float = 0.30) -> pd.DataFrame:
    df = df.copy()
    keep = [c for c in df.columns if df[c].isna().mean() <= threshold]
    return df[keep]


def preprocess(
    df: pd.DataFrame,
    *,
    duplicate_policy: DuplicatePolicy = "keep_first",
    drop_invalid: bool = True,
    convert_acres: bool = False,
    acre_rows: list[int] | None = None,
    add_indicators: bool = True,
) -> pd.DataFrame:
    """Full pre-processing chain used before committing to the database."""
    out = df.copy()
    out = fix_negative_values(out)
    out = fix_temp_swap(out)
    if convert_acres:
        out = convert_area_units(out, rows=acre_rows)
    if drop_invalid:
        out = drop_invalid_rows(out)
    out = resolve_duplicates(out, policy=duplicate_policy)
    if add_indicators:
        out = add_missing_indicators(out)
    return out
