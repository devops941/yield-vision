"""Ingestion: reading crop/weather/soil files, alias mapping and merging.

Real-world spreadsheets rarely use the canonical column names. This module maps
a wide set of aliases onto the canonical YieldVision schema, normalises season
labels and merges separately-uploaded crop, weather and soil files on the
documented keys.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

from src.config import load_aliases_config

WEATHER_COLUMNS = [
    "rainfall_mm",
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "humidity_pct",
]
SOIL_COLUMNS = [
    "soil_n_kgha",
    "soil_p_kgha",
    "soil_k_kgha",
    "soil_ph",
    "soil_oc_pct",
]
CROP_KEY = ["district", "season", "year"]


def _normalise(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[\(\)\[\]\{\}/\\]", " ", text)
    text = re.sub(r"[\s\-\.]+", "_", text)
    return text.strip("_")


def build_alias_lookup() -> dict[str, str]:
    """Return a mapping ``normalised_alias -> canonical_column``."""
    cfg = load_aliases_config()
    lookup: dict[str, str] = {}
    for canonical, aliases in cfg.get("aliases", {}).items():
        lookup[_normalise(canonical)] = canonical
        for alias in aliases or []:
            lookup[_normalise(alias)] = canonical
    return lookup


def map_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename ``df`` columns to canonical names.

    Returns the renamed frame and a mapping of ``original -> canonical`` for the
    columns that were recognised.
    """
    lookup = build_alias_lookup()
    rename: dict[str, str] = {}
    for col in df.columns:
        canonical = lookup.get(_normalise(col))
        if canonical and canonical not in rename.values():
            rename[col] = canonical
    mapped = df.rename(columns=rename)
    return mapped, rename


def normalise_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """Map alternative season spellings onto the canonical season labels."""
    if "season" not in df.columns:
        return df
    cfg = load_aliases_config()
    aliases = {k.lower(): v for k, v in cfg.get("season_aliases", {}).items()}
    df = df.copy()

    def _map(value):
        if pd.isna(value):
            return value
        key = str(value).strip().lower()
        return aliases.get(key, str(value).strip().title())

    df["season"] = df["season"].map(_map)
    return df


def read_table(source: str | Path | bytes | io.BytesIO, filename: str | None = None) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame.

    ``source`` may be a path, raw bytes or a file-like object. Excel detection is
    based on the filename extension.
    """
    name = filename or (str(source) if isinstance(source, (str, Path)) else "")
    suffix = Path(name).suffix.lower()

    if isinstance(source, (bytes, bytearray)):
        buffer: io.BytesIO | str | Path = io.BytesIO(source)
    else:
        buffer = source

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(buffer)
    if suffix in {".json"}:
        return pd.read_json(buffer)
    return pd.read_csv(buffer)


def load_crop_file(
    source: str | Path | bytes | io.BytesIO, filename: str | None = None
) -> pd.DataFrame:
    """Load a crop file and apply alias mapping and season normalisation."""
    raw = read_table(source, filename)
    mapped, _ = map_columns(raw)
    mapped = normalise_seasons(mapped)
    return mapped


def merge_sources(
    crop_df: pd.DataFrame,
    weather_df: pd.DataFrame | None = None,
    soil_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Left-join crop records with weather (district, season, year) and soil (district).

    Columns already present in ``crop_df`` are not overwritten; the merge only
    fills missing canonical weather/soil columns.
    """
    result = crop_df.copy()

    if weather_df is not None and not weather_df.empty:
        w, _ = map_columns(weather_df)
        w = normalise_seasons(w)
        keys = [k for k in CROP_KEY if k in w.columns and k in result.columns]
        if keys:
            w = w.drop_duplicates(subset=keys)
            overlap = [c for c in WEATHER_COLUMNS if c in w.columns and c not in result.columns]
            if overlap:
                result = result.merge(w[keys + overlap], on=keys, how="left")

    if soil_df is not None and not soil_df.empty:
        s, _ = map_columns(soil_df)
        keys = [k for k in ["district"] if k in s.columns and k in result.columns]
        if keys:
            s = s.drop_duplicates(subset=keys)
            overlap = [c for c in SOIL_COLUMNS if c in s.columns and c not in result.columns]
            if overlap:
                result = result.merge(s[keys + overlap], on=keys, how="left")

    return result


def canonicalise(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce canonical numeric columns to numeric dtype (errors -> NaN).

    Also derives ``yield_t_ha`` from ``production_t`` / ``area_ha`` when missing.
    """
    from src.data.synthetic import CANONICAL_COLUMNS

    df = df.copy()
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            continue
        if col in {"state", "district", "crop", "season"}:
            continue
        if col == "year":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "yield_t_ha" in df.columns:
        area = df.get("area_ha")
        production = df.get("production_t")
        if area is not None and production is not None:
            derived = production / area.replace(0, pd.NA)
            df["yield_t_ha"] = df["yield_t_ha"].fillna(derived)

    return df


def infer_missing_columns(df: pd.DataFrame) -> list[str]:
    """Return the canonical columns that are absent from ``df``."""
    from src.data.synthetic import CANONICAL_COLUMNS

    return [c for c in CANONICAL_COLUMNS if c not in df.columns]


def ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any absent canonical column so the feature pipeline can run.

    Prediction only needs the columns the model consumes. Missing numeric columns
    are filled with NaN (imputed downstream) and missing text columns with a
    placeholder, so a partial spreadsheet still predicts instead of raising
    KeyError.
    """
    import numpy as np

    from src.data.synthetic import CANONICAL_COLUMNS

    text_defaults = {"state": "unknown", "district": "unknown", "crop": "unknown", "season": "Kharif"}
    out = df.copy()
    for col in CANONICAL_COLUMNS:
        if col in out.columns:
            continue
        out[col] = text_defaults[col] if col in text_defaults else np.nan
    return out
