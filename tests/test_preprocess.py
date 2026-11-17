"""Tests for pre-processing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.preprocess import (
    add_missing_indicators,
    convert_area_units,
    drop_invalid_rows,
    fix_negative_values,
    fix_temp_swap,
    preprocess,
    resolve_duplicates,
)


def _df(**overrides):
    base = {
        "state": ["Punjab"],
        "district": ["Ludhiana"],
        "crop": ["rice"],
        "season": ["Kharif"],
        "year": [2020],
        "area_ha": [1000.0],
        "production_t": [3500.0],
        "yield_t_ha": [3.5],
        "rainfall_mm": [900.0],
        "temp_mean_c": [27.0],
        "temp_max_c": [34.0],
        "temp_min_c": [20.0],
        "soil_ph": [6.5],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_fix_negative_values():
    df = _df(area_ha=[-1000.0], rainfall_mm=[-50.0])
    out = fix_negative_values(df)
    assert out.loc[0, "area_ha"] == 1000.0
    assert out.loc[0, "rainfall_mm"] == 50.0


def test_fix_temp_swap():
    df = _df(temp_min_c=[40.0], temp_max_c=[20.0])
    out = fix_temp_swap(df)
    assert out.loc[0, "temp_min_c"] == 20.0
    assert out.loc[0, "temp_max_c"] == 40.0


def test_convert_area_units():
    df = _df(area_ha=[2471.0])
    out = convert_area_units(df)
    assert abs(out.loc[0, "area_ha"] - 1000.0) < 1.0


def test_drop_invalid_rows():
    df = pd.concat(
        [
            _df(),
            _df(year=[1900]),
            _df(area_ha=[-5.0]),
            _df(soil_ph=[14.0]),
        ],
        ignore_index=True,
    )
    out = drop_invalid_rows(df)
    assert len(out) == 1


def test_resolve_duplicates_average():
    df = pd.concat([_df(yield_t_ha=[3.0]), _df(yield_t_ha=[5.0])], ignore_index=True)
    out = resolve_duplicates(df, policy="average")
    assert len(out) == 1
    assert abs(out.loc[0, "yield_t_ha"] - 4.0) < 1e-9


def test_resolve_duplicates_keep_first():
    df = pd.concat([_df(yield_t_ha=[3.0]), _df(yield_t_ha=[5.0])], ignore_index=True)
    out = resolve_duplicates(df, policy="keep_first")
    assert out.loc[0, "yield_t_ha"] == 3.0


def test_add_missing_indicators():
    df = _df(soil_ph=[np.nan])
    out = add_missing_indicators(df)
    assert "soil_ph_missing" in out.columns
    assert out.loc[0, "soil_ph_missing"] == 1


def test_preprocess_pipeline(synthetic_df):
    out = preprocess(synthetic_df)
    # No negatives survive and no duplicate keys remain.
    assert (pd.to_numeric(out["area_ha"]) > 0).all()
    keys = ["district", "crop", "season", "year"]
    assert not out.duplicated(subset=keys).any()
