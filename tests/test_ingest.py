"""Tests for ingestion: alias mapping, season normalisation, merging."""

from __future__ import annotations

import io

import pandas as pd

from src.data.ingest import (
    canonicalise,
    load_crop_file,
    map_columns,
    merge_sources,
    normalise_seasons,
    read_table,
)


def test_alias_mapping():
    df = pd.DataFrame(
        {
            "State Name": ["Punjab"],
            "Dist": ["Ludhiana"],
            "Commodity": ["rice"],
            "Crop Season": ["Kharif"],
            "Yr": [2020],
            "Rainfall (mm)": [900],
            "Mean Temp": [27.0],
        }
    )
    mapped, rename = map_columns(df)
    assert "state" in mapped.columns
    assert "district" in mapped.columns
    assert "crop" in mapped.columns
    assert "season" in mapped.columns
    assert "year" in mapped.columns
    assert "rainfall_mm" in mapped.columns
    assert "temp_mean_c" in mapped.columns


def test_season_normalisation():
    df = pd.DataFrame({"season": ["kharif season", "WINTER", "zaid", "Rabi"]})
    out = normalise_seasons(df)
    assert out["season"].tolist() == ["Kharif", "Rabi", "Summer", "Rabi"]


def test_read_csv_bytes():
    csv = b"district,crop,season,year,rainfall_mm,temp_mean_c\nA,rice,Kharif,2020,900,27\n"
    df = read_table(csv, "x.csv")
    assert len(df) == 1
    assert df.loc[0, "district"] == "A"


def test_load_crop_file_end_to_end():
    csv = (
        b"State,District,Crop,Season,Year,Rainfall,Temp\n"
        b"Punjab,Ludhiana,Rice,kharif,2020,900,27\n"
    )
    df = load_crop_file(csv, "crop.csv")
    assert df.loc[0, "season"] == "Kharif"
    assert df.loc[0, "crop"] == "Rice"


def test_merge_sources():
    crop = pd.DataFrame(
        {
            "district": ["A", "A"],
            "season": ["Kharif", "Rabi"],
            "year": [2020, 2020],
            "crop": ["rice", "wheat"],
            "yield_t_ha": [3.5, 4.0],
        }
    )
    weather = pd.DataFrame(
        {
            "district": ["A", "A"],
            "season": ["Kharif", "Rabi"],
            "year": [2020, 2020],
            "rainfall_mm": [900, 200],
        }
    )
    soil = pd.DataFrame({"district": ["A"], "soil_ph": [6.5], "soil_n_kgha": [250]})
    merged = merge_sources(crop, weather, soil)
    assert "rainfall_mm" in merged.columns
    assert merged["rainfall_mm"].tolist() == [900, 200]
    assert "soil_ph" in merged.columns


def test_canonicalise_derives_yield():
    df = pd.DataFrame(
        {"area_ha": [100.0], "production_t": [350.0], "yield_t_ha": [None], "year": ["2020"]}
    )
    out = canonicalise(df)
    assert abs(out.loc[0, "yield_t_ha"] - 3.5) < 1e-9
    assert out.loc[0, "year"] == 2020
