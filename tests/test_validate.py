"""Tests for the validation rules (spec section 6.1) - one test per rule."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.validate import (
    check_crop_yield_range,
    check_duplicates,
    check_impossible_values,
    check_invalid_year,
    check_mandatory_columns,
    check_missingness,
    check_non_numeric,
    check_units,
    check_unknown_categories,
    check_yield_consistency,
    validate,
)


def _base_row(**overrides):
    row = {
        "state": "Punjab",
        "district": "Ludhiana",
        "crop": "rice",
        "season": "Kharif",
        "year": 2020,
        "area_ha": 1000.0,
        "production_t": 3500.0,
        "yield_t_ha": 3.5,
        "rainfall_mm": 900.0,
        "temp_mean_c": 27.0,
        "temp_max_c": 34.0,
        "temp_min_c": 20.0,
        "humidity_pct": 70.0,
        "soil_n_kgha": 250.0,
        "soil_p_kgha": 20.0,
        "soil_k_kgha": 200.0,
        "soil_ph": 6.5,
        "soil_oc_pct": 0.5,
        "fertilizer_kgha": 120.0,
        "irrigated_frac": 0.8,
        "pesticide_kgha": 2.0,
    }
    row.update(overrides)
    return row


def test_missing_mandatory_column():
    df = pd.DataFrame([_base_row()]).drop(columns=["rainfall_mm"])
    missing, issues = check_mandatory_columns(df)
    assert "rainfall_mm" in missing
    assert any(i.check == "missing_mandatory_column" for i in issues)


def test_non_numeric_detected():
    df = pd.DataFrame([_base_row(rainfall_mm="NA"), _base_row()])
    issues = check_non_numeric(df)
    assert len(issues) == 1
    assert issues[0].row_index == 0
    assert issues[0].severity == "warning"


def test_invalid_year():
    df = pd.DataFrame([_base_row(year=1900), _base_row(year=2020)])
    issues = check_invalid_year(df)
    assert len(issues) == 1
    assert issues[0].severity == "error"


def test_negative_area_is_error():
    df = pd.DataFrame([_base_row(area_ha=-500.0)])
    issues = check_impossible_values(df)
    assert any(i.column == "area_ha" and i.severity == "error" for i in issues)


def test_temp_min_greater_than_max():
    df = pd.DataFrame([_base_row(temp_min_c=40.0, temp_max_c=20.0)])
    issues = check_impossible_values(df)
    assert any("temperature" in i.message.lower() for i in issues)


def test_ph_out_of_range():
    df = pd.DataFrame([_base_row(soil_ph=12.0)])
    issues = check_impossible_values(df)
    assert any(i.column == "soil_ph" for i in issues)


def test_humidity_out_of_range():
    df = pd.DataFrame([_base_row(humidity_pct=150.0)])
    issues = check_impossible_values(df)
    assert any(i.column == "humidity_pct" for i in issues)


def test_crop_yield_range_flags_outlier():
    df = pd.DataFrame([_base_row(yield_t_ha=500.0)])
    issues = check_crop_yield_range(df)
    assert len(issues) == 1
    assert issues[0].severity == "warning"


def test_yield_consistency():
    df = pd.DataFrame([_base_row(area_ha=1000.0, production_t=1000.0, yield_t_ha=3.5)])
    issues = check_yield_consistency(df)
    assert len(issues) == 1
    assert issues[0].check == "yield_consistency"


def test_duplicates_detected():
    df = pd.DataFrame([_base_row(), _base_row()])
    issues = check_duplicates(df)
    assert len(issues) == 2


def test_missingness_flagged():
    rows = [_base_row() for _ in range(10)]
    df = pd.DataFrame(rows)
    df.loc[0:4, "soil_p_kgha"] = np.nan  # 50 % missing
    issues = check_missingness(df)
    assert any(i.column == "soil_p_kgha" for i in issues)


def test_unit_check_small_area():
    df = pd.DataFrame([_base_row(area_ha=20.0)])
    issues = check_units(df)
    assert any(i.check == "unit_check" for i in issues)


def test_unit_check_kg_production():
    df = pd.DataFrame([_base_row(area_ha=1000.0, production_t=500000.0, yield_t_ha=500.0)])
    issues = check_units(df)
    assert any(i.check == "unit_check" for i in issues)


def test_unknown_category():
    df = pd.DataFrame([_base_row(crop="quinoa"), _base_row(season="Autumn")])
    issues = check_unknown_categories(df)
    cols = {i.column for i in issues}
    assert "crop" in cols
    assert "season" in cols


def test_validate_full_report_counts(synthetic_df):
    report = validate(synthetic_df)
    assert report.n_rows == len(synthetic_df)
    assert report.n_errors > 0
    assert report.n_warnings > 0
    # Every injected error type should be represented by a check.
    checks = {i.check for i in report.issues}
    for expected in [
        "duplicate_record",
        "impossible_value",
        "non_numeric_value",
        "crop_yield_range",
        "unit_check",
    ]:
        assert expected in checks


def test_clean_data_has_no_errors(clean_df):
    report = validate(clean_df)
    assert report.n_errors == 0
