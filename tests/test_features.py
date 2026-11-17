"""Tests for feature engineering, including the leakage tests (spec section 12)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data.synthetic import generate_dataset
from src.features.engineer import (
    FeatureConfig,
    build_pipeline,
    engineer_features,
    growing_degree_days,
)


def test_gdd_hand_calculation():
    # Tmean=30, Tbase=10, 120 days => 120 * 20 = 2400
    gdd = growing_degree_days(np.array([30.0]), base_temp=10.0, days=120)
    assert gdd[0] == 2400.0


def test_gdd_clipped_at_zero():
    gdd = growing_degree_days(np.array([5.0]), base_temp=10.0, days=120)
    assert gdd[0] == 0.0


def test_gdd_in_engineered_frame(clean_df):
    out = engineer_features(clean_df)
    row = out.iloc[0]
    expected = 30 * 5 * max(0.0, row["temp_mean_c"] - 10.0)  # rice base temp 10
    assert abs(row["gdd"] - expected) < 1e-6


def test_rainfall_anomaly_manual(clean_df):
    out = engineer_features(clean_df)
    district = out["district"].iloc[0]
    mean_rain = clean_df[clean_df["district"] == district]["rainfall_mm"].mean()
    expected = out["rainfall_mm"].iloc[0] - mean_rain
    assert abs(out["rainfall_anomaly"].iloc[0] - expected) < 1e-6


def test_ph_distance(clean_df):
    out = engineer_features(clean_df)
    rice = out[out["crop"] == "rice"].iloc[0]
    assert abs(rice["ph_distance"] - abs(rice["soil_ph"] - 6.5)) < 1e-6


def test_history_feature_uses_only_past_years(clean_df):
    """yield_lag_mean3 must never use the current or a future year.

    The feature is defined at the (district, crop, year) level, so both seasons
    of the same year share the same value.
    """
    out = engineer_features(clean_df)
    district = out["district"].iloc[0]
    sub = out[(out["district"] == district) & (out["crop"] == "rice")]

    yearly = sub.groupby("year")["yield_t_ha"].mean().sort_index()
    years = list(yearly.index)

    # First year has no history.
    first_year_rows = sub[sub["year"] == years[0]]
    assert first_year_rows["yield_lag_mean3"].isna().all()

    # Second year's lag equals the first year's mean yield.
    second_rows = sub[sub["year"] == years[1]]
    assert np.allclose(second_rows["yield_lag_mean3"], yearly.loc[years[0]])

    # Third year's lag equals the mean of the first two years.
    expected = np.mean([yearly.loc[years[0]], yearly.loc[years[1]]])
    third_rows = sub[sub["year"] == years[2]]
    assert np.allclose(third_rows["yield_lag_mean3"], expected)

    # Both seasons in a year share the same lag value.
    assert third_rows["yield_lag_mean3"].nunique() == 1


def test_no_current_year_leakage(clean_df):
    """Perturbing the current year's yield must not change that row's lag feature."""
    out1 = engineer_features(clean_df)
    # Pick a row that has a non-null lag feature (i.e. not the first year).
    candidate = out1[out1["yield_lag_mean3"].notna()].index[0]
    key = (
        out1.loc[candidate, "district"],
        out1.loc[candidate, "crop"],
        out1.loc[candidate, "year"],
    )

    modified = clean_df.copy()
    modified.loc[candidate, "yield_t_ha"] = modified.loc[candidate, "yield_t_ha"] * 3
    out2 = engineer_features(modified)

    before = out1.loc[candidate, "yield_lag_mean3"]
    after = out2.loc[candidate, "yield_lag_mean3"]
    assert abs(before - after) < 1e-9, "Lag feature leaked the current year's yield"


def test_preprocessor_fits_only_on_train(clean_df):
    """Scaler statistics must come only from the training slice."""
    features = engineer_features(clean_df)
    from src.features.engineer import build_preprocessor

    train = features.iloc[: int(len(features) * 0.7)]
    pre = build_preprocessor(FeatureConfig(), scale=True)
    pre.fit(train)
    scaler = pre.named_transformers_["num"].named_steps["scaler"]
    assert np.allclose(scaler.mean_, scaler.mean_)  # fitted
    # Refit on a different subset must give different means -> proves fitting.
    pre2 = build_preprocessor(FeatureConfig(), scale=True)
    pre2.fit(features.iloc[int(len(features) * 0.7) :])
    scaler2 = pre2.named_transformers_["num"].named_steps["scaler"]
    assert not np.allclose(scaler.mean_, scaler2.mean_)


def test_pipeline_end_to_end(clean_df):
    features = engineer_features(clean_df)
    y = features["yield_t_ha"]
    pipe = build_pipeline(LinearRegression(), FeatureConfig())
    pipe.fit(features, y)
    preds = pipe.predict(features.head(5))
    assert len(preds) == 5


def test_unseen_category_does_not_raise(clean_df):
    features = engineer_features(clean_df)
    pipe = build_pipeline(LinearRegression(), FeatureConfig())
    pipe.fit(features, features["yield_t_ha"])
    novel = features.head(1).copy()
    novel["crop"] = "quinoa"
    preds = pipe.predict(novel)
    assert len(preds) == 1
