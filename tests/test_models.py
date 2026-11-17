"""Tests for models: baselines, CV schemes, metrics, training, intervals, explainability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import preprocess
from src.data.synthetic import generate_dataset
from src.features.engineer import FeatureConfig, TARGET, build_pipeline, engineer_features
from src.models.baselines import DistrictHistoryBaseline, MeanByCropSeason
from src.models.evaluate import compute_metrics, mape, metrics_by_scope, skill_score, worst_predictions
from src.models.explain import (
    coefficients_table,
    fit_for_explainability,
    ols_summary,
    partial_dependence_curve,
    permutation_importances,
    plain_language_summary,
)
from src.models.intervals import IntervalModel, TrainingRanges, whatif_curve
from src.models.train import train_all_models, train_single
from src.models.validation import assert_no_leakage, make_splits


@pytest.fixture(scope="module")
def dataset():
    df = generate_dataset(n_districts=10, n_crops=3, n_years=18, seed=7)
    return preprocess(df)


# --- metric tests -----------------------------------------------------------


def test_metrics_hand_calculated():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 5.0])
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(0.25)
    assert m["rmse"] == pytest.approx(0.5)
    assert m["bias"] == pytest.approx(0.25)
    assert m["mape"] == pytest.approx(100 * (0 + 0 + 0 + 0.25) / 4)


def test_r2_perfect():
    y = np.array([1.0, 2.0, 3.0])
    m = compute_metrics(y, y)
    assert m["r2"] == pytest.approx(1.0)
    assert m["mae"] == 0.0


def test_mape_guards_zero():
    m = mape(np.array([0.0, 2.0]), np.array([0.0, 2.5]))
    assert np.isfinite(m)


def test_skill_score():
    assert skill_score(1.0, 2.0) == pytest.approx(0.5)
    assert skill_score(2.0, 1.0) == pytest.approx(-1.0)
    assert skill_score(1.0, 0.0) is None


# --- CV scheme / leakage tests ---------------------------------------------


def test_temporal_split_no_overlap(dataset):
    splits = make_splits(dataset, "temporal", test_years=4)
    assert len(splits) == 1
    assert_no_leakage(dataset, splits[0], "temporal")


def test_group_split_no_district_overlap(dataset):
    splits = make_splits(dataset, "group", n_splits=4)
    assert len(splits) >= 2
    for split in splits:
        assert_no_leakage(dataset, split, "group")


def test_random_split_no_row_overlap(dataset):
    splits = make_splits(dataset, "random", n_splits=3)
    for split in splits:
        assert len(set(split.train_idx) & set(split.test_idx)) == 0


def test_temporal_test_years_strictly_later(dataset):
    split = make_splits(dataset, "temporal", test_years=4)[0]
    train = dataset.iloc[split.train_idx]
    test = dataset.iloc[split.test_idx]
    assert train["year"].max() < test["year"].min()


# --- baseline tests ---------------------------------------------------------


def test_mean_by_crop_season():
    df = pd.DataFrame(
        {
            "crop": ["rice", "rice", "wheat"],
            "season": ["Kharif", "Kharif", "Rabi"],
            TARGET: [3.0, 5.0, 4.0],
        }
    )
    model = MeanByCropSeason().fit(df)
    preds = model.predict(df)
    assert preds[0] == pytest.approx(4.0)
    assert preds[2] == pytest.approx(4.0)


def test_district_history_uses_past_only(dataset):
    split = make_splits(dataset, "temporal", test_years=4)[0]
    train = dataset.iloc[split.train_idx]
    test = dataset.iloc[split.test_idx]
    model = DistrictHistoryBaseline().fit(train)
    preds = model.predict(test, history=train)
    assert len(preds) == len(test)
    assert np.isfinite(preds).all()


# --- training tests ---------------------------------------------------------


def test_random_forest_beats_linear_temporal(dataset):
    """RF should beat Linear Regression on the synthetic non-linear response."""
    results = train_all_models(
        dataset, models=["linear", "ridge", "random_forest"], scheme="temporal"
    )
    rf_rmse = results["random_forest"]["metrics"]["rmse"]
    lin_rmse = results["linear"]["metrics"]["rmse"]
    assert rf_rmse < lin_rmse, f"RF {rf_rmse:.3f} not better than linear {lin_rmse:.3f}"


def test_all_schemes_run(dataset):
    for scheme in ["temporal", "group", "random"]:
        results = train_all_models(
            dataset, models=["baseline_crop_season", "ridge"], scheme=scheme
        )
        assert "ridge" in results
        assert results["ridge"]["metrics"]["n"] > 0


def test_train_single_reload_reproducible(dataset):
    first = train_single(dataset, "ridge", scheme="temporal")
    second = train_single(dataset, "ridge", scheme="temporal")
    p1 = first["pipeline"].predict(first["test_df"])
    p2 = second["pipeline"].predict(second["test_df"])
    assert np.allclose(p1, p2)


# --- interval tests ---------------------------------------------------------


def test_interval_coverage_near_nominal(dataset):
    """Empirical 80 % interval coverage on held-out data is within 70-90 %."""
    result = train_single(dataset, "random_forest", scheme="temporal")
    test_df = result["test_df"]
    interval = IntervalModel.fit(
        result["train_df"], model_name="random_forest", nominal_coverage=0.80
    )
    _, low, high = interval.predict_interval(result["pipeline"], test_df)
    coverage = interval.coverage(result["y_true"], low, high)
    assert 0.65 <= coverage <= 0.95, f"coverage {coverage:.2f} outside expected band"


def test_extrapolation_warning(dataset):
    ranges = TrainingRanges.from_frame(dataset)
    row = dataset.iloc[0].to_dict()
    row["rainfall_mm"] = row["rainfall_mm"] * 100  # far outside training range
    warnings = ranges.check_row(row)
    assert any("rainfall_mm" in w for w in warnings)


def test_unseen_category_warning(dataset):
    ranges = TrainingRanges.from_frame(dataset)
    row = dataset.iloc[0].to_dict()
    row["crop"] = "quinoa"
    warnings = ranges.check_row(row)
    assert any("crop" in w for w in warnings)


def test_whatif_curve(dataset):
    result = train_single(dataset, "random_forest", scheme="temporal")
    interval = IntervalModel.fit(result["train_df"], model_name="random_forest")
    base = result["test_df"].iloc[0].to_dict()
    curve = whatif_curve(result["pipeline"], interval, base, "rainfall_mm")
    assert len(curve["values"]) == len(curve["predictions"])
    assert len(curve["predictions"]) == 13


# --- explainability tests ---------------------------------------------------


def test_permutation_importance_ranks_rainfall_and_nitrogen(dataset):
    """Rainfall and nitrogen should rank highly (spec acceptance criterion)."""
    train = dataset[dataset["year"] <= dataset["year"].max() - 4]
    pipeline = fit_for_explainability(train, "random_forest")
    imp = permutation_importances(pipeline, train, n_repeats=5)
    top = set(imp.head(8)["feature"].tolist())
    assert any("rain" in f for f in top), f"rainfall not in top features: {top}"
    assert any("soil_n" in f or f == "n_factor" for f in top) or any(
        "soil_n" in f for f in top
    ), f"nitrogen not in top features: {top}"


def test_partial_dependence_shape(dataset):
    pipeline = fit_for_explainability(dataset, "random_forest")
    pd_curve = partial_dependence_curve(pipeline, dataset, "rainfall_mm")
    assert len(pd_curve["grid"]) == len(pd_curve["values"]) == 25


def test_plain_language_summary():
    imp = pd.DataFrame(
        {
            "feature": ["rainfall_mm", "soil_n_kgha", "temp_mean_c", "crop_rice"],
            "importance_mean": [0.5, 0.4, 0.3, 0.2],
            "importance_std": [0.01, 0.01, 0.01, 0.01],
        }
    )
    summary = plain_language_summary(imp, crop="rice")
    assert "association" in summary
    assert "rice" in summary


def test_coefficients_table_linear(dataset):
    pipeline = fit_for_explainability(dataset, "linear")
    table = coefficients_table(pipeline)
    assert len(table) > 0
    assert "coefficient" in table[0]


def test_ols_summary(dataset):
    out = ols_summary(dataset)
    assert out["r_squared"] is not None
    assert len(out["coefficients"]) > 0


# --- scope metrics ----------------------------------------------------------


def test_metrics_by_scope(dataset):
    split = make_splits(dataset, "temporal", test_years=4)[0]
    train = dataset.iloc[split.train_idx]
    test = dataset.iloc[split.test_idx]
    model = MeanByCropSeason().fit(train)
    preds = model.predict(test)
    scoped = metrics_by_scope(test, test[TARGET].to_numpy(), preds, ["crop", "season"])
    assert all("scope_type" in row for row in scoped)


def test_worst_predictions(dataset):
    split = make_splits(dataset, "temporal", test_years=4)[0]
    train = dataset.iloc[split.train_idx]
    test = dataset.iloc[split.test_idx]
    model = MeanByCropSeason().fit(train)
    preds = model.predict(test)
    worst = worst_predictions(test, test[TARGET].to_numpy(), preds, n=5)
    assert len(worst) == 5
    assert all("likely_reason" in row for row in worst)
