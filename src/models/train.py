"""Model training: Linear/Ridge, Random Forest, HistGradientBoosting (spec 6.3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from src.config import load_models_config
from src.features.engineer import FeatureConfig, TARGET, build_pipeline, engineer_features
from src.models.baselines import DistrictHistoryBaseline, MeanByCropSeason
from src.models.evaluate import compute_metrics
from src.models.validation import make_splits

MODEL_NAMES = ["baseline_crop_season", "baseline_district_history", "linear", "ridge", "random_forest", "hist_gradient_boosting"]

LINEAR_MODELS = {"linear", "ridge"}


def build_estimator(name: str, params: dict[str, Any] | None = None):
    params = params or {}
    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=params.get("alpha", 1.0), random_state=42)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=params.get("max_features", 1.0),
            random_state=params.get("random_state", 42),
            n_jobs=params.get("n_jobs", -1),
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=params.get("max_iter", 400),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", None),
            l2_regularization=params.get("l2_regularization", 0.0),
            random_state=params.get("random_state", 42),
        )
    raise ValueError(f"Unknown model: {name}")


def _fit_predict_pipeline(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict | None,
    feature_config: FeatureConfig,
):
    estimator = build_estimator(name, params)
    scale = name in LINEAR_MODELS
    pipeline = build_pipeline(estimator, feature_config, scale=scale)
    y_train = train_df[TARGET].to_numpy(dtype=float)
    pipeline.fit(train_df, y_train)
    preds = pipeline.predict(test_df)
    return pipeline, preds


def _filter_plausible(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose yield is outside the crop-specific plausible range."""
    from src.config import crop_settings

    if TARGET not in df.columns or "crop" not in df.columns:
        return df
    limits = df["crop"].map(
        lambda c: (crop_settings(str(c)).get("yield_min", 0.0), crop_settings(str(c)).get("yield_max", 1e9))
    )
    lo = limits.map(lambda t: t[0]).to_numpy(dtype=float)
    hi = limits.map(lambda t: t[1]).to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    keep = (y >= lo) & (y <= hi)
    return df.loc[keep].reset_index(drop=True)


def train_all_models(
    df: pd.DataFrame,
    *,
    models: list[str] | None = None,
    scheme: str = "temporal",
    feature_config: FeatureConfig | None = None,
    tune: bool = False,
    test_years: int = 4,
    exclude_outliers: bool = True,
) -> dict[str, dict[str, Any]]:
    """Train and evaluate the requested models under one CV scheme.

    Returns ``{model_name: {"metrics":..., "predictions":..., "y_true":..., ...}}``.
    """
    feature_config = feature_config or FeatureConfig()
    models = models or ["baseline_crop_season", "baseline_district_history", "linear", "ridge", "random_forest"]
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    if exclude_outliers:
        df = _filter_plausible(df)

    splits = make_splits(df, scheme, test_years=test_years)
    if not splits:
        raise ValueError(f"Not enough data to build '{scheme}' splits")

    results: dict[str, dict[str, Any]] = {}
    for name in models:
        all_true: list[float] = []
        all_pred: list[float] = []
        all_test: list = []
        fold_metrics: list[dict] = []
        pipeline = None

        for split in splits:
            train_df = df.iloc[split.train_idx].reset_index(drop=True)
            test_df = df.iloc[split.test_idx].reset_index(drop=True)
            if len(train_df) < 5 or len(test_df) == 0:
                continue

            if name == "baseline_crop_season":
                model = MeanByCropSeason().fit(train_df)
                preds = model.predict(test_df)
                pipeline = model
            elif name == "baseline_district_history":
                model = DistrictHistoryBaseline().fit(train_df)
                preds = model.predict(test_df, history=train_df)
                pipeline = model
            else:
                params = _tuned_params(name, train_df, feature_config) if tune else None
                pipeline, preds = _fit_predict_pipeline(
                    name, train_df, test_df, params, feature_config
                )

            y_true = test_df[TARGET].to_numpy(dtype=float)
            all_true.extend(y_true.tolist())
            all_pred.extend(np.asarray(preds, dtype=float).tolist())
            all_test.append(test_df)
            fold_metrics.append(compute_metrics(y_true, preds))

        y_true_arr = np.asarray(all_true, dtype=float)
        y_pred_arr = np.asarray(all_pred, dtype=float)
        metrics = compute_metrics(y_true_arr, y_pred_arr)
        test_frame = pd.concat(all_test, ignore_index=True) if all_test else pd.DataFrame()
        results[name] = {
            "model": name,
            "scheme": scheme,
            "metrics": metrics,
            "fold_metrics": fold_metrics,
            "pipeline": pipeline,
            "y_true": y_true_arr,
            "y_pred": y_pred_arr,
            "test_df": test_frame,
            "n_folds": len(fold_metrics),
        }
    return results


def _tuned_params(name: str, train_df: pd.DataFrame, feature_config: FeatureConfig) -> dict:
    """Light hyperparameter search using expanding-window time splits."""
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

    cfg = load_models_config()
    y = train_df[TARGET].to_numpy(dtype=float)
    valid = ~np.isnan(y)
    train, y = train_df.loc[valid].reset_index(drop=True), y[valid]

    if name == "ridge":
        grid = {"model__alpha": cfg.get("ridge", {}).get("alpha", [0.1, 1.0, 10.0])}
        n_iter = 3
    elif name == "random_forest":
        rf_cfg = cfg.get("random_forest", {})
        dists = rf_cfg.get("param_distributions", {})
        grid = {f"model__{k}": v for k, v in dists.items()}
        n_iter = min(rf_cfg.get("n_iter", 20), 12)
    elif name == "hist_gradient_boosting":
        hgb = cfg.get("hist_gradient_boosting", {})
        grid = {f"model__{k}": v for k, v in hgb.items() if k != "fixed"}
        n_iter = 6
    else:
        return {}

    years = np.sort(train["year"].unique())
    n_splits = min(3, max(2, len(years) - 1))
    cv = TimeSeriesSplit(n_splits=n_splits)
    estimator = build_estimator(name)
    scale = name in LINEAR_MODELS
    pipeline = build_pipeline(estimator, feature_config, scale=scale)
    order = np.argsort(train["year"].to_numpy(), kind="stable")
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=grid,
        n_iter=min(n_iter, max(1, _grid_size(grid))),
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )
    try:
        search.fit(train.iloc[order], y[order])
        return {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    except Exception:
        return {}


def _grid_size(grid: dict) -> int:
    size = 1
    for values in grid.values():
        size *= max(1, len(values))
    return size


def train_single(
    df: pd.DataFrame,
    name: str,
    *,
    scheme: str = "temporal",
    feature_config: FeatureConfig | None = None,
    params: dict | None = None,
) -> dict[str, Any]:
    """Train one model on the training portion and return the fitted pipeline."""
    feature_config = feature_config or FeatureConfig()
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    splits = make_splits(df, scheme)
    split = splits[0]
    train_df = df.iloc[split.train_idx].reset_index(drop=True)
    test_df = df.iloc[split.test_idx].reset_index(drop=True)
    pipeline, preds = _fit_predict_pipeline(name, train_df, test_df, params, feature_config)
    y_true = test_df[TARGET].to_numpy(dtype=float)
    return {
        "pipeline": pipeline,
        "metrics": compute_metrics(y_true, preds),
        "y_true": y_true,
        "y_pred": np.asarray(preds, dtype=float),
        "test_df": test_df,
        "train_df": train_df,
        "params": params or {},
    }
