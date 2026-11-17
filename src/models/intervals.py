"""Prediction intervals and extrapolation checks (spec section 6.6).

Intervals come from out-of-fold residual quantiles (the model is refit on the
training slice with K-fold and the residuals on the held-out folds give the
interval width). This avoids the optimistic, in-sample residuals that make an
80 % interval cover far less than 80 % of unseen rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.features.engineer import INPUT_COLUMNS, TARGET, build_pipeline


def _input_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the raw input columns (plus target when present)."""
    cols = [c for c in INPUT_COLUMNS if c in df.columns]
    if TARGET in df.columns:
        cols = cols + [TARGET]
    return df[cols].copy()


@dataclass
class TrainingRanges:
    """Per-feature min/max observed during training, used for extrapolation checks."""

    mins: dict[str, float]
    maxs: dict[str, float]
    categories: dict[str, set]

    @classmethod
    def from_frame(cls, df: pd.DataFrame, feature_config=None) -> "TrainingRanges":
        raw = _input_frame(df)
        mins: dict[str, float] = {}
        maxs: dict[str, float] = {}
        categories: dict[str, set] = {}
        for col in raw.columns:
            if col == TARGET:
                continue
            if pd.api.types.is_numeric_dtype(raw[col]):
                series = pd.to_numeric(raw[col], errors="coerce").dropna()
                if not series.empty:
                    mins[col] = float(series.min())
                    maxs[col] = float(series.max())
            else:
                categories[col] = set(raw[col].dropna().astype(str).unique())
        return cls(mins=mins, maxs=maxs, categories=categories)

    def check_row(self, row: pd.Series | dict, feature_config=None) -> list[str]:
        warnings: list[str] = []
        record = dict(row)
        for col, low in self.mins.items():
            if col not in record:
                continue
            value = record[col]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value < low:
                warnings.append(f"{col}={value:.3g} is below the training minimum {low:.3g}")
            elif value > self.maxs[col]:
                warnings.append(
                    f"{col}={value:.3g} is above the training maximum {self.maxs[col]:.3g}"
                )
        for col, seen in self.categories.items():
            if col not in record:
                continue
            value = record[col]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            if str(value) not in seen:
                warnings.append(f"{col}='{value}' was not seen during training")
        return warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "mins": self.mins,
            "maxs": self.maxs,
            "categories": {k: sorted(v) for k, v in self.categories.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingRanges":
        return cls(
            mins=payload.get("mins", {}),
            maxs=payload.get("maxs", {}),
            categories={k: set(v) for k, v in payload.get("categories", {}).items()},
        )


@dataclass
class IntervalModel:
    """Residual-quantile prediction interval."""

    residual_lo: float
    residual_hi: float
    nominal_coverage: float
    lower_q: float
    upper_q: float
    per_tree_spread: float | None = None
    ranges: TrainingRanges | None = None

    @classmethod
    def fit(
        cls,
        train_df: pd.DataFrame,
        *,
        model_name: str = "random_forest",
        params: dict | None = None,
        feature_config=None,
        nominal_coverage: float = 0.80,
        n_splits: int = 5,
    ) -> "IntervalModel":
        from sklearn.model_selection import KFold

        from src.models.train import build_estimator

        raw = _input_frame(train_df).dropna(subset=[TARGET]).reset_index(drop=True)
        y = raw[TARGET].to_numpy(dtype=float)
        X = raw.drop(columns=[TARGET])
        scale = model_name in {"linear", "ridge"}

        n_splits = min(n_splits, max(2, len(raw) // 10))
        residuals: list[float] = []
        if n_splits >= 2:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for tr, te in kf.split(X):
                pipeline = build_pipeline(
                    build_estimator(model_name, params), feature_config, scale=scale
                )
                pipeline.fit(X.iloc[tr], y[tr])
                preds = pipeline.predict(X.iloc[te])
                residuals.extend((y[te] - preds).tolist())
        else:  # pragma: no cover - tiny datasets
            pipeline = build_pipeline(
                build_estimator(model_name, params), feature_config, scale=scale
            )
            pipeline.fit(X, y)
            residuals.extend((y - pipeline.predict(X)).tolist())

        residuals = np.asarray(residuals, dtype=float)
        alpha = 1.0 - nominal_coverage
        lower_q, upper_q = alpha / 2.0, 1.0 - alpha / 2.0

        # Final model on all training data (used for the per-tree spread estimate).
        pipeline = build_pipeline(
            build_estimator(model_name, params), feature_config, scale=scale
        )
        pipeline.fit(X, y)
        spread = None
        model = pipeline.named_steps.get("model")
        if model is not None and hasattr(model, "estimators_"):
            try:
                feats = pipeline.named_steps["features"].transform(X)
                transformed = pipeline.named_steps["preprocessor"].transform(feats)
                per_tree = np.array([est.predict(transformed) for est in model.estimators_])
                spread = float(np.mean(np.std(per_tree, axis=0)))
            except Exception:
                spread = None

        return cls(
            residual_lo=float(np.quantile(residuals, lower_q)),
            residual_hi=float(np.quantile(residuals, upper_q)),
            nominal_coverage=nominal_coverage,
            lower_q=lower_q,
            upper_q=upper_q,
            per_tree_spread=spread,
            ranges=TrainingRanges.from_frame(train_df, feature_config),
        )

    def predict_interval(self, pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        preds = np.asarray(pipeline.predict(X), dtype=float)
        low = np.maximum(preds + self.residual_lo, 0.0)
        high = np.maximum(preds + self.residual_hi, low)
        return preds, low, high

    def coverage(self, y_true: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float)
        inside = (y_true >= low) & (y_true <= high)
        return float(np.mean(inside))

    def to_dict(self) -> dict[str, Any]:
        return {
            "residual_lo": self.residual_lo,
            "residual_hi": self.residual_hi,
            "nominal_coverage": self.nominal_coverage,
            "lower_q": self.lower_q,
            "upper_q": self.upper_q,
            "per_tree_spread": self.per_tree_spread,
            "ranges": self.ranges.to_dict() if self.ranges else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IntervalModel":
        ranges = payload.get("ranges")
        return cls(
            residual_lo=payload["residual_lo"],
            residual_hi=payload["residual_hi"],
            nominal_coverage=payload.get("nominal_coverage", 0.80),
            lower_q=payload.get("lower_q", 0.1),
            upper_q=payload.get("upper_q", 0.9),
            per_tree_spread=payload.get("per_tree_spread"),
            ranges=TrainingRanges.from_dict(ranges) if ranges else None,
        )


def extrapolation_warnings(ranges: TrainingRanges, row, feature_config=None) -> list[str]:
    if ranges is None:
        return []
    return ranges.check_row(row, feature_config)


def whatif_curve(
    pipeline,
    interval_model: IntervalModel | None,
    base_row: pd.Series | dict,
    feature: str,
    *,
    deltas: np.ndarray | None = None,
    feature_config=None,
) -> dict[str, Any]:
    """Sensitivity curve: vary one input holding others fixed."""
    row = dict(base_row)
    if feature == "rainfall_mm":
        base = float(row.get("rainfall_mm", 800.0))
        values = base * (1 + np.linspace(-0.3, 0.3, 13))
    elif feature == "temp_mean_c":
        base = float(row.get("temp_mean_c", 26.0))
        values = base + np.linspace(0, 3, 13)
    else:
        base = float(row.get(feature, 0.0))
        values = base + (
            deltas if deltas is not None else np.linspace(-0.3, 0.3, 13) * abs(base or 1.0)
        )

    preds, lows, highs = [], [], []
    for value in values:
        candidate = dict(row)
        candidate[feature] = float(value)
        if feature == "temp_mean_c":
            shift = float(value) - base
            if "temp_max_c" in candidate:
                candidate["temp_max_c"] = float(candidate["temp_max_c"]) + shift
            if "temp_min_c" in candidate:
                candidate["temp_min_c"] = float(candidate["temp_min_c"]) + shift
        frame = pd.DataFrame([candidate])
        if interval_model is not None:
            pred_arr, low_arr, high_arr = interval_model.predict_interval(pipeline, frame)
            preds.append(float(pred_arr[0]))
            lows.append(float(low_arr[0]))
            highs.append(float(high_arr[0]))
        else:
            preds.append(float(pipeline.predict(frame)[0]))
            lows.append(None)
            highs.append(None)

    baseline_pred = preds[len(preds) // 2] if preds else None
    return {
        "feature": feature,
        "values": [float(v) for v in values],
        "predictions": preds,
        "pred_low": lows,
        "pred_high": highs,
        "baseline_prediction": baseline_pred,
        "change": [None if baseline_pred is None else p - baseline_pred for p in preds],
    }
