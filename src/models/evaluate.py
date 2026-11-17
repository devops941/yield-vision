"""Evaluation metrics and scope breakdowns (spec section 6.4)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MIN_SCOPE_ROWS = 8


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, guarding zero and near-zero actuals."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-6
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true, y_pred) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None, "mape": None, "bias": None}
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2) if not np.isnan(r2) else None,
        "mape": mape(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
    }


def skill_score(rmse_model: float, rmse_baseline: float) -> float | None:
    """1 - RMSE_model / RMSE_baseline."""
    if rmse_baseline is None or rmse_model is None or rmse_baseline == 0:
        return None
    return float(1.0 - rmse_model / rmse_baseline)


def metrics_by_scope(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scope_cols: list[str],
    min_rows: int = MIN_SCOPE_ROWS,
) -> list[dict[str, Any]]:
    """Compute metrics grouped by one or more scope columns."""
    frame = df.copy()
    frame["_y_true"] = y_true
    frame["_y_pred"] = y_pred
    out: list[dict[str, Any]] = []
    for scope_col in scope_cols:
        if scope_col not in frame.columns:
            continue
        for value, group in frame.groupby(scope_col, dropna=False):
            if len(group) < min_rows:
                continue
            metrics = compute_metrics(group["_y_true"], group["_y_pred"])
            out.append(
                {
                    "scope_type": scope_col,
                    "scope_value": str(value),
                    **metrics,
                }
            )
    return out


def worst_predictions(
    df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, n: int = 10
) -> list[dict[str, Any]]:
    """Return the ``n`` worst predictions with a likely-reason annotation."""
    frame = df.copy()
    frame["y_true"] = y_true
    frame["y_pred"] = y_pred
    frame["abs_error"] = (frame["y_true"] - frame["y_pred"]).abs()
    frame = frame.sort_values("abs_error", ascending=False).head(n)

    reasons = _likely_reasons(frame)
    records = []
    for (idx, row), reason in zip(frame.iterrows(), reasons):
        records.append(
            {
                "row_index": int(idx),
                "district": row.get("district"),
                "crop": row.get("crop"),
                "season": row.get("season"),
                "year": _int(row.get("year")),
                "y_true": _round(row.get("y_true")),
                "y_pred": _round(row.get("y_pred")),
                "abs_error": _round(row.get("abs_error")),
                "likely_reason": reason,
            }
        )
    return records


def _likely_reasons(frame: pd.DataFrame) -> list[str]:
    reasons = []
    rain = frame.get("rainfall_mm")
    temp = frame.get("temp_mean_c")
    for i, (_, row) in enumerate(frame.iterrows()):
        if rain is not None and pd.notna(row.get("rainfall_mm")):
            p10, p90 = np.nanpercentile(pd.to_numeric(rain, errors="coerce"), [10, 90])
            if row["rainfall_mm"] < p10:
                reasons.append("unusually dry season")
                continue
            if row["rainfall_mm"] > p90:
                reasons.append("unusually wet season")
                continue
        if temp is not None and pd.notna(row.get("temp_mean_c")):
            p90t = np.nanpercentile(pd.to_numeric(temp, errors="coerce"), 90)
            if row["temp_mean_c"] > p90t:
                reasons.append("heat stress above threshold")
                continue
        reasons.append("possible data error or rare combination")
    return reasons


def comparison_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build a model x scheme comparison table from nested results."""
    rows = []
    for model_name, schemes in results.items():
        for scheme, metrics in schemes.items():
            rows.append(
                {
                    "model": model_name,
                    "scheme": scheme,
                    **{k: metrics.get(k) for k in ["n", "mae", "rmse", "r2", "mape", "bias", "skill_vs_baseline"]},
                }
            )
    return pd.DataFrame(rows)


def _round(value) -> float | None:
    try:
        f = float(value)
        return None if np.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
