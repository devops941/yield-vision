"""Explainability: importance, partial dependence and plain-language summary (6.5)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.features.engineer import INPUT_COLUMNS, TARGET, build_pipeline

CLIMATE_DRIVERS = ["rainfall_mm", "temp_mean_c", "temp_max_c", "gdd", "heat_stress_days"]
SOIL_DRIVERS = ["soil_n_kgha", "soil_p_kgha", "soil_k_kgha", "soil_ph", "soil_oc_pct"]


def _raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in INPUT_COLUMNS if c in df.columns]
    if TARGET in df.columns:
        cols = cols + [TARGET]
    return df[cols].copy()


def permutation_importances(
    pipeline,
    df: pd.DataFrame,
    *,
    feature_config=None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance on held-out data, computed on *raw input* columns.

    Because the pipeline contains the feature engineering step, permuting a raw
    column (e.g. ``rainfall_mm``) propagates to every derived feature that uses
    it, giving a directly interpretable driver ranking.
    """
    raw = _raw_frame(df).dropna(subset=[TARGET]).reset_index(drop=True)
    if len(raw) < 5:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    y = raw[TARGET].to_numpy(dtype=float)
    X = raw.drop(columns=[TARGET])
    result = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_root_mean_squared_error",
    )
    frame = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return frame.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def impurity_importance(pipeline) -> dict[str, float]:
    """Impurity importance mapped onto the transformed design matrix."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return {}
    importances = np.asarray(model.feature_importances_, dtype=float)
    try:
        names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        names = [f"feature_{i}" for i in range(len(importances))]
    if len(names) != len(importances):
        names = [f"feature_{i}" for i in range(len(importances))]
    return dict(zip(names, importances.tolist()))


def partial_dependence_curve(
    pipeline,
    df: pd.DataFrame,
    feature: str,
    *,
    feature_config=None,
    grid_resolution: int = 25,
) -> dict[str, Any]:
    """Compute a 1-D partial dependence curve for a raw input feature."""
    raw = _raw_frame(df)
    if feature not in raw.columns:
        return {"feature": feature, "grid": [], "values": []}
    base = raw.drop(columns=[TARGET], errors="ignore").copy()
    values = pd.to_numeric(base[feature], errors="coerce").dropna()
    if values.empty:
        return {"feature": feature, "grid": [], "values": []}
    grid = np.linspace(values.quantile(0.05), values.quantile(0.95), grid_resolution)
    preds = []
    for value in grid:
        perturbed = base.copy()
        perturbed[feature] = value
        preds.append(float(np.mean(pipeline.predict(perturbed))))
    return {
        "feature": feature,
        "grid": [float(g) for g in grid],
        "values": preds,
    }


def ice_curves(
    pipeline,
    df: pd.DataFrame,
    feature: str,
    *,
    feature_config=None,
    grid_resolution: int = 20,
    n_lines: int = 12,
) -> dict[str, Any]:
    """Individual conditional expectation curves for a sample of rows."""
    raw = _raw_frame(df)
    if feature not in raw.columns or raw.empty:
        return {"feature": feature, "grid": [], "lines": []}
    base_all = raw.drop(columns=[TARGET], errors="ignore")
    sample = base_all.sample(min(n_lines, len(base_all)), random_state=42)
    values = pd.to_numeric(base_all[feature], errors="coerce").dropna()
    grid = np.linspace(values.quantile(0.05), values.quantile(0.95), grid_resolution)
    lines = []
    for _, row in sample.iterrows():
        base = pd.DataFrame([row])
        line = []
        for value in grid:
            perturbed = base.copy()
            perturbed[feature] = value
            line.append(float(pipeline.predict(perturbed)[0]))
        lines.append(line)
    return {"feature": feature, "grid": [float(g) for g in grid], "lines": lines}


def plain_language_summary(importances: pd.DataFrame, crop: str | None = None) -> str:
    """Turn the importance table into an 'associated with' sentence."""
    if importances.empty:
        return "Not enough data to describe the drivers of yield."

    def _classify(name: str) -> str:
        base = str(name)
        if base.startswith(("rain", "temp", "gdd", "heat", "humidity")):
            return "climate"
        if base.startswith("soil"):
            return "soil"
        if base.startswith(("irrigated", "fertilizer", "pesticide", "area")):
            return "cultivation"
        if base.startswith("year"):
            return "time/technology"
        if base in {"crop", "season", "state", "district"}:
            return "crop/region"
        return "other"

    top = importances.head(6)
    labels = [_classify(f) for f in top["feature"]]
    seen: list[str] = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    crop_text = f" for {crop}" if crop else ""
    driver_text = " and ".join(seen[:3]) if seen else "several"
    examples = ", ".join(_pretty_feature(f) for f in top["feature"].head(3))
    return (
        f"{driver_text.capitalize()} factors were the strongest drivers of yield{crop_text} "
        f"in this dataset (top signals: {examples}). "
        "These results describe association in the data, not proof of causation."
    )


FEATURE_LABELS = {
    "crop": "crop",
    "season": "season",
    "state": "state",
    "district": "district",
    "year": "year",
    "area_ha": "area",
    "rainfall_mm": "rainfall",
    "temp_mean_c": "average temperature",
    "temp_max_c": "maximum temperature",
    "temp_min_c": "minimum temperature",
    "humidity_pct": "humidity",
    "soil_n_kgha": "soil nitrogen",
    "soil_p_kgha": "soil phosphorus",
    "soil_k_kgha": "soil potassium",
    "soil_ph": "soil pH",
    "soil_oc_pct": "soil organic carbon",
    "fertilizer_kgha": "fertilizer use",
    "irrigated_frac": "irrigated share",
    "pesticide_kgha": "pesticide use",
    "gdd": "growing degree days",
    "heat_stress_days": "heat-stress days",
}


def _pretty_feature(name: str) -> str:
    """Turn a canonical column name into something readable in prose."""
    return FEATURE_LABELS.get(str(name), str(name).replace("_", " "))


def coefficients_table(pipeline) -> list[dict[str, Any]]:
    """Linear-model coefficient table (on the transformed design matrix)."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return []
    try:
        names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        names = []
    coefs = np.asarray(model.coef_, dtype=float).ravel()
    if len(names) != len(coefs):
        names = [f"feature_{i}" for i in range(len(coefs))]
    return [
        {"feature": name, "coefficient": float(coef)}
        for name, coef in sorted(zip(names, coefs), key=lambda t: abs(t[1]), reverse=True)
    ]


def ols_summary(df: pd.DataFrame, feature_config=None) -> dict[str, Any]:
    """statsmodels OLS summary: coefficients, p-values and VIF for numeric features."""
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    from src.features.engineer import engineer_features, numeric_features

    features = engineer_features(df, feature_config)
    num_cols = [c for c in numeric_features(feature_config) if c in features.columns]
    data = features[num_cols + [TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < len(num_cols) + 5:
        return {"coefficients": [], "r_squared": None, "n": int(len(data))}

    X = sm.add_constant(data[num_cols])
    y = data[TARGET]
    try:
        model = sm.OLS(y, X).fit()
    except Exception as exc:  # pragma: no cover - singular design
        return {"coefficients": [], "r_squared": None, "n": int(len(data)), "error": str(exc)}

    coefficients = [
        {
            "feature": name,
            "coef": float(model.params[name]),
            "std_err": float(model.bse[name]),
            "t": float(model.tvalues[name]),
            "p_value": float(model.pvalues[name]),
        }
        for name in X.columns
    ]

    vif = []
    try:
        x_values = X.to_numpy(dtype=float)
        for i, name in enumerate(X.columns):
            if name == "const":
                continue
            vif.append({"feature": name, "vif": float(variance_inflation_factor(x_values, i))})
    except Exception:
        vif = []

    return {
        "coefficients": coefficients,
        "vif": vif,
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "n": int(len(data)),
    }


def fit_for_explainability(
    train_df: pd.DataFrame, model_name: str = "random_forest", feature_config=None
):
    from src.models.train import build_estimator

    scale = model_name in {"linear", "ridge"}
    pipeline = build_pipeline(build_estimator(model_name), feature_config, scale=scale)
    y = train_df[TARGET].to_numpy(dtype=float)
    pipeline.fit(train_df, y)
    return pipeline
