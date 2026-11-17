"""Feature engineering (spec section 6.2).

Builds the engineered feature table and a scikit-learn ``Pipeline`` /
``ColumnTransformer`` that fits imputers and scalers **inside** the
cross-validation folds (no leakage).

Key correctness properties, exercised by the tests:

* growing degree days (GDD) match a hand calculation,
* district rainfall/temperature anomalies use the district's long-term mean,
* the district rolling-history feature is shifted so it only ever uses *past*
  years (never the current or a future year),
* categorical encoders use ``handle_unknown="ignore"`` so unseen categories at
  prediction time do not raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import crop_settings

TARGET = "yield_t_ha"

# Raw input columns the model consumes (identifiers + drivers). ``production_t``
# is excluded because yield is derived from it, not predicted from it.
INPUT_COLUMNS = [
    "state",
    "district",
    "crop",
    "season",
    "year",
    "area_ha",
    "rainfall_mm",
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "humidity_pct",
    "soil_n_kgha",
    "soil_p_kgha",
    "soil_k_kgha",
    "soil_ph",
    "soil_oc_pct",
    "fertilizer_kgha",
    "irrigated_frac",
    "pesticide_kgha",
]

NUMERIC_FEATURES = [
    "rainfall_mm",
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "humidity_pct",
    "soil_n_kgha",
    "soil_p_kgha",
    "soil_k_kgha",
    "soil_ph",
    "soil_oc_pct",
    "fertilizer_kgha",
    "irrigated_frac",
    "pesticide_kgha",
    "gdd",
    "heat_stress_days",
    "rainfall_anomaly",
    "temp_anomaly",
    "ph_distance",
    "fertilizer_per_ha",
    "log_area",
    "year_index",
    "yield_lag_mean3",
    "rain_x_irrigation",
    "temp_x_crop_stress",
]

CATEGORICAL_FEATURES = ["crop", "season", "state", "district"]


@dataclass
class FeatureConfig:
    include_gdd: bool = True
    include_anomalies: bool = True
    include_lags: bool = True
    include_interactions: bool = True
    include_soil: bool = True
    include_categorical: bool = True
    add_missing_indicators: bool = True
    scale_linear: bool = True

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def growing_degree_days(
    temp_mean: np.ndarray | pd.Series, base_temp: float, days: int = 120
) -> np.ndarray:
    """Seasonal growing degree days.

    ``GDD = sum(max(0, Tmean - Tbase))``. With no daily series we approximate the
    season length as ``days`` and use the seasonal mean temperature, i.e.
    ``GDD = days * max(0, Tmean - Tbase)``.
    """
    tmean = np.asarray(temp_mean, dtype=float)
    return days * np.clip(tmean - base_temp, 0, None)


def _heat_stress_days(temp_max: np.ndarray | pd.Series, threshold: float) -> np.ndarray:
    tmax = np.asarray(temp_max, dtype=float)
    return np.clip(tmax - threshold, 0, None)


def _season_length(season: str) -> int:
    from src.config import load_crops_config

    seasons = load_crops_config().get("seasons", {})
    info = seasons.get(season, {})
    months = info.get("months")
    if months:
        return int(len(months) * 30)
    return 120


def engineer_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered features appended.

    Row order is preserved. ``reference_df`` (typically the training slice) is
    used to derive district climatology for the anomaly features so that train
    and test frames are transformed consistently; when omitted the frame's own
    statistics are used.
    """
    config = config or FeatureConfig()
    out = df.copy()
    out["_row_order"] = np.arange(len(out))

    crop_cfg = out["crop"].map(lambda c: crop_settings(str(c)))
    base_temp = crop_cfg.map(lambda c: c.get("base_temp_c", 10.0)).astype(float)
    heat_thresh = crop_cfg.map(lambda c: c.get("heat_stress_temp_c", 35.0)).astype(float)
    ph_opt = crop_cfg.map(lambda c: c.get("optimal_ph", 6.5)).astype(float)

    season_days = out["season"].map(lambda s: _season_length(str(s))).astype(float)

    if config.include_gdd:
        out["gdd"] = season_days.to_numpy() * np.clip(
            pd.to_numeric(out["temp_mean_c"], errors="coerce").to_numpy() - base_temp.to_numpy(),
            0,
            None,
        )
        out["heat_stress_days"] = np.clip(
            pd.to_numeric(out["temp_max_c"], errors="coerce").to_numpy() - heat_thresh.to_numpy(),
            0,
            None,
        )
    else:
        out["gdd"] = np.nan
        out["heat_stress_days"] = np.nan

    if config.include_anomalies and {"district", "rainfall_mm"}.issubset(out.columns):
        reference = reference_df if reference_df is not None else out
        district_mean_rain = (
            reference.groupby("district")["rainfall_mm"].mean()
        )
        district_mean_temp = (
            reference.groupby("district")["temp_mean_c"].mean()
        )
        out["rainfall_anomaly"] = (
            pd.to_numeric(out["rainfall_mm"], errors="coerce")
            - out["district"].map(district_mean_rain)
        )
        out["temp_anomaly"] = (
            pd.to_numeric(out["temp_mean_c"], errors="coerce")
            - out["district"].map(district_mean_temp)
        )
    else:
        out["rainfall_anomaly"] = np.nan
        out["temp_anomaly"] = np.nan

    if config.include_soil:
        out["ph_distance"] = (
            pd.to_numeric(out["soil_ph"], errors="coerce") - ph_opt
        ).abs()
    else:
        out["ph_distance"] = np.nan

    if {"fertilizer_kgha", "area_ha"}.issubset(out.columns):
        area = pd.to_numeric(out["area_ha"], errors="coerce").replace(0, np.nan)
        out["fertilizer_per_ha"] = pd.to_numeric(out["fertilizer_kgha"], errors="coerce") / area
        out["log_area"] = np.log1p(pd.to_numeric(out["area_ha"], errors="coerce").clip(lower=0))
    else:
        out["fertilizer_per_ha"] = np.nan
        out["log_area"] = np.nan

    if "year" in out.columns:
        year = pd.to_numeric(out["year"], errors="coerce")
        # Fixed epoch so the feature is identical across train/test frames.
        out["year_index"] = year - 2000.0
    else:
        out["year_index"] = np.nan

    if config.include_lags and TARGET in out.columns and {"district", "crop", "year"}.issubset(out.columns):
        # Aggregate to one value per (district, crop, year) first: multiple
        # seasons in the same year must not leak into each other, so the rolling
        # window is built on the *year* series and shifted by one year.
        yearly = (
            out.groupby(["district", "crop", "year"], as_index=False)[TARGET]
            .mean()
            .sort_values(["district", "crop", "year"], kind="stable")
        )
        yearly["yield_lag_mean3"] = yearly.groupby(["district", "crop"], sort=False)[
            TARGET
        ].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        out = out.merge(
            yearly[["district", "crop", "year", "yield_lag_mean3"]],
            on=["district", "crop", "year"],
            how="left",
        )
        out = out.sort_values("_row_order", kind="stable").reset_index(drop=True)
    else:
        out["yield_lag_mean3"] = np.nan

    if config.include_interactions:
        out["rain_x_irrigation"] = (
            pd.to_numeric(out["rainfall_mm"], errors="coerce")
            * pd.to_numeric(out["irrigated_frac"], errors="coerce")
        )
        out["temp_x_crop_stress"] = (
            pd.to_numeric(out["temp_max_c"], errors="coerce") - heat_thresh.to_numpy()
        )
    else:
        out["rain_x_irrigation"] = np.nan
        out["temp_x_crop_stress"] = np.nan

    return out.drop(columns=["_row_order"])


def numeric_features(config: FeatureConfig | None) -> list[str]:
    config = config or FeatureConfig()
    cols = list(NUMERIC_FEATURES)
    if not config.include_gdd:
        cols = [c for c in cols if c not in {"gdd", "heat_stress_days"}]
    if not config.include_anomalies:
        cols = [c for c in cols if c not in {"rainfall_anomaly", "temp_anomaly"}]
    if not config.include_lags:
        cols = [c for c in cols if c != "yield_lag_mean3"]
    if not config.include_soil:
        cols = [c for c in cols if c != "ph_distance"]
    if not config.include_interactions:
        cols = [c for c in cols if c not in {"rain_x_irrigation", "temp_x_crop_stress"}]
    return cols


def build_preprocessor(
    config: FeatureConfig | None = None, *, scale: bool | None = None
) -> ColumnTransformer:
    """Build the ColumnTransformer (impute + optional scale + one-hot).

    Operates on the *engineered* feature frame produced by
    :class:`FeatureEngineer`.
    """
    config = config or FeatureConfig()
    scale = config.scale_linear if scale is None else scale
    num_cols = numeric_features(config)

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=config.add_missing_indicators))
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, Any, Any]] = [
        ("num", Pipeline(numeric_steps), num_cols),
    ]
    if config.include_categorical:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn transformer that turns raw inputs into the engineered frame.

    Fitting stores the training frame so that district climatology (for the
    anomaly features) and the district-crop yearly yield history (for the
    leakage-safe rolling-lag feature) are derived **only** from training data and
    reused unchanged at transform time.
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        frame = X.copy()
        if y is not None and TARGET not in frame.columns:
            frame[TARGET] = np.asarray(y, dtype=float)
        self.reference_ = frame.copy()
        self._engineered_reference_ = engineer_features(frame, self.config)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        reference = getattr(self, "reference_", None)
        features = engineer_features(X, self.config, reference_df=reference)
        # Replace the (leak-free) lag with a lookup against the training history.
        if self.config.include_lags and hasattr(self, "_engineered_reference_"):
            features["yield_lag_mean3"] = self._lookup_lag(X)
        return features

    def _lookup_lag(self, X: pd.DataFrame) -> np.ndarray:
        ref = self._engineered_reference_
        history = (
            ref.dropna(subset=[TARGET])
            .groupby(["district", "crop", "year"])[TARGET]
            .mean()
            .reset_index()
        )
        lookup: dict[tuple, pd.Series] = {
            key: grp.sort_values("year")
            for key, grp in history.groupby(["district", "crop"])
        }
        out = []
        for _, row in X.iterrows():
            grp = lookup.get((row.get("district"), row.get("crop")))
            year = row.get("year")
            value = np.nan
            if grp is not None and pd.notna(year):
                past = grp[grp["year"] < year].tail(3)
                if len(past):
                    value = float(past[TARGET].mean())
            out.append(value)
        return np.asarray(out, dtype=float)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(feature_matrix(pd.DataFrame(columns=INPUT_COLUMNS), self.config).columns)


def build_pipeline(
    model: Any, config: FeatureConfig | None = None, *, scale: bool | None = None
) -> Pipeline:
    """Build a full ``Pipeline``: raw inputs -> features -> preprocessor -> model."""
    config = config or FeatureConfig()
    return Pipeline(
        [
            ("features", FeatureEngineer(config)),
            ("preprocessor", build_preprocessor(config, scale=scale)),
            ("model", model),
        ]
    )


def feature_matrix(df: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Return the engineered numeric + categorical design frame (not encoded)."""
    config = config or FeatureConfig()
    engineered = engineer_features(df, config)
    cols = numeric_features(config)
    if config.include_categorical:
        cols = cols + [c for c in CATEGORICAL_FEATURES if c in engineered.columns]
    cols = [c for c in cols if c in engineered.columns]
    return engineered[cols]


def feature_stats(df: pd.DataFrame, config: FeatureConfig | None = None) -> dict[str, Any]:
    """Descriptive statistics for the engineered feature frame."""
    matrix = feature_matrix(df, config)
    stats: dict[str, Any] = {}
    for col in matrix.columns:
        if pd.api.types.is_numeric_dtype(matrix[col]):
            series = matrix[col]
            stats[col] = {
                "mean": _f(series.mean()),
                "std": _f(series.std()),
                "min": _f(series.min()),
                "max": _f(series.max()),
                "missing": int(series.isna().sum()),
            }
        else:
            stats[col] = {
                "n_unique": int(matrix[col].nunique()),
                "missing": int(matrix[col].isna().sum()),
            }
    return stats


def _f(value) -> float | None:
    try:
        f = float(value)
        return None if np.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None
