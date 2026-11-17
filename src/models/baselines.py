"""Baseline models (spec section 6.3).

* Baseline A: mean yield for the crop-season in the training data.
* Baseline B: mean of the district-crop's previous 3 years (naive forecast).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.engineer import TARGET


class MeanByCropSeason:
    """Predict the training mean yield for each (crop, season) group."""

    def __init__(self) -> None:
        self.overall_: float = 0.0
        self.means_: dict[tuple, float] = {}

    def fit(self, df: pd.DataFrame) -> "MeanByCropSeason":
        self.overall_ = float(df[TARGET].mean())
        self.means_ = (
            df.groupby(["crop", "season"])[TARGET].mean().to_dict()
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(
            [
                self.means_.get((row.get("crop"), row.get("season")), self.overall_)
                for _, row in df.iterrows()
            ],
            dtype=float,
        )


class DistrictHistoryBaseline:
    """Predict the mean of the district-crop's previous 3 years.

    When there is no history the crop-season training mean is used as fallback.
    """

    def __init__(self, window: int = 3) -> None:
        self.window = window
        self.fallback_: MeanByCropSeason | None = None

    def fit(self, df: pd.DataFrame) -> "DistrictHistoryBaseline":
        self.fallback_ = MeanByCropSeason().fit(df)
        return self

    def predict(self, df: pd.DataFrame, history: pd.DataFrame | None = None) -> np.ndarray:
        """Predict using ``history`` (training data) as the source of past yields.

        For each row in ``df`` we average the yields of the same district and crop
        in ``history`` with a year strictly less than the row's year.
        """
        source = history if history is not None else df
        assert self.fallback_ is not None
        source = source.dropna(subset=[TARGET])
        grouped = {
            key: grp.sort_values("year")
            for key, grp in source.groupby(["district", "crop"])
        }
        preds = []
        for _, row in df.iterrows():
            key = (row.get("district"), row.get("crop"))
            grp = grouped.get(key)
            year = row.get("year")
            value = None
            if grp is not None and year is not None:
                past = grp[grp["year"] < year].tail(self.window)
                if len(past):
                    value = float(past[TARGET].mean())
            if value is None:
                value = self.fallback_.means_.get(
                    (row.get("crop"), row.get("season")), self.fallback_.overall_
                )
            preds.append(value)
        return np.array(preds, dtype=float)


def evaluate_baseline(model, train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    model.fit(train_df)
    if isinstance(model, DistrictHistoryBaseline):
        return model.predict(test_df, history=train_df)
    return model.predict(test_df)
