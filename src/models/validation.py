"""Cross-validation strategies (spec section 6.4).

Three evaluation schemes are reported and the *headline* is the temporal split:

1. ``temporal`` - train on years up to T, test on the last N years. This
   simulates real forecasting and is the headline scheme.
2. ``group`` - ``GroupKFold`` by district, testing generalisation to unseen
   districts.
3. ``random`` - plain ``KFold``, shown only for comparison and labelled
   optimistic because rows from the same district and year leak information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

SCHEMES = ("temporal", "group", "random")


@dataclass
class Split:
    scheme: str
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def temporal_splits(df: pd.DataFrame, test_years: int = 4) -> list[Split]:
    """Return a single temporal train/test split plus inner TimeSeriesSplit folds.

    The headline split trains on all years up to ``max_year - test_years`` and
    tests on the final ``test_years`` years.
    """
    years = np.sort(df["year"].unique())
    if len(years) <= test_years:
        # Fall back to a proportional split if there are too few years.
        cutoff = years[max(1, len(years) // 2)]
        test_mask = df["year"] >= cutoff
    else:
        cutoff = years[-test_years]
        test_mask = df["year"] >= cutoff

    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    return [Split("temporal", 0, train_idx, test_idx)]


def inner_time_series_splits(
    df: pd.DataFrame, train_idx: np.ndarray, n_splits: int = 3
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits over years for hyperparameter tuning.

    ``train_idx`` and the returned indices are *positional* (use with ``.iloc``).
    """
    from sklearn.model_selection import TimeSeriesSplit

    train = df.iloc[train_idx]
    years = np.sort(train["year"].unique())
    n_splits = min(n_splits, max(2, len(years) - 1))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    tss = TimeSeriesSplit(n_splits=n_splits)
    # Order rows by year so TimeSeriesSplit respects chronology (positional order).
    order = np.argsort(train["year"].to_numpy(), kind="stable")
    for tr_pos, te_pos in tss.split(order):
        splits.append((order[tr_pos], order[te_pos]))
    return splits


def group_splits(df: pd.DataFrame, n_splits: int = 5) -> list[Split]:
    """GroupKFold by district -> tests generalisation to unseen districts."""
    groups = df["district"].astype("category").cat.codes.to_numpy()
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        return []
    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    for fold, (tr, te) in enumerate(gkf.split(df, groups=groups)):
        splits.append(Split("group", fold, tr, te))
    return splits


def random_splits(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> list[Split]:
    """Plain KFold (optimistic: same district/year can appear in train and test)."""
    n_splits = min(n_splits, max(2, len(df) // 5))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        Split("random", fold, tr, te) for fold, (tr, te) in enumerate(kf.split(df))
    ]


def make_splits(
    df: pd.DataFrame,
    scheme: str,
    *,
    test_years: int = 4,
    n_splits: int = 5,
    seed: int = 42,
) -> list[Split]:
    if scheme == "temporal":
        return temporal_splits(df, test_years=test_years)
    if scheme == "group":
        return group_splits(df, n_splits=n_splits)
    if scheme == "random":
        return random_splits(df, n_splits=n_splits, seed=seed)
    raise ValueError(f"Unknown scheme: {scheme}")


def assert_no_leakage(df: pd.DataFrame, split: Split, scheme: str) -> None:
    """Assert the split is leak-free for the given scheme (used by tests)."""
    train = df.iloc[split.train_idx]
    test = df.iloc[split.test_idx]
    assert len(set(split.train_idx) & set(split.test_idx)) == 0, "train/test overlap"

    if scheme == "temporal":
        assert train["year"].max() < test["year"].min(), "temporal years overlap"
    elif scheme == "group":
        assert not (set(train["district"]) & set(test["district"])), "district overlap"
