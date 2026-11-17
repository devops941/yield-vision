"""Shared pytest fixtures for the YieldVision test-suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def synthetic_df():
    from src.data.synthetic import generate_dataset

    return generate_dataset(n_districts=8, n_crops=3, n_years=12, seed=7)


@pytest.fixture(scope="session")
def clean_df():
    from src.data.synthetic import generate_dataset

    return generate_dataset(n_districts=8, n_crops=3, n_years=12, seed=7, inject_errors=False)


@pytest.fixture()
def tmp_db(tmp_path):
    from src.data import db

    path = tmp_path / "test.db"
    db.init_db(path)
    return path
