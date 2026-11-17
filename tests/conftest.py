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


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    """TestClient bound to an isolated temporary SQLite database."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from src.data import db

    path = tmp_path / "api.db"
    engine = create_engine(f"sqlite:///{path}", future=True)
    db.Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, future=True)

    # Point the module-level singletons at the temporary database.
    monkeypatch.setattr(db, "_ENGINE", engine, raising=False)
    monkeypatch.setattr(db, "_SessionLocal", session_local, raising=False)

    # The report module and models use the same globals.
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as client:
        yield client
