"""MongoDB Atlas persistence layer for YieldVision.

MongoDB stores every user-facing artifact: uploaded datasets, the committed
record table, media uploads (image/video) with their extracted features and
predictions, single/batch predictions, model registry entries, and generated
reports. SQLite remains the fast local cache for the modelling pipeline.

The connection string is read from ``MONGO_URI`` (loaded from ``.env``). When it
is missing or the cluster is unreachable, every function degrades gracefully:
``mongo_enabled()`` returns ``False`` and callers fall back to SQLite only.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Any

_CLIENT = None
_DB = None
_STATUS: dict[str, Any] = {"enabled": False, "error": None, "db": None}


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional dependency
        return
    from src.config import PROJECT_ROOT

    load_dotenv(PROJECT_ROOT / ".env")


def _db_name() -> str:
    _load_env()
    return os.environ.get("MONGO_DB", "yieldvision")


def get_db():
    """Return the MongoDB database handle, or ``None`` when unavailable."""
    global _CLIENT, _DB
    if _DB is not None:
        return _DB
    _load_env()
    if os.environ.get("YIELDVISION_NO_MONGO"):
        _STATUS.update(enabled=False, error="disabled via YIELDVISION_NO_MONGO")
        return None
    uri = os.environ.get("MONGO_URI")
    if not uri:
        _STATUS.update(enabled=False, error="MONGO_URI not set")
        return None
    try:
        from pymongo import MongoClient

        _CLIENT = MongoClient(uri, serverSelectionTimeoutMS=6000, appname="yieldvision")
        _CLIENT.admin.command("ping")
        _DB = _CLIENT[_db_name()]
        _STATUS.update(enabled=True, error=None, db=_db_name())
        _ensure_indexes(_DB)
    except Exception as exc:  # pragma: no cover - network dependent
        _STATUS.update(enabled=False, error=f"{type(exc).__name__}: {exc}")
        _DB = None
    return _DB


def mongo_enabled() -> bool:
    return get_db() is not None


def status() -> dict[str, Any]:
    get_db()
    return dict(_STATUS)


def _ensure_indexes(db) -> None:
    try:
        db.records.create_index("record_id", unique=True)
        db.records.create_index([("crop", 1), ("year", 1)])
        db.uploads.create_index("upload_id", unique=True)
        db.predictions.create_index("created_at")
        db.media.create_index("media_id", unique=True)
        db.models.create_index("model_id", unique=True)
        db.reports.create_index("report_id", unique=True)
    except Exception:  # pragma: no cover - index creation is best-effort
        pass


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


# --------------------------------------------------------------------------
# Records (the committed dataset)
# --------------------------------------------------------------------------


def sync_records(df) -> int:
    """Replace the ``records`` collection with the rows of ``df``."""
    db = get_db()
    if db is None or df is None or len(df) == 0:
        return 0
    docs = _frame_to_docs(df)
    db.records.delete_many({})
    if docs:
        db.records.insert_many(docs, ordered=False)
    return len(docs)


def insert_records(df) -> int:
    db = get_db()
    if db is None or df is None or len(df) == 0:
        return 0
    docs = _frame_to_docs(df)
    if docs:
        db.records.insert_many(docs, ordered=False)
    return len(docs)


def load_records() -> list[dict]:
    db = get_db()
    if db is None:
        return []
    return list(db.records.find({}, {"_id": 0}))


def record_count() -> int:
    db = get_db()
    if db is None:
        return 0
    return db.records.estimated_document_count()


def _frame_to_docs(df) -> list[dict]:
    import pandas as pd

    frame = df.copy()
    for col in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            frame[col] = frame[col].astype(str)
    frame = frame.where(pd.notna(frame), None)
    return frame.to_dict(orient="records")


# --------------------------------------------------------------------------
# Uploads
# --------------------------------------------------------------------------


def save_upload(meta: dict) -> dict:
    db = get_db()
    if db is None:
        return meta
    doc = dict(meta)
    doc.setdefault("created_at", _now())
    db.uploads.update_one(
        {"upload_id": doc.get("upload_id")}, {"$set": doc}, upsert=True
    )
    return doc


def list_uploads() -> list[dict]:
    db = get_db()
    if db is None:
        return []
    return list(db.uploads.find({}, {"_id": 0}).sort("upload_id", -1))


# --------------------------------------------------------------------------
# Media (image / video) uploads with extracted features + predictions
# --------------------------------------------------------------------------


def save_media(doc: dict) -> dict:
    db = get_db()
    if db is None:
        return doc
    payload = dict(doc)
    payload.setdefault("created_at", _now())
    if db is not None:
        db.media.insert_one(dict(payload))
    return payload


def list_media(limit: int = 50) -> list[dict]:
    db = get_db()
    if db is None:
        return []
    return list(db.media.find({}, {"_id": 0}).sort("created_at", -1).limit(limit))


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------


def save_prediction(doc: dict) -> dict:
    db = get_db()
    if db is None:
        return doc
    payload = dict(doc)
    payload.setdefault("created_at", _now())
    db.predictions.insert_one(dict(payload))
    return payload


def list_predictions(limit: int = 50) -> list[dict]:
    db = get_db()
    if db is None:
        return []
    return list(db.predictions.find({}, {"_id": 0}).sort("created_at", -1).limit(limit))


# --------------------------------------------------------------------------
# Model registry mirror
# --------------------------------------------------------------------------


def save_model(doc: dict) -> dict:
    db = get_db()
    if db is None:
        return doc
    payload = dict(doc)
    payload.setdefault("created_at", _now())
    db.models.update_one({"model_id": payload.get("model_id")}, {"$set": payload}, upsert=True)
    return payload


def list_models() -> list[dict]:
    db = get_db()
    if db is None:
        return []
    return list(db.models.find({}, {"_id": 0}).sort("model_id", 1))


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


def save_report(doc: dict) -> dict:
    db = get_db()
    if db is None:
        return doc
    payload = dict(doc)
    payload.setdefault("created_at", _now())
    db.reports.update_one({"report_id": payload.get("report_id")}, {"$set": payload}, upsert=True)
    return payload


def list_reports() -> list[dict]:
    db = get_db()
    if db is None:
        return []
    return list(db.reports.find({}, {"_id": 0}).sort("report_id", -1))

