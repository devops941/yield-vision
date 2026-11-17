"""Shared FastAPI dependencies and in-process state helpers."""

from __future__ import annotations

import datetime as dt
import threading
from typing import Any, Iterator

from sqlalchemy.orm import Session

from src.data.db import get_session, init_db

# In-process job store for background training (single-user demo app).
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def db_session() -> Iterator[Session]:
    """FastAPI dependency that yields a SQLAlchemy session."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database() -> None:
    init_db()


def create_job(kind: str) -> str:
    job_id = f"{kind}-{dt.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "kind": kind,
            "status": "pending",
            "progress": 0,
            "message": "",
            "result": None,
            "error": None,
            "created_at": dt.datetime.now().isoformat(),
            "updated_at": dt.datetime.now().isoformat(),
        }
    return job_id


def update_job(job_id: str, **fields: Any) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job.update(fields)
        job["updated_at"] = dt.datetime.now().isoformat()


def get_job(job_id: str) -> dict[str, Any] | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def list_jobs(kind: str | None = None) -> list[dict[str, Any]]:
    with _JOBS_LOCK:
        jobs = [dict(j) for j in _JOBS.values()]
    if kind:
        jobs = [j for j in jobs if j["kind"] == kind]
    return sorted(jobs, key=lambda j: j["created_at"], reverse=True)
