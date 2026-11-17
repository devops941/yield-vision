"""Model registry: persist/load/activate joblib model versions (spec section 5 step 9)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import joblib
from sqlalchemy import select, update

from src.config import MODELS_DIR, ensure_dirs
from src.data.db import ModelMetric, ModelRegistry, get_session


def _next_version(session, name: str) -> int:
    stmt = select(ModelRegistry.version).where(ModelRegistry.name == name)
    versions = [v for v in session.scalars(stmt).all() if v is not None]
    return (max(versions) + 1) if versions else 1


def save_model(
    session,
    *,
    name: str,
    algo: str,
    pipeline,
    interval_model=None,
    params: dict | None = None,
    feature_list: list[str] | None = None,
    train_rows: int | None = None,
) -> int:
    """Persist a fitted pipeline (and its interval model) and register a version."""
    ensure_dirs()
    version = _next_version(session, name)
    file_path = MODELS_DIR / f"{name}_v{version}.joblib"
    payload = {
        "pipeline": pipeline,
        "interval_model": interval_model.to_dict() if interval_model is not None else None,
        "params": params or {},
        "name": name,
        "algo": algo,
        "version": version,
        "saved_at": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(),
    }
    joblib.dump(payload, file_path)

    entry = ModelRegistry(
        name=name,
        algo=algo,
        version=version,
        file_path=str(file_path),
        params_json=json.dumps(params or {}),
        feature_list_json=json.dumps(feature_list or []),
        is_active=0,
        train_rows=train_rows,
    )
    session.add(entry)
    session.flush()
    return entry.model_id


def load_model(file_path: str | Path) -> dict[str, Any]:
    return joblib.load(file_path)


def load_model_by_id(session, model_id: int) -> dict[str, Any]:
    entry = session.get(ModelRegistry, model_id)
    if entry is None or not entry.file_path:
        raise ValueError(f"Model {model_id} not found")
    return load_model(entry.file_path)


def add_metrics(session, model_id: int, scheme: str, metrics: dict[str, Any], *, scope_type: str = "overall", scope_value: str | None = None) -> int:
    row = ModelMetric(
        model_id=model_id,
        scheme=scheme,
        scope_type=scope_type,
        scope_value=scope_value,
        n=metrics.get("n"),
        mae=metrics.get("mae"),
        rmse=metrics.get("rmse"),
        r2=metrics.get("r2"),
        mape=metrics.get("mape"),
        bias=metrics.get("bias"),
        skill_vs_baseline=metrics.get("skill_vs_baseline"),
    )
    session.add(row)
    session.flush()
    return row.metric_id


def activate_model(session, model_id: int) -> None:
    session.execute(update(ModelRegistry).values(is_active=0))
    session.execute(update(ModelRegistry).where(ModelRegistry.model_id == model_id).values(is_active=1))
    session.flush()


def get_active_model(session) -> ModelRegistry | None:
    return session.scalars(
        select(ModelRegistry).where(ModelRegistry.is_active == 1).order_by(ModelRegistry.model_id.desc())
    ).first()


def list_models(session) -> list[dict[str, Any]]:
    entries = session.scalars(select(ModelRegistry).order_by(ModelRegistry.model_id.desc())).all()
    out = []
    for entry in entries:
        metrics = session.scalars(
            select(ModelMetric).where(
                ModelMetric.model_id == entry.model_id,
                ModelMetric.scope_type == "overall",
            )
        ).all()
        out.append(
            {
                "model_id": entry.model_id,
                "name": entry.name,
                "algo": entry.algo,
                "version": entry.version,
                "created_at": entry.created_at.isoformat() if entry.created_at else None,
                "is_active": bool(entry.is_active),
                "train_rows": entry.train_rows,
                "params": json.loads(entry.params_json) if entry.params_json else {},
                "metrics": {
                    m.scheme: {
                        "n": m.n,
                        "mae": m.mae,
                        "rmse": m.rmse,
                        "r2": m.r2,
                        "mape": m.mape,
                        "bias": m.bias,
                        "skill_vs_baseline": m.skill_vs_baseline,
                    }
                    for m in metrics
                },
            }
        )
    return out


def get_metrics(session, model_id: int, scheme: str | None = None, scope_type: str | None = None) -> list[dict[str, Any]]:
    stmt = select(ModelMetric).where(ModelMetric.model_id == model_id)
    if scheme:
        stmt = stmt.where(ModelMetric.scheme == scheme)
    if scope_type:
        stmt = stmt.where(ModelMetric.scope_type == scope_type)
    rows = session.scalars(stmt).all()
    return [
        {
            "scheme": r.scheme,
            "scope_type": r.scope_type,
            "scope_value": r.scope_value,
            "n": r.n,
            "mae": r.mae,
            "rmse": r.rmse,
            "r2": r.r2,
            "mape": r.mape,
            "bias": r.bias,
            "skill_vs_baseline": r.skill_vs_baseline,
        }
        for r in rows
    ]
