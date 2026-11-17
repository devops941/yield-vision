"""Model training, registry and evaluation endpoints (spec section 9)."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.deps import create_job, db_session, get_job, list_jobs, update_job
from backend.schemas import TrainRequest
from src.data import db as dbmod
from src.features.engineer import FeatureConfig
from src.models import registry
from src.models.service import train_and_register

router = APIRouter(prefix="/api/models", tags=["models"])


def _run_training(job_id: str, payload: TrainRequest) -> None:
    from src.data.db import get_session

    update_job(job_id, status="running", progress=5, message="Loading data")
    session = get_session()
    try:
        if payload.dataset == "sample":
            from src.data.preprocess import preprocess
            from src.data.synthetic import generate_dataset

            df = preprocess(generate_dataset())
        else:
            df = dbmod.load_records_df(session)
        if df.empty:
            update_job(job_id, status="failed", error="No data available for training")
            return

        feature_config = FeatureConfig(**payload.feature_config.model_dump())

        def progress(pct: int, message: str) -> None:
            update_job(job_id, progress=pct, message=message)

        result = train_and_register(
            df,
            models=payload.models,
            schemes=payload.schemes,
            feature_config=feature_config,
            tune=payload.tune,
            test_years=payload.test_years,
            progress=progress,
        )
        update_job(job_id, status="completed", progress=100, message="Training complete", result=result)
    except Exception as exc:  # pragma: no cover - background failure path
        update_job(job_id, status="failed", error=str(exc))
    finally:
        session.close()


@router.post("/train")
def start_training(
    payload: TrainRequest,
    background: BackgroundTasks,
    session: Session = Depends(db_session),
) -> dict:
    """Start a background training job; poll the returned job id for progress."""
    if payload.dataset == "database" and dbmod.load_records_df(session).empty:
        raise HTTPException(
            status_code=409,
            detail="No committed data. Upload a file or load the sample dataset first.",
        )
    job_id = create_job("training")
    background.add_task(_run_training, job_id, payload)
    return {"job_id": job_id, "status": "pending", "models": payload.models, "schemes": payload.schemes}


@router.get("/jobs")
def jobs() -> dict:
    """List training jobs."""
    return {"jobs": list_jobs("training")}


@router.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    """Poll a training job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@router.get("")
def list_models(session: Session = Depends(db_session)) -> dict:
    """Model registry with status and headline metrics."""
    return {"models": registry.list_models(session)}


@router.get("/compare")
def compare(
    scheme: str = Query(default="temporal"),
    session: Session = Depends(db_session),
) -> dict:
    """Side-by-side metric comparison for one scheme."""
    models = registry.list_models(session)
    rows = []
    for model in models:
        metrics = model["metrics"].get(scheme)
        if metrics:
            rows.append({"model_id": model["model_id"], "name": model["name"], "scheme": scheme, **metrics})
    rows.sort(key=lambda r: (r.get("rmse") is None, r.get("rmse") or 0.0))
    return {"scheme": scheme, "comparison": rows}


@router.get("/{model_id}/metrics")
def model_metrics(
    model_id: int,
    scheme: str | None = None,
    scope: str | None = None,
    session: Session = Depends(db_session),
) -> dict:
    """Detailed metrics, optionally filtered by scheme and scope type."""
    entry = session.get(dbmod.ModelRegistry, model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return {
        "model_id": model_id,
        "name": entry.name,
        "algo": entry.algo,
        "is_active": bool(entry.is_active),
        "metrics": registry.get_metrics(session, model_id, scheme=scheme, scope_type=scope),
    }


@router.get("/{model_id}/predictions")
def model_predictions(
    model_id: int,
    scheme: str = Query(default="temporal"),
    limit: int = 500,
    session: Session = Depends(db_session),
) -> dict:
    """Actual vs predicted values for a model under a scheme."""
    from sqlalchemy import select

    rows = session.scalars(
        select(dbmod.EvalPrediction)
        .where(dbmod.EvalPrediction.model_id == model_id, dbmod.EvalPrediction.scheme == scheme)
        .limit(limit)
    ).all()
    return {
        "model_id": model_id,
        "scheme": scheme,
        "points": [
            {"y_true": r.y_true, "y_pred": r.y_pred, "residual": (r.y_pred - r.y_true) if (r.y_pred is not None and r.y_true is not None) else None}
            for r in rows
        ],
    }


@router.get("/{model_id}/importance")
def model_importance(model_id: int, session: Session = Depends(db_session)) -> dict:
    """Feature importance table (impurity and permutation)."""
    from sqlalchemy import select

    rows = session.scalars(
        select(dbmod.FeatureImportanceRow).where(dbmod.FeatureImportanceRow.model_id == model_id)
    ).all()
    data = [
        {
            "feature": r.feature,
            "impurity_importance": r.impurity_importance,
            "permutation_importance": r.permutation_importance,
            "permutation_std": r.permutation_std,
        }
        for r in rows
    ]
    data.sort(key=lambda r: (r["permutation_importance"] is None, -(r["permutation_importance"] or 0.0)))
    return {"model_id": model_id, "importance": data}


@router.get("/{model_id}/pdp")
def model_pdp(
    model_id: int,
    feature: str = Query(default="rainfall_mm"),
    session: Session = Depends(db_session),
) -> dict:
    """Partial dependence curve for a feature using the persisted pipeline."""
    payload = registry.load_model_by_id(session, model_id)
    pipeline = payload.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=400, detail="This model has no persisted pipeline (baseline).")
    df = dbmod.load_records_df(session)
    if df.empty:
        raise HTTPException(status_code=409, detail="No committed data.")
    from src.models.explain import partial_dependence_curve

    curve = partial_dependence_curve(pipeline, df, feature)
    return {"model_id": model_id, **curve}


@router.get("/{model_id}/summary")
def model_summary(model_id: int, session: Session = Depends(db_session)) -> dict:
    """Plain-language summary of the drivers for a model."""
    from sqlalchemy import select

    import pandas as pd

    from src.models.explain import plain_language_summary

    rows = session.scalars(
        select(dbmod.FeatureImportanceRow).where(dbmod.FeatureImportanceRow.model_id == model_id)
    ).all()
    if not rows:
        return {"model_id": model_id, "summary": "No importance data available for this model."}
    frame = pd.DataFrame(
        [
            {
                "feature": r.feature,
                "importance_mean": r.permutation_importance
                if r.permutation_importance is not None
                else (r.impurity_importance or 0.0),
                "importance_std": r.permutation_std or 0.0,
            }
            for r in rows
        ]
    ).sort_values("importance_mean", ascending=False)
    return {"model_id": model_id, "summary": plain_language_summary(frame)}


@router.post("/{model_id}/activate")
def activate(model_id: int, session: Session = Depends(db_session)) -> dict:
    """Mark a model version as the active one used for predictions."""
    entry = session.get(dbmod.ModelRegistry, model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    registry.activate_model(session, model_id)
    return {"model_id": model_id, "is_active": True}
