"""Prediction endpoints: single, what-if and batch (spec sections 6.6 and 9)."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.deps import db_session
from backend.schemas import PredictRequest, WhatIfRequest
from src.data import db as dbmod
from src.data.ingest import canonicalise, load_crop_file
from src.data.validate import validate
from src.models import registry
from src.models.intervals import IntervalModel, whatif_curve

router = APIRouter(prefix="/api/predict", tags=["predict"])


def _active_payload(session: Session, model_id: int | None):
    if model_id is not None:
        return registry.load_model_by_id(session, model_id)
    entry = registry.get_active_model(session)
    if entry is None:
        raise HTTPException(
            status_code=409,
            detail="No active model. Train and activate a model first.",
        )
    return registry.load_model_by_id(session, entry.model_id)


def _predict_row(payload, pipeline, interval_model: IntervalModel | None) -> dict:
    frame = pd.DataFrame([payload])
    warnings = []
    if interval_model is not None and interval_model.ranges is not None:
        warnings = interval_model.ranges.check_row(payload)

    if interval_model is not None:
        preds, low, high = interval_model.predict_interval(pipeline, frame)
        y_pred, pred_low, pred_high = float(preds[0]), float(low[0]), float(high[0])
    else:
        y_pred = float(pipeline.predict(frame)[0])
        pred_low = pred_high = None

    return {
        "y_pred": y_pred,
        "pred_low": pred_low,
        "pred_high": pred_high,
        "interval_level": interval_model.nominal_coverage if interval_model else None,
        "warnings": warnings,
        "inputs": payload,
    }


@router.post("")
def predict_single(payload: PredictRequest, session: Session = Depends(db_session)) -> dict:
    """Single prediction with a prediction interval and extrapolation warnings."""
    stored = _active_payload(session, payload.model_id)
    pipeline = stored.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=400, detail="The selected model has no persisted pipeline.")
    interval_model = (
        IntervalModel.from_dict(stored["interval_model"]) if stored.get("interval_model") else None
    )

    body = payload.model_dump()
    model_id = body.pop("model_id", None)
    result = _predict_row(body, pipeline, interval_model)

    record = dbmod.UserPrediction(
        model_id=model_id,
        inputs_json=json.dumps(body),
        y_pred=result["y_pred"],
        pred_low=result["pred_low"],
        pred_high=result["pred_high"],
        warnings=json.dumps(result["warnings"]),
    )
    session.add(record)
    session.flush()

    result["prediction_id"] = record.prediction_id
    return result


@router.post("/whatif")
def predict_whatif(payload: WhatIfRequest, session: Session = Depends(db_session)) -> dict:
    """Sensitivity curve for one varied input, holding the rest fixed."""
    stored = _active_payload(session, payload.model_id)
    pipeline = stored.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=400, detail="The selected model has no persisted pipeline.")
    interval_model = (
        IntervalModel.from_dict(stored["interval_model"]) if stored.get("interval_model") else None
    )

    body = payload.model_dump()
    body.pop("model_id", None)
    feature = body.pop("feature")
    curve = whatif_curve(pipeline, interval_model, body, feature)
    return curve


@router.post("/batch")
async def predict_batch(
    file: UploadFile = File(...),
    model_id: int | None = None,
    session: Session = Depends(db_session),
) -> StreamingResponse:
    """Upload a CSV of conditions and download predictions with intervals."""
    stored = _active_payload(session, model_id)
    pipeline = stored.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=400, detail="The selected model has no persisted pipeline.")
    interval_model = (
        IntervalModel.from_dict(stored["interval_model"]) if stored.get("interval_model") else None
    )

    content = await file.read()
    try:
        df = canonicalise(load_crop_file(io.BytesIO(content), file.filename))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}") from exc

    report = validate(df)
    if report.has_blocking_errors:
        raise HTTPException(
            status_code=422,
            detail={"message": "Validation failed", "summary": report.summary(), "issues": [i.to_dict() for i in report.issues[:50]]},
        )

    frame = df.copy()
    if interval_model is not None:
        preds, low, high = interval_model.predict_interval(pipeline, frame)
        frame["y_pred"] = preds
        frame["pred_low"] = low
        frame["pred_high"] = high
        frame["interval_level"] = interval_model.nominal_coverage
    else:
        frame["y_pred"] = pipeline.predict(frame)

    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=yieldvision_predictions.csv"},
    )


@router.get("/history")
def prediction_history(limit: int = 50, session: Session = Depends(db_session)) -> dict:
    """Recent user predictions."""
    from sqlalchemy import select

    rows = session.scalars(
        select(dbmod.UserPrediction).order_by(dbmod.UserPrediction.prediction_id.desc()).limit(limit)
    ).all()
    return {
        "predictions": [
            {
                "prediction_id": r.prediction_id,
                "model_id": r.model_id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "inputs": json.loads(r.inputs_json) if r.inputs_json else {},
                "y_pred": r.y_pred,
                "pred_low": r.pred_low,
                "pred_high": r.pred_high,
                "warnings": json.loads(r.warnings) if r.warnings else [],
            }
            for r in rows
        ]
    }
