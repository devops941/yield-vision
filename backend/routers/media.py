"""Image and video upload endpoints: analyse a field, then predict yield.

Two-step flow:
1. ``POST /api/media/upload`` stores the file, extracts vegetation indices and
   returns them (plus a preview URL for images).
2. ``POST /api/media/{media_id}/predict`` maps those indices onto the active
   model's inputs and returns a yield prediction with an uncertainty range.

Every step is mirrored to MongoDB when a connection string is configured.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.deps import db_session
from src.config import DATA_DIR, ensure_dirs
from src.data import media as media_mod
from src.data import mongo
from src.models import registry
from src.models.intervals import IntervalModel

router = APIRouter(prefix="/api/media", tags=["media"])

ALLOWED = media_mod.IMAGE_SUFFIXES | media_mod.VIDEO_SUFFIXES


def _media_dir() -> Path:
    ensure_dirs()
    path = DATA_DIR / "media"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _next_media_id() -> int:
    docs = mongo.list_media(limit=10000)
    ids = [int(d.get("media_id", 0)) for d in docs]
    return (max(ids) + 1) if ids else 1


@router.post("/upload")
async def upload_media(
    file: UploadFile = File(...),
    state: str = Form("Punjab"),
    district: str = Form("Ludhiana"),
    crop: str = Form("rice"),
    season: str = Form("Kharif"),
    year: int = Form(2024),
) -> dict:
    """Store an image/video, extract vegetation indices and return a preview."""
    filename = file.filename or "field.jpg"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file '{suffix}'. Upload an image or video.",
        )

    content = await file.read()
    media_id = _next_media_id()
    stored = _media_dir() / f"media_{media_id}{suffix}"
    stored.write_bytes(content)

    try:
        analysis = media_mod.analyse_media(stored, filename)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not analyse media: {exc}") from exc

    doc = {
        "media_id": media_id,
        "filename": filename,
        "kind": analysis["kind"],
        "stored_path": str(stored),
        "frames": analysis.get("frames"),
        "indices": analysis["indices"],
        "vigour": analysis["vigour"],
        "frame_variation": analysis.get("frame_variation"),
        "context": {"state": state, "district": district, "crop": crop, "season": season, "year": year},
        "preview_url": f"/api/media/{media_id}/file" if analysis["kind"] == "image" else None,
    }
    mongo.save_media(doc)
    return doc


@router.get("/{media_id}/file")
def media_file(media_id: int) -> FileResponse:
    """Serve the stored media file (used for image previews)."""
    for doc in mongo.list_media(limit=10000):
        if int(doc.get("media_id", -1)) == media_id:
            path = Path(doc.get("stored_path", ""))
            if path.exists():
                return FileResponse(str(path))
            raise HTTPException(status_code=404, detail="Media file is missing on disk")
    raise HTTPException(status_code=404, detail=f"Media {media_id} not found")


@router.get("")
def list_media(limit: int = 50) -> dict:
    """Recent media analyses."""
    return {"media": mongo.list_media(limit=limit), "mongo": mongo.status()}


@router.post("/{media_id}/predict")
def predict_from_media(media_id: int, session: Session = Depends(db_session)) -> dict:
    """Predict yield from a stored media analysis using the active model."""
    docs = [d for d in mongo.list_media(limit=10000) if int(d.get("media_id", -1)) == media_id]
    if not docs:
        raise HTTPException(status_code=404, detail=f"Media {media_id} not found")
    doc = docs[0]

    entry = registry.get_active_model(session)
    if entry is None:
        raise HTTPException(
            status_code=409,
            detail="No active model. Train and activate a model first.",
        )
    stored = registry.load_model_by_id(session, entry.model_id)
    pipeline = stored.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=400, detail="The active model has no persisted pipeline.")
    interval_model = (
        IntervalModel.from_dict(stored["interval_model"]) if stored.get("interval_model") else None
    )

    analysis = {
        "vigour": doc.get("vigour", 0.5),
        "indices": doc.get("indices", {}),
    }
    features = media_mod.features_from_analysis(analysis, doc.get("context", {}))
    features.pop("_brightness", None)

    frame = pd.DataFrame([features])
    warnings = []
    if interval_model is not None and interval_model.ranges is not None:
        warnings = interval_model.ranges.check_row(features)

    if interval_model is not None:
        preds, low, high = interval_model.predict_interval(pipeline, frame)
        y_pred, pred_low, pred_high = float(preds[0]), float(low[0]), float(high[0])
    else:
        y_pred = float(pipeline.predict(frame)[0])
        pred_low = pred_high = None

    result = {
        "media_id": media_id,
        "model_id": entry.model_id,
        "model_name": entry.name,
        "vigour": doc.get("vigour"),
        "y_pred": y_pred,
        "pred_low": pred_low,
        "pred_high": pred_high,
        "interval_level": interval_model.nominal_coverage if interval_model else None,
        "derived_inputs": features,
        "warnings": warnings,
    }
    mongo.save_prediction(
        {
            "source": "media",
            "media_id": media_id,
            "model_id": entry.model_id,
            "inputs": features,
            "y_pred": y_pred,
            "pred_low": pred_low,
            "pred_high": pred_high,
            "warnings": warnings,
        }
    )
    return result
