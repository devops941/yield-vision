"""Feature engineering endpoints (spec section 9)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.deps import db_session
from backend.schemas import FeatureConfigIn
from src.data import db as dbmod
from src.data.db import FeatureSetRow
from src.features.engineer import FeatureConfig, engineer_features, feature_stats

router = APIRouter(prefix="/api/features", tags=["features"])


def _config_from_payload(payload: FeatureConfigIn) -> FeatureConfig:
    return FeatureConfig(**payload.model_dump())


@router.post("/build")
def build_features(
    payload: FeatureConfigIn,
    session: Session = Depends(db_session),
) -> dict:
    """Build the engineered feature set for the committed data and store its config."""
    df = dbmod.load_records_df(session)
    if df.empty:
        raise HTTPException(status_code=409, detail="No committed data to engineer features from.")

    config = _config_from_payload(payload)
    features = engineer_features(df, config)
    stats = feature_stats(df, config)

    row = FeatureSetRow(
        config_json=json.dumps(config.to_dict()),
        n_rows=int(len(features)),
        n_features=int(features.shape[1]),
    )
    session.add(row)
    session.flush()

    return {
        "feature_set_id": row.feature_set_id,
        "n_rows": int(features.shape[1] and len(features)),
        "n_features": int(features.shape[1]),
        "columns": list(features.columns),
        "stats": stats,
        "config": config.to_dict(),
    }


@router.get("/{feature_set_id}/preview")
def preview_features(
    feature_set_id: int,
    limit: int = 20,
    session: Session = Depends(db_session),
) -> dict:
    """Preview engineered rows and feature statistics for a stored feature set."""
    feature_set = session.get(FeatureSetRow, feature_set_id)
    if feature_set is None:
        raise HTTPException(status_code=404, detail=f"Feature set {feature_set_id} not found")

    df = dbmod.load_records_df(session)
    config = FeatureConfig(**json.loads(feature_set.config_json or "{}"))
    features = engineer_features(df, config)
    preview = features.head(limit).replace({np.nan: None}).to_dict(orient="records")
    for record in preview:
        for key, value in list(record.items()):
            if isinstance(value, np.integer):
                record[key] = int(value)
            elif isinstance(value, np.floating):
                record[key] = None if np.isnan(value) else float(value)

    return {
        "feature_set_id": feature_set_id,
        "n_rows": int(len(features)),
        "n_features": int(features.shape[1]),
        "preview": preview,
        "stats": feature_stats(df, config),
    }
