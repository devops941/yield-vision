"""Export endpoints: cleaned data CSV and results Excel (spec section 9)."""

from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.deps import db_session
from src.data import db as dbmod

router = APIRouter(prefix="/api/export", tags=["export"])


@router.get("/data.csv")
def export_data_csv(session: Session = Depends(db_session)) -> StreamingResponse:
    """Download the cleaned committed data as CSV."""
    df = dbmod.load_records_df(session)
    if df.empty:
        raise HTTPException(status_code=409, detail="No committed data to export.")
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=yieldvision_data.csv"},
    )


@router.get("/results.xlsx")
def export_results_xlsx(session: Session = Depends(db_session)) -> StreamingResponse:
    """Download an Excel workbook with data, model registry and metrics sheets."""
    from sqlalchemy import select

    data = dbmod.load_records_df(session)
    if data.empty:
        raise HTTPException(status_code=409, detail="No committed data to export.")

    models = pd.DataFrame(_model_rows(session))
    metrics = pd.DataFrame(_metric_rows(session))
    trends = pd.DataFrame(_trend_rows(session))

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="data", index=False)
        if not models.empty:
            models.to_excel(writer, sheet_name="models", index=False)
        if not metrics.empty:
            metrics.to_excel(writer, sheet_name="metrics", index=False)
        if not trends.empty:
            trends.to_excel(writer, sheet_name="trends", index=False)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=yieldvision_results.xlsx"},
    )


def _model_rows(session: Session) -> list[dict]:
    from src.models import registry

    return registry.list_models(session)


def _metric_rows(session: Session) -> list[dict]:
    from sqlalchemy import select

    rows = session.scalars(select(dbmod.ModelMetric)).all()
    return [
        {
            "model_id": r.model_id,
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


def _trend_rows(session: Session) -> list[dict]:
    from sqlalchemy import select

    rows = session.scalars(select(dbmod.TrendResult)).all()
    return [
        {
            "region_id": r.region_id,
            "crop_id": r.crop_id,
            "trend": r.trend,
            "p_value": r.p_value,
            "tau": r.tau,
            "sen_slope_t_ha_per_year": r.sen_slope_t_ha_per_year,
            "cv_yield": r.cv_yield,
        }
        for r in rows
    ]
