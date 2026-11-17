"""Trend explorer endpoints (spec section 9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.deps import db_session
from src.analysis.trends import (
    CLIMATE_VARIABLES,
    climate_yield_correlation,
    climate_yield_scatter,
    crop_comparison,
    top_bottom_regions,
    yield_time_series,
)
from src.data import db as dbmod

router = APIRouter(prefix="/api/explorer", tags=["explorer"])


def _load_df(session: Session) -> pd.DataFrame:
    df = dbmod.load_records_df(session)
    if df.empty:
        raise HTTPException(
            status_code=409,
            detail="No committed data. Upload a file or load the sample dataset first.",
        )
    return df


def _records(df: pd.DataFrame) -> list[dict]:
    out = df.replace({np.nan: None}).to_dict(orient="records")
    for row in out:
        for key, value in row.items():
            if isinstance(value, (np.integer,)):
                row[key] = int(value)
            elif isinstance(value, (np.floating,)):
                row[key] = None if np.isnan(value) else float(value)
    return out


@router.get("/summary")
def summary(session: Session = Depends(db_session)) -> dict:
    """Summary cards: rows, crops, regions, year range and mean yield."""
    df = _load_df(session)
    return {
        "rows": int(len(df)),
        "crops": sorted(df["crop"].dropna().unique().tolist()),
        "states": sorted(df["state"].dropna().unique().tolist()),
        "districts": sorted(df["district"].dropna().unique().tolist()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "mean_yield": float(df["yield_t_ha"].mean()),
        "median_yield": float(df["yield_t_ha"].median()),
        "n_crops": int(df["crop"].nunique()),
        "n_regions": int(df[["state", "district"]].drop_duplicates().shape[0]),
    }


@router.get("/filters")
def filters(session: Session = Depends(db_session)) -> dict:
    """Available filter options for the explorer UI."""
    df = _load_df(session)
    return {
        "crops": sorted(df["crop"].dropna().unique().tolist()),
        "states": sorted(df["state"].dropna().unique().tolist()),
        "districts": sorted(df["district"].dropna().unique().tolist()),
        "seasons": sorted(df["season"].dropna().unique().tolist()),
        "years": sorted(int(y) for y in df["year"].dropna().unique().tolist()),
        "climate_variables": CLIMATE_VARIABLES,
    }


@router.get("/trend")
def trend(
    crop: str | None = None,
    district: str | None = None,
    start: int | None = None,
    end: int | None = None,
    session: Session = Depends(db_session),
) -> dict:
    """Yield time series with a linear trend line plus Mann-Kendall statistics."""
    df = _load_df(session)
    series = yield_time_series(df, crop=crop, district=district, start=start, end=end)
    from src.analysis.trends import mann_kendall

    stats = mann_kendall(
        series["year"].to_numpy(dtype=float), series["mean_yield"].to_numpy(dtype=float)
    ) if len(series) else {"trend": "insufficient_data"}

    return {
        "filters": {"crop": crop, "district": district, "start": start, "end": end},
        "series": _records(series),
        "trend": stats,
    }


@router.get("/climate-yield")
def climate_yield(
    crop: str | None = None,
    variable: str = Query(default="rainfall_mm"),
    session: Session = Depends(db_session),
) -> dict:
    """Scatter data and Spearman correlation between a climate variable and yield."""
    df = _load_df(session)
    if variable not in df.columns:
        raise HTTPException(status_code=422, detail=f"Unknown variable '{variable}'")
    corr = climate_yield_correlation(df, variable=variable, crop=crop)
    scatter = climate_yield_scatter(df, variable=variable, crop=crop)
    return {"correlation": corr, "points": _records(scatter)}


@router.get("/regions")
def regions(
    crop: str | None = None, n: int = 5, session: Session = Depends(db_session)
) -> dict:
    """Top and bottom regions by mean yield."""
    df = _load_df(session)
    return top_bottom_regions(df, crop=crop, n=n)


@router.get("/crops")
def crops(session: Session = Depends(db_session)) -> dict:
    """Crop comparison table."""
    df = _load_df(session)
    return {"crops": _records(crop_comparison(df))}


@router.get("/records")
def records(
    crop: str | None = None,
    district: str | None = None,
    year: int | None = None,
    limit: int = 200,
    session: Session = Depends(db_session),
) -> dict:
    """Underlying records table with filters (for CSV export in the UI)."""
    df = _load_df(session)
    if crop:
        df = df[df["crop"] == crop]
    if district:
        df = df[df["district"] == district]
    if year:
        df = df[df["year"] == year]
    return {"count": int(len(df)), "records": _records(df.head(limit))}
