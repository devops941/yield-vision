"""Manual data-entry endpoints.

Lets a user type crop/weather/soil rows straight into the app instead of
uploading a spreadsheet. Entered rows are validated, pre-processed and written
to the same ``records`` table the upload flow uses, so every analytics page
(Explorer, Features, Train models, Accuracy, Predict, Reports) picks them up
with no extra wiring.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.deps import db_session
from backend.schemas import RecordEntryIn
from src.config import all_crops, all_seasons, crop_settings
from src.data import db as dbmod
from src.data import mongo
from src.data.preprocess import preprocess
from src.data.validate import validate

router = APIRouter(prefix="/api/records", tags=["records"])

# Field metadata drives the entry form so labels, units and grouping stay in
# one place and stay consistent across the whole app.
FIELD_GROUPS: list[dict[str, Any]] = [
    {
        "key": "identity",
        "title": "Where and when",
        "hint": "Location, crop and the season the record belongs to.",
        "fields": [
            {"name": "state", "label": "State", "type": "text", "required": True, "placeholder": "e.g. Punjab"},
            {"name": "district", "label": "District", "type": "text", "required": True, "placeholder": "e.g. Ludhiana"},
            {"name": "crop", "label": "Crop", "type": "select", "required": True, "options_key": "crops"},
            {"name": "season", "label": "Season", "type": "select", "required": True, "options_key": "seasons"},
            {"name": "year", "label": "Year", "type": "number", "required": True, "placeholder": "2024", "min": 1950, "max": 2100},
        ],
    },
    {
        "key": "output",
        "title": "Harvest",
        "hint": "Give the yield directly, or the area and production and we work it out.",
        "fields": [
            {"name": "yield_t_ha", "label": "Yield", "type": "number", "unit": "t/ha", "required": False, "step": 0.01, "placeholder": "e.g. 4.2"},
            {"name": "area_ha", "label": "Area", "type": "number", "unit": "ha", "required": False, "step": 1, "placeholder": "e.g. 1000"},
            {"name": "production_t", "label": "Production", "type": "number", "unit": "t", "required": False, "step": 1, "placeholder": "e.g. 4200"},
        ],
    },
    {
        "key": "weather",
        "title": "Weather",
        "hint": "Season totals and averages for the growing period.",
        "fields": [
            {"name": "rainfall_mm", "label": "Rainfall", "type": "number", "unit": "mm", "required": True, "step": 1, "placeholder": "e.g. 850"},
            {"name": "temp_mean_c", "label": "Average temperature", "type": "number", "unit": "°C", "required": False, "step": 0.1, "placeholder": "e.g. 26"},
            {"name": "temp_max_c", "label": "Maximum temperature", "type": "number", "unit": "°C", "required": False, "step": 0.1, "placeholder": "e.g. 34"},
            {"name": "temp_min_c", "label": "Minimum temperature", "type": "number", "unit": "°C", "required": False, "step": 0.1, "placeholder": "e.g. 18"},
            {"name": "humidity_pct", "label": "Humidity", "type": "number", "unit": "%", "required": False, "step": 1, "min": 0, "max": 100, "placeholder": "e.g. 65"},
        ],
    },
    {
        "key": "soil",
        "title": "Soil",
        "hint": "Soil test values. Leave any you do not have blank.",
        "fields": [
            {"name": "soil_n_kgha", "label": "Soil nitrogen", "type": "number", "unit": "kg/ha", "required": False, "step": 1, "placeholder": "e.g. 250"},
            {"name": "soil_p_kgha", "label": "Soil phosphorus", "type": "number", "unit": "kg/ha", "required": False, "step": 1, "placeholder": "e.g. 25"},
            {"name": "soil_k_kgha", "label": "Soil potassium", "type": "number", "unit": "kg/ha", "required": False, "step": 1, "placeholder": "e.g. 200"},
            {"name": "soil_ph", "label": "Soil pH", "type": "number", "required": False, "step": 0.1, "min": 3, "max": 10, "placeholder": "e.g. 6.5"},
            {"name": "soil_oc_pct", "label": "Organic carbon", "type": "number", "unit": "%", "required": False, "step": 0.01, "placeholder": "e.g. 0.6"},
        ],
    },
    {
        "key": "management",
        "title": "Management",
        "hint": "Optional. Improves the model when provided.",
        "fields": [
            {"name": "fertilizer_kgha", "label": "Fertilizer used", "type": "number", "unit": "kg/ha", "required": False, "step": 1, "placeholder": "e.g. 120"},
            {"name": "irrigated_frac", "label": "Irrigated share", "type": "number", "required": False, "step": 0.05, "min": 0, "max": 1, "placeholder": "0 – 1"},
            {"name": "pesticide_kgha", "label": "Pesticide used", "type": "number", "unit": "kg/ha", "required": False, "step": 0.1, "placeholder": "e.g. 2.5"},
        ],
    },
]

NUMERIC_FIELDS = [
    field["name"]
    for group in FIELD_GROUPS
    for field in group["fields"]
    if field["type"] == "number"
]
TEXT_FIELDS = ["state", "district", "crop", "season"]


@router.get("/meta")
def entry_meta(session: Session = Depends(db_session)) -> dict:
    """Field definitions plus pick-lists for the data-entry form."""
    districts = sorted(
        {
            row
            for row in session.scalars(select(dbmod.Region.district).distinct()).all()
            if row
        }
    )
    states = sorted(
        {row for row in session.scalars(select(dbmod.Region.state).distinct()).all() if row}
    )
    return {
        "groups": FIELD_GROUPS,
        "options": {
            "crops": all_crops(),
            "seasons": all_seasons(),
            "states": states,
            "districts": districts,
            "years": list(range(2005, 2027)),
        },
    }


@router.get("/recent")
def recent_records(
    limit: int = Query(default=25, ge=1, le=500),
    session: Session = Depends(db_session),
) -> dict:
    """The most recently committed records, newest first."""
    df = dbmod.load_records_df(session)
    if df.empty:
        return {"count": 0, "records": []}
    df = df.sort_values("record_id", ascending=False).head(limit)
    records = df.replace({np.nan: None}).to_dict(orient="records")
    for row in records:
        for key, value in list(row.items()):
            if isinstance(value, np.integer):
                row[key] = int(value)
            elif isinstance(value, np.floating):
                row[key] = None if np.isnan(value) else float(value)
    return {"count": len(df), "records": records}


def _rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Turn submitted rows into a canonical DataFrame, deriving yield if needed."""
    records: list[dict[str, Any]] = []
    for raw in rows:
        row: dict[str, Any] = {}
        for name in TEXT_FIELDS:
            value = raw.get(name)
            row[name] = None if value in (None, "") else str(value).strip()
        for name in NUMERIC_FIELDS:
            value = raw.get(name)
            if value in (None, ""):
                row[name] = np.nan
            else:
                try:
                    row[name] = float(value)
                except (TypeError, ValueError):
                    row[name] = np.nan
        records.append(row)

    df = pd.DataFrame(records, columns=TEXT_FIELDS + NUMERIC_FIELDS)

    # Yield can be derived when it is absent but area and production are present.
    if {"yield_t_ha", "area_ha", "production_t"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            derived = df["production_t"] / df["area_ha"].replace(0, np.nan)
        df["yield_t_ha"] = df["yield_t_ha"].fillna(derived)
    return df


@router.post("/entry")
def create_records(
    payload: RecordEntryIn,
    session: Session = Depends(db_session),
) -> dict:
    """Validate and commit manually entered rows.

    Blocking errors stop the write; warnings are returned for the user to review
    but do not prevent saving.
    """
    if not payload.rows:
        raise HTTPException(status_code=422, detail="Add at least one row to save.")

    df = _rows_to_frame([row.model_dump() for row in payload.rows])
    report = validate(df)

    if report.has_blocking_errors:
        return {
            "saved": False,
            "inserted": 0,
            "validation": report.to_dict(),
        }

    clean = preprocess(df)

    upload = dbmod.create_upload(session, "manual_entry", stored_path=None)
    upload.n_rows = report.n_rows
    upload.n_errors = report.n_errors
    upload.n_warnings = report.n_warnings
    upload.status = "committed"
    session.flush()

    inserted = dbmod.commit_records(session, upload.upload_id, clean)

    mongo.save_upload(
        {
            "upload_id": upload.upload_id,
            "filename": "manual_entry",
            "status": "committed",
            "n_rows": int(inserted),
        }
    )
    mongo.sync_records(dbmod.load_records_df(session))

    return {
        "saved": True,
        "inserted": inserted,
        "upload_id": upload.upload_id,
        "validation": report.to_dict(),
        "counts": dbmod.dataset_counts(session),
    }


@router.delete("/{record_id}")
def delete_record(record_id: int, session: Session = Depends(db_session)) -> dict:
    """Remove a single entered record."""
    record = session.get(dbmod.Record, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Record {record_id} not found")
    session.delete(record)
    session.flush()
    mongo.sync_records(dbmod.load_records_df(session))
    return {"deleted": record_id, "counts": dbmod.dataset_counts(session)}


@router.get("/crop-guide")
def crop_guide() -> dict:
    """Agronomic reference ranges, shown as gentle hints beside the form."""
    guide = {}
    for crop in all_crops():
        cfg = crop_settings(crop)
        guide[crop] = {
            "display_name": cfg.get("display_name", crop.title()),
            "yield_min": cfg.get("yield_min"),
            "yield_max": cfg.get("yield_max"),
            "optimal_ph": cfg.get("optimal_ph"),
            "rainfall_optimal_mm": cfg.get("rainfall_optimal_mm"),
        }
    return {"crops": guide}
