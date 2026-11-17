"""Report generation endpoints (spec section 9)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.deps import db_session
from backend.schemas import ReportRequest
from src.data import db as dbmod
from src.data import mongo
from src.report.generate import generate_report

router = APIRouter(prefix="/api/report", tags=["report"])


@router.post("")
def create_report(payload: ReportRequest, session: Session = Depends(db_session)) -> dict:
    """Generate a PDF report and register it in the database."""
    if dbmod.load_records_df(session).empty:
        raise HTTPException(status_code=409, detail="No committed data to report on.")

    try:
        path = generate_report(
            model_id=payload.model_id,
            include_sections=payload.include_sections,
            title=payload.title,
        )
    except Exception as exc:  # pragma: no cover - report engine failure
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}") from exc

    row = dbmod.ReportRow(file_path=str(path))
    session.add(row)
    session.flush()

    mongo.save_report(
        {
            "report_id": row.report_id,
            "file_path": str(path),
            "title": payload.title,
            "model_id": payload.model_id,
            "sections": payload.include_sections,
        }
    )

    return {
        "report_id": row.report_id,
        "file_path": str(path),
        "format": Path(path).suffix.lstrip("."),
        "download_url": f"/api/report/{row.report_id}",
    }


@router.get("/{report_id}")
def download_report(report_id: int, session: Session = Depends(db_session)) -> FileResponse:
    """Download a generated report by id."""
    row = session.get(dbmod.ReportRow, report_id)
    if row is None or not row.file_path:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    path = Path(row.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report file is missing on disk")

    media_type = "application/pdf" if path.suffix == ".pdf" else "text/html"
    return FileResponse(str(path), media_type=media_type, filename=path.name)


@router.get("")
def list_reports(session: Session = Depends(db_session)) -> dict:
    """List generated reports."""
    from sqlalchemy import select

    rows = session.scalars(select(dbmod.ReportRow).order_by(dbmod.ReportRow.report_id.desc())).all()
    return {
        "reports": [
            {
                "report_id": r.report_id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "file_path": r.file_path,
            }
            for r in rows
        ]
    }
