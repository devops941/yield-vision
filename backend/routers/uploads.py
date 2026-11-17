"""Upload, validation-issue and commit endpoints (spec section 9)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.deps import db_session
from backend.schemas import CommitRequest, IssuePatch
from src.config import DATA_DIR, ensure_dirs
from src.data import db as dbmod
from src.data import mongo
from src.data.ingest import canonicalise, load_crop_file
from src.data.preprocess import drop_high_missingness, preprocess
from src.data.synthetic import generate_dataset, write_template
from src.data.validate import validate

router = APIRouter(prefix="/api", tags=["uploads"])


def _uploads_dir() -> Path:
    ensure_dirs()
    return DATA_DIR / "uploads"


@router.get("/template", response_class=PlainTextResponse, tags=["uploads"])
def download_template() -> PlainTextResponse:
    """Download a CSV template with the expected canonical columns."""
    path = DATA_DIR / "sample" / "template.csv"
    write_template(path)
    return PlainTextResponse(
        path.read_text(encoding="utf-8"),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=yieldvision_template.csv"},
    )


@router.post("/uploads")
async def upload_file(
    file: UploadFile = File(...),
    session: Session = Depends(db_session),
) -> dict:
    """Upload a crop/weather/soil file; returns the upload id and validation summary."""
    content = await file.read()
    suffix = Path(file.filename or "upload.csv").suffix or ".csv"
    stored = _uploads_dir() / f"{Path(file.filename or 'upload').stem}{suffix}"
    stored.write_bytes(content)

    try:
        df = load_crop_file(stored, file.filename)
    except Exception as exc:  # pragma: no cover - unreadable file
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}") from exc

    df = canonicalise(df)
    report = validate(df)

    upload = dbmod.create_upload(session, file.filename or stored.name, str(stored))
    upload.n_rows = report.n_rows
    upload.n_errors = report.n_errors
    upload.n_warnings = report.n_warnings
    upload.status = "rejected" if report.has_blocking_errors else "validated"

    issue_dicts = [issue.to_dict() for issue in report.issues]
    dbmod.add_issues(session, upload.upload_id, issue_dicts)
    session.flush()

    return {
        "upload_id": upload.upload_id,
        "filename": upload.filename,
        "status": upload.status,
        "summary": {
            "n_rows": report.n_rows,
            "n_errors": report.n_errors,
            "n_warnings": report.n_warnings,
        },
        "blocking": report.has_blocking_errors,
        "missing_columns": report.missing_columns,
    }


@router.get("/uploads")
def list_uploads(session: Session = Depends(db_session)) -> list[dict]:
    """List all uploads."""
    uploads = session.scalars(select(dbmod.Upload).order_by(dbmod.Upload.upload_id.desc())).all()
    return [
        {
            "upload_id": u.upload_id,
            "filename": u.filename,
            "uploaded_at": u.uploaded_at.isoformat() if u.uploaded_at else None,
            "status": u.status,
            "n_rows": u.n_rows,
            "n_errors": u.n_errors,
            "n_warnings": u.n_warnings,
        }
        for u in uploads
    ]


@router.get("/uploads/{upload_id}/issues")
def list_issues(
    upload_id: int,
    severity: str | None = Query(default=None),
    status: str | None = Query(default=None),
    session: Session = Depends(db_session),
) -> dict:
    """List validation issues for an upload, optionally filtered."""
    upload = session.get(dbmod.Upload, upload_id)
    if upload is None:
        raise HTTPException(status_code=404, detail=f"Upload {upload_id} not found")

    stmt = select(dbmod.ValidationIssue).where(dbmod.ValidationIssue.upload_id == upload_id)
    if severity:
        stmt = stmt.where(dbmod.ValidationIssue.severity == severity)
    if status:
        stmt = stmt.where(dbmod.ValidationIssue.status == status)
    issues = session.scalars(stmt.order_by(dbmod.ValidationIssue.issue_id)).all()

    return {
        "upload_id": upload_id,
        "count": len(issues),
        "issues": [
            {
                "issue_id": i.issue_id,
                "row_index": i.row_index,
                "column": i.column,
                "value": i.value,
                "check": i.check,
                "severity": i.severity,
                "message": i.message,
                "suggestion": i.suggestion,
                "status": i.status,
            }
            for i in issues
        ],
    }


@router.patch("/uploads/{upload_id}/issues/{issue_id}")
def patch_issue(
    upload_id: int,
    issue_id: int,
    payload: IssuePatch,
    session: Session = Depends(db_session),
) -> dict:
    """Fix, ignore or exclude an issue and write an audit row."""
    issue = session.get(dbmod.ValidationIssue, issue_id)
    if issue is None or issue.upload_id != upload_id:
        raise HTTPException(status_code=404, detail=f"Issue {issue_id} not found for upload {upload_id}")

    old_value = issue.value
    status_map = {"fix": "fixed", "ignore": "ignored", "exclude": "excluded"}
    issue.status = status_map[payload.action]
    if payload.action == "fix" and payload.value is not None:
        issue.value = str(payload.value)

    dbmod.log_correction(
        session,
        upload_id=upload_id,
        issue_id=issue_id,
        action=payload.action,
        old_value=old_value,
        new_value=payload.value,
        note=payload.note,
    )
    session.flush()
    return {"issue_id": issue_id, "status": issue.status, "action": payload.action}


@router.post("/uploads/{upload_id}/commit")
def commit_upload(
    upload_id: int,
    payload: CommitRequest,
    session: Session = Depends(db_session),
) -> dict:
    """Pre-process the stored file and commit the cleaned rows to the database."""
    upload = session.get(dbmod.Upload, upload_id)
    if upload is None:
        raise HTTPException(status_code=404, detail=f"Upload {upload_id} not found")

    stored = Path(upload.stored_path) if upload.stored_path else None
    if stored is None or not stored.exists():
        raise HTTPException(status_code=400, detail="Stored file is missing; please re-upload")

    df = canonicalise(load_crop_file(stored, upload.filename))

    # Apply user decisions recorded against the issues before cleaning.
    issues = session.scalars(
        select(dbmod.ValidationIssue).where(dbmod.ValidationIssue.upload_id == upload_id)
    ).all()

    open_errors = [i for i in issues if i.severity == "error" and i.status == "open"]
    if open_errors:
        raise HTTPException(
            status_code=409,
            detail=(
                f"{len(open_errors)} error(s) still need a decision. "
                "Fix, ignore or exclude each one before committing."
            ),
        )

    for issue in issues:
        if issue.status != "fixed" or issue.column is None or issue.row_index is None:
            continue
        if issue.column not in df.columns or issue.row_index not in df.index:
            continue
        new_value: object = issue.value
        if pd.api.types.is_numeric_dtype(df[issue.column]):
            try:
                new_value = float(issue.value)
            except (TypeError, ValueError):
                new_value = np.nan
        df.at[issue.row_index, issue.column] = new_value

    clean = preprocess(df, duplicate_policy=payload.duplicate_policy)
    if payload.drop_high_missingness:
        clean = drop_high_missingness(clean)

    excluded_rows = {
        i.row_index for i in issues if i.status == "excluded" and i.row_index is not None
    }
    if excluded_rows:
        clean = clean.drop(index=[r for r in excluded_rows if r in clean.index], errors="ignore")

    inserted = dbmod.commit_records(session, upload_id, clean)
    upload.status = "committed"
    session.flush()

    mongo.save_upload(
        {
            "upload_id": upload_id,
            "filename": upload.filename,
            "status": upload.status,
            "n_rows": int(len(clean)),
        }
    )
    mongo.sync_records(dbmod.load_records_df(session))

    return {
        "upload_id": upload_id,
        "status": upload.status,
        "inserted": inserted,
    }


@router.post("/datasets/sample")
def load_sample(session: Session = Depends(db_session)) -> dict:
    """Generate the built-in synthetic dataset and commit it to the database."""
    df = generate_dataset()
    clean = preprocess(df)

    upload = dbmod.create_upload(session, "synthetic_sample.csv", stored_path=None)
    report = validate(df)
    upload.n_rows = report.n_rows
    upload.n_errors = report.n_errors
    upload.n_warnings = report.n_warnings
    upload.status = "committed"
    session.flush()

    inserted = dbmod.commit_records(session, upload.upload_id, clean)
    mongo.sync_records(dbmod.load_records_df(session))
    return {
        "upload_id": upload.upload_id,
        "status": "committed",
        "inserted": inserted,
        "validation": {
            "n_rows": report.n_rows,
            "n_errors": report.n_errors,
            "n_warnings": report.n_warnings,
        },
        "counts": dbmod.dataset_counts(session),
        "mongo": mongo.status(),
    }
