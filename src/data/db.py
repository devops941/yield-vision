"""SQLAlchemy 2.0 models and repositories for YieldVision (SQLite)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from src.config import DB_PATH


class Base(DeclarativeBase):
    pass


class Region(Base):
    __tablename__ = "regions"
    region_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    state: Mapped[str | None] = mapped_column(String)
    district: Mapped[str | None] = mapped_column(String)
    lat: Mapped[float | None] = mapped_column(Float)
    lon: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (UniqueConstraint("state", "district", name="uq_region"),)


class Crop(Base):
    __tablename__ = "crops"
    crop_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    crop_name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    base_temp_c: Mapped[float | None] = mapped_column(Float)
    optimal_ph: Mapped[float | None] = mapped_column(Float)
    yield_min: Mapped[float | None] = mapped_column(Float)
    yield_max: Mapped[float | None] = mapped_column(Float)


class Upload(Base):
    __tablename__ = "uploads"
    upload_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str | None] = mapped_column(String)
    uploaded_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))
    status: Mapped[str] = mapped_column(String, default="validated")
    n_rows: Mapped[int] = mapped_column(Integer, default=0)
    n_errors: Mapped[int] = mapped_column(Integer, default=0)
    n_warnings: Mapped[int] = mapped_column(Integer, default=0)
    stored_path: Mapped[str | None] = mapped_column(String)


class ValidationIssue(Base):
    __tablename__ = "validation_issues"
    issue_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    upload_id: Mapped[int] = mapped_column(ForeignKey("uploads.upload_id"), index=True)
    row_index: Mapped[int | None] = mapped_column(Integer)
    column: Mapped[str | None] = mapped_column(String)
    value: Mapped[str | None] = mapped_column(Text)
    check: Mapped[str | None] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String)  # 'error' | 'warning'
    message: Mapped[str | None] = mapped_column(Text)
    suggestion: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, default="open")  # open|fixed|ignored|excluded


class CorrectionAudit(Base):
    __tablename__ = "corrections_audit"
    audit_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    issue_id: Mapped[int | None] = mapped_column(ForeignKey("validation_issues.issue_id"))
    upload_id: Mapped[int | None] = mapped_column(ForeignKey("uploads.upload_id"))
    action: Mapped[str | None] = mapped_column(String)
    old_value: Mapped[str | None] = mapped_column(Text)
    new_value: Mapped[str | None] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))


class Record(Base):
    __tablename__ = "records"
    record_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    upload_id: Mapped[int | None] = mapped_column(ForeignKey("uploads.upload_id"))
    region_id: Mapped[int | None] = mapped_column(ForeignKey("regions.region_id"), index=True)
    crop_id: Mapped[int | None] = mapped_column(ForeignKey("crops.crop_id"), index=True)
    season: Mapped[str | None] = mapped_column(String)
    year: Mapped[int | None] = mapped_column(Integer, index=True)
    area_ha: Mapped[float | None] = mapped_column(Float)
    production_t: Mapped[float | None] = mapped_column(Float)
    yield_t_ha: Mapped[float | None] = mapped_column(Float)
    rainfall_mm: Mapped[float | None] = mapped_column(Float)
    temp_mean_c: Mapped[float | None] = mapped_column(Float)
    temp_max_c: Mapped[float | None] = mapped_column(Float)
    temp_min_c: Mapped[float | None] = mapped_column(Float)
    humidity_pct: Mapped[float | None] = mapped_column(Float)
    soil_n: Mapped[float | None] = mapped_column(Float)
    soil_p: Mapped[float | None] = mapped_column(Float)
    soil_k: Mapped[float | None] = mapped_column(Float)
    soil_ph: Mapped[float | None] = mapped_column(Float)
    soil_oc: Mapped[float | None] = mapped_column(Float)
    fertilizer_kgha: Mapped[float | None] = mapped_column(Float)
    irrigated_frac: Mapped[float | None] = mapped_column(Float)
    pesticide_kgha: Mapped[float | None] = mapped_column(Float)


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    model_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String)
    algo: Mapped[str | None] = mapped_column(String)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))
    file_path: Mapped[str | None] = mapped_column(String)
    params_json: Mapped[str | None] = mapped_column(Text)
    feature_list_json: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[int] = mapped_column(Integer, default=0)
    train_rows: Mapped[int | None] = mapped_column(Integer)


class ModelMetric(Base):
    __tablename__ = "model_metrics"
    metric_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("model_registry.model_id"), index=True)
    scheme: Mapped[str] = mapped_column(String)
    scope_type: Mapped[str] = mapped_column(String, default="overall")
    scope_value: Mapped[str | None] = mapped_column(String)
    n: Mapped[int | None] = mapped_column(Integer)
    mae: Mapped[float | None] = mapped_column(Float)
    rmse: Mapped[float | None] = mapped_column(Float)
    r2: Mapped[float | None] = mapped_column(Float)
    mape: Mapped[float | None] = mapped_column(Float)
    bias: Mapped[float | None] = mapped_column(Float)
    skill_vs_baseline: Mapped[float | None] = mapped_column(Float)


class EvalPrediction(Base):
    __tablename__ = "eval_predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("model_registry.model_id"), index=True)
    record_id: Mapped[int | None] = mapped_column(Integer)
    scheme: Mapped[str] = mapped_column(String, index=True)
    y_true: Mapped[float | None] = mapped_column(Float)
    y_pred: Mapped[float | None] = mapped_column(Float)
    pred_low: Mapped[float | None] = mapped_column(Float)
    pred_high: Mapped[float | None] = mapped_column(Float)


class FeatureImportanceRow(Base):
    __tablename__ = "feature_importance"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("model_registry.model_id"), index=True)
    feature: Mapped[str] = mapped_column(String)
    impurity_importance: Mapped[float | None] = mapped_column(Float)
    permutation_importance: Mapped[float | None] = mapped_column(Float)
    permutation_std: Mapped[float | None] = mapped_column(Float)


class UserPrediction(Base):
    __tablename__ = "user_predictions"
    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int | None] = mapped_column(ForeignKey("model_registry.model_id"))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))
    inputs_json: Mapped[str | None] = mapped_column(Text)
    y_pred: Mapped[float | None] = mapped_column(Float)
    pred_low: Mapped[float | None] = mapped_column(Float)
    pred_high: Mapped[float | None] = mapped_column(Float)
    warnings: Mapped[str | None] = mapped_column(Text)


class TrendResult(Base):
    __tablename__ = "trend_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    region_id: Mapped[int | None] = mapped_column(Integer, index=True)
    crop_id: Mapped[int | None] = mapped_column(Integer, index=True)
    trend: Mapped[str | None] = mapped_column(String)
    p_value: Mapped[float | None] = mapped_column(Float)
    tau: Mapped[float | None] = mapped_column(Float)
    sen_slope_t_ha_per_year: Mapped[float | None] = mapped_column(Float)
    cv_yield: Mapped[float | None] = mapped_column(Float)


class ReportRow(Base):
    __tablename__ = "reports"
    report_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))
    file_path: Mapped[str | None] = mapped_column(String)
    scope_json: Mapped[str | None] = mapped_column(Text)


class FeatureSetRow(Base):
    __tablename__ = "feature_sets"
    feature_set_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None))
    config_json: Mapped[str | None] = mapped_column(Text)
    n_rows: Mapped[int | None] = mapped_column(Integer)
    n_features: Mapped[int | None] = mapped_column(Integer)


# ---------------------------------------------------------------------------
# Engine / session helpers
# ---------------------------------------------------------------------------

_ENGINE = None
_SessionLocal = None


def get_engine(db_path: str | Path | None = None):
    global _ENGINE, _SessionLocal
    if db_path is not None:
        engine = create_engine(f"sqlite:///{db_path}", future=True)
        Base.metadata.create_all(engine)
        return engine
    if _ENGINE is None:
        _ENGINE = create_engine(f"sqlite:///{DB_PATH}", future=True)
        Base.metadata.create_all(_ENGINE)
    return _ENGINE


def get_session(db_path: str | Path | None = None) -> Session:
    global _SessionLocal
    if db_path is not None:
        engine = get_engine(db_path)
        return sessionmaker(bind=engine, future=True)()
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), future=True)
    return _SessionLocal()


def init_db(db_path: str | Path | None = None) -> None:
    """Create all tables."""
    get_engine(db_path)


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------


def get_or_create_region(session: Session, state: str | None, district: str | None) -> Region:
    stmt = select(Region).where(Region.state == state, Region.district == district)
    region = session.scalars(stmt).first()
    if region is None:
        region = Region(state=state, district=district)
        session.add(region)
        session.flush()
    return region


def get_or_create_crop(session: Session, crop_name: str) -> Crop:
    from src.config import crop_settings

    stmt = select(Crop).where(Crop.crop_name == crop_name)
    crop = session.scalars(stmt).first()
    if crop is None:
        cfg = crop_settings(crop_name)
        crop = Crop(
            crop_name=crop_name,
            base_temp_c=cfg.get("base_temp_c"),
            optimal_ph=cfg.get("optimal_ph"),
            yield_min=cfg.get("yield_min"),
            yield_max=cfg.get("yield_max"),
        )
        session.add(crop)
        session.flush()
    return crop


def create_upload(session: Session, filename: str, stored_path: str | None = None) -> Upload:
    upload = Upload(filename=filename, stored_path=stored_path, status="validated")
    session.add(upload)
    session.flush()
    return upload


def add_issues(session: Session, upload_id: int, issues: Iterable[dict[str, Any]]) -> int:
    count = 0
    for issue in issues:
        session.add(
            ValidationIssue(
                upload_id=upload_id,
                row_index=issue.get("row_index"),
                column=issue.get("column"),
                value=str(issue.get("value")) if issue.get("value") is not None else None,
                check=issue.get("check"),
                severity=issue.get("severity", "warning"),
                message=issue.get("message"),
                suggestion=issue.get("suggestion"),
                status=issue.get("status", "open"),
            )
        )
        count += 1
    session.flush()
    return count


def commit_records(session: Session, upload_id: int, df) -> int:
    """Bulk-insert a cleaned DataFrame into ``records``, upserting regions/crops."""
    inserted = 0
    region_cache: dict[tuple, int] = {}
    crop_cache: dict[str, int] = {}

    for _, row in df.iterrows():
        state = row.get("state")
        district = row.get("district")
        key = (state, district)
        if key not in region_cache:
            region_cache[key] = get_or_create_region(session, state, district).region_id
        crop_name = str(row.get("crop"))
        if crop_name not in crop_cache:
            crop_cache[crop_name] = get_or_create_crop(session, crop_name).crop_id

        session.add(
            Record(
                upload_id=upload_id,
                region_id=region_cache[key],
                crop_id=crop_cache[crop_name],
                season=row.get("season"),
                year=_to_int(row.get("year")),
                area_ha=_to_float(row.get("area_ha")),
                production_t=_to_float(row.get("production_t")),
                yield_t_ha=_to_float(row.get("yield_t_ha")),
                rainfall_mm=_to_float(row.get("rainfall_mm")),
                temp_mean_c=_to_float(row.get("temp_mean_c")),
                temp_max_c=_to_float(row.get("temp_max_c")),
                temp_min_c=_to_float(row.get("temp_min_c")),
                humidity_pct=_to_float(row.get("humidity_pct")),
                soil_n=_to_float(row.get("soil_n_kgha")),
                soil_p=_to_float(row.get("soil_p_kgha")),
                soil_k=_to_float(row.get("soil_k_kgha")),
                soil_ph=_to_float(row.get("soil_ph")),
                soil_oc=_to_float(row.get("soil_oc_pct")),
                fertilizer_kgha=_to_float(row.get("fertilizer_kgha")),
                irrigated_frac=_to_float(row.get("irrigated_frac")),
                pesticide_kgha=_to_float(row.get("pesticide_kgha")),
            )
        )
        inserted += 1
    session.flush()
    return inserted


def load_records_df(session: Session, upload_id: int | None = None):
    """Return all committed records as a canonical DataFrame."""
    import pandas as pd

    stmt = select(Record, Region, Crop).join(
        Region, Record.region_id == Region.region_id, isouter=True
    ).join(Crop, Record.crop_id == Crop.crop_id, isouter=True)
    if upload_id is not None:
        stmt = stmt.where(Record.upload_id == upload_id)

    rows = []
    for record, region, crop in session.execute(stmt).all():
        rows.append(
            {
                "record_id": record.record_id,
                "upload_id": record.upload_id,
                "state": region.state if region else None,
                "district": region.district if region else None,
                "crop": crop.crop_name if crop else None,
                "season": record.season,
                "year": record.year,
                "area_ha": record.area_ha,
                "production_t": record.production_t,
                "yield_t_ha": record.yield_t_ha,
                "rainfall_mm": record.rainfall_mm,
                "temp_mean_c": record.temp_mean_c,
                "temp_max_c": record.temp_max_c,
                "temp_min_c": record.temp_min_c,
                "humidity_pct": record.humidity_pct,
                "soil_n_kgha": record.soil_n,
                "soil_p_kgha": record.soil_p,
                "soil_k_kgha": record.soil_k,
                "soil_ph": record.soil_ph,
                "soil_oc_pct": record.soil_oc,
                "fertilizer_kgha": record.fertilizer_kgha,
                "irrigated_frac": record.irrigated_frac,
                "pesticide_kgha": record.pesticide_kgha,
            }
        )
    return pd.DataFrame(rows)


def log_correction(
    session: Session,
    *,
    upload_id: int | None,
    issue_id: int | None,
    action: str,
    old_value: Any = None,
    new_value: Any = None,
    note: str | None = None,
) -> CorrectionAudit:
    audit = CorrectionAudit(
        upload_id=upload_id,
        issue_id=issue_id,
        action=action,
        old_value=None if old_value is None else str(old_value),
        new_value=None if new_value is None else str(new_value),
        note=note,
    )
    session.add(audit)
    session.flush()
    return audit


def _to_float(value) -> float | None:
    try:
        import math

        if value is None:
            return None
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _to_int(value) -> int | None:
    f = _to_float(value)
    return int(f) if f is not None else None


def dataset_counts(session: Session) -> dict[str, Any]:
    """Summary counts used by the explorer summary endpoint."""
    n_records = session.scalar(select(func.count()).select_from(Record)) or 0
    n_regions = session.scalar(select(func.count()).select_from(Region)) or 0
    n_crops = session.scalar(select(func.count()).select_from(Crop)) or 0
    min_year = session.scalar(select(func.min(Record.year)))
    max_year = session.scalar(select(func.max(Record.year)))
    mean_yield = session.scalar(select(func.avg(Record.yield_t_ha)))
    return {
        "rows": int(n_records),
        "regions": int(n_regions),
        "crops": int(n_crops),
        "year_min": int(min_year) if min_year is not None else None,
        "year_max": int(max_year) if max_year is not None else None,
        "mean_yield": float(mean_yield) if mean_yield is not None else None,
    }
