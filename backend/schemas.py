"""Pydantic v2 request/response schemas for the YieldVision API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SCHEMES = ("temporal", "group", "random")
MODEL_CHOICES = (
    "baseline_crop_season",
    "baseline_district_history",
    "linear",
    "ridge",
    "random_forest",
    "hist_gradient_boosting",
)


class FeatureConfigIn(BaseModel):
    include_gdd: bool = True
    include_anomalies: bool = True
    include_lags: bool = True
    include_interactions: bool = True
    include_soil: bool = True
    include_categorical: bool = True
    add_missing_indicators: bool = True


class TrainRequest(BaseModel):
    models: list[str] = Field(
        default_factory=lambda: ["baseline_crop_season", "baseline_district_history", "ridge", "random_forest"]
    )
    schemes: list[str] = Field(default_factory=lambda: ["temporal", "group", "random"])
    feature_config: FeatureConfigIn = Field(default_factory=FeatureConfigIn)
    tune: bool = False
    test_years: int = 4
    dataset: Literal["database", "sample"] = "database"

    @field_validator("models")
    @classmethod
    def _valid_models(cls, value: list[str]) -> list[str]:
        for name in value:
            if name not in MODEL_CHOICES:
                raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_CHOICES}")
        return value

    @field_validator("schemes")
    @classmethod
    def _valid_schemes(cls, value: list[str]) -> list[str]:
        for name in value:
            if name not in SCHEMES:
                raise ValueError(f"Unknown scheme '{name}'. Choose from {SCHEMES}")
        return value


class PredictRequest(BaseModel):
    state: str = "Punjab"
    district: str = "Ludhiana"
    crop: str = "rice"
    season: str = "Kharif"
    year: int = Field(default=2024, ge=1950, le=2100)
    area_ha: float = Field(default=1000.0, gt=0)
    rainfall_mm: float = Field(default=850.0, ge=0)
    temp_mean_c: float = Field(default=26.0, ge=-10, le=60)
    temp_max_c: float = Field(default=34.0, ge=-10, le=60)
    temp_min_c: float = Field(default=18.0, ge=-30, le=50)
    humidity_pct: float = Field(default=65.0, ge=0, le=100)
    soil_n_kgha: float = Field(default=250.0, ge=0)
    soil_p_kgha: float = Field(default=25.0, ge=0)
    soil_k_kgha: float = Field(default=200.0, ge=0)
    soil_ph: float = Field(default=6.5, ge=3, le=10)
    soil_oc_pct: float = Field(default=0.6, ge=0)
    fertilizer_kgha: float = Field(default=120.0, ge=0)
    irrigated_frac: float = Field(default=0.6, ge=0, le=1)
    pesticide_kgha: float = Field(default=2.5, ge=0)
    model_id: int | None = None


class WhatIfRequest(PredictRequest):
    feature: Literal["rainfall_mm", "temp_mean_c", "soil_n_kgha", "soil_ph", "irrigated_frac", "fertilizer_kgha"] = "rainfall_mm"


class IssuePatch(BaseModel):
    action: Literal["fix", "ignore", "exclude"]
    value: Any | None = None
    note: str | None = None


class CommitRequest(BaseModel):
    duplicate_policy: Literal["keep_first", "keep_last", "average"] = "keep_first"
    drop_high_missingness: bool = False


class ReportRequest(BaseModel):
    model_id: int | None = None
    title: str = "YieldVision Analysis Report"
    include_sections: list[str] = Field(
        default_factory=lambda: ["summary", "trends", "models", "explainability", "predictions"]
    )


class RecordEntry(BaseModel):
    """One manually entered crop/weather/soil row."""

    state: str | None = None
    district: str | None = None
    crop: str | None = None
    season: str | None = None
    year: int | None = None
    yield_t_ha: float | None = None
    area_ha: float | None = None
    production_t: float | None = None
    rainfall_mm: float | None = None
    temp_mean_c: float | None = None
    temp_max_c: float | None = None
    temp_min_c: float | None = None
    humidity_pct: float | None = None
    soil_n_kgha: float | None = None
    soil_p_kgha: float | None = None
    soil_k_kgha: float | None = None
    soil_ph: float | None = None
    soil_oc_pct: float | None = None
    fertilizer_kgha: float | None = None
    irrigated_frac: float | None = None
    pesticide_kgha: float | None = None


class RecordEntryIn(BaseModel):
    rows: list[RecordEntry] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    database: str
