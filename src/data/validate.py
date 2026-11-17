"""Validation rules (spec section 6.1).

Every check produces a structured issue with ``row_index``, ``column``,
``value``, ``check``, ``severity``, ``message`` and ``suggestion``. Issues are
collected into a :class:`ValidationReport` which the API serialises and the UI
renders as a filterable table.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import all_crops, all_seasons, crop_settings

MANDATORY_KEY = ["district", "crop", "season", "year"]
MIN_YEAR = 1950
MAX_MISSING_FRACTION = 0.30
YIELD_CONSISTENCY_TOL = 0.05
ACRE_TO_HA = 2.471


@dataclass
class Issue:
    row_index: int | None
    column: str | None
    value: Any
    check: str
    severity: str  # 'error' | 'warning'
    message: str
    suggestion: str | None = None
    status: str = "open"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    n_rows: int
    issues: list[Issue] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def has_blocking_errors(self) -> bool:
        return self.n_errors > 0

    def summary(self) -> str:
        return (
            f"{self.n_rows:,} rows: {self.n_errors} errors, {self.n_warnings} warnings"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "summary": self.summary(),
            "missing_columns": self.missing_columns,
            "issues": [i.to_dict() for i in self.issues],
        }


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def check_mandatory_columns(df: pd.DataFrame) -> tuple[list[str], list[Issue]]:
    """Check presence of mandatory columns. Returns (missing, issues)."""
    missing: list[str] = []
    issues: list[Issue] = []

    for col in MANDATORY_KEY:
        if col not in df.columns or df[col].isna().all():
            missing.append(col)

    has_yield = "yield_t_ha" in df.columns and not df["yield_t_ha"].isna().all()
    has_area_prod = (
        "area_ha" in df.columns
        and "production_t" in df.columns
        and not (df["area_ha"].isna().all() or df["production_t"].isna().all())
    )
    if not (has_yield or has_area_prod):
        missing.append("yield_t_ha (or area_ha + production_t)")

    for col in ["rainfall_mm"]:
        if col not in df.columns or df[col].isna().all():
            missing.append(col)

    temp_present = any(
        c in df.columns and not df[c].isna().all()
        for c in ["temp_mean_c", "temp_max_c", "temp_min_c"]
    )
    if not temp_present:
        missing.append("temperature (temp_mean_c/temp_max_c/temp_min_c)")

    for col in missing:
        issues.append(
            Issue(
                row_index=None,
                column=col,
                value=None,
                check="missing_mandatory_column",
                severity="error",
                message=f"Mandatory column '{col}' is missing or entirely empty.",
                suggestion="Download the CSV template and provide this column.",
            )
        )
    return missing, issues


def check_non_numeric(df: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    numeric_cols = [
        c
        for c in df.columns
        if c not in {"state", "district", "crop", "season"}
    ]
    for col in numeric_cols:
        series = df[col]
        coerced = pd.to_numeric(series, errors="coerce")
        bad = series.notna() & coerced.isna()
        for idx in df.index[bad]:
            issues.append(
                Issue(
                    row_index=int(idx),
                    column=col,
                    value=series.loc[idx],
                    check="non_numeric_value",
                    severity="warning",
                    message=f"Non-numeric value in numeric column '{col}'.",
                    suggestion="Value converted to NaN; correct or impute it.",
                )
            )
    return issues


def check_invalid_year(df: pd.DataFrame, current_year: int | None = None) -> list[Issue]:
    if "year" not in df.columns:
        return []
    current_year = current_year or dt.date.today().year
    issues: list[Issue] = []
    years = _numeric(df["year"])
    bad = years.notna() & ((years < MIN_YEAR) | (years > current_year))
    for idx in df.index[bad]:
        issues.append(
            Issue(
                row_index=int(idx),
                column="year",
                value=df["year"].loc[idx],
                check="invalid_year",
                severity="error",
                message=f"Year {df['year'].loc[idx]} is outside the valid range "
                f"{MIN_YEAR}-{current_year}.",
                suggestion="Correct the year or exclude the row.",
            )
        )
    return issues


def check_impossible_values(df: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []

    def _neg(col: str, label: str):
        if col not in df.columns:
            return
        vals = _numeric(df[col])
        bad = vals.notna() & (vals < 0)
        for idx in df.index[bad]:
            issues.append(
                Issue(
                    row_index=int(idx),
                    column=col,
                    value=df[col].loc[idx],
                    check="impossible_value",
                    severity="error",
                    message=f"Negative {label} is not possible.",
                    suggestion="Use the absolute value or exclude the row.",
                )
            )

    _neg("area_ha", "area")
    _neg("production_t", "production")
    _neg("rainfall_mm", "rainfall")

    if {"temp_min_c", "temp_max_c"}.issubset(df.columns):
        tmin, tmax = _numeric(df["temp_min_c"]), _numeric(df["temp_max_c"])
        bad = tmin.notna() & tmax.notna() & (tmin > tmax)
        for idx in df.index[bad]:
            issues.append(
                Issue(
                    row_index=int(idx),
                    column="temp_min_c",
                    value=df["temp_min_c"].loc[idx],
                    check="impossible_value",
                    severity="error",
                    message="Minimum temperature exceeds maximum temperature.",
                    suggestion="Swap the two values.",
                )
            )

    if "soil_ph" in df.columns:
        ph = _numeric(df["soil_ph"])
        bad = ph.notna() & ((ph < 3.0) | (ph > 10.0))
        for idx in df.index[bad]:
            issues.append(
                Issue(
                    row_index=int(idx),
                    column="soil_ph",
                    value=df["soil_ph"].loc[idx],
                    check="impossible_value",
                    severity="error",
                    message="Soil pH outside the plausible range 3-10.",
                    suggestion="Correct the pH value or exclude the row.",
                )
            )

    if "humidity_pct" in df.columns:
        hum = _numeric(df["humidity_pct"])
        bad = hum.notna() & ((hum < 0) | (hum > 100))
        for idx in df.index[bad]:
            issues.append(
                Issue(
                    row_index=int(idx),
                    column="humidity_pct",
                    value=df["humidity_pct"].loc[idx],
                    check="impossible_value",
                    severity="error",
                    message="Humidity outside the valid range 0-100 %.",
                    suggestion="Correct the humidity value.",
                )
            )
    return issues


def check_crop_yield_range(df: pd.DataFrame) -> list[Issue]:
    if "yield_t_ha" not in df.columns or "crop" not in df.columns:
        return []
    issues: list[Issue] = []
    yields = _numeric(df["yield_t_ha"])
    for idx in df.index:
        crop = df["crop"].loc[idx]
        yld = yields.loc[idx]
        if pd.isna(yld) or pd.isna(crop):
            continue
        cfg = crop_settings(str(crop))
        lo, hi = cfg.get("yield_min", 0.1), cfg.get("yield_max", 15.0)
        if yld < lo or yld > hi:
            issues.append(
                Issue(
                    row_index=int(idx),
                    column="yield_t_ha",
                    value=df["yield_t_ha"].loc[idx],
                    check="crop_yield_range",
                    severity="warning",
                    message=f"Yield {yld:.2f} t/ha is outside the plausible range "
                    f"for {crop} ({lo}-{hi} t/ha).",
                    suggestion="Flagged as an outlier; review or exclude.",
                )
            )
    return issues


def check_yield_consistency(df: pd.DataFrame) -> list[Issue]:
    required = {"area_ha", "production_t", "yield_t_ha"}
    if not required.issubset(df.columns):
        return []
    issues: list[Issue] = []
    area, prod, yld = (
        _numeric(df["area_ha"]),
        _numeric(df["production_t"]),
        _numeric(df["yield_t_ha"]),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        computed = prod / area.replace(0, np.nan)
        rel = (yld - computed).abs() / yld.replace(0, np.nan)
    bad = rel.notna() & (rel > YIELD_CONSISTENCY_TOL)
    for idx in df.index[bad]:
        issues.append(
            Issue(
                row_index=int(idx),
                column="yield_t_ha",
                value=df["yield_t_ha"].loc[idx],
                check="yield_consistency",
                severity="warning",
                message=f"Stated yield differs from production/area by "
                f"{rel.loc[idx] * 100:.1f} % (> {YIELD_CONSISTENCY_TOL * 100:.0f} %).",
                suggestion="Recompute yield as production/area or verify inputs.",
            )
        )
    return issues


def check_duplicates(df: pd.DataFrame) -> list[Issue]:
    keys = [k for k in MANDATORY_KEY if k in df.columns]
    if len(keys) < 4:
        return []
    issues: list[Issue] = []
    dup_mask = df.duplicated(subset=keys, keep=False)
    for idx in df.index[dup_mask]:
        issues.append(
            Issue(
                row_index=int(idx),
                column=", ".join(keys),
                value=None,
                check="duplicate_record",
                severity="warning",
                message="Duplicate record for the same district/crop/season/year.",
                suggestion="Keep first, keep last or average the duplicates.",
            )
        )
    return issues


def check_missingness(df: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    n = len(df)
    if n == 0:
        return issues
    for col in df.columns:
        frac = df[col].isna().mean()
        if frac > MAX_MISSING_FRACTION:
            issues.append(
                Issue(
                    row_index=None,
                    column=col,
                    value=round(frac, 3),
                    check="high_missingness",
                    severity="warning",
                    message=f"Column '{col}' is missing in {frac * 100:.0f} % of rows.",
                    suggestion="Exclude the feature or choose an imputation strategy.",
                )
            )
    return issues


def check_units(df: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []

    if "production_t" in df.columns and "area_ha" in df.columns:
        area = _numeric(df["area_ha"])
        prod = _numeric(df["production_t"])
        # If production/area implies a yield far above plausible, likely kg vs t.
        with np.errstate(divide="ignore", invalid="ignore"):
            implied = prod / area.replace(0, np.nan)
        bad = implied.notna() & (implied > 60)
        for idx in df.index[bad]:
            issues.append(
                Issue(
                    row_index=int(idx),
                    column="production_t",
                    value=df["production_t"].loc[idx],
                    check="unit_check",
                    severity="warning",
                    message="Implied yield exceeds 60 t/ha; production may be in kg.",
                    suggestion="If production is in kg, divide by 1000.",
                )
            )

    if "area_ha" in df.columns:
        area = _numeric(df["area_ha"])
        # District-level area is usually thousands of hectares; very small values
        # or values ~2.471x the district median hint at acres or unit mixups.
        small = area.notna() & (area > 0) & (area < 50)
        for idx in df.index[small]:
            issues.append(
                Issue(
                    row_index=int(idx),
                    column="area_ha",
                    value=df["area_ha"].loc[idx],
                    check="unit_check",
                    severity="warning",
                    message="Area value is unusually small for a district-level record.",
                    suggestion=f"If the value is in acres, multiply by {ACRE_TO_HA:.3f}.",
                )
            )

        if "district" in df.columns:
            for district, group in df.groupby("district"):
                med = area.loc[group.index].median()
                if pd.isna(med) or med <= 0:
                    continue
                ratio = area.loc[group.index] / med
                acre_like = ratio.between(2.3, 2.6)
                for idx in group.index[acre_like]:
                    issues.append(
                        Issue(
                            row_index=int(idx),
                            column="area_ha",
                            value=df["area_ha"].loc[idx],
                            check="unit_check",
                            severity="warning",
                            message=f"Area is ~{ratio.loc[idx]:.2f}x the district median, "
                            "suggesting the value may be in acres.",
                            suggestion=f"Divide by {ACRE_TO_HA:.3f} to convert acres to hectares.",
                        )
                    )
    return issues


def check_unknown_categories(df: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    known_crops = {c.lower() for c in all_crops()}
    known_seasons = {s.lower() for s in all_seasons()}

    if "crop" in df.columns:
        for idx in df.index:
            crop = df["crop"].loc[idx]
            if pd.notna(crop) and str(crop).strip().lower() not in known_crops:
                issues.append(
                    Issue(
                        row_index=int(idx),
                        column="crop",
                        value=crop,
                        check="unknown_category",
                        severity="warning",
                        message=f"Unknown crop label '{crop}'.",
                        suggestion="Add it to config/crops.yaml or map it to a known crop.",
                    )
                )
    if "season" in df.columns:
        for idx in df.index:
            season = df["season"].loc[idx]
            if pd.notna(season) and str(season).strip().lower() not in known_seasons:
                issues.append(
                    Issue(
                        row_index=int(idx),
                        column="season",
                        value=season,
                        check="unknown_category",
                        severity="warning",
                        message=f"Unknown season label '{season}'.",
                        suggestion="Map it to Kharif, Rabi or Summer.",
                    )
                )
    return issues


def validate(df: pd.DataFrame) -> ValidationReport:
    """Run every validation rule and return a :class:`ValidationReport`."""
    report = ValidationReport(n_rows=len(df))

    missing, col_issues = check_mandatory_columns(df)
    report.missing_columns = missing
    report.issues.extend(col_issues)

    # Row-level checks still run on whatever data is available.
    report.issues.extend(check_non_numeric(df))
    report.issues.extend(check_invalid_year(df))
    report.issues.extend(check_impossible_values(df))
    report.issues.extend(check_crop_yield_range(df))
    report.issues.extend(check_yield_consistency(df))
    report.issues.extend(check_duplicates(df))
    report.issues.extend(check_missingness(df))
    report.issues.extend(check_units(df))
    report.issues.extend(check_unknown_categories(df))
    return report


# ---------------------------------------------------------------------------
# Auto-fix helpers used by the correction workflow
# ---------------------------------------------------------------------------


def autofix_negative(series: pd.Series) -> pd.Series:
    return series.abs()


def autofix_swap_temps(df: pd.DataFrame) -> pd.DataFrame:
    if not {"temp_min_c", "temp_max_c"}.issubset(df.columns):
        return df
    tmin, tmax = _numeric(df["temp_min_c"]), _numeric(df["temp_max_c"])
    swap = tmin.notna() & tmax.notna() & (tmin > tmax)
    df = df.copy()
    df.loc[swap, "temp_min_c"] = tmax[swap]
    df.loc[swap, "temp_max_c"] = tmin[swap]
    return df


def autofix_acres_to_ha(series: pd.Series) -> pd.Series:
    return _numeric(series) / ACRE_TO_HA
