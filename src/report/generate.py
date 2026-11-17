"""PDF report generation (spec sections 6.9 and 9).

Builds matplotlib figures, renders a Jinja2 HTML template and converts it to PDF
with WeasyPrint. Everything degrades gracefully: if a section has no data it is
simply omitted from the report.
"""

from __future__ import annotations

import base64
import datetime as dt
import io
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

from src.analysis.trends import (
    climate_yield_correlation,
    district_crop_trends,
    top_bottom_regions,
    yield_time_series,
)
from src.config import OUTPUTS_DIR, PROJECT_ROOT, ensure_dirs
from src.data.db import get_session
from src.features.engineer import FeatureConfig
from src.models.explain import partial_dependence_curve, permutation_importances
from src.models.registry import get_metrics, load_model_by_id, list_models

TEMPLATE_DIR = PROJECT_ROOT / "src" / "report"

ACCENT = "#22d3a6"
ACCENT2 = "#6366f1"
INK = "#0f172a"


def _fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight", transparent=False)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK)
    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=10, color=INK)
    ax.grid(True, alpha=0.25, linestyle="--")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def trend_figure(df: pd.DataFrame, crop: str | None = None) -> str | None:
    series = yield_time_series(df, crop=crop)
    if series.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(series["year"], series["mean_yield"], marker="o", color=ACCENT2, linewidth=2, label="Mean yield")
    if series["trend_line"].notna().any():
        ax.plot(series["year"], series["trend_line"], linestyle="--", color=ACCENT, label="Trend")
    _style_axes(ax, f"Yield over time{' - ' + crop if crop else ''}", "Year", "Yield (t/ha)")
    ax.legend(frameon=False)
    return _fig_to_base64(fig)


def climate_figure(df: pd.DataFrame, variable: str = "rainfall_mm", crop: str | None = None) -> str | None:
    sub = df.copy()
    if crop:
        sub = sub[sub["crop"] == crop]
    sub = sub.dropna(subset=[variable, "yield_t_ha"])
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.scatter(sub[variable], sub["yield_t_ha"], s=12, alpha=0.5, color=ACCENT2)
    corr = climate_yield_correlation(df, variable=variable, crop=crop)
    _style_axes(
        ax,
        f"{variable} vs yield (Spearman r={corr.get('spearman_r') or 0:.2f})",
        variable,
        "Yield (t/ha)",
    )
    return _fig_to_base64(fig)


def region_figure(df: pd.DataFrame, crop: str | None = None) -> str | None:
    regions = top_bottom_regions(df, crop=crop, n=6)
    top = regions.get("top", [])
    if not top:
        return None
    labels = [f"{r['district']}" for r in top]
    values = [r["mean_yield"] for r in top]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(labels[::-1], values[::-1], color=ACCENT)
    _style_axes(ax, "Top regions by mean yield", "Mean yield (t/ha)", "District")
    return _fig_to_base64(fig)


def importance_figure(pipeline, df: pd.DataFrame, top: int = 10) -> str | None:
    try:
        imp = permutation_importances(pipeline, df, n_repeats=5)
    except Exception:
        return None
    if imp.empty:
        return None
    imp = imp.head(top)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(imp["feature"][::-1], imp["importance_mean"][::-1], color=ACCENT2)
    _style_axes(ax, "Permutation importance (raw inputs)", "Increase in RMSE when permuted", "Feature")
    return _fig_to_base64(fig)


def pdp_figure(pipeline, df: pd.DataFrame, feature: str = "rainfall_mm") -> str | None:
    curve = partial_dependence_curve(pipeline, df, feature)
    if not curve.get("grid"):
        return None
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(curve["grid"], curve["values"], color=ACCENT, linewidth=2)
    _style_axes(ax, f"Partial dependence: {feature}", feature, "Predicted yield (t/ha)")
    return _fig_to_base64(fig)


def predictions_figure(points: list[dict]) -> str | None:
    if not points:
        return None
    y_true = [p["y_true"] for p in points if p.get("y_true") is not None and p.get("y_pred") is not None]
    y_pred = [p["y_pred"] for p in points if p.get("y_true") is not None and p.get("y_pred") is not None]
    if not y_true:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, s=12, alpha=0.5, color=ACCENT2)
    lo = min(min(y_true), min(y_pred))
    hi = max(max(y_true), max(y_pred))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=INK, linewidth=1)
    _style_axes(ax, "Actual vs predicted yield", "Actual (t/ha)", "Predicted (t/ha)")
    return _fig_to_base64(fig)


def build_context(model_id: int | None = None, include_sections: list[str] | None = None) -> dict[str, Any]:
    """Assemble all data needed to render the report."""
    include_sections = include_sections or ["summary", "trends", "models", "explainability", "predictions"]
    session = get_session()
    try:
        from src.data.db import load_records_df

        df = load_records_df(session)
        models = list_models(session)

        active_id = model_id
        if active_id is None:
            active = next((m for m in models if m["is_active"]), None)
            active_id = active["model_id"] if active else (models[0]["model_id"] if models else None)

        context: dict[str, Any] = {
            "title": "YieldVision Analysis Report",
            "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "include_sections": include_sections,
            "n_rows": int(len(df)),
            "n_crops": int(df["crop"].nunique()) if not df.empty else 0,
            "n_regions": int(df[["state", "district"]].drop_duplicates().shape[0]) if not df.empty else 0,
            "year_range": (
                f"{int(df['year'].min())}-{int(df['year'].max())}" if not df.empty else "n/a"
            ),
            "mean_yield": float(df["yield_t_ha"].mean()) if not df.empty else None,
            "models": models,
            "figures": {},
        }

        if not df.empty:
            trends = district_crop_trends(df)
            increasing = int((trends["trend"] == "increasing").sum())
            context["trend_summary"] = {
                "n_series": int(len(trends)),
                "increasing": increasing,
                "decreasing": int((trends["trend"] == "decreasing").sum()),
                "no_trend": int((trends["trend"] == "no_trend").sum()),
                "median_sen_slope": float(trends["sen_slope_t_ha_per_year"].median()),
            }
            context["trend_table"] = (
                trends.sort_values("sen_slope_t_ha_per_year", ascending=False).head(12).round(4).to_dict(orient="records")
            )
            context["figures"]["trend"] = trend_figure(df)
            context["figures"]["climate"] = climate_figure(df)
            context["figures"]["regions"] = region_figure(df)
            context["crop_comparison"] = (
                df.groupby("crop", as_index=False)
                .agg(mean_yield=("yield_t_ha", "mean"), n=("yield_t_ha", "size"))
                .round(3)
                .to_dict(orient="records")
            )

        if active_id is not None:
            payload = load_model_by_id(session, active_id)
            pipeline = payload.get("pipeline")
            interval = payload.get("interval_model")
            entry = session.get(__import__("src.data.db", fromlist=["ModelRegistry"]).ModelRegistry, active_id)
            context["active_model"] = {
                "model_id": active_id,
                "name": entry.name if entry else None,
                "algo": entry.algo if entry else None,
                "interval_level": interval.get("nominal_coverage") if interval else None,
                "interval_low": interval.get("residual_lo") if interval else None,
                "interval_high": interval.get("residual_hi") if interval else None,
                "metrics": get_metrics(session, active_id),
            }
            if pipeline is not None and not df.empty:
                if "explainability" in include_sections:
                    context["figures"]["importance"] = importance_figure(pipeline, df)
                    context["figures"]["pdp_rainfall"] = pdp_figure(pipeline, df, "rainfall_mm")
                    context["figures"]["pdp_nitrogen"] = pdp_figure(pipeline, df, "soil_n_kgha")
                if "predictions" in include_sections:
                    from sqlalchemy import select

                    from src.data.db import EvalPrediction

                    rows = session.scalars(
                        select(EvalPrediction)
                        .where(EvalPrediction.model_id == active_id)
                        .limit(400)
                    ).all()
                    points = [{"y_true": r.y_true, "y_pred": r.y_pred} for r in rows]
                    context["figures"]["predictions"] = predictions_figure(points)

        return context
    finally:
        session.close()


def render_html(context: dict[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("template.html.j2")
    return template.render(**context)


def generate_report(
    model_id: int | None = None,
    *,
    include_sections: list[str] | None = None,
    title: str = "YieldVision Analysis Report",
) -> Path:
    """Render the report to a PDF in the outputs directory and return its path."""
    ensure_dirs()
    context = build_context(model_id=model_id, include_sections=include_sections)
    context["title"] = title
    html = render_html(context)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = OUTPUTS_DIR / f"yieldvision_report_{timestamp}.pdf"
    html_path = OUTPUTS_DIR / f"yieldvision_report_{timestamp}.html"
    html_path.write_text(html, encoding="utf-8")

    try:
        from weasyprint import HTML

        HTML(string=html, base_url=str(TEMPLATE_DIR)).write_pdf(str(pdf_path))
        return pdf_path
    except Exception:  # pragma: no cover - PDF engine fallback
        # If WeasyPrint is unavailable, return the HTML rendering instead.
        return html_path
