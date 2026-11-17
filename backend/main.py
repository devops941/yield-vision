"""YieldVision FastAPI application entry point.

Run with ``uvicorn backend.main:app --reload`` and open http://localhost:8000.
Swagger docs are at /docs.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.deps import init_database
from backend.routers import (
    explorer,
    export,
    features,
    models,
    predict,
    report,
    uploads,
)
from src.config import DB_PATH, PROJECT_ROOT

VERSION = "1.0.0"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_database()
    yield


app = FastAPI(
    title="YieldVision API",
    description=(
        "Climate and soil-driven crop yield prediction engine: upload, validate, "
        "explore trends, engineer features, train and compare models, predict with "
        "uncertainty, and export results."
    ),
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (
    uploads.router,
    explorer.router,
    features.router,
    models.router,
    predict.router,
    export.router,
    report.router,
):
    app.include_router(router)


@app.get("/health", tags=["system"])
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "version": VERSION, "database": str(DB_PATH)}


# --- Static frontend -------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/{page}.html", include_in_schema=False)
    def page(page: str) -> FileResponse:
        candidate = FRONTEND_DIR / f"{page}.html"
        if candidate.exists():
            return FileResponse(str(candidate))
        return FileResponse(str(FRONTEND_DIR / "index.html"))
