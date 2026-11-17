"""API tests for the FastAPI backend (spec section 12)."""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import generate_dataset


def _sample_csv(n_rows: int = 60) -> bytes:
    df = generate_dataset(n_districts=4, n_crops=2, n_years=8, seed=3, inject_errors=False)
    buffer = io.StringIO()
    df.head(n_rows).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_openapi_lists_endpoints(api_client):
    schema = api_client.get("/openapi.json").json()
    paths = schema["paths"]
    for expected in ["/api/uploads", "/api/explorer/summary", "/api/models/train", "/api/predict", "/api/report"]:
        assert expected in paths


def test_template_download(api_client):
    r = api_client.get("/api/template")
    assert r.status_code == 200
    assert "district" in r.text and "crop" in r.text


def test_upload_validate_and_issues(api_client):
    r = api_client.post(
        "/api/uploads",
        files={"file": ("crop.csv", _sample_csv(), "text/csv")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["n_rows"] > 0
    upload_id = body["upload_id"]

    issues = api_client.get(f"/api/uploads/{upload_id}/issues")
    assert issues.status_code == 200
    assert "issues" in issues.json()


def test_upload_unknown_id_returns_404(api_client):
    r = api_client.get("/api/uploads/9999/issues")
    assert r.status_code == 404


def test_load_sample_and_explorer(api_client):
    r = api_client.post("/api/datasets/sample")
    assert r.status_code == 200, r.text
    assert r.json()["inserted"] > 0

    summary = api_client.get("/api/explorer/summary").json()
    assert summary["rows"] > 0
    assert summary["n_crops"] > 0

    trend = api_client.get("/api/explorer/trend", params={"crop": "rice"}).json()
    assert "series" in trend and "trend" in trend

    climate = api_client.get("/api/explorer/climate-yield", params={"variable": "rainfall_mm"}).json()
    assert "correlation" in climate

    regions = api_client.get("/api/explorer/regions").json()
    assert "top" in regions


def test_explorer_without_data_returns_409(api_client):
    r = api_client.get("/api/explorer/summary")
    assert r.status_code == 409


def test_features_build_and_preview(api_client):
    api_client.post("/api/datasets/sample")
    r = api_client.post("/api/features/build", json={"include_gdd": True, "include_lags": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_features"] > 10
    assert "gdd" in body["columns"]

    preview = api_client.get(f"/api/features/{body['feature_set_id']}/preview")
    assert preview.status_code == 200
    assert len(preview.json()["preview"]) > 0


def test_train_predict_and_export(api_client):
    api_client.post("/api/datasets/sample")

    r = api_client.post(
        "/api/models/train",
        json={
            "models": ["baseline_crop_season", "ridge", "random_forest"],
            "schemes": ["temporal", "group", "random"],
            "test_years": 3,
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    # Poll until the background job finishes.
    status = None
    for _ in range(60):
        status = api_client.get(f"/api/models/jobs/{job_id}").json()
        if status["status"] in {"completed", "failed"}:
            break
        time.sleep(0.5)
    assert status is not None and status["status"] == "completed", status

    models = api_client.get("/api/models").json()["models"]
    assert len(models) >= 3
    active = next((m for m in models if m["is_active"]), None)
    assert active is not None

    compare = api_client.get("/api/models/compare", params={"scheme": "temporal"}).json()
    assert len(compare["comparison"]) >= 3

    model_id = active["model_id"]
    metrics = api_client.get(f"/api/models/{model_id}/metrics").json()
    assert metrics["metrics"]

    importance = api_client.get(f"/api/models/{model_id}/importance").json()
    assert len(importance["importance"]) > 0

    # Single prediction with an interval.
    pred = api_client.post(
        "/api/predict",
        json={"crop": "rice", "district": "Ludhiana", "state": "Punjab", "season": "Kharif", "year": 2024},
    )
    assert pred.status_code == 200, pred.text
    body = pred.json()
    assert body["y_pred"] is not None

    # What-if sensitivity curve.
    whatif = api_client.post(
        "/api/predict/whatif",
        json={"crop": "rice", "feature": "rainfall_mm"},
    )
    assert whatif.status_code == 200
    assert len(whatif.json()["predictions"]) == 13

    # Export CSV and Excel.
    csv = api_client.get("/api/export/data.csv")
    assert csv.status_code == 200
    assert b"yield_t_ha" in csv.content

    xlsx = api_client.get("/api/export/results.xlsx")
    assert xlsx.status_code == 200
    assert len(xlsx.content) > 0


def test_extrapolation_warning_in_prediction(api_client):
    api_client.post("/api/datasets/sample")
    api_client.post(
        "/api/models/train",
        json={"models": ["random_forest"], "schemes": ["temporal"], "test_years": 3},
    )
    # Wait for training.
    for _ in range(60):
        jobs = api_client.get("/api/models/jobs").json()["jobs"]
        if jobs and jobs[0]["status"] == "completed":
            break
        time.sleep(0.5)

    pred = api_client.post(
        "/api/predict",
        json={"crop": "rice", "rainfall_mm": 99999.0},
    )
    assert pred.status_code == 200, pred.text
    assert any("rainfall_mm" in w for w in pred.json()["warnings"])


def test_predict_without_model_returns_409(api_client):
    api_client.post("/api/datasets/sample")
    r = api_client.post("/api/predict", json={"crop": "rice"})
    assert r.status_code == 409


def test_invalid_train_request_returns_422(api_client):
    api_client.post("/api/datasets/sample")
    r = api_client.post("/api/models/train", json={"models": ["not_a_model"]})
    assert r.status_code == 422


def test_batch_prediction_returns_csv(api_client):
    api_client.post("/api/datasets/sample")
    api_client.post(
        "/api/models/train",
        json={"models": ["ridge"], "schemes": ["temporal"], "test_years": 3},
    )
    for _ in range(60):
        jobs = api_client.get("/api/models/jobs").json()["jobs"]
        if jobs and jobs[0]["status"] == "completed":
            break
        time.sleep(0.5)

    r = api_client.post(
        "/api/predict/batch",
        files={"file": ("batch.csv", _sample_csv(20), "text/csv")},
    )
    assert r.status_code == 200, r.text
    assert b"y_pred" in r.content


def test_report_generation_and_download(api_client):
    api_client.post("/api/datasets/sample")
    api_client.post(
        "/api/models/train",
        json={"models": ["ridge"], "schemes": ["temporal"], "test_years": 3},
    )
    for _ in range(60):
        jobs = api_client.get("/api/models/jobs").json()["jobs"]
        if jobs and jobs[0]["status"] == "completed":
            break
        time.sleep(0.5)

    r = api_client.post("/api/report", json={"title": "Test Report"})
    assert r.status_code == 200, r.text
    report_id = r.json()["report_id"]

    download = api_client.get(f"/api/report/{report_id}")
    assert download.status_code == 200
    assert len(download.content) > 0
