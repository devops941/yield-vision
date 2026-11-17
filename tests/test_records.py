"""API tests for the manual data-entry endpoints."""

from __future__ import annotations


def _good_row(**overrides):
    row = {
        "state": "Punjab",
        "district": "Ludhiana",
        "crop": "rice",
        "season": "Kharif",
        "year": 2023,
        "yield_t_ha": 4.2,
        "rainfall_mm": 850.0,
        "temp_mean_c": 26.0,
        "humidity_pct": 65.0,
        "soil_ph": 6.5,
    }
    row.update(overrides)
    return row


def test_records_meta_lists_fields_and_options(api_client):
    r = api_client.get("/api/records/meta")
    assert r.status_code == 200
    data = r.json()
    assert [g["key"] for g in data["groups"]] == [
        "identity",
        "output",
        "weather",
        "soil",
        "management",
    ]
    assert "rice" in data["options"]["crops"]
    assert "Kharif" in data["options"]["seasons"]
    # Every field carries a label and a type so the UI can render it blindly.
    for group in data["groups"]:
        for field in group["fields"]:
            assert field["label"]
            assert field["type"] in {"text", "number", "select"}


def test_crop_guide_returns_reference_ranges(api_client):
    r = api_client.get("/api/records/crop-guide")
    assert r.status_code == 200
    crops = r.json()["crops"]
    assert "rice" in crops
    assert crops["rice"]["yield_min"] < crops["rice"]["yield_max"]


def test_entered_record_feeds_analytics(api_client):
    """A typed record must show up in the explorer summary and record list."""
    before = api_client.get("/api/explorer/summary")
    if before.status_code == 409:
        before_rows = 0
    else:
        before_rows = before.json()["rows"]

    r = api_client.post("/api/records/entry", json={"rows": [_good_row()]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["saved"] is True
    assert body["inserted"] == 1
    assert body["counts"]["rows"] == before_rows + 1

    summary = api_client.get("/api/explorer/summary").json()
    assert summary["rows"] == before_rows + 1
    assert "rice" in summary["crops"]

    recent = api_client.get("/api/records/recent?limit=5").json()
    assert recent["count"] >= 1
    assert recent["records"][0]["district"] == "Ludhiana"


def test_entered_yield_can_be_derived_from_area_and_production(api_client):
    r = api_client.post(
        "/api/records/entry",
        json={"rows": [_good_row(yield_t_ha=None, area_ha=1000.0, production_t=4200.0)]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["saved"] is True

    recent = api_client.get("/api/records/recent?limit=1").json()["records"][0]
    assert recent["yield_t_ha"] == 4.2


def test_entry_with_missing_required_field_is_not_saved(api_client):
    r = api_client.post(
        "/api/records/entry",
        json={"rows": [_good_row(rainfall_mm=None, yield_t_ha=None)]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["saved"] is False
    assert body["inserted"] == 0
    assert body["validation"]["n_errors"] >= 1


def test_entry_with_empty_rows_is_rejected(api_client):
    r = api_client.post("/api/records/entry", json={"rows": []})
    assert r.status_code == 422


def test_delete_record_removes_it_from_analytics(api_client):
    saved = api_client.post("/api/records/entry", json={"rows": [_good_row()]}).json()
    assert saved["saved"] is True
    record_id = api_client.get("/api/records/recent?limit=1").json()["records"][0]["record_id"]

    r = api_client.delete(f"/api/records/{record_id}")
    assert r.status_code == 200
    assert r.json()["deleted"] == record_id
    assert r.json()["counts"]["rows"] == saved["counts"]["rows"] - 1

    missing = api_client.delete(f"/api/records/{record_id}")
    assert missing.status_code == 404
