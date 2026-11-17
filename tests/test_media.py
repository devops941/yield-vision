"""Tests for the image/video analysis and MongoDB persistence layers."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data import media, mongo


@pytest.fixture()
def canopy_image(tmp_path):
    arr = np.zeros((160, 160, 3), dtype=np.uint8)
    arr[..., 1] = 185
    arr[..., 0] = 60
    arr[..., 2] = 55
    path = tmp_path / "canopy.jpg"
    Image.fromarray(arr).save(path)
    return path


@pytest.fixture()
def soil_image(tmp_path):
    arr = np.zeros((160, 160, 3), dtype=np.uint8)
    arr[..., 0] = 150
    arr[..., 1] = 112
    arr[..., 2] = 82
    path = tmp_path / "soil.jpg"
    Image.fromarray(arr).save(path)
    return path


def test_canopy_is_greener_than_soil(canopy_image, soil_image):
    canopy = media.analyse_image(canopy_image)
    soil = media.analyse_image(soil_image)
    assert canopy["vigour"] > soil["vigour"]
    assert canopy["indices"]["green_fraction"] > soil["indices"]["green_fraction"]
    assert 0.0 <= canopy["vigour"] <= 1.0


def test_features_from_analysis_are_bounded(canopy_image):
    analysis = media.analyse_image(canopy_image)
    features = media.features_from_analysis(analysis, {"crop": "rice", "district": "Ludhiana"})
    assert features["crop"] == "rice"
    assert 0.0 <= features["irrigated_frac"] <= 1.0
    assert 400 <= features["rainfall_mm"] <= 1500
    assert features["soil_ph"] >= 0


def test_analyse_media_rejects_unknown_type(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with pytest.raises(ValueError):
        media.analyse_media(bad, "notes.txt")


def test_mongo_disabled_is_graceful(monkeypatch):
    monkeypatch.setenv("YIELDVISION_NO_MONGO", "1")
    monkeypatch.setattr(mongo, "_DB", None, raising=False)
    assert mongo.get_db() is None
    assert mongo.status()["enabled"] is False
    assert mongo.sync_records(None) == 0
    assert mongo.load_records() == []
