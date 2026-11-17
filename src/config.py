"""Shared configuration loading helpers for YieldVision."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DB_PATH = DATA_DIR / "yieldvision.db"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@functools.lru_cache(maxsize=None)
def load_crops_config() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "crops.yaml")


@functools.lru_cache(maxsize=None)
def load_aliases_config() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "aliases.yaml")


@functools.lru_cache(maxsize=None)
def load_models_config() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "models.yaml")


def crop_settings(crop: str) -> dict[str, Any]:
    """Return agronomic settings for ``crop`` merged over the defaults."""
    cfg = load_crops_config()
    defaults = dict(cfg.get("defaults", {}))
    crops = cfg.get("crops", {})
    key = str(crop).strip().lower()
    specific = crops.get(key)
    if specific is None:
        for name, settings in crops.items():
            if settings.get("display_name", "").lower() == key:
                specific = settings
                break
    if specific:
        defaults.update(specific)
    return defaults


def all_crops() -> list[str]:
    return list(load_crops_config().get("crops", {}).keys())


def all_seasons() -> list[str]:
    return list(load_crops_config().get("seasons", {}).keys())


def ensure_dirs() -> None:
    for path in (DATA_DIR / "uploads", DATA_DIR / "processed", MODELS_DIR, OUTPUTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
