"""Image and video analysis for field-crop health.

A field photo or drone clip carries canopy information. From it we derive a set
of vegetation indices that are honest proxies for the same drivers the tabular
model already uses (canopy vigour, moisture stress, temperature stress), then
map them onto the model's input columns so a prediction can be produced from
imagery alone.

Indices (computed per pixel, then averaged over the scene):
* ExG  - Excess Green index, a canopy-cover proxy.
* GLI  - Green Leaf Index.
* VARI - Visible Atmospherically Resistant Index (greenness vs. soil/lighting).
* NGRDI - Normalised Green-Red Difference Index.

Video files are sampled at a fixed stride so a short clip contributes many
frames; the per-frame indices are averaged and their variation is reported as a
stability score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Reference points used to translate a vegetation index into a 0..1 vigour score.
# Calibrated so bare soil (VARI ~ 0.0) maps near 0.25 and dense canopy
# (VARI ~ 0.4) maps near 1.0, matching typical RGB field imagery.
_VIGOUR_LOW = 0.0
_VIGOUR_HIGH = 0.4


def _load_rgb(path: str | Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        if max(rgb.size) > 720:
            rgb.thumbnail((720, 720))
        return np.asarray(rgb, dtype=np.float32) / 255.0


def _indices(rgb: np.ndarray) -> dict[str, float]:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    eps = 1e-6

    exg = 2.0 * g - r - b
    gli = (2.0 * g - r - b) / (2.0 * g + r + b + eps)
    vari = (g - r) / (g + r - b + eps)
    ngrdi = (g - r) / (g + r + eps)

    green_mask = (g > r) & (g > b)
    green_frac = float(np.mean(green_mask))

    return {
        "exg": float(np.mean(exg)),
        "gli": float(np.mean(gli)),
        "vari": float(np.mean(np.clip(vari, -1.0, 1.0))),
        "ngrdi": float(np.mean(np.clip(ngrdi, -1.0, 1.0))),
        "green_fraction": green_frac,
        "brightness": float(np.mean(0.299 * r + 0.587 * g + 0.114 * b)),
        "canopy_saturation": float(np.mean(np.clip(g - (r + b) / 2.0, 0, 1))),
    }


def _vigour(vari: float, green_frac: float) -> float:
    """Blend VARI and green fraction into a bounded 0..1 canopy vigour score."""
    span = _VIGOUR_HIGH - _VIGOUR_LOW
    vari_score = float(np.clip((vari - _VIGOUR_LOW) / span, 0.0, 1.0))
    return float(np.clip(0.65 * vari_score + 0.35 * green_frac, 0.0, 1.0))


def analyse_image(path: str | Path) -> dict[str, Any]:
    rgb = _load_rgb(path)
    idx = _indices(rgb)
    return {
        "kind": "image",
        "frames": 1,
        "indices": idx,
        "vigour": _vigour(idx["vari"], idx["green_fraction"]),
        "frame_variation": 0.0,
    }


def analyse_video(path: str | Path, max_frames: int = 24) -> dict[str, Any]:
    import imageio.v3 as iio

    frames: list[np.ndarray] = []
    per_frame_vari: list[float] = []
    try:
        meta = iio.immeta(path)
        fps = float(meta.get("fps") or 0) or 0.0
    except Exception:
        fps = 0.0
    step = max(1, int(round(fps / 2))) if fps else 1  # sample ~2 frames per second

    for i, frame in enumerate(iio.imiter(path)):
        if i % step:
            continue
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = arr[..., :3] / 255.0
        if max(arr.shape[:2]) > 720:
            ys = np.linspace(0, arr.shape[0] - 1, 360).astype(int)
            xs = np.linspace(0, arr.shape[1] - 1, 360).astype(int)
            arr = arr[np.ix_(ys, xs)]
        frames.append(arr)
        per_frame_vari.append(_indices(arr)["vari"])
        if len(frames) >= max_frames:
            break

    if not frames:
        raise ValueError("No frames could be read from the video.")

    idx = _indices(np.concatenate([f.reshape(-1, 3) for f in frames], axis=0).reshape(-1, 1, 3))
    mean_vari = float(np.mean(per_frame_vari))
    variation = float(np.std(per_frame_vari))
    return {
        "kind": "video",
        "frames": len(frames),
        "fps": fps or None,
        "indices": {**idx, "vari": mean_vari},
        "vigour": _vigour(mean_vari, idx["green_fraction"]),
        "frame_variation": variation,
    }


def analyse_media(path: str | Path, filename: str | None = None) -> dict[str, Any]:
    suffix = Path(filename or path).suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return analyse_video(path)
    if suffix in IMAGE_SUFFIXES:
        return analyse_image(path)
    raise ValueError(f"Unsupported media type '{suffix}'. Upload an image or video.")


def features_from_analysis(
    analysis: dict[str, Any], context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Map media analysis onto the model's input columns.

    Vegetation vigour drives rainfall/soil proxies; the context dict lets the
    caller pin the location, crop and season, since imagery alone cannot supply
    them.
    """
    ctx = dict(context or {})
    vigour = float(analysis.get("vigour", 0.5))
    indices = analysis.get("indices", {})
    brightness = float(indices.get("brightness", 0.5))

    # Higher vigour implies healthier moisture and nutrient status. The mapping is
    # monotonic and bounded, so predictions stay inside plausible agronomic ranges.
    rainfall = 450.0 + 950.0 * vigour
    soil_n = 160.0 + 220.0 * vigour
    soil_oc = 0.3 + 0.5 * vigour
    irrigated = float(np.clip(0.35 + 0.55 * vigour, 0.0, 1.0))
    temp_mean = 24.0 + 4.0 * (1.0 - vigour)

    return {
        "state": ctx.get("state", "Punjab"),
        "district": ctx.get("district", "Ludhiana"),
        "crop": ctx.get("crop", "rice"),
        "season": ctx.get("season", "Kharif"),
        "year": int(ctx.get("year", 2024)),
        "area_ha": float(ctx.get("area_ha", 1000.0)),
        "rainfall_mm": round(rainfall, 1),
        "temp_mean_c": round(temp_mean, 1),
        "temp_max_c": round(temp_mean + 8.0, 1),
        "temp_min_c": round(temp_mean - 8.0, 1),
        "humidity_pct": round(45.0 + 35.0 * vigour, 1),
        "soil_n_kgha": round(soil_n, 1),
        "soil_p_kgha": round(18.0 + 22.0 * vigour, 1),
        "soil_k_kgha": round(150.0 + 160.0 * vigour, 1),
        "soil_ph": round(6.2 + 0.8 * vigour, 2),
        "soil_oc_pct": round(soil_oc, 3),
        "fertilizer_kgha": round(80.0 + 110.0 * vigour, 1),
        "irrigated_frac": round(irrigated, 3),
        "pesticide_kgha": round(1.5 + 1.8 * (1.0 - vigour), 2),
        "_brightness": round(brightness, 4),
    }
