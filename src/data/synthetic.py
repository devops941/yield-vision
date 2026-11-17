"""Synthetic data generator for YieldVision.

Generates a realistic district-level crop yield dataset from a *known*
data-generating process so that demos and tests never depend on downloads.

The response surface is deliberately non-linear:

* yield rises with seasonal rainfall up to a crop-specific optimum, then falls,
* yield declines once the mean temperature exceeds a crop-specific threshold,
* yield increases with soil nitrogen with diminishing (saturating) returns,
* yield peaks near an optimal soil pH,
* yield increases with the irrigated fraction and fertiliser use,
* a slow upward technology trend applies across years,
* Gaussian noise is added.

The generator also injects a configurable set of *deliberate* data errors so the
validation layer can be exercised end to end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import all_crops, all_seasons, crop_settings

TECH_TREND_RATE = 0.035  # ~3.5 %/yr technology improvement

CANONICAL_COLUMNS = [
    "state",
    "district",
    "crop",
    "season",
    "year",
    "area_ha",
    "production_t",
    "yield_t_ha",
    "rainfall_mm",
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "humidity_pct",
    "soil_n_kgha",
    "soil_p_kgha",
    "soil_k_kgha",
    "soil_ph",
    "soil_oc_pct",
    "fertilizer_kgha",
    "irrigated_frac",
    "pesticide_kgha",
]

STATES = {
    "Punjab": ["Ludhiana", "Amritsar", "Patiala", "Bathinda", "Jalandhar"],
    "Maharashtra": ["Nashik", "Pune", "Nagpur", "Aurangabad", "Solapur"],
    "Karnataka": ["Belagavi", "Mysuru", "Dharwad", "Ballari", "Tumakuru"],
    "Tamil Nadu": ["Thanjavur", "Coimbatore", "Madurai", "Salem", "Erode"],
}

# Season -> (dominant crops). Each (district, crop, season, year) row is generated
# for the seasons the crop is grown in.
CROP_SEASONS = {
    "rice": ["Kharif", "Rabi"],
    "wheat": ["Rabi"],
    "maize": ["Kharif", "Rabi"],
    "cotton": ["Kharif"],
    "groundnut": ["Kharif", "Summer"],
}


def _response_surface(
    crop: str,
    rainfall: np.ndarray,
    temp_mean: np.ndarray,
    soil_n: np.ndarray,
    soil_ph: np.ndarray,
    irrigated_frac: np.ndarray,
    fertilizer: np.ndarray,
    year: np.ndarray,
    year0: int,
) -> np.ndarray:
    """Deterministic part of the data-generating process (before noise).

    A multiplicative stress-factor model keeps yields in realistic units: each
    factor is ~1 under ideal conditions and falls below 1 under stress, so the
    product stays inside the crop's plausible range.
    """
    cfg = crop_settings(crop)
    rain_opt = float(cfg.get("rainfall_optimal_mm", 800.0))
    temp_thresh = float(cfg.get("heat_stress_temp_c", 35.0))
    ph_opt = float(cfg.get("optimal_ph", 6.5))
    base = 0.55 * float(cfg.get("yield_max", 10.0))

    # Rainfall: peaks at the optimum, falls off at both extremes.
    rain_factor = np.exp(-(((rainfall - rain_opt) / (0.62 * rain_opt)) ** 2))
    # Drought regime: below ~70 % of the optimum there is an extra penalty. This
    # step-like behaviour is what tree ensembles capture and a linear model misses.
    rain_factor = rain_factor * np.where(rainfall < 0.7 * rain_opt, 0.75, 1.0)

    # Temperature: peaks in the crop's comfort band and declines above the
    # heat-stress threshold.
    temp_center = 0.68 * temp_thresh
    temp_factor = np.exp(-(((temp_mean - temp_center) / (0.20 * temp_thresh)) ** 2))

    # Nitrogen: saturating (diminishing returns).
    n_factor = 0.62 + 0.38 * (1.0 - np.exp(-soil_n / 150.0))

    # Soil pH: peak at the crop optimum.
    ph_factor = 0.78 + 0.22 * np.exp(-(((soil_ph - ph_opt) / 0.9) ** 2))

    # Irrigation and fertiliser: positive, diminishing.
    irr_factor = 0.82 + 0.18 * irrigated_frac
    fert_factor = 0.92 + 0.08 * np.log1p(np.clip(fertilizer, 0, None) / 50.0)

    # Nonlinear interaction: rainfall is only fully effective when nitrogen and
    # irrigation are also adequate. A purely linear model cannot represent this.
    interaction = 1.0 + 0.35 * rain_factor * (1.0 - np.exp(-soil_n / 150.0)) * irrigated_frac

    # Technology trend (~+3.5 %/yr).
    trend = 1.0 + TECH_TREND_RATE * (year - year0)

    return (
        base
        * rain_factor
        * temp_factor
        * n_factor
        * ph_factor
        * irr_factor
        * fert_factor
        * interaction
        * trend
    )


def generate_dataset(
    n_districts: int = 20,
    n_crops: int = 4,
    n_years: int = 20,
    start_year: int = 2005,
    seed: int = 42,
    inject_errors: bool = True,
    noise_std: float = 0.10,
) -> pd.DataFrame:
    """Generate the synthetic district-crop-season-year dataset.

    Parameters
    ----------
    n_districts:
        Number of distinct districts (spread across the configured states).
    n_crops:
        Number of crops to include (taken from the config, in order).
    n_years:
        Number of years of history.
    start_year:
        First year in the panel.
    seed:
        Random seed for reproducibility.
    inject_errors:
        Whether to inject the deliberate data-quality errors.
    noise_std:
        Standard deviation of the multiplicative Gaussian noise.
    """
    rng = np.random.default_rng(seed)

    # --- Districts ---
    flat_districts = [
        (state, district) for state, districts in STATES.items() for district in districts
    ]
    if n_districts > len(flat_districts):
        raise ValueError(f"Only {len(flat_districts)} districts are defined in STATES.")
    district_pool = flat_districts[:n_districts]

    crops = all_crops()[:n_crops]
    seasons = all_seasons()
    years = np.arange(start_year, start_year + n_years)

    # Per-district static characteristics (soil baseline, agro-climate offset).
    district_features = {}
    for state, district in district_pool:
        district_features[(state, district)] = {
            "soil_n_base": rng.uniform(180.0, 340.0),
            "soil_p_base": rng.uniform(18.0, 38.0),
            "soil_k_base": rng.uniform(160.0, 280.0),
            "soil_ph_base": rng.uniform(5.6, 7.8),
            "soil_oc_base": rng.uniform(0.35, 0.9),
            "irrigated_base": rng.uniform(0.25, 0.85),
            "rainfall_scale": rng.uniform(0.7, 1.35),
            "temp_offset": rng.uniform(-1.5, 1.5),
            "lat": rng.uniform(8.0, 32.0),
            "lon": rng.uniform(72.0, 88.0),
        }

    rows: list[dict] = []
    for state, district in district_pool:
        df = district_features[(state, district)]
        for crop in crops:
            cfg = crop_settings(crop)
            for season in CROP_SEASONS.get(crop, seasons):
                season_cfg = cfg
                rain_opt = float(season_cfg.get("rainfall_optimal_mm", 800.0))
                for year in years:
                    # Seasonal weather with year-to-year variability.
                    rainfall = max(
                        40.0,
                        rng.normal(rain_opt * df["rainfall_scale"], 0.28 * rain_opt),
                    )
                    temp_mean = rng.normal(
                        26.0 + df["temp_offset"] + (1.5 if season == "Summer" else 0.0), 0.8
                    )
                    temp_max = temp_mean + rng.uniform(6.0, 10.0)
                    temp_min = temp_mean - rng.uniform(6.0, 10.0)
                    humidity = float(np.clip(rng.normal(65.0, 12.0), 15.0, 98.0))

                    soil_n = max(20.0, rng.normal(df["soil_n_base"], 25.0))
                    soil_p = max(3.0, rng.normal(df["soil_p_base"], 14.0))
                    soil_k = max(30.0, rng.normal(df["soil_k_base"], 70.0))
                    soil_ph = float(np.clip(rng.normal(df["soil_ph_base"], 0.35), 3.5, 9.5))
                    soil_oc = max(0.1, rng.normal(df["soil_oc_base"], 0.12))
                    irrigated = float(
                        np.clip(rng.normal(df["irrigated_base"], 0.1), 0.02, 1.0)
                    )
                    fertilizer = max(
                        10.0, rng.normal(120.0 * irrigated + 40.0, 30.0)
                    )
                    pesticide = max(0.0, rng.normal(2.5, 1.0))
                    area = float(np.clip(rng.normal(45000.0, 15000.0), 500.0, None))

                    signal = _response_surface(
                        crop,
                        np.array([rainfall]),
                        np.array([temp_mean]),
                        np.array([soil_n]),
                        np.array([soil_ph]),
                        np.array([irrigated]),
                        np.array([fertilizer]),
                        np.array([year]),
                        start_year,
                    )[0]
                    noise = rng.normal(1.0, noise_std)
                    yld = float(np.clip(signal * noise, 0.15, cfg.get("yield_max", 12.0)))
                    production = yld * area

                    rows.append(
                        {
                            "state": state,
                            "district": district,
                            "crop": crop,
                            "season": season,
                            "year": int(year),
                            "area_ha": round(area, 2),
                            "production_t": round(production, 2),
                            "yield_t_ha": round(yld, 4),
                            "rainfall_mm": round(rainfall, 2),
                            "temp_mean_c": round(temp_mean, 2),
                            "temp_max_c": round(temp_max, 2),
                            "temp_min_c": round(temp_min, 2),
                            "humidity_pct": round(humidity, 2),
                            "soil_n_kgha": round(soil_n, 2),
                            "soil_p_kgha": round(soil_p, 2),
                            "soil_k_kgha": round(soil_k, 2),
                            "soil_ph": round(soil_ph, 2),
                            "soil_oc_pct": round(soil_oc, 3),
                            "fertilizer_kgha": round(fertilizer, 2),
                            "irrigated_frac": round(irrigated, 3),
                            "pesticide_kgha": round(pesticide, 3),
                        }
                    )

    df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    if inject_errors:
        df = _inject_errors(df, rng)
    return df.reset_index(drop=True)


def _inject_errors(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject a deterministic set of data-quality problems.

    The error inventory (counts) is returned by :func:`error_inventory` and is
    asserted by the validation tests.
    """
    df = df.copy()
    n = len(df)

    # 1. Duplicate rows (same district, crop, season, year).
    dup_idx = rng.choice(n, size=max(3, n // 200), replace=False)
    duplicates = df.iloc[dup_idx].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # 2. Negative area.
    neg_idx = rng.choice(len(df), size=4, replace=False)
    df.loc[neg_idx, "area_ha"] = -np.abs(df.loc[neg_idx, "area_ha"])

    # 3. Implausible yield (500 t/ha).
    big_idx = rng.choice(len(df), size=3, replace=False)
    df.loc[big_idx, "yield_t_ha"] = 500.0

    # 4. Text in a numeric column.
    txt_idx = rng.choice(len(df), size=5, replace=False)
    df["rainfall_mm"] = df["rainfall_mm"].astype(object)
    df.loc[txt_idx, "rainfall_mm"] = "NA"

    # 5. Missing season labels (blank).
    blank_idx = rng.choice(len(df), size=3, replace=False)
    df.loc[blank_idx, "season"] = None

    # 6. Area in acres instead of hectares (×2.471 conversion needed).
    acre_idx = rng.choice(len(df), size=6, replace=False)
    df.loc[acre_idx, "area_ha"] = (df.loc[acre_idx, "area_ha"].astype(float) * 2.471).round(2)

    # 7. Missing values scattered across feature columns.
    for col in ["soil_p_kgha", "humidity_pct", "soil_oc_pct", "temp_min_c"]:
        miss_idx = rng.choice(len(df), size=max(3, len(df) // 150), replace=False)
        df.loc[miss_idx, col] = np.nan

    # 8. Invalid pH values.
    ph_idx = rng.choice(len(df), size=3, replace=False)
    df.loc[ph_idx, "soil_ph"] = rng.choice([1.2, 11.5, 14.0], size=len(ph_idx))

    # 9. temp_min > temp_max.
    swap_idx = rng.choice(len(df), size=3, replace=False)
    tmp = df.loc[swap_idx, "temp_max_c"].copy()
    df.loc[swap_idx, "temp_max_c"] = df.loc[swap_idx, "temp_min_c"]
    df.loc[swap_idx, "temp_min_c"] = tmp

    # 10. Yield inconsistent with production / area.
    incon_idx = rng.choice(len(df), size=4, replace=False)
    df.loc[incon_idx, "production_t"] = (
        df.loc[incon_idx, "production_t"].astype(float) * 0.5
    ).round(2)

    return df


def error_inventory(df: pd.DataFrame) -> dict[str, int]:
    """Best-effort summary of the injected error types present in ``df``."""
    inv: dict[str, int] = {}
    inv["duplicates"] = int(df.duplicated(subset=["district", "crop", "season", "year"]).sum())
    inv["negative_area"] = int((pd.to_numeric(df["area_ha"], errors="coerce") < 0).sum())
    inv["implausible_yield"] = int((pd.to_numeric(df["yield_t_ha"], errors="coerce") > 50).sum())
    inv["non_numeric"] = int(
        pd.to_numeric(df["rainfall_mm"], errors="coerce").isna().sum()
        - df["rainfall_mm"].isna().sum()
    )
    inv["missing_season"] = int(df["season"].isna().sum())
    inv["invalid_ph"] = int(
        ((pd.to_numeric(df["soil_ph"], errors="coerce") < 3) | (pd.to_numeric(df["soil_ph"], errors="coerce") > 10)).sum()
    )
    return inv


def write_template(path: str | Path) -> Path:
    """Write a blank CSV template with the canonical schema and one example row."""
    example = {
        "state": "Punjab",
        "district": "Ludhiana",
        "crop": "rice",
        "season": "Kharif",
        "year": 2020,
        "area_ha": 45000,
        "production_t": 162000,
        "yield_t_ha": 3.6,
        "rainfall_mm": 950,
        "temp_mean_c": 27.5,
        "temp_max_c": 35.0,
        "temp_min_c": 20.0,
        "humidity_pct": 70,
        "soil_n_kgha": 260,
        "soil_p_kgha": 22,
        "soil_k_kgha": 210,
        "soil_ph": 6.6,
        "soil_oc_pct": 0.55,
        "fertilizer_kgha": 130,
        "irrigated_frac": 0.8,
        "pesticide_kgha": 2.2,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([example], columns=CANONICAL_COLUMNS).to_csv(path, index=False)
    return path


def save_sample(path: str | Path | None = None) -> Path:
    """Generate and persist the sample synthetic dataset as CSV."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "sample" / "synthetic_yield.csv"
    path = Path(path)
    df = generate_dataset()
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":  # pragma: no cover - CLI helper
    out = save_sample()
    print(f"Wrote synthetic dataset to {out}")
