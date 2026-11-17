"""Command-line pipeline: ingest -> validate -> features -> train -> evaluate.

Usage:
    python -m src.pipeline --source synthetic
    python -m src.pipeline --source file --path data/sample/crop.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.config import ensure_dirs
from src.data.ingest import canonicalise, load_crop_file
from src.data.preprocess import preprocess
from src.data.validate import validate
from src.features.engineer import FeatureConfig
from src.models.service import train_and_register


def load_source(source: str, path: str | None) -> pd.DataFrame:
    if source == "synthetic":
        from src.data.synthetic import generate_dataset

        return generate_dataset()
    if source == "file":
        if not path:
            raise SystemExit("--path is required when --source file")
        return canonicalise(load_crop_file(path, Path(path).name))
    raise SystemExit(f"Unknown source: {source}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="YieldVision end-to-end pipeline")
    parser.add_argument("--source", choices=["synthetic", "file"], default="synthetic")
    parser.add_argument("--path", default=None, help="Path to a crop CSV when --source file")
    parser.add_argument(
        "--models",
        default="baseline_crop_season,baseline_district_history,ridge,random_forest",
    )
    parser.add_argument("--schemes", default="temporal,group,random")
    parser.add_argument("--test-years", type=int, default=4)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--no-train", action="store_true", help="Only run ingest/validate/features")
    args = parser.parse_args(argv)

    ensure_dirs()
    print(f"[1/5] Loading data from '{args.source}'…")
    raw = load_source(args.source, args.path)
    print(f"      {len(raw):,} raw rows")

    print("[2/5] Validating…")
    report = validate(raw)
    print(f"      {report.summary()}")

    print("[3/5] Pre-processing…")
    clean = preprocess(raw)
    print(f"      {len(clean):,} clean rows")

    # Persist clean records into SQLite and mirror to MongoDB
    from src.data import db as dbmod, mongo

    session = dbmod.get_session()
    try:
        filename = Path(args.path).name if args.path else f"{args.source}_dataset.csv"
        upload = dbmod.create_upload(session, filename, stored_path=None)
        upload.n_rows = report.n_rows
        upload.n_errors = report.n_errors
        upload.n_warnings = report.n_warnings
        upload.status = "committed"
        session.flush()
        dbmod.commit_records(session, upload.upload_id, clean)
        session.commit()
        mongo.sync_records(dbmod.load_records_df(session))
    finally:
        session.close()

    print("[4/5] Engineering features…")
    config = FeatureConfig()
    from src.features.engineer import engineer_features

    features = engineer_features(clean, config)
    print(f"      {features.shape[1]} columns")

    if args.no_train:
        print("[5/5] Skipping training (--no-train).")
        return 0

    print("[5/5] Training and evaluating models…")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    result = train_and_register(
        clean,
        models=models,
        schemes=schemes,
        feature_config=config,
        tune=args.tune,
        test_years=args.test_years,
        progress=lambda pct, msg: print(f"      {pct:3d}% {msg}"),
    )

    print("\nHeadline (temporal) metrics:")
    for model in result["models"]:
        m = model["metrics"]
        rmse = f"{m['rmse']:.3f}" if m.get("rmse") is not None else "n/a"
        r2 = f"{m['r2']:.3f}" if m.get("r2") is not None else "n/a"
        print(f"  {model['name']:28s} rmse={rmse} r2={r2}")

    print("\nDone. Start the app with: uvicorn backend.main:app --reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
