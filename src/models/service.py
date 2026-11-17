"""Training service: orchestrate training, evaluation and registry persistence.

Kept separate from the FastAPI router so it can run inside a background task and
be unit tested without the web layer.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from src.data.db import EvalPrediction, FeatureImportanceRow, get_session
from src.features.engineer import FeatureConfig
from src.models import registry
from src.models.evaluate import metrics_by_scope
from src.models.explain import impurity_importance, permutation_importances
from src.models.intervals import IntervalModel
from src.models.train import train_all_models

ProgressFn = Callable[[int, str], None]


def train_and_register(
    df: pd.DataFrame,
    *,
    models: list[str],
    schemes: list[str],
    feature_config: FeatureConfig,
    tune: bool = False,
    test_years: int = 4,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Train every model under every scheme, persist versions, metrics and predictions."""
    session = get_session()
    registered: list[dict[str, Any]] = []
    comparison: dict[str, dict[str, Any]] = {}
    baseline_rmse: dict[str, float] = {}

    try:
        total = max(1, len(models) * len(schemes))
        step = 0
        per_model_predictions: dict[str, dict[str, np.ndarray]] = {}

        for scheme in schemes:
            results = train_all_models(
                df,
                models=models,
                scheme=scheme,
                feature_config=feature_config,
                tune=tune,
                test_years=test_years,
            )
            # Baseline RMSE for the skill score (best of the two baselines).
            base_rmses = [
                r["metrics"]["rmse"]
                for name, r in results.items()
                if name.startswith("baseline_") and r["metrics"]["rmse"] is not None
            ]
            best_baseline = min(base_rmses) if base_rmses else None

            for name, result in results.items():
                step += 1
                if progress:
                    progress(int(100 * step / total), f"{name} / {scheme}")
                metrics = dict(result["metrics"])
                if best_baseline and metrics.get("rmse") is not None:
                    metrics["skill_vs_baseline"] = 1.0 - metrics["rmse"] / best_baseline
                comparison.setdefault(name, {})[scheme] = metrics
                per_model_predictions.setdefault(name, {})[scheme] = {
                    "y_true": result["y_true"],
                    "y_pred": result["y_pred"],
                    "pipeline": result.get("pipeline"),
                    "test_df": result.get("test_df"),
                }

        # Persist one version per model, using the headline (temporal) scheme metrics.
        headline = "temporal" if "temporal" in schemes else schemes[0]
        for name in models:
            if name not in comparison:
                continue
            scheme_results = comparison[name]
            metrics = scheme_results.get(headline) or next(iter(scheme_results.values()))

            interval_model = None
            pipeline = per_model_predictions.get(name, {}).get(headline, {}).get("pipeline")
            params: dict[str, Any] = {}
            if pipeline is not None:
                try:
                    interval_model = IntervalModel.fit(
                        df,
                        model_name=name,
                        params=None,
                        feature_config=feature_config,
                    )
                except Exception:
                    interval_model = None

            model_id = registry.save_model(
                session,
                name=name,
                algo=name,
                pipeline=pipeline,
                interval_model=interval_model,
                params=params,
                feature_list=list(feature_config.to_dict().keys()),
                train_rows=int(len(df)),
            )

            for scheme, scheme_metrics in scheme_results.items():
                registry.add_metrics(session, model_id, scheme, scheme_metrics)

            # Scope breakdowns on the headline scheme predictions.
            preds = per_model_predictions.get(name, {}).get(headline)
            if preds is not None and preds.get("test_df") is not None:
                scoped = metrics_by_scope(
                    preds["test_df"], preds["y_true"], preds["y_pred"], ["crop", "season", "state", "year"]
                )
                for scope_row in scoped:
                    registry.add_metrics(
                        session,
                        model_id,
                        headline,
                        scope_row,
                        scope_type=scope_row["scope_type"],
                        scope_value=scope_row["scope_value"],
                    )
                _store_predictions(session, model_id, headline, preds)

            if pipeline is not None:
                _store_importance(session, model_id, pipeline, df)

            registered.append(
                {
                    "model_id": model_id,
                    "name": name,
                    "version": 1,
                    "headline_scheme": headline,
                    "metrics": metrics,
                }
            )

        session.commit()

        # Activate the non-baseline model with the best headline RMSE.
        candidates = [
            r
            for r in registered
            if not r["name"].startswith("baseline_") and r["metrics"].get("rmse") is not None
        ]
        if candidates:
            best = min(candidates, key=lambda r: r["metrics"]["rmse"])
            registry.activate_model(session, best["model_id"])
            session.commit()

        return {"models": registered, "comparison": comparison, "headline_scheme": headline}
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _store_predictions(session, model_id: int, scheme: str, preds: dict[str, Any]) -> None:
    y_true = np.asarray(preds["y_true"], dtype=float)
    y_pred = np.asarray(preds["y_pred"], dtype=float)
    test_df = preds.get("test_df")
    districts = test_df["district"].tolist() if test_df is not None and "district" in test_df else []
    crops = test_df["crop"].tolist() if test_df is not None and "crop" in test_df else []
    years = test_df["year"].tolist() if test_df is not None and "year" in test_df else []
    n = min(len(y_true), 500)
    for i in range(n):
        session.add(
            EvalPrediction(
                model_id=model_id,
                scheme=scheme,
                district=str(districts[i]) if i < len(districts) else None,
                crop=str(crops[i]) if i < len(crops) else None,
                year=int(years[i]) if i < len(years) and years[i] is not None else None,
                y_true=float(y_true[i]),
                y_pred=float(y_pred[i]),
            )
        )
    session.flush()


def _store_importance(session, model_id: int, pipeline, df: pd.DataFrame) -> None:
    try:
        impurity = impurity_importance(pipeline)
    except Exception:
        impurity = {}
    try:
        perm = permutation_importances(pipeline, df, n_repeats=5)
    except Exception:
        perm = pd.DataFrame()

    perm_map = (
        {row["feature"]: (row["importance_mean"], row["importance_std"]) for _, row in perm.iterrows()}
        if not perm.empty
        else {}
    )
    features = sorted(set(impurity) | set(perm_map))
    for feature in features:
        imp_value = impurity.get(feature)
        perm_value, perm_std = perm_map.get(feature, (None, None))
        session.add(
            FeatureImportanceRow(
                model_id=model_id,
                feature=feature,
                impurity_importance=imp_value,
                permutation_importance=perm_value,
                permutation_std=perm_std,
            )
        )
    session.flush()
