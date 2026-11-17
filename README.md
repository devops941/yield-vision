# YieldVision

**A Climate and Soil-Driven Crop Yield Prediction Engine.**

YieldVision turns raw district-level crop records into validated data, historical
trend insight, engineered climate features, trained regression models and
uncertainty-aware yield forecasts — all served through a FastAPI backend and a
plain HTML/JavaScript + Plotly.js frontend. It also predicts yield from a field
photo or video. There is **no Streamlit**; Python owns every piece of data
processing, feature engineering, modelling and report logic, and the frontend
only displays results.

---

## Highlights

- **Guided data workflow** — upload CSV/Excel/JSON, validate against a rule book,
  review every issue, then commit a clean dataset to SQLite.
- **Historical trend explorer** — per-year yield trajectories, Mann-Kendall trend
  tests, Sen's slope, climate correlations and regional rankings.
- **Leakage-safe feature engineering** — growing degree days, climate anomalies,
  district history (past years only), interactions and soil features, all fitted
  inside cross-validation folds via scikit-learn `Pipeline` / `ColumnTransformer`.
- **Model training and comparison** — baselines → Linear/Ridge → Random Forest →
  Histogram Gradient Boosting, evaluated under temporal, GroupKFold and random
  schemes with a skill score against the best baseline.
- **Honest predictions** — point estimate plus an empirical 80% prediction
  interval, and extrapolation warnings when inputs fall outside the training range.
- **Explainability** — permutation and impurity importance, partial dependence
  curves and a plain-language summary.
- **Image and video prediction** — upload a field photo or clip; vegetation
  indices (ExG, GLI, VARI, NGRDI) are extracted and mapped onto the model inputs
  to predict yield from imagery alone.
- **Reports and exports** — a PDF analysis report, cleaned-data CSV, a results
  workbook and batch prediction downloads.
- **MongoDB Atlas storage** — uploads, records, media, predictions, model
  registry entries and reports are mirrored to MongoDB (with a SQLite cache).
- **Light, full-width UI** — a white canvas with a leaf-green palette, plain
  wording, and a layout that fills the screen.

---

## Stack

| Layer | Technology |
|---|---|
| Data / numeric | pandas 3.1, NumPy 2.5 |
| Machine learning | scikit-learn 1.9 |
| Statistics | SciPy 1.18, statsmodels 0.15, pymannkendall 1.4 |
| Backend | FastAPI 0.163, Uvicorn, Pydantic 2.14 |
| Database | SQLite via SQLAlchemy 2.1 + MongoDB Atlas via pymongo 4.20 |
| Reports | Jinja2, matplotlib, WeasyPrint |
| Media | Pillow, imageio (image and video analysis) |
| Frontend | HTML, vanilla JavaScript, Plotly.js |
| Tests | pytest 9.2 |

Python **3.11+** (developed and tested on 3.13).

---

## Quick start

```bash
git clone https://github.com/devops941/yield-vision.git
cd yield-vision

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the whole pipeline on synthetic data, then start the app
python -m src.pipeline --source synthetic
uvicorn backend.main:app --reload --port 8000
```

Open <http://localhost:8000>. Interactive API docs are at `/docs`.

### MongoDB (optional)

Storage is mirrored to MongoDB Atlas. Put the connection string in a `.env`
file at the project root (it is git-ignored):

```bash
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
MONGO_DB=yieldvision
```

If `MONGO_URI` is missing or the cluster is unreachable the app still runs; the
SQLite cache handles everything. `GET /health` reports the MongoDB status. Tests
set `YIELDVISION_NO_MONGO=1` so they stay offline.

### Or load data from the browser

1. Open the **Dashboard** and press **Load sample dataset**; or use
   **Data entry** to type in one record; or go to **Upload & Validate** and drop
   your own CSV.
2. Review the validation issues, fix/ignore/exclude as needed, then **Commit**.
3. Walk through **Trends → Features → Train models**.
4. Activate a model, then use **Predict**, **Accuracy**, **Explainability** and
   **Reports & export**.
5. On **Image & video**, drop a field photo or clip to predict from imagery.
   On **Predict**, upload an Excel or CSV to see every row predicted in a table.

---

## Command line pipeline

```bash
# Synthetic dataset (default)
python -m src.pipeline --source synthetic

# Your own CSV
python -m src.pipeline --source file --path data/sample/crop.csv

# Choose models, schemes, tuning
python -m src.pipeline --models ridge,random_forest,hist_gradient_boosting \
    --schemes temporal,group --test-years 5 --tune
```

---

## Data schema

Canonical columns (aliases such as `rain`, `rainfall`, `precip` are mapped
automatically during ingest):

| Column | Meaning | Required |
|---|---|---|
| `state`, `district` | Administrative location | yes |
| `crop`, `season` | Crop and season | yes |
| `year` | Harvest year | yes |
| `yield_t_ha` | Yield in tonnes per hectare (target) | yes |
| `rainfall_mm` | Seasonal rainfall | recommended |
| `temp_mean_c`, `temp_max_c`, `temp_min_c` | Temperatures | recommended |
| `humidity_pct` | Relative humidity | optional |
| `soil_n_kgha`, `soil_p_kgha`, `soil_k_kgha`, `soil_ph`, `soil_oc_pct` | Soil properties | optional |
| `fertilizer_kgha`, `irrigated_frac`, `pesticide_kgha`, `area_ha` | Management | optional |

A ready-to-fill template is available at `/api/template`.

---

## Project layout

```
yield-vision/
├── backend/                # FastAPI application
│   ├── main.py             # app factory, static mounting, /docs
│   ├── schemas.py          # Pydantic request/response models
│   ├── deps.py             # shared dependencies (DB session)
│   └── routers/            # uploads, explorer, features, models,
│                           # predict, media, export, report
├── frontend/               # HTML + JS + Plotly.js (no Streamlit)
│   ├── index.html          # dashboard
│   ├── entry.html          # type in a record by hand
│   ├── upload.html         # upload, validate, review, commit
│   ├── media.html          # image and video prediction
│   ├── explorer.html       # trend explorer
│   ├── features.html       # feature engineering
│   ├── models.html         # training and registry
│   ├── accuracy.html       # accuracy dashboard
│   ├── explain.html        # explainability
│   ├── predict.html        # prediction, what-if, batch
│   ├── reports.html        # reports and export
│   ├── css/style.css       # 2026 dark-glass design system
│   └── js/app.js           # API client, shell, Plotly theming
├── src/
│   ├── config.py           # paths and settings
│   ├── pipeline.py         # end-to-end CLI
│   ├── data/               # synthetic, ingest, validate, preprocess, db
│   ├── features/           # engineer.py (sklearn pipeline)
│   ├── models/             # train, baselines, evaluate, explain,
│   │                       # intervals, registry, service
│   └── analysis/           # Mann-Kendall, Sen's slope, correlations
│   └── report/             # Jinja2 template + PDF generation
├── tests/                  # pytest suite (validation, leakage, models, API)
└── requirements.txt
```

---

## Testing

```bash
pytest                            # full suite
pytest --cov=src --cov=backend    # with coverage
```

The suite covers every validation rule, leakage guarantees (temporal split has no
overlap, GroupKFold keeps districts disjoint, imputers/scalers fitted on training
folds only), model skill on synthetic non-linear data, interval coverage,
extrapolation warnings, audit-trail writes and the full API surface.

---

## Notes

- `data/` holds the SQLite database, uploaded files, saved models and generated
  reports; it is created on first run.
- The synthetic generator injects the full set of error types so the validation
  workflow can be exercised end to end without real data.
- No model, scaler or imputer is ever fitted on the held-out test set.
