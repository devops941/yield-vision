# YieldVision: Climate and Soil-Driven Crop Yield Prediction Engine

**Type:** Final-year project | **Domain:** Agriculture & Food | **Level:** Intermediate
**Stack:** Python 3.11, Pandas, NumPy, scikit-learn, SciPy, statsmodels, FastAPI, SQLite, HTML/JS + Plotly.js. **No Streamlit.**

> **Instructions for Claude Code:** Read this whole document first. Build the project phase by phase (Section 11). After each phase, run the tests and confirm the acceptance criteria before moving on. Do not skip phases. Ask me only if something is truly blocking; otherwise use the defaults stated here. Do NOT use Streamlit anywhere; the web layer is FastAPI + static HTML/JavaScript frontend. Python owns all data processing, feature engineering, modelling and report logic; the frontend only displays results. Do NOT add computer vision or deep learning; they are not needed for this project.

---

## 1. Problem Statement

Crop yield depends on rainfall, temperature, soil nutrients and pH, irrigation, fertilizer use and crop type. Historical records sit in separate spreadsheets (crop statistics, weather, soil reports), so farm planners and agricultural researchers cannot easily see yield trends, test how conditions change yield, or compare predictive models fairly.

YieldVision is a Python application that lets a user **upload crop, weather and soil data, validates and cleans it, explores historical trends, engineers features, trains and compares prediction models (Linear/Ridge Regression vs Random Forest, optionally Gradient Boosting), predicts yield with an uncertainty range for new conditions, and exports results and a report.**

## 2. Objectives

1. Build a working web application: upload, validate, explore, engineer features, train, compare, predict, export.
2. Provide a reliable input process with **clear validation messages** and handling of missing, duplicate and invalid values.
3. Implement the logic as **reusable Python modules**: validation, feature engineering, baselines, regression models, cross-validation, evaluation, explainability, prediction intervals.
4. Compare models honestly with **MAE, RMSE, R^2, MAPE** using time-based and group-based cross-validation (no leakage).
5. Present trends, accuracy and explanations through charts, summary cards and plain-language text.
6. Let users review/correct records, run single and what-if predictions, batch-predict from a file, and export results and a PDF report.

## 3. Scope

**In scope:** district/region-level crop yield prediction for a few crops, seasonal climate and soil features, model comparison, prediction with intervals, what-if analysis, dashboard, report, tests.
**Out of scope:** satellite/image-based yield estimation, deep learning, crop simulation models (DSSAT, APSIM), farm-level IoT, multi-user authentication (future work).

**Default task:** predict **yield (tonnes per hectare)** for a given region, crop, season and year, using seasonal weather, soil and cultivation features.

## 4. Data Sources

| Data | Source | Format | Notes |
|---|---|---|---|
| Crop area, production, yield | Open Government Data (data.gov.in) "District-wise, season-wise crop production statistics"; Ministry of Agriculture & Farmers Welfare; ICRISAT district-level database; FAO (FAOSTAT) for country-level | CSV / Excel | Yield = production / area; choose 3-5 crops (e.g. rice, wheat, maize, cotton, groundnut) and 1-3 states |
| Rainfall, temperature | IMD gridded data (via `imdlib`), NASA POWER (free API), or Open-Meteo historical API | CSV / JSON | Aggregate to **seasonal** features (Kharif Jun-Oct, Rabi Nov-Mar, Summer Apr-May) |
| Soil properties | Soil Health Card portal data (N, P, K, pH, organic carbon), ISRIC SoilGrids (free API/raster), or district soil reports | CSV | Treated as mostly static per region |
| Cultivation variables | Fertilizer use, irrigated area fraction, pesticide use (from state/district statistics) | CSV | Optional columns |

**Fallback (mandatory):** Build `src/data/synthetic.py` generating **20 districts x 4 crops x 20 years x seasons** with a known data-generating process: yield rises with rainfall up to an optimum then drops (non-linear), declines when temperature exceeds a crop-specific threshold during critical months, increases with nitrogen with diminishing returns, peaks near an optimal soil pH, increases with irrigation fraction, has a slow upward technology trend over years, plus noise, and **deliberately injected errors** (duplicates, negative area, yield of 500 t/ha, text in numeric columns, missing seasons, area in acres vs hectares, blank cells). The whole project must run end to end on synthetic data so demo and tests never depend on downloads. Real data plugs in through the same schema.

### Standard input schema (CSV/Excel, one row per region-crop-season-year)

`state, district, crop, season, year, area_ha, production_t, yield_t_ha (optional, computed if missing), rainfall_mm, temp_mean_c, temp_max_c, temp_min_c, humidity_pct (optional), soil_n_kgha, soil_p_kgha, soil_k_kgha, soil_ph, soil_oc_pct, fertilizer_kgha (optional), irrigated_frac (optional), pesticide_kgha (optional)`

Mandatory: `district, crop, season, year` and either `yield_t_ha` or both `area_ha` and `production_t`, plus at least `rainfall_mm` and one temperature column. Weather and soil can alternatively be uploaded as separate files and are joined on `(district, season, year)` and `district`. The app provides a downloadable template and an alias table for column names.

## 5. System Flow

```
User uploads crop / weather / soil files (or loads sample synthetic data)
        |
        v
[1] Input management: file checks, alias mapping, merge of crop + weather + soil
        |
        v
[2] Validation: schema, ranges, duplicates, missingness, units -> issue report (row, column, problem, suggestion)
        |
        v
[3] User reviews issues: accept auto-fix, edit value, or exclude row (audit trail)
        |
        v
[4] Commit to SQLite (records, regions, crops)
        |
        v
[5] Historical Trend Explorer: yield over time, crop/region comparison, climate vs yield scatter, summary cards
        |
        v
[6] Feature engineering + preprocessing pipeline (scikit-learn Pipeline / ColumnTransformer)
        |
        v
[7] Model training: baselines -> Ridge/Linear -> Random Forest -> (Gradient Boosting) with tuned hyperparameters, time/group CV
        |
        v
[8] Evaluation + comparison: MAE, RMSE, R2, MAPE, skill vs baseline; errors by crop/region/year; explainability
        |
        v
[9] Model registry: save versions (joblib) with metrics, mark one as active
        |
        v
[10] Prediction module: single, what-if (sensitivity), batch file; prediction interval
        |
        v
[11] FastAPI REST layer  -->  [12] Web dashboard
        |
        v
[13] Export: CSV/Excel of results, predictions, PDF report
```

## 6. Methods (precise definitions)

### 6.1 Validation rules (each produces a message with row, column, value, reason, suggestion)

| Check | Rule | Severity |
|---|---|---|
| Missing mandatory column | `district, crop, season, year`, yield source, rainfall, temperature absent | Error (blocks upload) |
| Non-numeric value | "NA", text in numeric columns | Warning, converted to NaN |
| Invalid year | Outside 1950 to current year | Error (row) |
| Impossible values | negative area, production or rainfall; `temp_min > temp_max`; pH outside 3-10; humidity outside 0-100 | Error (row/cell) |
| Crop-specific yield range | Yield outside configured plausible range (e.g. rice 0.2-12 t/ha) | Warning, flagged as outlier |
| Yield consistency | If area, production and yield all present, `|yield - production/area| / yield > 5%` | Warning |
| Duplicates | Same `(district, crop, season, year)` | Warning; options: keep first, keep last, average |
| Missingness | Feature missing in more than 30% of rows | Warning; feature excluded or imputed (user chooses) |
| Unit check | area/production magnitudes suggesting acres or kg instead of ha or tonnes | Warning with suggested conversion |
| Unknown category | New crop or season label not in config | Warning (mapped or added) |

All issues are stored in `validation_issues`; the UI shows a filterable table and a summary ("1,240 rows: 8 errors, 61 warnings").

### 6.2 Feature engineering

- **Target:** `yield_t_ha` (primary). Optional experiment: `log1p(yield)` to handle skew.
- **Climate features (seasonal):** total rainfall, rainfall in critical months, number of rainy days (if daily data), mean/max/min temperature, **growing degree days** `GDD = sum(max(0, (Tmax+Tmin)/2 - Tbase))` with crop-specific base temperature in config, heat-stress days above a threshold, rainfall deficit relative to the district's long-term mean (anomaly), temperature anomaly.
- **Soil features:** N, P, K, pH, organic carbon, pH distance from crop optimum.
- **Cultivation features:** fertilizer per hectare, irrigated fraction, pesticide, area sown (log).
- **Time features:** `year` (captures technology trend) and optionally a rolling mean of the district's previous 3 years' yield (**shifted so only past years are used**).
- **Categorical features:** crop, season, state, district (one-hot or ordinal; high-cardinality district via one-hot with `handle_unknown="ignore"` or via lat/lon-free target encoding fitted **only inside training folds**).
- **Interactions (for the linear model):** rainfall x irrigation, temperature x crop.
- **Pipeline:** `ColumnTransformer` + `SimpleImputer` (median, add missing-indicator) + `StandardScaler` for linear models only; everything fitted inside the cross-validation folds via `sklearn.pipeline.Pipeline`.

### 6.3 Models

1. **Baseline A (mean by crop-season):** average yield for that crop and season in the training data.
2. **Baseline B (district history):** mean of the district-crop's previous 3 years (naive forecast).
3. **Linear Regression and Ridge** (standardized features; `statsmodels` OLS once for coefficient table, p-values, VIF).
4. **Random Forest Regressor:** tuned with `RandomizedSearchCV` over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`.
5. **Optional:** `HistGradientBoostingRegressor` for comparison.

### 6.4 Validation strategy (critical, must be implemented exactly)

Report **three evaluation schemes** and make clear which is the headline:
1. **Temporal split (headline):** train on years up to T, test on the last 3-4 years (simulates real forecasting). Inner tuning uses `TimeSeriesSplit` over years.
2. **Group K-fold by district (`GroupKFold`):** tests generalization to **unseen districts**.
3. **Random K-fold:** shown only for comparison and labelled as *optimistic* because rows from the same district and year leak information.

Metrics: **MAE, RMSE, R^2, MAPE** (guard zeros), **skill score vs best baseline** `1 - RMSE_model / RMSE_baseline`, and bias (mean error). Also report errors by crop, season, state and year, and the 10 worst predictions with likely reasons (drought year, flood, data errors).

### 6.5 Explainability

Impurity importance (Random Forest), **permutation importance** on held-out data (preferred), partial dependence and ICE plots for rainfall, temperature, nitrogen, pH; OLS coefficients for the linear model. Plain-language summary: "rainfall and nitrogen were the strongest drivers of rice yield in this dataset". State that this describes association in the data, not proof of causation.

### 6.6 Prediction with uncertainty

- **Prediction interval:** for Random Forest use the spread of per-tree predictions combined with out-of-fold residual quantiles; for linear models use residual quantiles from cross-validation. Report e.g. 80% interval by crop.
- **Extrapolation warning:** if any input lies outside the training range (min-max per feature) or a category is unseen, show a warning and widen/flag the interval.
- **What-if / sensitivity:** vary one input (e.g. rainfall -30% to +30%, temperature +0 to +3 C) holding others fixed and plot predicted yield change.
- **Batch prediction:** upload a CSV of conditions, validate with the same rules, return predictions with intervals as a downloadable file.

### 6.7 Trend explorer analytics

Yield time series per crop/region, Mann-Kendall trend and Sen's slope of yield (t/ha per year) per district-crop (`pymannkendall`), coefficient of variation (yield stability), climate-yield correlation (Spearman) with significance, and top/bottom regions.

## 7. Technology Stack

| Layer | Choice | Reason |
|---|---|---|
| Language | Python 3.11 | Core data, feature and ML logic |
| Data | pandas, numpy | Tabular data, merges, grouping |
| ML | scikit-learn | Pipelines, Ridge, Random Forest, GroupKFold, TimeSeriesSplit, metrics, permutation importance |
| Statistics | SciPy, statsmodels, pymannkendall | OLS summary, correlation, trend tests |
| File handling | openpyxl, python-multipart | Excel read/write, uploads |
| Backend | **FastAPI** + Uvicorn, Pydantic v2 | REST API, validated schemas, auto docs at `/docs` |
| Frontend | **HTML + CSS + JavaScript (no framework), Plotly.js** | Interactive charts without Streamlit |
| Database | SQLite via SQLAlchemy 2.0 | Zero-setup single-user storage; PostgreSQL-ready |
| Model persistence | joblib | Versioned model files |
| Reports | Jinja2 + matplotlib figures + weasyprint (PDF) | Downloadable report |
| Background jobs | FastAPI `BackgroundTasks` | Model training without blocking the UI |
| Config | PyYAML | Crop settings, ranges, aliases, hyperparameter grids |
| Testing | pytest, httpx TestClient | Unit and API tests |
| Quality / tooling | ruff, black, Git, pre-commit | Code quality |
| Deployment | Docker (optional), Uvicorn, GitHub | Reproducible demo |

Use the smallest library set that works; no deep learning, no OpenCV, no extra databases.

## 8. Database Design (SQLite)

```sql
CREATE TABLE regions (
  region_id INTEGER PRIMARY KEY AUTOINCREMENT,
  state TEXT, district TEXT,
  lat REAL, lon REAL,
  UNIQUE(state, district)
);

CREATE TABLE crops (
  crop_id INTEGER PRIMARY KEY AUTOINCREMENT,
  crop_name TEXT UNIQUE NOT NULL,
  base_temp_c REAL,              -- for GDD
  optimal_ph REAL,
  yield_min REAL, yield_max REAL -- plausible range for validation
);

CREATE TABLE uploads (
  upload_id INTEGER PRIMARY KEY AUTOINCREMENT,
  filename TEXT, uploaded_at DATETIME,
  status TEXT,                   -- 'validated','committed','rejected'
  n_rows INTEGER, n_errors INTEGER, n_warnings INTEGER
);

CREATE TABLE records (
  record_id INTEGER PRIMARY KEY AUTOINCREMENT,
  upload_id INTEGER REFERENCES uploads(upload_id),
  region_id INTEGER REFERENCES regions(region_id),
  crop_id INTEGER REFERENCES crops(crop_id),
  season TEXT, year INTEGER,
  area_ha REAL, production_t REAL, yield_t_ha REAL,
  rainfall_mm REAL, temp_mean_c REAL, temp_max_c REAL, temp_min_c REAL, humidity_pct REAL,
  soil_n REAL, soil_p REAL, soil_k REAL, soil_ph REAL, soil_oc REAL,
  fertilizer_kgha REAL, irrigated_frac REAL, pesticide_kgha REAL,
  is_excluded INTEGER DEFAULT 0,
  is_outlier INTEGER DEFAULT 0,
  UNIQUE(region_id, crop_id, season, year)
);

CREATE TABLE validation_issues (
  issue_id INTEGER PRIMARY KEY AUTOINCREMENT,
  upload_id INTEGER REFERENCES uploads(upload_id),
  row_number INTEGER, column_name TEXT, value TEXT,
  severity TEXT,                 -- 'error' | 'warning'
  rule TEXT, message TEXT, suggestion TEXT,
  status TEXT DEFAULT 'open'     -- 'open' | 'fixed' | 'ignored' | 'excluded'
);

CREATE TABLE corrections_audit (
  audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
  record_id INTEGER, column_name TEXT,
  old_value TEXT, new_value TEXT,
  reason TEXT, changed_at DATETIME
);

CREATE TABLE feature_sets (
  feature_set_id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at DATETIME,
  config_json TEXT,              -- features chosen, imputation, encoding
  n_rows INTEGER, n_features INTEGER,
  data_path TEXT                 -- parquet/csv snapshot of engineered data
);

CREATE TABLE model_registry (
  model_id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_name TEXT,               -- 'baseline_mean','baseline_history','ridge','linear','random_forest','hist_gb'
  feature_set_id INTEGER REFERENCES feature_sets(feature_set_id),
  crop_scope TEXT,               -- 'all' or specific crop
  params_json TEXT,
  trained_at DATETIME,
  model_path TEXT,
  is_active INTEGER DEFAULT 0,
  status TEXT                    -- 'queued','training','done','failed'
);

CREATE TABLE model_metrics (
  model_id INTEGER REFERENCES model_registry(model_id),
  scheme TEXT,                   -- 'temporal','group_kfold','random_kfold'
  scope TEXT,                    -- 'overall','crop:<name>','season:<name>','state:<name>','year:<y>'
  mae REAL, rmse REAL, r2 REAL, mape REAL, bias REAL, skill_vs_baseline REAL,
  PRIMARY KEY (model_id, scheme, scope)
);

CREATE TABLE eval_predictions (
  model_id INTEGER REFERENCES model_registry(model_id),
  record_id INTEGER REFERENCES records(record_id),
  scheme TEXT,
  y_true REAL, y_pred REAL, pred_low REAL, pred_high REAL,
  PRIMARY KEY (model_id, record_id, scheme)
);

CREATE TABLE feature_importance (
  model_id INTEGER REFERENCES model_registry(model_id),
  feature TEXT,
  impurity_importance REAL,
  permutation_importance REAL, permutation_std REAL,
  PRIMARY KEY (model_id, feature)
);

CREATE TABLE user_predictions (
  prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id INTEGER REFERENCES model_registry(model_id),
  created_at DATETIME,
  inputs_json TEXT,
  y_pred REAL, pred_low REAL, pred_high REAL,
  warnings TEXT                  -- e.g. 'rainfall outside training range'
);

CREATE TABLE trend_results (
  region_id INTEGER, crop_id INTEGER,
  trend TEXT, p_value REAL, tau REAL,
  sen_slope_t_ha_per_year REAL, cv_yield REAL,
  PRIMARY KEY (region_id, crop_id)
);

CREATE TABLE reports (
  report_id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at DATETIME, file_path TEXT, scope_json TEXT
);
```

Indexes: `records(region_id, crop_id, year)`, `model_metrics(model_id)`, `eval_predictions(model_id, scheme)`.

## 9. Backend API and Frontend

### REST endpoints (FastAPI, JSON, Pydantic-validated)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/template` | Download CSV template |
| POST | `/api/uploads` | Upload crop/weather/soil file(s); returns upload id and validation summary |
| GET | `/api/uploads/{id}/issues?severity=&status=` | List validation issues |
| PATCH | `/api/uploads/{id}/issues/{issue_id}` | Fix value / ignore / exclude (writes audit row) |
| POST | `/api/uploads/{id}/commit` | Pre-process and commit to database |
| POST | `/api/datasets/sample` | Load built-in synthetic dataset |
| GET | `/api/explorer/summary` | Summary cards (rows, crops, regions, year range, mean yield) |
| GET | `/api/explorer/trend?crop=&district=&start=&end=` | Yield time series and trend result |
| GET | `/api/explorer/climate-yield?crop=&variable=` | Scatter data and correlation |
| POST | `/api/features/build` | Build engineered feature set with config |
| GET | `/api/features/{id}/preview` | Preview rows and feature stats |
| POST | `/api/models/train` | Train selected models (background task); returns model ids |
| GET | `/api/models` | Model registry with status |
| GET | `/api/models/compare?scheme=` | Side-by-side metrics |
| GET | `/api/models/{id}/metrics?scheme=&scope=` | Detailed metrics |
| GET | `/api/models/{id}/predictions?scheme=` | Actual vs predicted |
| GET | `/api/models/{id}/importance` | Feature importance |
| GET | `/api/models/{id}/pdp?feature=` | Partial dependence data |
| POST | `/api/models/{id}/activate` | Set active model |
| POST | `/api/predict` | Single prediction with interval and warnings |
| POST | `/api/predict/whatif` | Sensitivity curve for one varied input |
| POST | `/api/predict/batch` | Upload file, get predictions file |
| GET | `/api/export/data.csv`, `/api/export/results.xlsx` | Cleaned data and results |
| POST | `/api/report`, GET `/api/report/{id}` | Generate and download PDF report |
| GET | `/health` | Health check |

### Frontend pages (static HTML + JS, served from `/`)

1. **Upload and Validate:** drag-drop upload, template download, summary cards, issues table with inline fix/ignore/exclude, "Commit data".
2. **Trend Explorer:** filters (crop, state, district, years); yield time series with trend line, region comparison bars, climate-vs-yield scatter, summary cards, table of underlying records with CSV export.
3. **Features:** choose feature options (GDD, anomalies, interactions, imputation), preview engineered table and distributions.
4. **Models:** pick models, start training, progress/status, registry table with activate button.
5. **Accuracy Dashboard:** metric comparison table (switch between temporal / group / random schemes), actual-vs-predicted plot, residual plot, error by crop/season/year, worst predictions list.
6. **Explainability:** permutation importance bars, partial dependence curves, plain-language summary.
7. **Predict:** form for single prediction with interval and warnings, what-if sensitivity chart, batch upload and download.
8. **Reports and Export:** generate PDF report, download CSV/Excel, history.

## 10. Project Structure

```
yieldvision/
|-- PROJECT_SPEC_YIELDVISION.md
|-- README.md
|-- requirements.txt
|-- config/
|   |-- crops.yaml            # base temps, optimal pH, yield ranges, seasons
|   |-- aliases.yaml          # column aliases
|   `-- models.yaml           # hyperparameter grids, CV settings
|-- data/
|   |-- sample/               # synthetic + template files
|   |-- uploads/
|   |-- processed/
|   `-- yieldvision.db
|-- models/                   # joblib files (versioned)
|-- src/
|   |-- data/
|   |   |-- synthetic.py
|   |   |-- ingest.py         # reading, alias mapping, merging crop/weather/soil
|   |   |-- validate.py       # rules in 6.1
|   |   |-- preprocess.py     # units, duplicates, imputation flags
|   |   `-- db.py             # SQLAlchemy models + repositories
|   |-- features/
|   |   `-- engineer.py       # GDD, anomalies, lags, interactions, pipeline builder
|   |-- models/
|   |   |-- baselines.py
|   |   |-- train.py          # Ridge, RF, HistGB, tuning
|   |   |-- validation.py     # temporal, GroupKFold, random CV
|   |   |-- evaluate.py       # metrics, by-scope breakdowns
|   |   |-- explain.py        # importance, PDP
|   |   |-- intervals.py      # prediction intervals, extrapolation checks
|   |   `-- registry.py       # save/load/activate model versions
|   |-- analysis/
|   |   `-- trends.py         # Mann-Kendall, Sen slope, CV, correlations
|   |-- report/
|   |   |-- generate.py
|   |   `-- template.html.j2
|   `-- pipeline.py           # CLI: ingest -> features -> train -> evaluate
|-- backend/
|   |-- main.py
|   |-- routers/              # uploads, explorer, features, models, predict, export
|   |-- schemas.py
|   `-- deps.py
|-- frontend/
|   |-- index.html, upload.html, explorer.html, features.html, models.html, accuracy.html, explain.html, predict.html, reports.html
|   |-- css/style.css
|   `-- js/
|-- tests/
|-- notebooks/                # EDA
`-- outputs/
```

Commands: `python -m src.pipeline --source synthetic|file --path data.csv`, then `uvicorn backend.main:app --reload` and open `http://localhost:8000`.

## 11. Development Plan (12 weeks)

| Phase | Weeks | Deliverables | Acceptance criteria |
|---|---|---|---|
| 1. Setup and study | 1 | Repo, venv, requirements, config YAMLs, notes on yield drivers, GDD, CV strategies | `pip install -r requirements.txt` works, `pytest` runs |
| 2. Data layer and synthetic data | 2 | `synthetic.py` with injected errors, `ingest.py`, `db.py`, schema | Synthetic dataset loads; alias mapping works; tables created |
| 3. Validation and preprocessing | 3-4 | `validate.py`, `preprocess.py` | **Every injected error type detected** with correct row/column/message; duplicates and unit issues handled; tests pass |
| 4. Trend explorer analytics | 5 | `trends.py` plus explorer queries | Mann-Kendall detects the synthetic technology trend; Sen slope within 20% of truth |
| 5. Feature engineering | 6 | `engineer.py` and sklearn pipeline | **Leakage test passes** (rolling history uses only past years); GDD matches hand calculation; no unfitted transforms outside folds |
| 6. Baselines and linear models | 7 | `baselines.py`, Ridge/OLS | Baselines and Ridge evaluated under all three schemes; temporal scheme is headline |
| 7. Random Forest, tuning, evaluation | 8-9 | `train.py`, `validation.py`, `evaluate.py`, `explain.py`, `intervals.py` | RF beats baselines on synthetic non-linear data; metrics by scope stored; permutation importance ranks rainfall and nitrogen high; interval coverage near nominal (e.g. 80%) on test set |
| 8. Registry and backend API | 10 | `registry.py`, FastAPI routers, background training | API tests pass; `/docs` lists all endpoints; training runs as background job; prediction returns interval and warnings |
| 9. Frontend | 11 | HTML pages with Plotly.js | User can upload, fix, commit, explore, train, compare, predict, what-if, export entirely in the browser; no console errors |
| 10. Report, tests, real data | 12 | PDF report, README, coverage over 70%, real-data run, demo script | Pipeline and app work unchanged on real CSVs; report downloads correctly |

## 12. Testing Plan

- **Validation tests:** one test per rule in 6.1; the synthetic error file produces exactly the expected issue counts.
- **Feature tests:** GDD on a hand-made temperature series; anomaly features equal manual values; district-history feature never uses the current or future year.
- **Leakage tests:** temporal split has no overlap and test years are strictly later; group split puts no district in both train and test; imputers and scalers are fitted only on training folds (assert via pipeline inspection).
- **Model tests:** on synthetic data with a known non-linear response, RF has lower temporal-CV RMSE than Linear Regression; saved model reloads and reproduces predictions exactly.
- **Metric tests:** MAE/RMSE/R^2/MAPE on tiny arrays with hand-calculated answers.
- **Interval tests:** empirical coverage of the 80% interval on held-out data is between 70% and 90%.
- **Extrapolation test:** an input far outside training range returns a warning.
- **API tests:** upload, issues, commit, train, compare, predict, what-if, batch, export endpoints return correct statuses and schemas; invalid requests return 422 with clear details.
- **Audit test:** every correction creates a `corrections_audit` row.
- **Smoke test:** full pipeline on synthetic data; app starts and serves the home page.

## 13. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Data leakage inflating accuracy (same district and year in train and test) | Temporal and GroupKFold as headline; random K-fold labelled optimistic; leakage tests |
| Small dataset for some crops/regions | Show sample counts per scope; avoid reporting metrics for scopes with fewer than a minimum number of rows; simpler models/regularization |
| Technology and policy trends confound climate effect | Include year trend feature, use anomalies, discuss in report |
| Random Forest cannot extrapolate beyond training range | Compare with linear model, extrapolation warnings, state limitation |
| Weather and soil granularity mismatch (district yield vs grid weather) | Aggregate grid cells to district mean; document method |
| Missing soil data | Median imputation with missing indicator; test sensitivity with and without soil features |
| Inconsistent units/columns in real files | Alias table, unit checks, template file |
| Overstating causality | Phrase as "associated with"; use "predicted" not "will be"; show intervals |
| Training blocks the UI | Background tasks with status polling |

## 14. Final Deliverables Checklist

- [ ] Working code in repo with README and setup steps
- [ ] SQLite database with data, registry, metrics, predictions and audit trail
- [ ] Validation module with clear messages and correction workflow
- [ ] Feature engineering, baselines, Ridge/Linear, Random Forest (optional HistGB), three evaluation schemes
- [ ] Prediction with intervals, what-if analysis, batch prediction
- [ ] FastAPI backend with Swagger docs
- [ ] Web dashboard (HTML + Plotly.js), no Streamlit
- [ ] CSV/Excel export and PDF report
- [ ] Final project report (Introduction, Literature Review, Data and Methodology, Implementation, Results and Discussion, Conclusion and Future Work, References)
- [ ] Presentation slides and demo script
- [ ] Future work: satellite NDVI features, crop-specific deep models, authentication and roles, PostgreSQL, farmer-facing mobile view, climate-change scenario simulation, multi-year forecasting

## 15. Claude Code Starter Prompts

Use in order:

1. `Read PROJECT_SPEC_YIELDVISION.md. Set up the repo structure from Section 10, requirements.txt, the three YAML config files and a README. Do Phase 1 only. Do not use Streamlit, deep learning or OpenCV.`
2. `Implement Phase 2: synthetic data generator with the known data-generating process and injected errors from Section 4, ingestion with alias mapping and merging, and the SQLAlchemy schema from Section 8, with pytest tests.`
3. `Implement Phase 3: validate.py and preprocess.py following the rule table in Section 6.1. Tests must confirm every injected error is caught with the right row, column and message.`
4. `Implement Phase 4: trend analytics (Mann-Kendall, Sen slope, CV, climate-yield correlations) per 6.7.`
5. `Implement Phase 5: feature engineering per 6.2 as a scikit-learn pipeline, with the leakage tests from Section 12.`
6. `Implement Phase 6: baselines, Linear/Ridge and the OLS coefficient table, evaluated under temporal, GroupKFold and random schemes per 6.4.`
7. `Implement Phase 7: Random Forest with RandomizedSearchCV, optional HistGradientBoosting, full evaluation by scope, permutation importance, partial dependence, and prediction intervals with extrapolation warnings per 6.5 and 6.6.`
8. `Implement Phase 8: model registry with joblib versions and the FastAPI backend with all endpoints in Section 9, background training, Pydantic schemas, audit-trail corrections and API tests.`
9. `Implement Phase 9: static HTML + JavaScript frontend pages with Plotly.js for every page in Section 9. No Streamlit.`
10. `Generate the PDF report, finish README, raise test coverage to 70%+, add an optional Dockerfile.`
11. `Add support for real data (data.gov.in crop production, IMD/NASA POWER weather, Soil Health Card or SoilGrids soil): document column mapping, extend aliases, and test with a sample real-format CSV.`

## 16. References (starting points)

- Breiman, L. (2001): Random Forests
- Hastie, Tibshirani, Friedman: The Elements of Statistical Learning (regularization, cross-validation)
- Roberts et al. (2017): Cross-validation strategies for data with temporal, spatial, hierarchical structure
- Lobell & Burke (2010): on the use of statistical models to predict crop yield responses to climate change
- Ritchie & Nanney: growing degree day concept; FAO crop water requirement guidelines
- Documentation: data.gov.in, ICRISAT district database, IMD, NASA POWER, Open-Meteo, ISRIC SoilGrids, scikit-learn (GroupKFold, TimeSeriesSplit, permutation_importance), FastAPI, Pydantic, Plotly.js
