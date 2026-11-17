# AGENTS.md — YieldVision working notes

## What this is
Climate/soil-driven crop-yield prediction engine. Python owns all data work;
the frontend is plain HTML + vanilla JS + Plotly.js (no Streamlit, no build step).

## Commands
- Run the app: `python -m uvicorn backend.main:app --host 0.0.0.0 --port 12000`
- Full pipeline: `python -m src.pipeline --source synthetic`
- Tests: `python -m pytest -q`  (full suite takes ~11 min; training-heavy)
- Single fast test file: `python -m pytest tests/test_media.py -q`
- Lint: `python -m ruff check <files>` (no repo-wide ruff config; B008 is expected
  for FastAPI `Depends` defaults and is not an error here)

## Layout
- `src/data/` — ingest, validation, preprocessing, SQLite (`db.py`), MongoDB (`mongo.py`), media (`media.py`)
- `src/features/engineer.py` — all engineered features; fitted inside CV folds
- `src/models/` — training service, registry, prediction intervals
- `backend/routers/` — uploads, explorer, features, models, predict, media, export, report
- `frontend/` — one HTML page per section; shared `css/style.css` + `js/app.js`
- `tests/` — pytest suite

## Conventions
- Frontend pages share nav from `app.js`; use `fieldLabel(name)` for friendly
  column names and `modelLabel(name)` for model names instead of `fmt.title(name)`.
- Asset links carry a `?v=YYYY` query (`/static/js/app.js?v=2026`) so browsers
  pick up JS/CSS changes instead of serving a stale cached copy.
- UI is a light theme: white canvas, leaf-green accents, full-width (no max-width cap).
  CSS variables live in `:root` of `frontend/css/style.css`.
- MongoDB mirroring is best-effort: every `mongo.*` call must no-op when the
  cluster is unreachable, so SQLite always remains the source of truth.
- `MONGO_URI`/`MONGO_DB` come from `.env` (git-ignored). `.env.example` is the template.
- Tests set `YIELDVISION_NO_MONGO=1` so they never touch Atlas.
- `ensure_canonical_columns()` fills absent canonical columns so partial
  spreadsheets still predict (feature code indexes columns directly).

## Do not commit
Runtime artifacts are git-ignored and untracked: `models/*.joblib`, `data/*.db`,
`data/uploads/`, `data/media/`, `data/processed/`, `outputs/`. Only
`data/sample/template.csv` under `data/` is tracked.
