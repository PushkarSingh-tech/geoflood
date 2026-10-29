# GeoFlood AI: Urban Road Waterlogging Risk (simple capstone version)

FastAPI + vanilla JS dashboard (Leaflet map, Chart.js charts) with a scikit-learn pipeline.
**Educational project. Not an official flood warning system.**

The bundled data is a **SYNTHETIC demo** (fictional street grid, generated labels). Its predictions are
for UI testing only and say nothing about real flood risk. The app shows a banner whenever demo data is loaded.

## Run

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # optional: pip install catboost
python scripts/make_demo_data.py                      # (re)creates synthetic demo data
python -m ml.train                                    # trains, writes models/ and reports/
uvicorn app.main:app --reload                         # open http://127.0.0.1:8000
python -m pytest -q                                   # tests
```

Leaflet and Chart.js load from CDNs (unpkg / jsDelivr). For fully offline use, download them into
`app/static/vendor/` and change the `<script>`/`<link>` URLs in `index.html`. The map has no basemap by
default (works offline); tick "Basemap tiles" to add OpenStreetMap tiles (attribution is included).

## Dataset contract (`data/sample/roads.csv` + `roads.geojson`)

| column | required | notes |
|---|---|---|
| `road_id` | yes | unique; must match GeoJSON `properties.road_id` |
| `lat`, `lon` | recommended | segment midpoint; enables the geographic train/test split |
| `rainfall_1h/3h/24h` (mm), `elevation` (m), `slope` (deg), `drainage_density`, `impervious_pct` (0-100), `historical_flood_count` | any subset | missing columns are simply not used; out-of-range values become missing and are median-imputed |
| `flooded` | for training | binary 0/1: waterlogging observed in the label period |
| `data_source` | recommended | free text; values starting with `SYNTHETIC` switch on demo mode |

Leakage rule: `historical_flood_count` must only count events **before** the period of the `flooded` label.
Edit `FEATURE_RANGES` in `ml/config.py` to add/remove features. Roads missing from the GeoJSON are not drawn
(geometry is never invented).

## Using real data

1. Road geometry: export from OpenStreetMap (e.g. Overpass/OSMnx) as GeoJSON LineStrings (ODbL, credit "© OpenStreetMap contributors").
2. Rainfall: gauge or reanalysis data for the label event(s) (check licence, e.g. IMD, GPM IMERG, ERA5).
3. Elevation/slope: SRTM or Copernicus DEM sampled per segment. Drainage and impervious surface: municipal drainage maps, land-cover products.
4. Labels: municipal complaint logs, waterlogging reports or sensor data, mapped to segments. Prefer labels spanning several events and years.
5. Save as `roads.csv` / `roads.geojson` in a folder, run `python scripts/validate_data.py yourfile.csv`, then
   `GEOFLOOD_DATA_DIR=path/to/folder python -m ml.train` and start the server with the same variable (see `.env.example`;
   the app reads real environment variables, it does not auto-load `.env`).

There is deliberately no upload endpoint: files are imported by placing them in the data folder.

## Method and honesty notes

- Target: binary waterlogging occurrence. The score estimates that label's outcome, not flood depth or severity.
- Models compared: majority-prior Dummy, Logistic Regression, and CatBoost (if installed) or HistGradientBoosting. The gradient-boosting model is the fixed primary model; it is **not** chosen by test score.
- Validation: geographic hold-out (northernmost 30% by latitude) when `lat` exists, otherwise random stratified (reported as optimistic). No time-based split is possible with a single event. No hyperparameter tuning or separate validation set.
- Class weights are used only when the training positive rate is below 20%.
- Metrics at threshold 0.5: precision, recall, F1, PR-AUC, ROC-AUC, Brier, confusion matrix, 5-bin calibration, permutation importance (`reports/`). Deployed model is refit on all labelled rows, so dashboard scores on labelled roads are in-sample.
- Risk categories are fixed score bins (<0.25 Low, <0.5 Moderate, <0.75 High, else Very High), not calibrated depth classes.
- Scenario endpoint replaces rainfall on all roads at once; it is a hypothetical, never a forecast, and may be physically unrealistic.

## API

`GET /api/health` · `GET /api/dashboard` · `GET /api/roads` (filters: `category`, `min_probability`, `q`) ·
`GET /api/roads/{road_id}` · `POST /api/predict` · `POST /api/scenario` · `GET /api/model-info`. Interactive docs at `/docs`.
