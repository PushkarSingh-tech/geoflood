"""GeoFlood AI - FastAPI backend. Run: uvicorn app.main:app --reload"""
from __future__ import annotations
import json, os, time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from ml import preprocess as pp
from ml.config import DATA_DIR, CSV_NAME, GEOJSON_NAME, MODEL_DIR, FEATURE_RANGES, RISK_COLORS, categorize
from ml.predict import Predictor, read_report

STATIC = Path(__file__).parent / "static"
CATEGORIES = ["Low", "Moderate", "High", "Very High"]
state: dict = {"df": None, "geo": {}, "predictor": None, "scores": None, "load_errors": []}


def load_state():
    state.update(df=None, geo={}, predictor=None, scores=None, load_errors=[])
    csv, gj = DATA_DIR / CSV_NAME, DATA_DIR / GEOJSON_NAME
    if csv.exists():
        df = pp.clean(pp.load_csv(csv))
        state["df"] = df.set_index("road_id", drop=False).rename_axis(None)
        state["csv_mtime"] = time.strftime("%Y-%m-%d %H:%M", time.gmtime(csv.stat().st_mtime))
    else:
        state["load_errors"].append(f"Dataset not found: {csv.name}. See README 'Data import'.")
    if gj.exists():
        try:
            fc = json.loads(gj.read_text())
            state["geo"] = {f["properties"]["road_id"]: f["geometry"] for f in fc["features"]}
        except Exception as e:  # malformed file must not crash startup
            state["load_errors"].append(f"Could not read {gj.name}: {e}")
    else:
        state["load_errors"].append(f"Road geometry not found: {gj.name}. Map will be empty.")
    try:
        state["predictor"] = Predictor(MODEL_DIR)
    except Exception:
        state["load_errors"].append("No trained model found. Run: python -m ml.train")
    if state["predictor"] is not None and state["df"] is not None:
        feats = state["predictor"].features
        state["scores"] = pd.Series(state["predictor"].predict_df(state["df"]), index=state["df"].index)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_state()
    yield


app = FastAPI(title="GeoFlood AI", version="1.0", lifespan=lifespan,
              description="Educational road waterlogging risk model. Not an official flood warning system.")


def _mode() -> str:
    p = state["predictor"]
    return p.meta["data_mode"] if p else "NO_MODEL"


def _need_model():
    if state["predictor"] is None:
        raise HTTPException(503, "No trained model available. Run `python -m ml.train` and restart.")


def _row_feats(row) -> dict:
    return {k: (None if pd.isna(row[k]) else round(float(row[k]), 3))
            for k in FEATURE_RANGES if k in row.index}


def _road_record(rid: str) -> dict:
    row = state["df"].loc[rid]
    rec = {"road_id": rid, "features": _row_feats(row),
           "observed_label": None if "flooded" not in row.index or pd.isna(row["flooded"]) else int(row["flooded"]),
           "data_source": str(row.get("data_source", "unknown"))}
    if state["scores"] is not None:
        p = float(state["scores"][rid])
        rec.update(probability=round(p, 4), category=categorize(p), color=RISK_COLORS[categorize(p)])
    return rec


# ---------- schemas ----------
class PredictRequest(BaseModel):
    rainfall_1h: Optional[float] = Field(None, ge=0, le=200)
    rainfall_3h: Optional[float] = Field(None, ge=0, le=400)
    rainfall_24h: Optional[float] = Field(None, ge=0, le=1000)
    elevation: Optional[float] = Field(None, ge=-500, le=9000)
    slope: Optional[float] = Field(None, ge=0, le=90)
    drainage_density: Optional[float] = Field(None, ge=0, le=50)
    impervious_pct: Optional[float] = Field(None, ge=0, le=100)
    historical_flood_count: Optional[float] = Field(None, ge=0, le=1000)

    @model_validator(mode="after")
    def rainfall_consistent(self):
        a, b, c = self.rainfall_1h, self.rainfall_3h, self.rainfall_24h
        if a is not None and b is not None and a > b:
            raise ValueError("rainfall_1h cannot exceed rainfall_3h.")
        if b is not None and c is not None and b > c:
            raise ValueError("rainfall_3h cannot exceed rainfall_24h.")
        if a is not None and c is not None and a > c:
            raise ValueError("rainfall_1h cannot exceed rainfall_24h.")
        return self


class ScenarioRequest(BaseModel):
    rainfall_1h: float = Field(..., ge=0, le=200)
    rainfall_3h: float = Field(..., ge=0, le=400)
    rainfall_24h: float = Field(..., ge=0, le=1000)

    @model_validator(mode="after")
    def consistent(self):
        if not (self.rainfall_1h <= self.rainfall_3h <= self.rainfall_24h):
            raise ValueError("Rainfall must satisfy 1h <= 3h <= 24h.")
        return self


# ---------- endpoints ----------
@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": state["predictor"] is not None,
            "dataset_loaded": state["df"] is not None, "mode": _mode(), "issues": state["load_errors"]}


@app.get("/api/dashboard")
def dashboard():
    if state["df"] is None:
        raise HTTPException(503, "Dataset not loaded. " + " ".join(state["load_errors"]))
    n = len(state["df"])
    out = {"mode": _mode(), "total_roads": n, "dataset_updated": state.get("csv_mtime"),
           "model_trained_at": state["predictor"].meta["trained_at"] if state["predictor"] else None,
           "observed_flood_rate": float(state["df"]["flooded"].mean()) if "flooded" in state["df"] else None,
           "counts": None, "high_risk": []}
    if state["scores"] is not None:
        cats = state["scores"].map(categorize)
        out["counts"] = [{"category": c, "count": int((cats == c).sum()),
                          "percent": round(100 * float((cats == c).sum()) / n, 1), "color": RISK_COLORS[c]}
                         for c in CATEGORIES]
        top = state["scores"][cats.isin(["High", "Very High"])].sort_values(ascending=False).head(15)
        out["high_risk"] = [{"road_id": r, "probability": round(float(p), 3), "category": categorize(p)}
                            for r, p in top.items()]
    return out


@app.get("/api/roads")
def roads(category: Optional[str] = Query(None, description="Comma-separated categories"),
          min_probability: float = Query(0.0, ge=0, le=1),
          q: Optional[str] = Query(None, max_length=50, description="road_id substring")):
    """GeoJSON FeatureCollection. Roads without geometry are omitted (never invented)."""
    if state["df"] is None:
        raise HTTPException(503, "Dataset not loaded.")
    wanted = {c.strip() for c in category.split(",")} if category else None
    if wanted and not wanted <= set(CATEGORIES):
        raise HTTPException(422, f"category must be among {CATEGORIES}")
    feats = []
    for rid in state["df"].index:
        if q and q.lower() not in rid.lower():
            continue
        geom = state["geo"].get(rid)
        if geom is None:
            continue
        rec = _road_record(rid)
        if state["scores"] is not None:
            if rec["probability"] < min_probability or (wanted and rec["category"] not in wanted):
                continue
        feats.append({"type": "Feature", "geometry": geom, "properties": rec})
    return {"type": "FeatureCollection", "mode": _mode(), "features": feats}


@app.get("/api/roads/{road_id}")
def road(road_id: str):
    if state["df"] is None or road_id not in state["df"].index:
        raise HTTPException(404, f"Road '{road_id}' not found.")
    return _road_record(road_id)


@app.post("/api/predict")
def predict(req: PredictRequest):
    _need_model()
    vals = req.model_dump()
    if all(v is None for v in vals.values()):
        raise HTTPException(422, "Provide at least one feature value.")
    res = state["predictor"].predict_one(vals)
    res.update(kind="single-input prediction", mode=_mode(),
               note="Missing inputs were filled with training medians. This is a model score, not a flood warning.")
    return res


@app.post("/api/scenario")
def scenario(req: ScenarioRequest):
    """Re-score all roads with rainfall replaced. Hypothetical what-if; never stored, never a forecast."""
    _need_model()
    if state["df"] is None:
        raise HTTPException(503, "Dataset not loaded.")
    df = state["df"].copy()
    df["rainfall_1h"], df["rainfall_3h"], df["rainfall_24h"] = req.rainfall_1h, req.rainfall_3h, req.rainfall_24h
    sc = pd.Series(state["predictor"].predict_df(df), index=df.index)
    cats = sc.map(categorize)
    base_cats = state["scores"].map(categorize)
    return {"kind": "SCENARIO (hypothetical, not a forecast)", "mode": _mode(),
            "rainfall": req.model_dump(),
            "counts": [{"category": c, "count": int((cats == c).sum()), "baseline_count": int((base_cats == c).sum())}
                       for c in CATEGORIES],
            "roads": {r: {"probability": round(float(p), 4), "category": categorize(p)} for r, p in sc.items()},
            "warning": ("Rainfall is changed for every road at once while terrain/drainage stay fixed; "
                        "this may not be physically realistic. Results are model outputs only.")}


@app.get("/api/model-info")
def model_info():
    _need_model()
    return {"metadata": state["predictor"].meta, "metrics": read_report("metrics.json"),
            "feature_importance": read_report("feature_importance.json"),
            "disclaimer": "Educational model. Not an official flood warning system."}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
