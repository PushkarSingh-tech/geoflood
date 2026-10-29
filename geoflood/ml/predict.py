"""Model loading and prediction."""
from __future__ import annotations
import json
import joblib, numpy as np, pandas as pd
from .config import MODEL_DIR, REPORT_DIR, categorize
from . import preprocess as pp


class Predictor:
    def __init__(self, model_dir=MODEL_DIR):
        self.model = joblib.load(model_dir / "model.joblib")
        self.meta = json.loads((model_dir / "metadata.json").read_text())
        self.features = self.meta["features"]

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        d = pp.engineer(pp.clean(df))
        for c in self.features:
            if c not in d.columns:
                d[c] = np.nan  # imputed with training median
        return self.model.predict_proba(d[self.features])[:, 1]

    def predict_one(self, values: dict) -> dict:
        p = float(self.predict_df(pd.DataFrame([values]))[0])
        missing = [f for f in self.features if values.get(f) is None and f != "rain_intensity_ratio"]
        return {"probability": round(p, 4), "category": categorize(p), "imputed_features": missing}


def read_report(name: str):
    f = REPORT_DIR / name
    return json.loads(f.read_text()) if f.exists() else None
