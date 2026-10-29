"""Data loading, validation, cleaning."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import FEATURE_RANGES, TARGET, ID_COL

MAX_ROWS = 200_000


def load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=MAX_ROWS)
    df.columns = [c.strip() for c in df.columns]
    return df


def available_features(df: pd.DataFrame) -> list[str]:
    return [f for f in FEATURE_RANGES if f in df.columns and df[f].notna().any()]


def validate_dataframe(df: pd.DataFrame) -> dict:
    """Return a report: errors (blocking) and warnings."""
    errors, warnings = [], []
    if ID_COL not in df.columns:
        errors.append(f"Missing required column '{ID_COL}'.")
    elif df[ID_COL].duplicated().any():
        warnings.append(f"{int(df[ID_COL].duplicated().sum())} duplicate road_id rows (first kept).")
    feats = available_features(df)
    if not feats:
        errors.append("No known feature columns found. Expected any of: " + ", ".join(FEATURE_RANGES))
    for f in feats:
        col = pd.to_numeric(df[f], errors="coerce")
        lo, hi = FEATURE_RANGES[f]
        bad = int(((col < lo) | (col > hi)).sum())
        miss = int(col.isna().sum())
        if bad:
            warnings.append(f"{f}: {bad} values outside [{lo}, {hi}] will be set to missing.")
        if miss:
            warnings.append(f"{f}: {miss} missing values will be imputed (median).")
    if TARGET in df.columns:
        vals = set(pd.to_numeric(df[TARGET], errors="coerce").dropna().unique())
        if not vals <= {0, 1}:
            errors.append(f"Target '{TARGET}' must be binary 0/1 for this pipeline; found {sorted(vals)[:6]}.")
    else:
        warnings.append(f"No '{TARGET}' column: prediction only, training is not possible.")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "n_rows": int(len(df)), "features": feats}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, coerce numerics, null out out-of-range values. Never mutates input."""
    out = df.copy()
    if ID_COL in out.columns:
        out = out.drop_duplicates(subset=ID_COL, keep="first")
    for f, (lo, hi) in FEATURE_RANGES.items():
        if f in out.columns:
            col = pd.to_numeric(out[f], errors="coerce")
            out[f] = col.where((col >= lo) & (col <= hi), np.nan)
    if TARGET in out.columns:
        out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    return out.reset_index(drop=True)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Simple derived features (computed only if inputs exist; no label-derived features)."""
    out = df.copy()
    if {"rainfall_1h", "rainfall_24h"} <= set(out.columns):
        out["rain_intensity_ratio"] = out["rainfall_1h"] / (out["rainfall_24h"] + 1.0)
    return out


DERIVED = ["rain_intensity_ratio"]


def model_columns(df: pd.DataFrame) -> list[str]:
    cols = available_features(df)
    if "rain_intensity_ratio" in df.columns:
        cols = cols + DERIVED
    return cols
