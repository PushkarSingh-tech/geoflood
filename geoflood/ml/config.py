"""Central configuration: paths, features, valid ranges, risk categories."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("GEOFLOOD_DATA_DIR", ROOT / "data" / "sample"))
MODEL_DIR = Path(os.getenv("GEOFLOOD_MODEL_DIR", ROOT / "models"))
REPORT_DIR = Path(os.getenv("GEOFLOOD_REPORT_DIR", ROOT / "reports"))
CSV_NAME = "roads.csv"
GEOJSON_NAME = "roads.geojson"

TARGET = "flooded"  # binary: 1 = waterlogging observed, 0 = not observed
ID_COL = "road_id"

# feature -> (min, max). Values outside the range are treated as invalid (-> missing).
# Remove a feature here if your dataset does not have it; the pipeline adapts.
FEATURE_RANGES = {
    "rainfall_1h": (0.0, 200.0),   # mm
    "rainfall_3h": (0.0, 400.0),   # mm
    "rainfall_24h": (0.0, 1000.0), # mm
    "elevation": (-500.0, 9000.0), # m
    "slope": (0.0, 90.0),          # degrees
    "drainage_density": (0.0, 50.0),  # km of drain per km2
    "impervious_pct": (0.0, 100.0),
    "historical_flood_count": (0.0, 1000.0),  # events BEFORE the label period only
}
RAINFALL_FEATURES = ["rainfall_1h", "rainfall_3h", "rainfall_24h"]

# (upper bound exclusive, name). Thresholds are fixed conventions on the model score,
# not calibrated flood-depth classes.
RISK_BINS = [(0.25, "Low"), (0.50, "Moderate"), (0.75, "High"), (1.01, "Very High")]
RISK_COLORS = {"Low": "#2e9d57", "Moderate": "#e5c100", "High": "#f08a24", "Very High": "#d1342f"}


def categorize(p: float) -> str:
    for upper, name in RISK_BINS:
        if p < upper:
            return name
    return "Very High"
