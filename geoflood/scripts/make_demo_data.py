"""Create a SYNTHETIC demo dataset (fictional street grid, NOT real roads). For UI testing only."""
import json
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(7)
N = 16          # nodes per side
STEP = 0.002    # degrees between nodes
out = Path(__file__).resolve().parent.parent / "data" / "sample"
out.mkdir(parents=True, exist_ok=True)

def elev(x, y):  # smooth fictional terrain with a low basin
    return 12 + 6 * np.sin(x / 4.0) + 5 * np.cos(y / 5.0) - 7 * np.exp(-((x - 9) ** 2 + (y - 6) ** 2) / 12)

segs = []
for i in range(N):
    for j in range(N):
        if i + 1 < N: segs.append(((i, j), (i + 1, j)))
        if j + 1 < N: segs.append(((i, j), (i, j + 1)))

rows, feats = [], []
for k, ((x1, y1), (x2, y2)) in enumerate(segs):
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    e = elev(mx, my) + rng.normal(0, 0.4)
    slope = abs(rng.normal(2.0, 1.2))
    drain = np.clip(rng.normal(6, 2.5) - 0.1 * (e < 8) * 3, 0.5, 20)
    imp = np.clip(rng.normal(60, 20) + 5 * (abs(mx - 8) < 4), 5, 98)
    r24 = (60 + 9 * mx + 4 * my) * rng.uniform(0.9, 1.1)  # fictional rainfall gradient across the grid
    r3, r1 = r24 * rng.uniform(0.35, 0.5), r24 * rng.uniform(0.12, 0.2)
    z = -2.6 + 0.025 * (r24 - 120) + 0.03 * (imp - 50) - 0.22 * (e - 12) - 0.25 * (drain - 6) - 0.3 * slope
    p = 1 / (1 + np.exp(-z))
    flooded = int(rng.random() < p)
    hist = int(rng.poisson(0.2 + 3.0 * p))  # prior-season counts (before label event)
    rid = f"SYN-{k:04d}"
    rows.append(dict(road_id=rid, lat=round(my * STEP, 6), lon=round(mx * STEP, 6),
                     rainfall_1h=round(r1, 1), rainfall_3h=round(r3, 1), rainfall_24h=round(r24, 1),
                     elevation=round(e, 2), slope=round(slope, 2), drainage_density=round(drain, 2),
                     impervious_pct=round(imp, 1), historical_flood_count=hist,
                     flooded=flooded, data_source="SYNTHETIC_DEMO"))
    feats.append({"type": "Feature",
                  "properties": {"road_id": rid, "data_source": "SYNTHETIC_DEMO"},
                  "geometry": {"type": "LineString",
                               "coordinates": [[round(x1 * STEP, 6), round(y1 * STEP, 6)],
                                               [round(x2 * STEP, 6), round(y2 * STEP, 6)]]}})

pd.DataFrame(rows).to_csv(out / "roads.csv", index=False)
(out / "roads.geojson").write_text(json.dumps(
    {"type": "FeatureCollection", "name": "SYNTHETIC demo grid - not real roads", "features": feats}))
print(f"Wrote {len(rows)} synthetic segments, flooded rate {np.mean([r['flooded'] for r in rows]):.2f}")
