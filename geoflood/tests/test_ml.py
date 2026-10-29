import numpy as np, pandas as pd
from ml import preprocess as pp
from ml.config import DATA_DIR, CSV_NAME, categorize
from ml.train import train
from ml.predict import Predictor


def test_clean_nulls_out_of_range():
    df = pd.DataFrame({"road_id": ["a", "b", "a"], "rainfall_1h": [10, -5, 10], "slope": [3, 120, 3]})
    out = pp.clean(df)
    assert len(out) == 2 and np.isnan(out.loc[1, "rainfall_1h"]) and np.isnan(out.loc[1, "slope"])


def test_validate_reports_errors():
    assert not pp.validate_dataframe(pd.DataFrame({"x": [1]}))["ok"]
    bad_target = pd.DataFrame({"road_id": ["a"], "slope": [1], "flooded": [3]})
    assert not pp.validate_dataframe(bad_target)["ok"]
    assert pp.validate_dataframe(pd.DataFrame({"road_id": ["a"], "slope": [1]}))["ok"]


def test_categorize_boundaries():
    assert [categorize(p) for p in (0, .25, .5, .75, 1)] == ["Low", "Moderate", "High", "Very High", "Very High"]


def test_train_and_predict(tmp_path):
    out = train(DATA_DIR / CSV_NAME, tmp_path / "m", tmp_path / "r")
    assert "geographic" in out["meta"]["validation"]
    assert {"dummy_prior", "logistic_regression"} <= set(out["metrics"])
    pred = Predictor(tmp_path / "m")
    r = pred.predict_one({"rainfall_1h": 20, "rainfall_3h": 50, "rainfall_24h": 100})  # others missing
    assert 0 <= r["probability"] <= 1 and r["imputed_features"]
