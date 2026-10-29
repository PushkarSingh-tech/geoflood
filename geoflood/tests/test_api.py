import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # runs startup so data/model are loaded
        yield c


def test_health(client):
    j = client.get("/api/health").json()
    assert j["status"] == "ok"


def test_dashboard_and_roads(client):
    d = client.get("/api/dashboard").json()
    assert d["total_roads"] > 0 and sum(c["count"] for c in d["counts"]) == d["total_roads"]
    fc = client.get("/api/roads", params={"category": "High,Very High"}).json()
    assert fc["type"] == "FeatureCollection"
    assert all(f["properties"]["category"] in ("High", "Very High") for f in fc["features"])
    assert client.get("/api/roads", params={"category": "Bogus"}).status_code == 422


def test_road_detail(client):
    assert client.get("/api/roads/SYN-0001").status_code == 200
    assert client.get("/api/roads/nope").status_code == 404


def test_predict_validation(client):
    ok = client.post("/api/predict", json={"rainfall_1h": 20, "rainfall_3h": 50, "rainfall_24h": 100})
    assert ok.status_code == 200 and 0 <= ok.json()["probability"] <= 1
    assert client.post("/api/predict", json={"rainfall_1h": -1}).status_code == 422
    assert client.post("/api/predict", json={"rainfall_1h": 90, "rainfall_3h": 40}).status_code == 422
    assert client.post("/api/predict", json={}).status_code == 422


def test_scenario(client):
    r = client.post("/api/scenario", json={"rainfall_1h": 60, "rainfall_3h": 120, "rainfall_24h": 250})
    j = r.json()
    assert r.status_code == 200 and "not a forecast" in j["kind"] and j["roads"]
    assert client.post("/api/scenario", json={"rainfall_1h": 90, "rainfall_3h": 40, "rainfall_24h": 100}).status_code == 422


def test_model_info(client):
    assert client.get("/api/model-info").json()["metadata"]["data_mode"]
