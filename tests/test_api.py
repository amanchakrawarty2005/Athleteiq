from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200


def test_predict_match():
    r = client.post("/predict-match", json={
        "home_form": 1.5,
        "away_form": 1.2,
        "home_fixture_density": 2,
        "away_fixture_density": 2,
        "elo_delta": 0.0,
        "home_advantage": 1,
        "B365H": 2.0,
        "B365D": 3.4,
        "B365A": 3.8
    })
    assert r.status_code == 200
    assert "prediction" in r.json()
    assert "probabilities" in r.json()


def test_rate_player():
    r = client.get("/rate-player/30893")
    assert r.status_code == 200
    assert "performance_score" in r.json()


def test_cluster_archetypes():
    r = client.get("/cluster-archetypes")
    assert r.status_code == 200
    assert len(r.json()) > 0


def test_top_performers():
    r = client.get("/top-performers?metric=finishing&limit=5")
    assert r.status_code == 200
    assert len(r.json()) == 5