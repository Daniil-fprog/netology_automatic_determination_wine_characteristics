from fastapi.testclient import TestClient

from wine_characteristics.api import app


SAMPLE = "Красное вино с ароматом вишни и пряностей, плотным вкусом и мягкими танинами."


def test_health_and_model_info():
    with TestClient(app) as client:
        health = client.get("/health")
        info = client.get("/model/info")

        assert health.status_code == 200
        assert health.json() == {"status": "ok", "model_loaded": True}
        assert info.status_code == 200
        assert info.json()["n_samples"] > 0


def test_predict_contract():
    with TestClient(app) as client:
        response = client.post("/predict", json={"description": SAMPLE})

        assert response.status_code == 200
        payload = response.json()
        assert "categories" in payload
        assert "aroma_notes" in payload
        assert "taste_notes" in payload
        assert "model_mode" in payload


def test_batch_and_validation():
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"descriptions": [SAMPLE, SAMPLE]})
        invalid = client.post("/predict", json={"description": "коротко"})

        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 2
        assert invalid.status_code == 422


def test_metrics_endpoint():
    with TestClient(app) as client:
        client.post("/predict", json={"description": SAMPLE})
        response = client.get("/metrics")

        assert response.status_code == 200
        assert "wine_api_predictions_total" in response.text
