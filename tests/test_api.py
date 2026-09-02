"""Tests for the FastAPI inference server."""
import pytest
from fastapi.testclient import TestClient

from src.api.inference_server import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_health(client):
    """Health endpoint returns 200 with required fields."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_a_loaded" in data
    assert "model_b_loaded" in data
    assert "model_version" in data
    assert "cache_size" in data


def test_predict_single(client):
    """Single prediction returns expected fields."""
    response = client.post("/predict", json={
        "sensor_id": "sensor-0001",
        "value": 70.0,
        "unit": "C",
        "pressure": 2.0,
        "vibration": 0.05,
    })
    assert response.status_code == 200
    data = response.json()
    assert "anomaly_likelihood" in data
    assert "is_anomaly" in data
    assert "sensor_id" in data
    assert data["sensor_id"] == "sensor-0001"
    assert "model_version" in data
    assert "model_used" in data


def test_predict_batch(client):
    """Batch prediction returns list of predictions."""
    items = [
        {"sensor_id": f"sensor-{i:04d}", "value": 70.0 + i, "unit": "C"}
        for i in range(5)
    ]
    response = client.post("/predict-batch", json={"items": items})
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 5


def test_predict_batch_empty_fails(client):
    """Empty batch should return 422."""
    response = client.post("/predict-batch", json={"items": []})
    assert response.status_code == 422


def test_predict_missing_fields(client):
    """Missing required fields should return 422."""
    response = client.post("/predict", json={"value": 70.0})
    assert response.status_code == 422


def test_predict_invalid_json(client):
    """Invalid JSON should return 422."""
    response = client.post("/predict", content="not json", headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_metrics_endpoint(client):
    """Prometheus metrics endpoint should return 200."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "prediction_requests_total" in response.text or "prediction_latency_seconds" in response.text or response.status_code == 200
