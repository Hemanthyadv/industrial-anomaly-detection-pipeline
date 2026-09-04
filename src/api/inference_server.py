"""FastAPI inference service with model loading, caching, A/B routing, and Prometheus metrics.

Endpoints:
    POST /predict         - Single sensor prediction
    POST /predict-batch   - Batch prediction (up to 1000 items)
    GET  /health          - Service health check
    GET  /metrics         - Prometheus metrics (auto-mounted)
"""
import logging
import os
import time
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import make_asgi_app

from src.monitoring import (
    ANOMALIES_DETECTED,
    MODEL_INFO,
    PREDICTION_LATENCY,
    PREDICTION_REQUESTS,
)

LOG = logging.getLogger("inference-api")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/isolation_forest.joblib")
MODEL_B_PATH = os.getenv("MODEL_B_PATH", "models/lof.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "baseline")
MODEL_A_WEIGHT = float(os.getenv("MODEL_A_WEIGHT", "0.8"))
MODEL_B_WEIGHT = float(os.getenv("MODEL_B_WEIGHT", "0.2"))

cache: TTLCache = TTLCache(maxsize=10000, ttl=30)

app = FastAPI(title="Industrial Sensor Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/metrics", make_asgi_app())


# --- Pydantic Models ---

class SensorReading(BaseModel):
    """Single sensor reading for prediction."""
    sensor_id: str
    value: float
    unit: str = "C"
    pressure: float = 2.0
    vibration: float = 0.05


class BatchRequest(BaseModel):
    """Batch of sensor readings (1-1000 items)."""
    items: List[SensorReading] = Field(min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    """Prediction result for a single sensor reading."""
    sensor_id: str
    anomaly_likelihood: float
    is_anomaly: bool
    model_version: str
    model_used: str = "model_a"


# --- Model Loading ---

def load_model(path: str) -> Optional[Any]:
    """Load a model from disk. Returns None if file doesn't exist."""
    if not os.path.exists(path):
        LOG.warning("Model file not found: %s", path)
        return None
    try:
        model = joblib.load(path)
        LOG.info("Loaded model from %s", path)
        return model
    except Exception as exc:
        LOG.error("Failed to load model from %s: %s", path, exc)
        return None


def _is_pipeline(model) -> bool:
    """Check if model is a sklearn Pipeline (has scaler built in)."""
    try:
        from sklearn.pipeline import Pipeline
        return isinstance(model, Pipeline)
    except ImportError:
        return False


def _is_autoencoder(model) -> bool:
    """Check if model is our autoencoder bundle dict."""
    return isinstance(model, dict) and model.get("type") == "autoencoder"


MODEL_A = load_model(MODEL_PATH)
MODEL_B = load_model(MODEL_B_PATH)

if MODEL_A is not None:
    MODEL_INFO.info({"model_path": MODEL_PATH, "version": MODEL_VERSION})


# --- Scoring ---

def _score_with_model(model: Any, x: np.ndarray) -> tuple:
    """Score a single observation with any supported model type.

    Returns (anomaly_likelihood: float, is_anomaly: bool).
    """
    if _is_autoencoder(model):
        scaler = model["scaler"]
        mlp = model["mlp"]
        threshold = model["threshold"]
        x_scaled = scaler.transform(x)
        reconstructed = mlp.predict(x_scaled)
        recon_error = float(np.mean((x_scaled - reconstructed) ** 2))
        likelihood = min(1.0, recon_error / max(threshold * 2, 1e-6))
        is_anomaly = recon_error > threshold
        return likelihood, bool(is_anomaly)

    # sklearn Pipeline or raw model
    if _is_pipeline(model):
        decision = float(model.decision_function(x)[0])
        prediction = int(model.predict(x)[0])
    else:
        decision = float(model.decision_function(x)[0])
        prediction = int(model.predict(x)[0])

    # Convert decision function to 0-1 likelihood using sigmoid
    likelihood = float(1.0 / (1.0 + np.exp(5.0 * decision)))
    is_anomaly = prediction == -1
    return likelihood, is_anomaly


def select_model() -> tuple:
    """Select model based on A/B routing weights.

    Returns (model, label).
    """
    import random
    if MODEL_A is not None and MODEL_B is not None:
        total = MODEL_A_WEIGHT + MODEL_B_WEIGHT
        if random.random() < (MODEL_A_WEIGHT / total):
            return MODEL_A, "model_a"
        return MODEL_B, "model_b"
    if MODEL_A is not None:
        return MODEL_A, "model_a"
    if MODEL_B is not None:
        return MODEL_B, "model_b"
    return None, "none"


def score(item: SensorReading) -> Dict[str, Any]:
    """Score a single sensor reading."""
    key = (item.sensor_id, item.value, item.pressure, item.vibration)
    if key in cache:
        return cache[key]

    model, model_label = select_model()
    x = np.array([[item.value, item.pressure, item.vibration]], dtype=float)

    if model is None:
        # No model available - return error-indicating response
        result = {
            "sensor_id": item.sensor_id,
            "anomaly_likelihood": -1.0,
            "is_anomaly": False,
            "model_version": "unavailable",
            "model_used": "none",
        }
        cache[key] = result
        return result

    likelihood, is_anomaly = _score_with_model(model, x)

    if is_anomaly:
        ANOMALIES_DETECTED.labels(model_version=MODEL_VERSION).inc()

    result = {
        "sensor_id": item.sensor_id,
        "anomaly_likelihood": round(likelihood, 6),
        "is_anomaly": is_anomaly,
        "model_version": MODEL_VERSION,
        "model_used": model_label,
    }
    cache[key] = result
    return result


# --- Endpoints ---

@app.get("/health")
def health():
    """Service health check."""
    return {
        "status": "ok",
        "model_a_loaded": MODEL_A is not None,
        "model_b_loaded": MODEL_B is not None,
        "model_version": MODEL_VERSION,
        "cache_size": len(cache),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(item: SensorReading):
    """Single sensor prediction endpoint."""
    start = time.perf_counter()
    PREDICTION_REQUESTS.labels(endpoint="predict").inc()
    try:
        result = score(item)
        return result
    except Exception as exc:
        LOG.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="prediction failed") from exc
    finally:
        PREDICTION_LATENCY.labels(endpoint="predict").observe(
            time.perf_counter() - start
        )


@app.post("/predict-batch")
def predict_batch(batch: BatchRequest):
    """Batch prediction endpoint (1-1000 items)."""
    start = time.perf_counter()
    PREDICTION_REQUESTS.labels(endpoint="predict-batch").inc(len(batch.items))
    try:
        predictions = [score(item) for item in batch.items]
        return {"predictions": predictions}
    except Exception as exc:
        LOG.error("Batch prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="batch prediction failed") from exc
    finally:
        PREDICTION_LATENCY.labels(endpoint="predict-batch").observe(
            time.perf_counter() - start
        )
