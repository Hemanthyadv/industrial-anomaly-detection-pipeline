# Quick Start Guide

Follow these steps to run the complete Industrial IoT platform locally.

---

## 1. Prerequisites

- **Python:** 3.11 or higher
- **Docker Desktop:** with Docker Compose v2+
- **System Memory:** Minimum 8 GB RAM recommended

---

## 2. Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-username/industrial-realtime-platform.git
cd industrial-realtime-platform

# Create Python virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Copy environment file
cp .env.example .env
```

---

## 3. Generate Historical Data & Train ML Models

Before starting the containers, generate training data and produce model artifacts:

```bash
# 1. Generate 7-day multi-sensor historical dataset (120 sensors)
python -m src.kafka.sensor_simulator --days 7 --sensors 120 --output data/sample_sensor_data.csv

# 2. Train Isolation Forest, LOF, and Autoencoder
python -m src.ml.train_models --output-dir models --data data/sample_sensor_data.csv

# 3. Train Prophet time-series forecasting models
python -m src.ml.forecasting --data data/sample_sensor_data.csv --output-dir models
```

---

## 4. Start the Complete Stack with Docker Compose

```bash
docker compose up --build -d
```

Check the status of all services:
```bash
docker compose ps
```

---

## 5. Access UI Services

| Service | URL | Default Credentials |
|---|---|---|
| **FastAPI Swagger Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | None |
| **Prometheus Dashboard** | [http://localhost:9090](http://localhost:9090) | None |
| **Grafana UI** | [http://localhost:3000](http://localhost:3000) | `admin` / `admin` |
| **MLflow Server** | [http://localhost:5000](http://localhost:5000) | None |

---

## 6. Test Inference Endpoints

### Single Sensor Prediction:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_id": "sensor-0001",
    "value": 70.2,
    "unit": "C",
    "pressure": 2.02,
    "vibration": 0.051
  }'
```

### Batch Prediction:
```bash
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"sensor_id": "sensor-0001", "value": 70.2, "unit": "C", "pressure": 2.02, "vibration": 0.051},
      {"sensor_id": "sensor-0002", "value": 115.8, "unit": "C", "pressure": 3.80, "vibration": 0.190}
    ]
  }'
```

---

## 7. Stream Telemetry to Kafka

To simulate live streaming at 1,000 messages/sec into Kafka:
```bash
python -m src.kafka.producer
```

---

## 8. Run Automated Tests & Benchmarks

```bash
# Run pytest with coverage
pytest -q --cov=src --cov-report=term-missing

# Benchmark API latency (p50/p95/p99)
python scripts/benchmark_api.py --requests 500

# Benchmark ML evaluation metrics
python scripts/benchmark_ml.py
```

---

## 9. Stopping & Teardown

```bash
# Stop containers
docker compose stop

# Stop and remove containers and networks
docker compose down

# Stop and remove volumes (clean reset)
docker compose down -v
```

