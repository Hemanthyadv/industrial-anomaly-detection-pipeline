# Industrial Real-Time IoT Data Streaming, Feature Store & Predictive Intelligence Platform

[![CI](https://github.com/your-username/industrial-realtime-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/industrial-realtime-platform/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A production-grade reference architecture for real-time Industrial IoT telemetry ingestion, stream processing, windowed feature engineering, Delta Lake ACID feature store, multi-model unsupervised anomaly detection, time-series forecasting, low-latency FastAPI inference, and Prometheus/Grafana observability.

---

## 1. What This Project Does

This platform ingests high-frequency telemetry from a simulated fleet of 100+ industrial sensors (monitoring temperature, pressure, and vibration across distributed sites), enforces Avro schema validation, processes sliding and tumbling event-time windows in Apache Spark Structured Streaming, persists versioned features into a Delta Lake feature store, trains unsupervised anomaly detectors (Isolation Forest, LOF, Neural Autoencoders) and time-series forecasters (Prophet), serves real-time inferences under 100ms with dynamic A/B routing via FastAPI, and monitors pipeline health, data freshness, model drift, and Kafka lag in Prometheus and Grafana.

---

## 2. Why It Exists

Modern industrial environments (refineries, manufacturing plants, smart grids) generate high-velocity telemetry where equipment degradation manifests as subtle statistical deviations—rate-of-change surges, vibration spikes, drift, or signal freeze. Traditional batch processing fails to detect anomalies before catastrophic failure occurs.

This platform bridges the gap between raw real-time streaming and ML-driven predictive maintenance:
- **Low Latency & High Throughput:** Ingests up to 100,000+ msg/sec with bounded buffers and backpressure.
- **Strict Data Contracts:** Schema validation using binary Avro prevents corrupt data propagation.
- **ACID Feature Storage:** Delta Lake ensures consistent feature retrieval for both batch training and real-time inference.
- **Multi-Model Anomaly Intelligence:** Evaluates multiple unsupervised algorithms with real-time scoring and prediction caching.
- **End-to-End Observability:** Metrics, SLO alerting, and 4 Grafana dashboards provisioned out of the box.

---

## 3. Architecture & Data Flow

```text
                               +------------------------------------------+
                               |     Industrial Sensors Fleet (120+)      |
                               |    [Temperature, Pressure, Vibration]    |
                               +--------------------+---------------------+
                                                    |
                                                    v
                               +------------------------------------------+
                               |    Sensor Simulator / Avro Validator     |
                               |    - Injected Spikes, Drift, Freezes     |
                               |    - fastavro strict schema validation   |
                               +--------------------+---------------------+
                                                    |
                                       Idempotent Producer (acks=all)
                                                    |
                                                    v
+----------------------------------------------------------------------------------------------------+
|                                      Apache Kafka Cluster                                          |
|  Topics:                                                                                           |
|  * raw_sensor_data (12 partitions)                                                                 |
|  * processed_events (12 partitions)                                                                |
|  * anomalies (12 partitions)                                                                       |
+--------------------+-------------------------------------------------------------------------------+
                     |
                     v
+----------------------------------------------------+       +---------------------------------------+
|             Spark Structured Streaming             |       |        MLflow Tracking Server         |
|  - 1-min & 5-min Tumbling/Sliding Windows          |       |  - Parameter & Metric Tracking        |
|  - Rolling Stats (Mean, Min, Max, Var, Skew, Kurt) |       |  - Artifacts & Model Registry         |
|  - Watermarking (10m) & Late-Data Quarantine       |       |  - Versioning (Staging / Production)  |
+--------------------+-------------------------------+       +-------------------+-------------------+
                     |                                                           |
                     v                                                           v
+----------------------------------------------------+       +---------------------------------------+
|             Delta Lake Feature Store               |       |         Model Training Engine         |
|  - ACID Writes & Versioned Partitions              |------>|  - Isolation Forest (StandardScaler)  |
|  - Thread-Safe JSON Feature Registry               |       |  - Local Outlier Factor (Novelty)     |
|  - Time-Aligned Lags (t-1, t-5, t-60)              |       |  - Neural Autoencoder (MLP)           |
+----------------------------------------------------+       |  - Prophet Time-Series Forecasting    |
                                                             +-------------------+-------------------+
                                                                                 |
                                                                                 v
+----------------------------------------------------+       +---------------------------------------+
|            Prometheus & Alertmanager               |<------|         FastAPI Inference API         |
|  - Metrics: Throughput, Latency, Freshness, Drift  |       |  - POST /predict (p95 < 100ms)        |
|  - 7 Production Alert Rules (YAML)                 |       |  - POST /predict-batch (1-1000 items) |
|  - Scrapes API (/metrics) & Pipeline Exporters     |       |  - In-Memory TTLCache & A/B Routing   |
+--------------------+-------------------------------+       |  - Health & Readiness Checks          |
                     |                                       +---------------------------------------+
                     v
+----------------------------------------------------------------------------------------------------+
|                                    Grafana 11 Dashboards                                           |
|  1. Real-Time Sensor Metrics  |  2. Anomaly Detection  |  3. Model Performance  | 4. System Health |
+----------------------------------------------------------------------------------------------------+
```

---

## 4. Key Architectural Decisions

| Decision | Selected Technology | Alternatives Evaluated | Rationale |
|---|---|---|---|
| **Stream Engine** | **Apache Spark Structured Streaming** | Apache Flink | Unified API across streaming transformations and historical Delta Lake feature extraction; familiar operational model for data teams. |
| **Storage Layer** | **Delta Lake** | Apache Iceberg, Raw Parquet | Native ACID transactions, versioned time-travel, compaction (`OPTIMIZE`), and seamless integration with PySpark streaming sinks. |
| **Primary Anomaly Model** | **Isolation Forest with StandardScaler** | One-Class SVM, K-Means | Linear time complexity, highly effective on multi-modal continuous sensor telemetry, robust to high dimensional tabular signals without labels. |
| **Density Comparison** | **Local Outlier Factor (LOF)** | DBSCAN | `novelty=True` enables scoring real-time samples against dense local neighborhood clusters. |
| **Reconstruction Model** | **Neural Autoencoder (MLPRegressor)** | PyTorch Deep Autoencoder | Lightweight deployment footprint in containerized inference services without heavy GPU/Torch runtime dependencies. |
| **Forecasting** | **Facebook Prophet** | LSTM / GRU | Built-in diurnal/weekly seasonality handling, interpretable trend change points, and native probabilistic prediction intervals. |
| **Message Ingestion** | **Apache Kafka + fastavro** | RabbitMQ, MQTT | High partition parallelism, zero-loss idempotence (`acks=all`), with client-side Avro contract validation. |
| **Inference Server** | **FastAPI + Uvicorn + TTLCache** | Flask, Triton | Asynchronous ASGI handling, native Pydantic type safety, integrated Prometheus metrics, sub-100ms response latencies. |

---

## 5. Technology Stack

- **Streaming & Messaging:** Apache Kafka 7.7 (Confluent), Apache Zookeeper, `confluent-kafka`
- **Data Serialization:** Apache Avro, `fastavro`
- **Stream Processing & Storage:** Apache Spark 3.5.3, Delta Lake 3.2.0, PySpark
- **Machine Learning & Forecasting:** scikit-learn 1.5, Prophet, NumPy, Pandas, Joblib
- **Experiment Tracking & Registry:** MLflow 2.15 (SQLite Backend + Artifact Store)
- **API & Serving:** FastAPI, Uvicorn, Pydantic v2, Cachetools
- **Observability:** Prometheus 2.54, Grafana 11.2 (Auto-provisioned Dashboards), Alertmanager Rules
- **Containerization & CI/CD:** Docker, Docker Compose, GitHub Actions, Pytest, Ruff

---

## 6. Getting Started & Quick Run

### Prerequisites
- Docker & Docker Compose
- Python 3.11+

### Step 1: Clone and Configure Environment
```bash
git clone https://github.com/your-username/industrial-realtime-platform.git
cd industrial-realtime-platform
cp .env.example .env
```

### Step 2: Generate 7-Day Sensor Training Dataset
```bash
# Generate 7 days of telemetry for 120 sensors (1,209,600 rows)
python -m src.kafka.sensor_simulator --days 7 --sensors 120 --output data/sample_sensor_data.csv
```

### Step 3: Train Anomaly Detection & Forecasting Models
```bash
# Trains Isolation Forest, LOF, and Autoencoder; saves artifacts to models/
python -m src.ml.train_models --output-dir models --data data/sample_sensor_data.csv

# Train time-series forecasting pipeline (Prophet)
python -m src.ml.forecasting --data data/sample_sensor_data.csv --output-dir models
```

### Step 4: Launch Complete Stack with Docker Compose
```bash
docker compose up --build -d
```

### Step 5: Verify Services
| Service | Endpoint | Credentials | Description |
|---|---|---|---|
| **FastAPI Swagger Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | None | Interactive prediction API |
| **Prometheus UI** | [http://localhost:9090](http://localhost:9090) | None | Metrics, targets, and active alert rules |
| **Grafana Dashboards** | [http://localhost:3000](http://localhost:3000) | `admin` / `admin` | 4 auto-provisioned dashboards |
| **MLflow UI** | [http://localhost:5000](http://localhost:5000) | None | Experiment runs, params, and registered models |
| **Kafka Broker** | `localhost:9092` | None | Raw and processed Kafka topics |

---

## 7. Real-Time Streaming & Inference Verification

### Send a Live Prediction Request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_id": "sensor-0001",
    "value": 72.4,
    "unit": "C",
    "pressure": 2.05,
    "vibration": 0.048
  }'
```
**Response:**
```json
{
  "sensor_id": "sensor-0001",
  "anomaly_likelihood": 0.041289,
  "is_anomaly": false,
  "model_version": "baseline",
  "model_used": "model_a"
}
```

### Stream Live Sensor Telemetry to Kafka:
```bash
python -m src.kafka.producer
```

---

## 8. Automated Testing & Benchmarking

### Run the Test Suite with Coverage:
```bash
pip install -r requirements-dev.txt
pytest -q --cov=src --cov-report=term-missing
```

### Run Benchmarks:
```bash
# 1. API Latency Benchmark (p50, p95, p99)
python scripts/benchmark_api.py --url http://localhost:8000/predict --requests 1000

# 2. ML Models Evaluation Benchmark
python scripts/benchmark_ml.py --data data/sample_sensor_data.csv --output-dir models

# 3. Kafka Producer Throughput Benchmark (requires running Kafka)
python scripts/benchmark_producer.py --bootstrap-servers localhost:9092 --messages 10000
```

---

## 9. Grafana Dashboards

The stack auto-provisions four dedicated Grafana dashboards:
1. **Real-Time Sensor Metrics (`sensor-realtime`):** Request throughput, p50/p95/p99 latency, anomaly rate, data freshness.
2. **Anomaly Detection (`anomaly-detection`):** Anomaly detections over time by model version, data quality check failures.
3. **Model Performance (`model-performance`):** Model drift score, active ML model metadata, forecast MAE.
4. **System Health (`system-health`):** API health status, Kafka consumer lag, messages produced, null rates.

---

## 10. Repository Structure

```text
├── .github/workflows/ci.yml      # GitHub Actions CI & scheduled model retraining
├── data/
│   ├── sample_sensor_data.csv    # 7-day historical dataset with synthetic anomalies
│   └── feature_registry.json     # Thread-safe JSON feature metadata registry
├── docker/
│   ├── Dockerfile.api            # Lightweight non-root API container
│   ├── Dockerfile.spark          # Spark 3.5.3 container with Kafka & Delta JARs
│   ├── Dockerfile.mlflow         # MLflow server container
│   ├── alert_rules.yml           # 7 Prometheus alerting rules
│   ├── prometheus.yml            # Prometheus scrape config & alert rules loader
│   └── grafana-provisioning/     # Automated datasource & dashboard providers
├── grafana_dashboards/           # 4 auto-provisioned Grafana JSON dashboard templates
├── notebooks/
│   └── end_to_end_walkthrough.ipynb # 11-step complete runnable ML walkthrough
├── scripts/
│   ├── benchmark_api.py          # API latency benchmarking harness
│   ├── benchmark_ml.py           # ML models benchmarking & comparison script
│   ├── benchmark_producer.py     # Kafka producer throughput test
│   ├── train.py                  # CLI training entry point
│   ├── register_model.py         # MLflow model registration helper
│   └── promote_model.py          # MLflow model stage promotion helper
├── src/
│   ├── alerting.py               # Prometheus alert definitions & YAML exporter
│   ├── monitoring.py             # Centralized Prometheus metrics registry
│   ├── api/
│   │   └── inference_server.py   # FastAPI service with caching, A/B routing & metrics
│   ├── kafka/
│   │   ├── avro_validator.py     # fastavro schema validator and serializer
│   │   ├── producer.py           # Idempotent batching producer with backpressure
│   │   ├── schema.avsc           # SensorEvent Avro schema contract
│   │   └── sensor_simulator.py   # Deterministic telemetry simulator & anomaly injector
│   ├── ml/
│   │   ├── forecasting.py        # Prophet time-series forecasting pipeline
│   │   ├── hyperparameter_tuning.py # Unsupervised anomaly model tuning
│   │   ├── mlflow_setup.py       # MLflow registry & A/B model routing helpers
│   │   ├── model_evaluation.py   # Anomaly & forecasting metric evaluators
│   │   └── train_models.py       # Isolation Forest, LOF & Autoencoder training
│   └── spark/
│       ├── feature_engineering.py# Streaming window aggregations & batch lag features
│       ├── feature_store.py      # Delta Lake feature store adapter & registry
│       ├── quality_checks.py     # Data quality validation & quarantine filters
│       └── spark_streaming.py    # Spark Structured Streaming pipeline
├── tests/                        # Full unit and integration test suite
├── docker-compose.yml            # Complete multi-service Docker Compose stack
├── features.yaml                 # Formal feature catalog definitions
├── requirements.txt              # Production dependencies
└── requirements-dev.txt          # Development, testing, and linting dependencies
```

---

## 11. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

