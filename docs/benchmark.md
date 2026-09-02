# Reproducible Benchmarking Procedures & Methodology

This document outlines the exact execution instructions to benchmark the Industrial IoT platform across ingestion throughput, API latency, and machine learning accuracy.

---

## 1. Benchmarking Tooling Overview

Three reproducible benchmark utilities are located in `scripts/`:
1. `scripts/benchmark_api.py` — Evaluates FastAPI endpoint latency ($p50$, $p95$, $p99$, mean, min, max) under HTTP load.
2. `scripts/benchmark_producer.py` — Evaluates Kafka producer publication rate (msg/sec) and delivery callback efficiency.
3. `scripts/benchmark_ml.py` — Evaluates model training duration, F1/ROC-AUC classification performance, and per-sample inference latency.

---

## 2. Benchmark Execution

### 2.1 API Latency Benchmark (FastAPI)
Measures the response latency distribution for single and batch predictions.

```bash
# Ensure API service is running (locally or via Docker)
# Local: uvicorn src.api.inference_server:app --port 8000
# Docker: docker compose up -d api

# Run 1,000 requests against /predict
python scripts/benchmark_api.py --url http://localhost:8000/predict --requests 1000

# Run 5,000 requests against /predict
python scripts/benchmark_api.py --url http://localhost:8000/predict --requests 5000
```

#### Output Metrics Captured:
- Total requests and success/error counts
- $p50$, $p95$, and $p99$ response latencies in milliseconds
- Mean, minimum, and maximum round-trip latency

---

### 2.2 Ingestion & Producer Throughput Benchmark (Kafka)
Measures message serialization, batch aggregation, and Kafka broker write throughput.

```bash
# Requires Kafka broker active on port 9092
python scripts/benchmark_producer.py --bootstrap-servers localhost:9092 --messages 50000
```

#### Output Metrics Captured:
- Total produced messages
- Successfully delivered messages via delivery callbacks
- Elapsed time and effective messages/sec throughput

---

### 2.3 ML Anomaly Detection Benchmark
Trains all candidate models (Isolation Forest, LOF, Autoencoder) on historical sensor data, computes precision, recall, F1, ROC-AUC, FPR, and benchmarks in-memory inference speed.

```bash
python scripts/benchmark_ml.py --data data/sample_sensor_data.csv --output-dir models
```

---

## 3. Multi-Tier Workload Benchmark Plan

| Test Scale | Configuration | Target Hardware | Execution Strategy |
|---|---|---|---|
| **Tier 1: 1,000 msg/s** | 1 Producer, 12 Partitions | Single Developer Laptop (4 cores, 8GB RAM) | Direct execution via `python -m src.kafka.producer` |
| **Tier 2: 10,000 msg/s** | 4 Concurrent Producers, 12 Partitions | Single Server / VM (8 cores, 16GB RAM) | Multi-process producer pool with batching |
| **Tier 3: 100,000 msg/s** | 10 Producer nodes, 24 Partitions | Distributed Cluster (Kubernetes / EKS) | Distributed pod workers + 3-node Kafka broker cluster |
| **Tier 4: 1,000,000 msg/s** | 50 Producer nodes, 64 Partitions | High-Throughput Cloud Cluster | Multi-broker MSK cluster + Spark on EMR / Databricks |

> **Note:** Performance numbers for distributed tiers (100K+ msg/s) should only be published after verification on target cloud infrastructure with provisioned IOPS.

