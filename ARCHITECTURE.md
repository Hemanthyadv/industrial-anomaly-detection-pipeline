# Industrial Real-Time Platform: System Architecture & Design

## 1. End-to-End Data Flow

```text
                                +-------------------------------------------+
                                |             Industrial Sensors            |
                                |     120+ Sensors across 4 Site Clusters   |
                                +---------------------+---------------------+
                                                      |
                                                      v
                                +-------------------------------------------+
                                |      Sensor Simulator / Avro Encoder      |
                                |  - Deterministic diurnal patterns         |
                                |  - Injected anomalies (Spike, Drift, etc) |
                                |  - Binary serialization via fastavro      |
                                +---------------------+---------------------+
                                                      |
                                         acks=all, lz4, idempotent
                                                      |
                                                      v
+-------------------------------------------------------------------------------------------------------+
|                                         Apache Kafka Cluster                                          |
|                                                                                                       |
|  * raw_sensor_data    (12 partitions, replication=1, retention=7d)                                    |
|  * processed_events  (12 partitions, replication=1, retention=7d)                                    |
|  * anomalies         (12 partitions, replication=1, retention=30d)                                   |
+----------------------+--------------------------------------------------------------------------------+
                       |
                       | Event-Time Structured Streaming
                       v
+------------------------------------------------------+        +---------------------------------------+
|             Spark Structured Streaming               |        |        MLflow Tracking Server         |
|  - Watermarking (10m delay)                          |        |  - Model Versioning (Alias / Stage)   |
|  - 1-min & 5-min Tumbling Windows                    |        |  - Parameter & Metric Tracking        |
|  - Quality Filtering (Nulls, ID Regex, Outliers)     |        |  - Artifact Storage                   |
|  - Stateful Lag Extraction (t-1, t-5, t-60)          |        +-------------------+-------------------+
+----------------------+-------------------------------+                            |
                       |                                                            |
                       v                                                            v
+------------------------------------------------------+        +---------------------------------------+
|              Delta Lake Feature Store                |        |        Model Training Pipeline        |
|  - ACID Versioned Sinks (/data/delta/features_5m)    |------->|  - Isolation Forest + StandardScaler  |
|  - Thread-Safe JSON Feature Catalog                  |        |  - Local Outlier Factor (Novelty)     |
|  - Point-in-time training feature extraction         |        |  - Neural Autoencoder (MLP)           |
+------------------------------------------------------+        |  - Prophet Time-Series Forecaster     |
                                                                +-------------------+-------------------+
                                                                                    |
                                                                                    v
+------------------------------------------------------+        +---------------------------------------+
|              Prometheus & Alertmanager               |<-------|         FastAPI Inference API         |
|  - Scrapes /metrics on port 8000                     |        |  - POST /predict (p95 < 100ms)        |
|  - Evaluates 7 SLO Alert Rules                       |        |  - POST /predict-batch (1-1000 items) |
|  - Tracks throughput, latency, lag, drift, errors    |        |  - In-Memory TTLCache (30s)           |
+----------------------+-------------------------------+        |  - Dynamic A/B Model Routing          |
                       |                                        +---------------------------------------+
                       v
+-------------------------------------------------------------------------------------------------------+
|                                        Grafana 11 Dashboards                                          |
|  1. Real-Time Sensor Metrics    2. Anomaly Detection    3. Model Performance    4. System Health      |
+-------------------------------------------------------------------------------------------------------+
```

---

## 2. Ingestion & Data Contracts (Avro)

Telemetry payloads strictly adhere to the Avro binary contract defined in `src/kafka/schema.avsc`:

```json
{
  "type": "record",
  "name": "SensorEvent",
  "namespace": "industrial.iot",
  "fields": [
    {"name": "timestamp", "type": "long", "logicalType": "timestamp-millis"},
    {"name": "sensor_id", "type": "string"},
    {"name": "value", "type": "double"},
    {"name": "unit", "type": "string"},
    {"name": "metadata", "type": {"type": "map", "values": "string"}}
  ]
}
```

### Producer Invariants:
- **Idempotence:** `enable.idempotence=True` prevents duplicate message delivery during network retries.
- **Partition Key Affinity:** Messages use `key=sensor_id` to guarantee in-order delivery per sensor partition.
- **Bounded Buffering & Backpressure:** Bounded in-flight queue (`50,000` messages) with explicit `BufferError` handling and callback polling.

---

## 3. Stream Processing & Window Semantics

The streaming pipeline is implemented in PySpark Structured Streaming (`src/spark/spark_streaming.py`):

1. **Watermarking:** `withWatermark("timestamp", "10 minutes")` drops data arriving later than 10 minutes past event-time.
2. **5-Minute Tumbling Windows:**
   - Computes summary statistics: `rolling_mean_5m`, `rolling_min_5m`, `rolling_max_5m`, `rolling_var_5m`, `rolling_stddev_5m`, `rolling_skewness_5m`, `rolling_kurtosis_5m`.
3. **1-Hour Tumbling Windows:**
   - Computes macro trends: `rolling_mean_1h`, `rolling_min_1h`, `rolling_max_1h`, `rolling_var_1h`.
4. **Time-Aligned Lag Semantics:**
   - Lags ($t-1, t-5, t-60$) are computed over 1-minute time-bucketed indices rather than arbitrary row-count offsets, preventing distortions during transmission dropouts.
5. **ACID Delta Sink:** Stream micro-batches are appended atomically to `/data/delta/features_5m` with checkpoint locations for recovery.

---

## 4. Machine Learning & Anomaly Detection

### Unsupervised Models Evaluated:
1. **Isolation Forest (Primary Baseline):**
   - Preprocessed with `StandardScaler` to normalize multi-modal scales (Temperature $\sim 70^\circ\text{C}$, Pressure $\sim 2.0\text{ bar}$, Vibration $\sim 0.05\text{ mm/s}$).
   - Fast $O(n \log n)$ training and $O(\text{trees})$ sub-millisecond inference.
2. **Local Outlier Factor (LOF):**
   - Configured with `novelty=True` for real-time scoring.
   - Evaluates local density ratios compared to neighboring points.
3. **Neural Autoencoder (MLP-Based):**
   - Trained to reconstruct nominal sensor profiles ($3 \to 32 \to 16 \to 8 \to 16 \to 32 \to 3$).
   - High reconstruction MSE ($\frac{1}{d} \sum (x_i - \hat{x}_i)^2 > \text{threshold}$) flags complex multi-variable coupling anomalies.

### Time-Series Forecasting (Prophet):
- Predicts future sensor values $24\text{ hours}$ ahead.
- Generates 95% confidence intervals ($\hat{y}_{\text{lower}}, \hat{y}_{\text{upper}}$) to trigger proactive preventative alarms prior to threshold breaches.

---

## 5. Inference Service & A/B Routing

The FastAPI serving layer (`src/api/inference_server.py`):
- **Dynamic A/B Routing:** Bounded random selection parameterized by `MODEL_A_WEIGHT` (e.g. 0.8) and `MODEL_B_WEIGHT` (e.g. 0.2).
- **Prediction Caching:** In-memory `TTLCache(maxsize=10000, ttl=30)` eliminates redundant computation for repeated telemetry queries.
- **Unified Prometheus Metrics:** Directly updates global metric instances (`PREDICTION_REQUESTS`, `PREDICTION_LATENCY`, `ANOMALIES_DETECTED`).

---

## 6. Observability & SLO Alerting

Prometheus actively scrapes the application stack every 15 seconds. Rules in `docker/alert_rules.yml` evaluate seven operational conditions:
1. **`SensorDataStale`:** Telemetry freshness $> 300\text{s}$ (Critical).
2. **`KafkaLagHigh`:** Total consumer lag $> 10,000$ messages (Warning).
3. **`APIPredictionLatencyHigh`:** p95 latency $> 100\text{ms}$ over 5 minutes (Warning).
4. **`AnomalyRateAbnormal`:** Anomaly rate $> 15\%$ over 10 minutes (Warning).
5. **`DataQualityFailure`:** Quality error rate $> 0.1/\text{s}$ (Critical).
6. **`ModelDriftDetected`:** Drift score $> 0.5$ (Warning).
7. **`ModelPerformanceDegraded`:** Anomaly rate outside $[0.1\%, 30\%]$ (Critical).

---

## 7. Scaling Model & Cloud Capacity Estimation

To scale from $1\text{K msg/s}$ to $100\text{K+}$ or $1\text{M msg/s}$:
- **Kafka:** Increase topic partition count from 12 to 64; scale broker cluster across multiple availability zones.
- **Spark:** Scale executors horizontally; increase shuffle partitions to match core count.
- **Inference:** Deploy FastAPI as a Kubernetes Deployment with Horizontal Pod Autoscaler (HPA) targeting CPU utilization $< 70\%$.

