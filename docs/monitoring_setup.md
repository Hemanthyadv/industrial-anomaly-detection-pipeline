# Monitoring & Observability Setup Guide

This guide documents the Prometheus metrics collection, alerting engine, and Grafana visualization setup for the Industrial IoT platform.

---

## 1. Architecture

The observability stack comprises:
- **Centralized Metrics Registry:** `src/monitoring.py` defines all application-level Prometheus Counters, Gauges, and Histograms.
- **FastAPI `/metrics`:** Exposes Prometheus exposition format via ASGI mounting (`prometheus_client.make_asgi_app()`).
- **Prometheus Server (`prom/prometheus:v2.54.1`):** Scrapes API targets every 15s and evaluates alerting rules.
- **Grafana (`grafana/grafana:11.2.0`):** Pre-provisioned with Prometheus datasource and 4 operational dashboards.

---

## 2. Metric Catalog

### Producer & Ingestion Metrics:
| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `sensor_messages_produced_total` | Counter | `topic` | Total events published to Kafka |
| `sensor_producer_errors_total` | Counter | `error_type` | Delivery failures and serialization errors |
| `sensor_producer_retries_total` | Counter | — | In-flight retry attempts |

### Processing & Health Metrics:
| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `sensor_messages_processed_total` | Counter | — | Total events processed by Spark stream |
| `kafka_consumer_lag` | Gauge | `topic`, `partition` | Consumer group lag behind partition high watermark |
| `data_freshness_seconds` | Gauge | — | Elapsed seconds since newest event timestamp |

### Data Quality Metrics:
| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `data_quality_null_rate` | Gauge | `field` | Ratio of null values in processing window |
| `data_quality_outlier_rate` | Gauge | — | Fraction of statistical $3\sigma$ outliers |
| `data_quality_failures_total` | Counter | `check_type` | Total records failing data quality checks |

### Machine Learning & Inference Metrics:
| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `prediction_requests_total` | Counter | `endpoint` | Total HTTP prediction requests received |
| `prediction_latency_seconds` | Histogram | `endpoint` | End-to-end inference latency ($[5\text{ms}, 1\text{s}]$ buckets) |
| `sensor_anomalies_detected_total`| Counter | `model_version` | Cumulative anomaly detections |
| `sensor_anomaly_rate` | Gauge | — | Current anomaly ratio over rolling window |
| `model_drift_score` | Gauge | — | Statistical distance from baseline feature distribution |
| `forecast_error_metric` | Gauge | `sensor_type` | Mean Absolute Error against ground truth |
| `ml_model_info` | Info | `model_path`, `version` | Active model deployment metadata |

---

## 3. Prometheus Alerting Rules

Alert rules are defined in `docker/alert_rules.yml` and loaded into Prometheus automatically:

```yaml
groups:
  - name: industrial_iot_alerts
    rules:
      - alert: SensorDataStale
        expr: 'data_freshness_seconds > 300'
        for: 2m
        labels: { severity: critical }
      - alert: KafkaLagHigh
        expr: 'sum(kafka_consumer_lag) > 10000'
        for: 5m
        labels: { severity: warning }
      - alert: APIPredictionLatencyHigh
        expr: 'histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m])) > 0.1'
        for: 5m
        labels: { severity: warning }
      - alert: AnomalyRateAbnormal
        expr: 'sensor_anomaly_rate > 0.15'
        for: 10m
        labels: { severity: warning }
      - alert: DataQualityFailure
        expr: 'rate(data_quality_failures_total[5m]) > 0.1'
        for: 5m
        labels: { severity: critical }
      - alert: ModelDriftDetected
        expr: 'model_drift_score > 0.5'
        for: 15m
        labels: { severity: warning }
      - alert: ModelPerformanceDegraded
        expr: 'sensor_anomaly_rate > 0.3 or sensor_anomaly_rate < 0.001'
        for: 30m
        labels: { severity: critical }
```

---

## 4. Grafana Dashboards

Grafana dashboards are located in `grafana_dashboards/` and auto-loaded via `docker/grafana-provisioning/dashboards/dashboards.yml`:
1. **`sensor-overview.json` (Real-Time Sensor Metrics):** Throughput, p50/p95/p99 latency, anomaly rate, data freshness.
2. **`anomaly-detection.json` (Anomaly Detection):** Detections by model version, anomaly rate, quality failures.
3. **`model-performance.json` (Model Performance):** Drift indicator, active model info, forecast error.
4. **`system-health.json` (System Health):** Service uptime, Kafka lag, producer error rates.

