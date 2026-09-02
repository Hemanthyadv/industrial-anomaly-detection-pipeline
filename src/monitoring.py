"""Prometheus metric definitions for the entire pipeline.

All metrics are defined here as a single source of truth.
Components import and update these metrics.
"""
from prometheus_client import Counter, Gauge, Histogram, Info

# --- Producer metrics ---
MESSAGES_PRODUCED = Counter(
    "sensor_messages_produced_total",
    "Total sensor messages produced to Kafka",
    ["topic"],
)
PRODUCER_ERRORS = Counter(
    "sensor_producer_errors_total",
    "Producer errors (validation, delivery failures)",
    ["error_type"],
)
PRODUCER_RETRIES = Counter(
    "sensor_producer_retries_total",
    "Producer retry attempts",
)

# --- Consumer / processing metrics ---
MESSAGES_PROCESSED = Counter(
    "sensor_messages_processed_total",
    "Sensor messages processed by the streaming pipeline",
)
KAFKA_LAG = Gauge(
    "kafka_consumer_lag",
    "Kafka consumer group lag (messages behind)",
    ["topic", "partition"],
)
DATA_FRESHNESS = Gauge(
    "data_freshness_seconds",
    "Seconds since the most recent event was ingested",
)

# --- Data quality metrics ---
NULL_RATE = Gauge(
    "data_quality_null_rate",
    "Fraction of records with null values in the current window",
    ["field"],
)
OUTLIER_RATE = Gauge(
    "data_quality_outlier_rate",
    "Fraction of records flagged as statistical outliers",
)
QUALITY_FAILURES = Counter(
    "data_quality_failures_total",
    "Total data quality check failures",
    ["check_type"],
)

# --- Anomaly detection metrics ---
ANOMALIES_DETECTED = Counter(
    "sensor_anomalies_detected_total",
    "Anomalies detected by the ML model",
    ["model_version"],
)
ANOMALY_RATE = Gauge(
    "sensor_anomaly_rate",
    "Current anomaly rate (fraction of recent predictions that are anomalous)",
)

# --- Inference API metrics ---
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total prediction requests to the API",
    ["endpoint"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0),
)

# --- Model metrics ---
MODEL_INFO = Info(
    "ml_model",
    "Current loaded model information",
)
MODEL_DRIFT = Gauge(
    "model_drift_score",
    "Model drift indicator (higher = more drift from training distribution)",
)
FORECAST_ERROR = Gauge(
    "forecast_error_metric",
    "Forecast error (MAE) when ground truth becomes available",
    ["sensor_type"],
)
