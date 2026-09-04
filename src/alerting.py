"""Prometheus alert rule definitions.

This module defines alert rules as Python data structures and provides
a function to export them as Prometheus-compatible YAML.
"""
import yaml
from pathlib import Path
from typing import List, Dict, Any


RULES: List[Dict[str, Any]] = [
    {
        "alert": "SensorDataStale",
        "expr": "data_freshness_seconds > 300",
        "for": "2m",
        "labels": {"severity": "critical"},
        "annotations": {
            "summary": "No sensor data received for over 5 minutes",
            "description": "data_freshness_seconds is {{ $value }}s, exceeding the 300s threshold.",
        },
    },
    {
        "alert": "KafkaLagHigh",
        "expr": "sum(kafka_consumer_lag) > 10000",
        "for": "5m",
        "labels": {"severity": "warning"},
        "annotations": {
            "summary": "Kafka consumer lag is too high",
            "description": "Total consumer lag is {{ $value }}, exceeding 10000 messages.",
        },
    },
    {
        "alert": "APIPredictionLatencyHigh",
        "expr": 'histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m])) > 0.1',
        "for": "5m",
        "labels": {"severity": "warning"},
        "annotations": {
            "summary": "API p95 prediction latency exceeds 100ms",
            "description": "p95 latency is {{ $value }}s.",
        },
    },
    {
        "alert": "AnomalyRateAbnormal",
        "expr": "sensor_anomaly_rate > 0.15",
        "for": "10m",
        "labels": {"severity": "warning"},
        "annotations": {
            "summary": "Anomaly rate is abnormally high",
            "description": "Anomaly rate is {{ $value }}, exceeding 15%.",
        },
    },
    {
        "alert": "DataQualityFailure",
        "expr": "rate(data_quality_failures_total[5m]) > 0.1",
        "for": "5m",
        "labels": {"severity": "critical"},
        "annotations": {
            "summary": "Data quality check failures detected",
            "description": "Data quality failure rate is {{ $value }}/s.",
        },
    },
    {
        "alert": "ModelDriftDetected",
        "expr": "model_drift_score > 0.5",
        "for": "15m",
        "labels": {"severity": "warning"},
        "annotations": {
            "summary": "Model drift score exceeds threshold",
            "description": "Drift score is {{ $value }}.",
        },
    },
    {
        "alert": "ModelPerformanceDegraded",
        "expr": "sensor_anomaly_rate > 0.3 or sensor_anomaly_rate < 0.001",
        "for": "30m",
        "labels": {"severity": "critical"},
        "annotations": {
            "summary": "Model performance may have degraded",
            "description": "Anomaly rate ({{ $value }}) is outside expected operating range.",
        },
    },
]


def export_rules_yaml(output_path: str = "docker/alert_rules.yml") -> str:
    """Export alert rules as a Prometheus-compatible YAML file.

    Args:
        output_path: File path to write the YAML rules.

    Returns:
        The output file path.
    """
    rules_doc = {
        "groups": [
            {
                "name": "industrial_iot_alerts",
                "rules": RULES,
            }
        ]
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(rules_doc, f, default_flow_style=False, sort_keys=False)
    return str(path)


if __name__ == "__main__":
    out = export_rules_yaml()
    print(f"Alert rules exported to {out}")
