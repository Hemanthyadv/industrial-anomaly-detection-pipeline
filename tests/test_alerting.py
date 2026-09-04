"""Tests for alert rule export and definitions."""
import os
import tempfile
from src.alerting import RULES, export_rules_yaml


def test_alert_rules_exist():
    assert len(RULES) == 7
    for rule in RULES:
        assert "alert" in rule
        assert "expr" in rule
        assert "for" in rule
        assert "labels" in rule
        assert "annotations" in rule


def test_export_rules_yaml():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "alert_rules.yml")
        exported = export_rules_yaml(out_path)
        assert os.path.exists(exported)
        with open(exported, "r") as f:
            content = f.read()
            assert "industrial_iot_alerts" in content
            assert "SensorDataStale" in content
