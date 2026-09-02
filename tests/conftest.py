"""Shared test fixtures for the Industrial IoT platform."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_sensor_event():
    """A valid sensor event matching the Avro schema."""
    return {
        "timestamp": 1693000000000,
        "sensor_id": "sensor-0001",
        "value": 72.5,
        "unit": "C",
        "metadata": {"site": "site-00", "sensor_type": "temperature", "source": "simulator"},
    }


@pytest.fixture
def sample_anomaly_event():
    """A sensor event with anomalous values."""
    return {
        "timestamp": 1693000000000,
        "sensor_id": "sensor-0017",
        "value": 126.0,  # temperature spike
        "unit": "C",
        "metadata": {"site": "site-01", "sensor_type": "temperature", "source": "simulator", "is_anomaly": "1"},
    }


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_csv(temp_dir):
    """Generate a small sample CSV for testing."""
    from src.kafka.sensor_simulator import generate
    csv_path = os.path.join(temp_dir, "test_data.csv")
    generate(days=1, sensors=6, output=csv_path, seed=42)
    return csv_path
