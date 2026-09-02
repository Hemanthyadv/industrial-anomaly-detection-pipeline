"""Tests for the sensor simulator."""
import csv
import os
import random
import tempfile

from src.kafka.sensor_simulator import generate, generate_sensor_value, inject_anomaly, stream_events


def test_sensor_value_is_numeric():
    """generate_sensor_value returns a float for all sensor types."""
    rng = random.Random(1)
    for kind in ("temperature", "pressure", "vibration"):
        value = generate_sensor_value(1, 10, kind, rng)
        assert isinstance(value, float)


def test_temperature_range():
    """Temperature values should be roughly in [55, 85] under normal conditions."""
    rng = random.Random(42)
    values = [generate_sensor_value(0, m, "temperature", rng) for m in range(1440)]
    assert all(40 < v < 100 for v in values), f"Out of range: min={min(values)}, max={max(values)}"


def test_pressure_range():
    """Pressure values should be roughly in [1.0, 3.0] under normal conditions."""
    rng = random.Random(42)
    values = [generate_sensor_value(0, m, "pressure", rng) for m in range(1440)]
    assert all(0.5 < v < 4.0 for v in values)


def test_vibration_range():
    """Vibration values should be roughly in [0.0, 0.15] under normal conditions."""
    rng = random.Random(42)
    values = [generate_sensor_value(0, m, "vibration", rng) for m in range(1440)]
    assert all(-0.1 < v < 0.3 for v in values)


def test_anomaly_injection_spike():
    """Spike anomaly should trigger at specific conditions."""
    rng = random.Random(42)
    value = 70.0
    # minute=2500, sensor=0 -> 2500 % 2500 == 0 and 0 % 17 == 0 -> spike
    result_val, is_anomaly, atype = inject_anomaly(value, 2500, 0, "temperature", rng)
    assert is_anomaly is True
    assert atype == "spike"
    assert result_val == value * 1.8


def test_anomaly_injection_normal():
    """Normal conditions should not inject anomalies."""
    rng = random.Random(42)
    value = 70.0
    result_val, is_anomaly, atype = inject_anomaly(value, 100, 1, "temperature", rng)
    assert is_anomaly is False
    assert atype is None
    assert result_val == value


def test_generate_creates_csv():
    """generate() creates a valid CSV file with the correct columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        count = generate(days=1, sensors=3, output=path, seed=42)
        assert os.path.exists(path)
        assert count == 1440 * 3  # 1 day * 1440 min * 3 sensors

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            assert "timestamp" in headers
            assert "sensor_id" in headers
            assert "value" in headers
            assert "is_anomaly" in headers
            assert "anomaly_type" in headers
            row = next(reader)
            assert row["sensor_id"].startswith("sensor-")


def test_generate_reproducible():
    """Same seed produces same output."""
    from datetime import datetime, timezone
    fixed_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = os.path.join(tmpdir, "a.csv")
        path2 = os.path.join(tmpdir, "b.csv")
        generate(days=1, sensors=2, output=path1, seed=42, start_time=fixed_time)
        generate(days=1, sensors=2, output=path2, seed=42, start_time=fixed_time)
        with open(path1) as f1, open(path2) as f2:
            assert f1.read() == f2.read()


def test_stream_events_yields_dicts():
    """stream_events yields dicts matching the Avro schema."""
    gen = stream_events(sensors=3, seed=42)
    events = [next(gen) for _ in range(9)]  # 3 sensors * 3 iterations
    for event in events:
        assert "timestamp" in event
        assert "sensor_id" in event
        assert "value" in event
        assert "unit" in event
        assert "metadata" in event
        assert isinstance(event["timestamp"], int)
        assert isinstance(event["value"], float)
