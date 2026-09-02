"""Tests for Avro schema validation."""
import pytest

from src.kafka.avro_validator import deserialize_event, serialize_event, validate_event


def test_valid_event_passes(sample_sensor_event):
    """A valid sensor event should pass validation."""
    assert validate_event(sample_sensor_event) is True


def test_serialization_roundtrip(sample_sensor_event):
    """Serialize and deserialize should produce the original event."""
    data = serialize_event(sample_sensor_event)
    assert isinstance(data, bytes)
    assert len(data) > 0

    result = deserialize_event(data)
    assert result["sensor_id"] == sample_sensor_event["sensor_id"]
    assert result["value"] == pytest.approx(sample_sensor_event["value"])
    assert result["unit"] == sample_sensor_event["unit"]
    assert result["timestamp"] == sample_sensor_event["timestamp"]


def test_missing_field_fails():
    """Missing required fields should fail validation."""
    bad_event = {"sensor_id": "sensor-0001", "value": 70.0}
    with pytest.raises(ValueError):
        validate_event(bad_event)


def test_wrong_type_fails():
    """Wrong field types should fail validation."""
    bad_event = {
        "timestamp": "not-a-number",  # should be long
        "sensor_id": "sensor-0001",
        "value": 70.0,
        "unit": "C",
        "metadata": {},
    }
    with pytest.raises((ValueError, TypeError)):
        validate_event(bad_event)
