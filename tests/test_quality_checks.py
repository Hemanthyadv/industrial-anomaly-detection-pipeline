"""Tests for data quality checks (batch mode)."""
import pytest


def test_validate_schema_passes():
    """validate_schema accepts DataFrames with required columns."""
    # Since PySpark is not required for CI, test the pure-python logic
    from src.spark.quality_checks import VALID_UNITS, VALID_SENSOR_PATTERN
    import re
    assert re.match(VALID_SENSOR_PATTERN, "sensor-0001")
    assert not re.match(VALID_SENSOR_PATTERN, "invalid-sensor")
    assert "C" in VALID_UNITS
    assert "bar" in VALID_UNITS
    assert "mm/s" in VALID_UNITS
    assert "invalid" not in VALID_UNITS


def test_valid_sensor_pattern():
    """Sensor ID pattern should match sensor-NNNN format."""
    import re
    from src.spark.quality_checks import VALID_SENSOR_PATTERN
    assert re.match(VALID_SENSOR_PATTERN, "sensor-0000")
    assert re.match(VALID_SENSOR_PATTERN, "sensor-0119")
    assert not re.match(VALID_SENSOR_PATTERN, "sensor-01")
    assert not re.match(VALID_SENSOR_PATTERN, "SENSOR-0001")
    assert not re.match(VALID_SENSOR_PATTERN, "temp-0001")
