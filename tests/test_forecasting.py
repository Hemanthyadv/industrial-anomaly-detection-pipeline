"""Tests for forecasting helpers."""
import pandas as pd
from src.ml.forecasting import prepare_prophet_data


def test_prepare_prophet_data():
    data = {
        "timestamp": ["2026-01-01 00:00:00", "2026-01-01 00:01:00"],
        "sensor_id": ["sensor-0000", "sensor-0000"],
        "value": [70.1, 70.3],
    }
    df = pd.DataFrame(data)
    prophet_df = prepare_prophet_data(df, "sensor-0000")
    assert "ds" in prophet_df.columns
    assert "y" in prophet_df.columns
    assert len(prophet_df) == 2