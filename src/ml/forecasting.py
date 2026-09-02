"""Time-series forecasting pipeline using Prophet.

Trains per-sensor forecasting models on historical data and provides
predictions with confidence intervals.

Design decisions:
- Prophet over LSTM: handles seasonality natively, provides confidence intervals,
  doesn't require GPU, simpler to deploy.
- Per-sensor training for sensor-specific patterns.
- Baseline comparison using naive last-value forecast.
"""
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
try:
    import mlflow
except ImportError:
    mlflow = None
import numpy as np
import pandas as pd

from .model_evaluation import evaluate_forecast

LOG = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def prepare_prophet_data(
    df: pd.DataFrame,
    sensor_id: str,
    value_col: str = "value",
) -> pd.DataFrame:
    """Prepare data for Prophet: requires 'ds' and 'y' columns.

    Args:
        df: Raw sensor DataFrame with timestamp and value columns.
        sensor_id: Sensor to filter for.
        value_col: Column name for the sensor value.

    Returns:
        Prophet-formatted DataFrame with 'ds' and 'y' columns.
    """
    sensor_data = df[df["sensor_id"] == sensor_id].copy()
    sensor_data["ds"] = pd.to_datetime(sensor_data["timestamp"])
    sensor_data["y"] = sensor_data[value_col].astype(float)
    sensor_data = sensor_data[["ds", "y"]].sort_values("ds").reset_index(drop=True)
    return sensor_data


def train_forecast_model(
    df: pd.DataFrame,
    sensor_id: str,
    train_fraction: float = 0.8,
    forecast_periods: int = 1440,  # 1 day at 1-min frequency
) -> Dict:
    """Train a Prophet model for a single sensor.

    Uses temporal train/validation split to avoid data leakage.

    Args:
        df: Full sensor DataFrame.
        sensor_id: Sensor to train on.
        train_fraction: Fraction of data for training (rest for validation).
        forecast_periods: Number of periods to forecast.

    Returns:
        Dictionary with model, metrics, forecast DataFrame, and baseline metrics.
    """
    try:
        from prophet import Prophet
    except ImportError:
        LOG.warning("Prophet not installed. Install with: pip install prophet")
        return {"error": "Prophet not installed", "sensor_id": sensor_id}

    prophet_data = prepare_prophet_data(df, sensor_id)

    if len(prophet_data) < 100:
        LOG.warning("Insufficient data for sensor %s (%d rows)", sensor_id, len(prophet_data))
        return {"error": "Insufficient data", "sensor_id": sensor_id}

    # Temporal split (no leakage)
    split_idx = int(len(prophet_data) * train_fraction)
    train_data = prophet_data.iloc[:split_idx]
    val_data = prophet_data.iloc[split_idx:]

    LOG.info(
        "Training Prophet for %s: train=%d, val=%d",
        sensor_id, len(train_data), len(val_data),
    )

    # Train Prophet
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05,
    )
    model.fit(train_data)

    # Forecast on validation period
    future = model.make_future_dataframe(periods=len(val_data), freq="min")
    forecast = model.predict(future)

    # Evaluate on validation set
    val_forecast = forecast.iloc[split_idx:]
    if len(val_forecast) > len(val_data):
        val_forecast = val_forecast.iloc[:len(val_data)]

    actual = val_data["y"].values[:len(val_forecast)]
    predicted = val_forecast["yhat"].values[:len(actual)]

    metrics = evaluate_forecast(actual, predicted)

    # Baseline: naive last-value forecast
    baseline_pred = np.roll(actual, 1)
    baseline_pred[0] = actual[0]
    baseline_metrics = evaluate_forecast(actual, baseline_pred)

    LOG.info(
        "Sensor %s - Prophet MAE: %.4f, RMSE: %.4f | Baseline MAE: %.4f, RMSE: %.4f",
        sensor_id,
        metrics["mae"],
        metrics["rmse"],
        baseline_metrics["mae"],
        baseline_metrics["rmse"],
    )

    return {
        "sensor_id": sensor_id,
        "model": model,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "forecast": forecast,
        "train_size": len(train_data),
        "val_size": len(val_data),
    }


def train_forecasting_pipeline(
    data_path: str = "data/sample_sensor_data.csv",
    output_dir: str = "models",
    n_sensors: int = 3,
) -> Dict:
    """Train forecast models for multiple representative sensors.

    Args:
        data_path: Path to historical CSV data.
        output_dir: Directory to save model artifacts.
        n_sensors: Number of sensors to train (one per type).

    Returns:
        Dictionary of results per sensor.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    LOG.info("Loaded %d rows from %s", len(df), data_path)

    # Pick representative sensors (one per type)
    sensor_ids = []
    for sensor_type in ["temperature", "pressure", "vibration"]:
        type_sensors = df[df["sensor_type"] == sensor_type]["sensor_id"].unique()
        if len(type_sensors) > 0:
            sensor_ids.append(type_sensors[0])

    if not sensor_ids:
        # Fallback: use first n_sensors
        sensor_ids = df["sensor_id"].unique()[:n_sensors]

    results = {}
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("industrial-forecasting")
    except Exception:
        LOG.warning("MLflow not available for forecasting")

    for sid in sensor_ids:
        with mlflow.start_run(run_name=f"prophet-{sid}"):
            result = train_forecast_model(df, sid)
            if "error" not in result:
                mlflow.log_params({"sensor_id": sid, "model_type": "Prophet"})
                mlflow.log_metrics(result["metrics"])
                mlflow.log_metrics({f"baseline_{k}": v for k, v in result["baseline_metrics"].items()})

                # Save model
                model_path = output / f"prophet_{sid}.joblib"
                joblib.dump(result["model"], model_path)
                LOG.info("Saved forecast model to %s", model_path)

            results[sid] = result

    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train forecasting models")
    p.add_argument("--data", default="data/sample_sensor_data.csv")
    p.add_argument("--output-dir", default="models")
    p.add_argument("--n-sensors", type=int, default=3)
    args = p.parse_args()
    results = train_forecasting_pipeline(args.data, args.output_dir, args.n_sensors)
    for sid, res in results.items():
        if "error" not in res:
            print(f"{sid}: MAE={res['metrics']['mae']:.4f} RMSE={res['metrics']['rmse']:.4f}")
        else:
            print(f"{sid}: {res['error']}")
