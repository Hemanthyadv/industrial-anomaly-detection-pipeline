"""Deterministic-capable industrial sensor data generator.

Supports:
- CSV batch generation for historical training data (7+ days)
- Streaming generator for live Kafka production
- Multiple anomaly patterns: spike, drift, freeze, dropout
- Configurable sensor count, rate, and random seed
"""
import argparse
import csv
import math
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Generator, Optional

UNITS = {"temperature": "C", "pressure": "bar", "vibration": "mm/s"}
SENSOR_COUNT = int(os.getenv("SENSOR_COUNT", "120"))


def generate_sensor_value(
    sensor_index: int, minute: int, kind: str, rng: random.Random
) -> float:
    """Generate a realistic sensor reading with seasonal variation and noise.

    Args:
        sensor_index: Index of the sensor (determines phase offset).
        minute: Minute offset from simulation start.
        kind: Sensor type - 'temperature', 'pressure', or 'vibration'.
        rng: Seeded random number generator for reproducibility.

    Returns:
        Simulated sensor value as float.
    """
    phase = sensor_index * 0.17
    seasonal = math.sin((minute / 1440) * 2 * math.pi + phase)
    noise = rng.gauss(
        0, {"temperature": 1.0, "pressure": 0.08, "vibration": 0.01}[kind]
    )
    baseline = {"temperature": 70, "pressure": 2.0, "vibration": 0.05}[kind]
    amplitude = {"temperature": 5, "pressure": 0.25, "vibration": 0.02}[kind]
    return baseline + amplitude * seasonal + noise


def inject_anomaly(
    value: float, minute: int, sensor_index: int, kind: str, rng: random.Random
) -> tuple:
    """Inject synthetic anomalies for model evaluation.

    Anomaly patterns:
    - Spike: 1.8x multiplicative spike (periodic)
    - Drift: Gradual offset increasing over time
    - Freeze: Value stuck at a constant
    - Noise burst: Large random noise addition

    Args:
        value: Original sensor value.
        minute: Current minute in simulation.
        sensor_index: Sensor index.
        kind: Sensor type.
        rng: Random generator.

    Returns:
        Tuple of (modified_value, is_anomaly: bool, anomaly_type: str or None).
    """
    # Spike anomaly: periodic, deterministic
    if minute % 2500 == 0 and sensor_index % 17 == 0:
        return value * 1.8, True, "spike"

    # Drift anomaly: slow drift over a window
    if 5000 <= minute <= 5100 and sensor_index % 23 == 0:
        drift_factor = (minute - 5000) / 50.0
        baseline = {"temperature": 70, "pressure": 2.0, "vibration": 0.05}[kind]
        return value + drift_factor * baseline * 0.15, True, "drift"

    # Freeze anomaly: stuck value for a window
    if 7200 <= minute <= 7210 and sensor_index % 19 == 0:
        baseline = {"temperature": 70, "pressure": 2.0, "vibration": 0.05}[kind]
        return baseline, True, "freeze"

    # Noise burst: random large deviation
    if minute % 3600 == 1800 and sensor_index % 31 == 0:
        noise_scale = {"temperature": 15.0, "pressure": 1.0, "vibration": 0.15}[kind]
        return value + rng.gauss(0, noise_scale), True, "noise_burst"

    return value, False, None


def generate(
    days: int = 7,
    sensors: int = 120,
    output: str = "data/sample_sensor_data.csv",
    seed: int = 42,
    start_time: Optional[datetime] = None,
) -> int:
    """Generate a historical CSV dataset for training.

    Args:
        days: Number of days to simulate.
        sensors: Number of sensors.
        output: Output CSV file path.
        seed: Random seed for reproducibility.
        start_time: Optional base timestamp for deterministic outputs.

    Returns:
        Total number of rows generated.
    """
    rng = random.Random(seed)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = start_time or datetime.now(timezone.utc)
    start = base - timedelta(days=days)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "sensor_id",
                "value",
                "unit",
                "sensor_type",
                "site",
                "is_anomaly",
                "anomaly_type",
            ]
        )
        for minute in range(days * 1440):
            ts = start + timedelta(minutes=minute)
            for sensor in range(sensors):
                kind = ("temperature", "pressure", "vibration")[sensor % 3]
                value = generate_sensor_value(sensor, minute, kind, rng)
                value, is_anomaly, anomaly_type = inject_anomaly(
                    value, minute, sensor, kind, rng
                )
                writer.writerow(
                    [
                        ts.isoformat(),
                        f"sensor-{sensor:04d}",
                        round(value, 5),
                        UNITS[kind],
                        kind,
                        f"site-{sensor % 4:02d}",
                        int(is_anomaly),
                        anomaly_type or "",
                    ]
                )
                count += 1
    return count


def stream_events(
    sensors: int = 120,
    seed: int = 42,
    rate_per_sec: int = 1000,
) -> Generator[Dict, None, None]:
    """Generate sensor events as dicts for streaming to Kafka.

    Each event matches the Avro schema: timestamp (epoch ms), sensor_id, value, unit, metadata.

    Args:
        sensors: Number of sensors to simulate.
        seed: Random seed.
        rate_per_sec: Target messages per second (used for pacing externally).

    Yields:
        Sensor event dictionaries.
    """
    rng = random.Random(seed)
    minute = 0
    while True:
        ts = datetime.now(timezone.utc)
        for sensor in range(sensors):
            kind = ("temperature", "pressure", "vibration")[sensor % 3]
            value = generate_sensor_value(sensor, minute, kind, rng)
            value, is_anomaly, anomaly_type = inject_anomaly(
                value, minute, sensor, kind, rng
            )
            yield {
                "timestamp": int(ts.timestamp() * 1000),
                "sensor_id": f"sensor-{sensor:04d}",
                "value": round(value, 5),
                "unit": UNITS[kind],
                "metadata": {
                    "site": f"site-{sensor % 4:02d}",
                    "sensor_type": kind,
                    "source": "simulator",
                    "is_anomaly": str(int(is_anomaly)),
                    "anomaly_type": anomaly_type or "",
                },
            }
        minute += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate industrial sensor data")
    p.add_argument("--days", type=int, default=7, help="Days of data to generate")
    p.add_argument("--sensors", type=int, default=120, help="Number of sensors")
    p.add_argument("--output", default="data/sample_sensor_data.csv", help="Output CSV path")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = p.parse_args()
    print(f"Generated {generate(args.days, args.sensors, args.output, args.seed):,} rows")
