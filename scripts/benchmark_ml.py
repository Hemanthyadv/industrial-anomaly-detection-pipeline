"""ML model evaluation benchmark.

Measures anomaly detection and forecasting model performance.

Run with: python scripts/benchmark_ml.py [--data PATH] [--output-dir DIR]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")


def benchmark_ml(data_path: str = "data/sample_sensor_data.csv", output_dir: str = "models") -> dict:
    """Run ML benchmark: train, evaluate, measure inference speed."""
    from src.ml.train_models import train_all, make_dataset, load_training_data
    import joblib

    # Load data
    try:
        X, y, _ = load_training_data(data_path)
    except Exception:
        X, y = make_dataset()

    # Train
    start = time.monotonic()
    results = train_all(output_dir=output_dir)
    train_time = time.monotonic() - start

    # Inference speed
    model_path = Path(output_dir) / "isolation_forest.joblib"
    inference_times = []
    if model_path.exists():
        model = joblib.load(model_path)
        x_single = X[:1]
        for _ in range(1000):
            t0 = time.perf_counter()
            model.predict(x_single)
            inference_times.append(time.perf_counter() - t0)

    benchmark = {
        "training_time_seconds": round(train_time, 2),
        "model_metrics": {
            name: {k: round(v, 4) if isinstance(v, float) else v
                   for k, v in metrics.items() if isinstance(v, (int, float))}
            for name, metrics in results.items()
        },
    }

    if inference_times:
        inference_times.sort()
        n = len(inference_times)
        benchmark["inference_latency"] = {
            "p50_ms": round(inference_times[int(n * 0.50)] * 1000, 4),
            "p95_ms": round(inference_times[int(n * 0.95)] * 1000, 4),
            "p99_ms": round(inference_times[int(n * 0.99)] * 1000, 4),
        }

    return benchmark


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sample_sensor_data.csv")
    p.add_argument("--output-dir", default="models")
    args = p.parse_args()

    results = benchmark_ml(args.data, args.output_dir)
    print("\n=== ML Benchmark ===")
    print(json.dumps(results, indent=2))
