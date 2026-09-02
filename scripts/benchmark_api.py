"""API latency benchmark script.

Measures p50, p95, p99 prediction latency.
Run with: python scripts/benchmark_api.py [--url URL] [--requests N]
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request


def benchmark(url: str, n_requests: int = 1000) -> dict:
    """Run benchmark against the prediction API."""
    payload = json.dumps({
        "sensor_id": "sensor-0001",
        "value": 72.5,
        "unit": "C",
        "pressure": 2.1,
        "vibration": 0.04,
    }).encode()

    latencies = []
    errors = 0

    print(f"Running {n_requests} requests against {url}...")
    for i in range(n_requests):
        start = time.perf_counter()
        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
            latencies.append(time.perf_counter() - start)
        except Exception:
            errors += 1
            latencies.append(time.perf_counter() - start)

        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_requests}")

    latencies.sort()
    n = len(latencies)

    results = {
        "total_requests": n_requests,
        "successful": n_requests - errors,
        "errors": errors,
        "p50_ms": round(latencies[int(n * 0.50)] * 1000, 2) if latencies else 0,
        "p95_ms": round(latencies[int(n * 0.95)] * 1000, 2) if latencies else 0,
        "p99_ms": round(latencies[int(n * 0.99)] * 1000, 2) if latencies else 0,
        "mean_ms": round(statistics.mean(latencies) * 1000, 2) if latencies else 0,
        "min_ms": round(min(latencies) * 1000, 2) if latencies else 0,
        "max_ms": round(max(latencies) * 1000, 2) if latencies else 0,
    }
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Benchmark the prediction API")
    p.add_argument("--url", default="http://localhost:8000/predict")
    p.add_argument("--requests", type=int, default=1000)
    args = p.parse_args()

    results = benchmark(args.url, args.requests)
    print("\n=== Benchmark Results ===")
    for key, value in results.items():
        print(f"  {key}: {value}")
