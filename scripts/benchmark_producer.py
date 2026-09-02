"""Kafka producer throughput benchmark.

Measures message production rate at various target throughputs.
Requires a running Kafka broker.

Run with: python scripts/benchmark_producer.py [--bootstrap-servers HOST:PORT]
"""
import argparse
import json
import sys
import time

sys.path.insert(0, ".")


def benchmark_producer(bootstrap_servers: str, n_messages: int = 10000) -> dict:
    """Benchmark Kafka producer throughput."""
    try:
        from confluent_kafka import Producer
    except ImportError:
        return {"error": "confluent-kafka not installed"}

    producer = Producer({
        "bootstrap.servers": bootstrap_servers,
        "enable.idempotence": True,
        "acks": "all",
        "linger.ms": 10,
        "batch.num.messages": 1000,
        "compression.type": "lz4",
    })

    topic = "benchmark_test"
    delivered = 0
    errors = 0

    def on_delivery(err, msg):
        nonlocal delivered, errors
        if err:
            errors += 1
        else:
            delivered += 1

    start = time.monotonic()
    for i in range(n_messages):
        payload = json.dumps({
            "timestamp": int(time.time() * 1000),
            "sensor_id": f"sensor-{i % 120:04d}",
            "value": 70.0 + (i % 10),
            "unit": "C",
            "metadata": {"source": "benchmark"},
        }).encode()

        try:
            producer.produce(topic, value=payload, on_delivery=on_delivery)
        except BufferError:
            producer.poll(0.1)
            producer.produce(topic, value=payload, on_delivery=on_delivery)

        if i % 1000 == 0:
            producer.poll(0)

    producer.flush(30)
    elapsed = time.monotonic() - start

    return {
        "messages": n_messages,
        "delivered": delivered,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 3),
        "throughput_msg_per_sec": round(n_messages / elapsed, 1) if elapsed > 0 else 0,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap-servers", default="localhost:9092")
    p.add_argument("--messages", type=int, default=10000)
    args = p.parse_args()

    results = benchmark_producer(args.bootstrap_servers, args.messages)
    print("\n=== Producer Benchmark ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
