"""Kafka producer with Avro validation, batching, retries, bounded queue, backpressure, and metrics."""
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Iterable, Optional

from confluent_kafka import KafkaError, KafkaException, Producer

LOG = logging.getLogger("sensor-producer")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

TOPIC = os.getenv("KAFKA_RAW_TOPIC", "raw_sensor_data")
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
PRODUCER_RATE = int(os.getenv("PRODUCER_RATE", "1000"))
SENSOR_COUNT = int(os.getenv("SENSOR_COUNT", "120"))

# Producer metrics (updated in-process, exposed via monitoring module when available)
_metrics = {
    "sent": 0,
    "errors": 0,
    "retries": 0,
    "delivery_failures": 0,
}


def delivery_report(err, msg):
    """Callback invoked per message delivery or failure."""
    if err:
        _metrics["delivery_failures"] += 1
        LOG.error("delivery failed topic=%s partition=%s error=%s", msg.topic(), msg.partition(), err)
    else:
        LOG.debug("delivered topic=%s partition=%s offset=%s", msg.topic(), msg.partition(), msg.offset())


def create_producer() -> Producer:
    """Create a Kafka producer with idempotent, batching, and retry configuration."""
    return Producer(
        {
            "bootstrap.servers": BOOTSTRAP,
            "enable.idempotence": True,
            "acks": "all",
            "retries": 10,
            "retry.backoff.ms": 100,
            "retry.backoff.max.ms": 5000,
            "linger.ms": 10,
            "batch.num.messages": 1000,
            "queue.buffering.max.messages": 100000,
            "queue.buffering.max.kbytes": 1048576,
            "compression.type": "lz4",
            "message.send.max.retries": 10,
        }
    )


def produce_events(events: Iterable[dict], max_inflight: int = 50000) -> int:
    """Produce a batch of events to Kafka with backpressure handling.

    Args:
        events: Iterable of sensor event dictionaries.
        max_inflight: Maximum in-flight messages before applying backpressure.

    Returns:
        Number of messages successfully queued.
    """
    from .avro_validator import validate_event

    producer = create_producer()
    sent = 0
    for event in events:
        # Validate against Avro schema
        try:
            validate_event(event)
        except ValueError as exc:
            _metrics["errors"] += 1
            LOG.warning("Schema validation failed, skipping event: %s", exc)
            continue

        payload = json.dumps(event, separators=(",", ":")).encode()
        key = event.get("sensor_id", "").encode()

        # Backpressure: wait if queue is full
        while True:
            try:
                producer.produce(
                    TOPIC,
                    value=payload,
                    key=key,
                    on_delivery=delivery_report,
                )
                break
            except BufferError:
                LOG.debug("Producer queue full, applying backpressure")
                producer.poll(0.1)
                time.sleep(0.01)

        sent += 1
        _metrics["sent"] = sent

        # Poll for delivery callbacks periodically
        if sent % 1000 == 0:
            producer.poll(0)

    remaining = producer.flush(30)
    if remaining > 0:
        LOG.warning("flush timeout: %d messages still in queue", remaining)
    LOG.info("produced=%d delivery_failures=%d", sent, _metrics["delivery_failures"])
    return sent


def get_metrics() -> dict:
    """Return current producer metrics."""
    return dict(_metrics)


def main():
    """Run continuous streaming producer at configurable rate."""
    from .sensor_simulator import stream_events

    running = True

    def stop(*_):
        nonlocal running
        running = False
        LOG.info("Shutdown signal received, draining...")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    LOG.info(
        "Starting producer: bootstrap=%s topic=%s rate=%d sensors=%d",
        BOOTSTRAP, TOPIC, PRODUCER_RATE, SENSOR_COUNT,
    )

    producer = create_producer()
    gen = stream_events(sensors=SENSOR_COUNT, seed=7)
    sent = 0
    batch_start = time.monotonic()
    batch_count = 0

    from .avro_validator import validate_event

    while running:
        try:
            event = next(gen)
        except StopIteration:
            break

        try:
            validate_event(event)
        except ValueError:
            _metrics["errors"] += 1
            continue

        payload = json.dumps(event, separators=(",", ":")).encode()
        key = event.get("sensor_id", "").encode()

        while True:
            try:
                producer.produce(
                    TOPIC, value=payload, key=key, on_delivery=delivery_report
                )
                break
            except BufferError:
                producer.poll(0.1)
                time.sleep(0.01)

        sent += 1
        batch_count += 1
        _metrics["sent"] = sent

        if sent % 1000 == 0:
            producer.poll(0)

        # Rate limiting
        if batch_count >= PRODUCER_RATE:
            elapsed = time.monotonic() - batch_start
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
            LOG.info("rate=%d/s total=%d errors=%d", batch_count, sent, _metrics["errors"])
            batch_start = time.monotonic()
            batch_count = 0

    remaining = producer.flush(30)
    LOG.info("Shutdown complete: produced=%d remaining=%d", sent, remaining)


if __name__ == "__main__":
    main()
