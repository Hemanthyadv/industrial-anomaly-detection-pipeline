"""Spark Structured Streaming job: Kafka -> Feature Engineering -> Delta Lake.

Reads raw sensor events from Kafka, applies windowed feature engineering,
data quality checks, and writes results to Delta Lake.
"""
import os
import logging

from pyspark.sql import SparkSession, functions as F, types as T

LOG = logging.getLogger(__name__)

# Configuration via environment variables
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
KAFKA_RAW_TOPIC = os.getenv("KAFKA_RAW_TOPIC", "raw_sensor_data")
KAFKA_PROCESSED_TOPIC = os.getenv("KAFKA_PROCESSED_TOPIC", "processed_events")
KAFKA_ANOMALY_TOPIC = os.getenv("KAFKA_ANOMALY_TOPIC", "anomalies")
CHECKPOINT_BASE = os.getenv("SPARK_CHECKPOINT_DIR", "/checkpoints")
DELTA_BASE = os.getenv("DELTA_DATA_DIR", "/data/delta")
SHUFFLE_PARTITIONS = os.getenv("SPARK_SHUFFLE_PARTITIONS", "8")
WATERMARK_DELAY = os.getenv("SPARK_WATERMARK_DELAY", "10 minutes")

# JSON schema for raw sensor events from Kafka
SENSOR_SCHEMA = T.StructType(
    [
        T.StructField("timestamp", T.LongType()),  # epoch millis
        T.StructField("sensor_id", T.StringType()),
        T.StructField("value", T.DoubleType()),
        T.StructField("unit", T.StringType()),
        T.StructField(
            "metadata", T.MapType(T.StringType(), T.StringType())
        ),
    ]
)


def build_spark() -> SparkSession:
    """Build a SparkSession with Kafka and Delta Lake support."""
    return (
        SparkSession.builder.appName("IndustrialSensorFeatureEngineering")
        .config("spark.sql.shuffle.partitions", SHUFFLE_PARTITIONS)
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.streaming.schemaInference", "true")
        .getOrCreate()
    )


def run():
    """Main streaming pipeline entrypoint."""
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # --- Read from Kafka ---
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", KAFKA_RAW_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # --- Parse JSON and convert timestamp ---
    parsed = (
        raw.select(
            F.from_json(F.col("value").cast("string"), SENSOR_SCHEMA).alias("data")
        )
        .select("data.*")
        .withColumn(
            "timestamp",
            # Convert epoch millis (long) to proper TimestampType
            (F.col("timestamp") / 1000).cast(T.TimestampType()),
        )
    )

    # --- Schema validation ---
    clean = parsed.filter(
        F.col("sensor_id").isNotNull()
        & F.col("value").isNotNull()
        & F.col("timestamp").isNotNull()
        & F.col("unit").isNotNull()
    )

    # --- Add watermark for late data handling ---
    watermarked = clean.withWatermark("timestamp", WATERMARK_DELAY)

    # --- 5-minute tumbling window features ---
    from .feature_engineering import add_5min_tumbling_features, add_1hour_tumbling_features

    features_5m = add_5min_tumbling_features(watermarked)
    features_1h = add_1hour_tumbling_features(watermarked)

    # --- Write 5-min features to Delta Lake ---
    (
        features_5m.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", f"{CHECKPOINT_BASE}/features_5m")
        .option("path", f"{DELTA_BASE}/features_5m")
        .queryName("features_5m")
        .start()
    )

    # --- Write 1-hour features to Delta Lake ---
    (
        features_1h.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", f"{CHECKPOINT_BASE}/features_1h")
        .option("path", f"{DELTA_BASE}/features_1h")
        .queryName("features_1h")
        .start()
    )

    # --- Write processed events to Kafka ---
    (
        features_5m.select(
            F.col("sensor_id").alias("key"),
            F.to_json(F.struct("*")).alias("value"),
        )
        .writeStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("topic", KAFKA_PROCESSED_TOPIC)
        .option("checkpointLocation", f"{CHECKPOINT_BASE}/processed_kafka")
        .outputMode("append")
        .queryName("processed_events")
        .start()
    )

    LOG.info("Streaming queries started. Awaiting termination...")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    run()
