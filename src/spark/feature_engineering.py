"""Streaming-compatible feature engineering for sensor data.

Uses Spark Structured Streaming windowed aggregations:
- groupBy(F.window(...), "sensor_id").agg(...) for tumbling/sliding windows
- foreachBatch for stateful lag features

This module provides both streaming-compatible and batch-compatible feature functions.
"""
try:
    from pyspark.sql import DataFrame, SparkSession, functions as F
    from pyspark.sql.types import DoubleType, StringType, StructField, StructType, TimestampType
except ImportError:
    DataFrame = SparkSession = F = DoubleType = StringType = StructField = StructType = TimestampType = None


def add_5min_tumbling_features(df: DataFrame) -> DataFrame:
    """Compute 5-minute tumbling window features per sensor.

    Features: mean, min, max, variance, stddev, skewness, kurtosis.

    Args:
        df: Streaming or batch DataFrame with columns: timestamp, sensor_id, value.

    Returns:
        Aggregated DataFrame with one row per (window, sensor_id).
    """
    return (
        df.groupBy(
            F.window("timestamp", "5 minutes"),
            "sensor_id",
        )
        .agg(
            F.mean("value").alias("rolling_mean_5m"),
            F.min("value").alias("rolling_min_5m"),
            F.max("value").alias("rolling_max_5m"),
            F.variance("value").alias("rolling_var_5m"),
            F.stddev("value").alias("rolling_stddev_5m"),
            F.skewness("value").alias("rolling_skewness_5m"),
            F.kurtosis("value").alias("rolling_kurtosis_5m"),
            F.count("value").alias("event_count_5m"),
        )
        .withColumn("window_start", F.col("window.start"))
        .withColumn("window_end", F.col("window.end"))
        .drop("window")
    )


def add_1min_tumbling_features(df: DataFrame) -> DataFrame:
    """Compute 1-minute tumbling window features per sensor.

    Used as the base granularity for lag features.

    Args:
        df: Streaming or batch DataFrame with columns: timestamp, sensor_id, value.

    Returns:
        Aggregated DataFrame with one row per (1-min window, sensor_id).
    """
    return (
        df.groupBy(
            F.window("timestamp", "1 minute"),
            "sensor_id",
        )
        .agg(
            F.mean("value").alias("value_mean_1m"),
            F.min("value").alias("value_min_1m"),
            F.max("value").alias("value_max_1m"),
            F.count("value").alias("event_count_1m"),
        )
        .withColumn("window_start", F.col("window.start"))
        .withColumn("window_end", F.col("window.end"))
        .drop("window")
    )


def add_1hour_tumbling_features(df: DataFrame) -> DataFrame:
    """Compute 1-hour tumbling window features per sensor.

    Features: mean, min, max, variance.

    Args:
        df: Streaming or batch DataFrame with columns: timestamp, sensor_id, value.

    Returns:
        Aggregated DataFrame with one row per (1-hour window, sensor_id).
    """
    return (
        df.groupBy(
            F.window("timestamp", "1 hour"),
            "sensor_id",
        )
        .agg(
            F.mean("value").alias("rolling_mean_1h"),
            F.min("value").alias("rolling_min_1h"),
            F.max("value").alias("rolling_max_1h"),
            F.variance("value").alias("rolling_var_1h"),
        )
        .withColumn("window_start", F.col("window.start"))
        .withColumn("window_end", F.col("window.end"))
        .drop("window")
    )


def add_batch_lag_features(df: DataFrame) -> DataFrame:
    """Add time-aligned lag features for batch DataFrames.

    Lag semantics: t-1, t-5, t-60 refer to the value from
    1, 5, and 60 minutes prior for the same sensor.
    This is computed using self-joins on time-aligned 1-minute windows,
    NOT row-number based lags.

    IMPORTANT: This function is for BATCH processing only.
    For streaming, use the foreachBatch approach in spark_streaming.py.

    Args:
        df: Batch DataFrame with columns: sensor_id, window_start, value_mean_1m.

    Returns:
        DataFrame with added lag_t1, lag_t5, lag_t60, rate_of_change, acceleration columns.
    """
    from pyspark.sql.window import Window

    w = Window.partitionBy("sensor_id").orderBy("window_start")

    result = df.withColumn(
        "lag_t1", F.lag("value_mean_1m", 1).over(w)
    ).withColumn(
        "lag_t5", F.lag("value_mean_1m", 5).over(w)
    ).withColumn(
        "lag_t60", F.lag("value_mean_1m", 60).over(w)
    )

    # Rate of change (first derivative): current - previous
    result = result.withColumn(
        "rate_of_change",
        F.col("value_mean_1m") - F.coalesce(F.col("lag_t1"), F.col("value_mean_1m")),
    )

    # Acceleration (second derivative): change of rate of change
    result = result.withColumn(
        "prev_rate_of_change", F.lag("rate_of_change", 1).over(w)
    ).withColumn(
        "acceleration",
        F.col("rate_of_change") - F.coalesce(F.col("prev_rate_of_change"), F.lit(0.0)),
    ).drop("prev_rate_of_change")

    return result
