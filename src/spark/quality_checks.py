"""Streaming-compatible data quality checks.

Provides validation, null detection, outlier detection, and quarantine
functionality for the sensor data pipeline.
"""
import logging
from typing import Dict, Set

try:
    from pyspark.sql import DataFrame, functions as F
except ImportError:
    DataFrame = None
    F = None

LOG = logging.getLogger(__name__)

# Valid sensor ID pattern: sensor-NNNN
VALID_SENSOR_PATTERN = r"^sensor-\d{4}$"
VALID_UNITS: Set[str] = {"C", "bar", "mm/s"}


def validate_schema(df: DataFrame) -> DataFrame:
    """Validate that required columns exist with correct names.

    Args:
        df: Input DataFrame.

    Returns:
        The input DataFrame (unchanged) if valid.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"timestamp", "sensor_id", "value", "unit"}
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def add_quality_flags(df: DataFrame) -> DataFrame:
    """Add data quality flag columns to a DataFrame.

    Flags added:
    - is_null_value: value is null
    - is_null_sensor_id: sensor_id is null
    - is_invalid_sensor_id: sensor_id doesn't match expected pattern
    - is_invalid_unit: unit not in valid set
    - is_timestamp_invalid: timestamp is null or in the far future
    - is_quality_failure: any quality check failed

    Args:
        df: DataFrame with sensor columns.

    Returns:
        DataFrame with quality flag columns added.
    """
    valid_units_col = F.array(*[F.lit(u) for u in VALID_UNITS])

    result = (
        df.withColumn("is_null_value", F.col("value").isNull())
        .withColumn("is_null_sensor_id", F.col("sensor_id").isNull())
        .withColumn(
            "is_invalid_sensor_id",
            ~F.col("sensor_id").rlike(VALID_SENSOR_PATTERN),
        )
        .withColumn(
            "is_invalid_unit",
            ~F.array_contains(valid_units_col, F.col("unit")),
        )
        .withColumn(
            "is_timestamp_invalid",
            F.col("timestamp").isNull()
            | (F.col("timestamp") > F.current_timestamp() + F.expr("INTERVAL 1 HOUR")),
        )
    )

    # Composite flag: any check failed
    result = result.withColumn(
        "is_quality_failure",
        F.col("is_null_value")
        | F.col("is_null_sensor_id")
        | F.col("is_invalid_sensor_id")
        | F.col("is_invalid_unit")
        | F.col("is_timestamp_invalid"),
    )

    return result


def split_good_bad(df: DataFrame):
    """Split DataFrame into good records and quarantined bad records.

    Args:
        df: DataFrame with is_quality_failure column.

    Returns:
        Tuple of (good_df, bad_df).
    """
    flagged = add_quality_flags(df)
    good = flagged.filter(~F.col("is_quality_failure"))
    bad = flagged.filter(F.col("is_quality_failure"))
    return good, bad


def compute_quality_metrics_batch(df: DataFrame) -> Dict[str, float]:
    """Compute quality metrics for a batch of data.

    Used inside foreachBatch for streaming quality monitoring.

    Args:
        df: Batch DataFrame.

    Returns:
        Dictionary of quality metric values.
    """
    total = df.count()
    if total == 0:
        return {"total": 0, "null_rate": 0.0, "invalid_rate": 0.0}

    null_count = df.filter(F.col("value").isNull()).count()
    invalid_sensor = df.filter(~df["sensor_id"].rlike(VALID_SENSOR_PATTERN)).count()

    return {
        "total": total,
        "null_rate": null_count / total,
        "invalid_sensor_rate": invalid_sensor / total,
    }


def add_outlier_flag_batch(df: DataFrame, sigma_threshold: float = 3.0) -> DataFrame:
    """Flag statistical outliers in a batch DataFrame.

    Uses per-sensor mean/stddev computed on the batch.
    This function is for BATCH or foreachBatch use only.

    Args:
        df: DataFrame with sensor_id and value columns.
        sigma_threshold: Number of standard deviations for outlier threshold.

    Returns:
        DataFrame with is_statistical_outlier column added.
    """
    stats = df.groupBy("sensor_id").agg(
        F.avg("value").alias("_mu"),
        F.stddev("value").alias("_sigma"),
    )
    result = (
        df.join(stats, "sensor_id")
        .withColumn(
            "is_statistical_outlier",
            F.abs(F.col("value") - F.col("_mu"))
            > sigma_threshold * F.coalesce(F.col("_sigma"), F.lit(0.0)),
        )
        .drop("_mu", "_sigma")
    )
    return result
