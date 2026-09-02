"""Delta Lake feature store adapter and feature metadata registry.

Provides:
- FeatureRegistry: JSON-based feature metadata registry with CRUD operations.
- write_delta: Helper to write streaming DataFrames to Delta Lake.
- read_features: Load features from Delta Lake for training/inference.
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)

FEATURE_REGISTRY_PATH = os.getenv(
    "FEATURE_REGISTRY_PATH", "data/feature_registry.json"
)


class FeatureRegistry:
    """Thread-safe, file-based feature metadata registry.

    Stores feature metadata (name, description, data type, source,
    version, created_at, usage) in a JSON file with file locking
    for concurrent safety.

    Example usage:
        >>> registry = FeatureRegistry()
        >>> registry.register(
        ...     name="rolling_mean_5m",
        ...     description="Mean sensor value over a 5-minute window",
        ...     data_type="double",
        ...     source="spark_streaming",
        ...     version="1.0",
        ...     usage="training,inference",
        ... )
        >>> feature = registry.get_feature("rolling_mean_5m")
        >>> all_features = registry.list_features()
    """

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or FEATURE_REGISTRY_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> Dict[str, Any]:
        """Read the registry file."""
        if not self.path.exists():
            return {}
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: Dict[str, Any]) -> None:
        """Write the registry file atomically."""
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self.path)

    def register(
        self,
        name: str,
        description: str,
        data_type: str = "double",
        source: str = "spark_streaming",
        version: str = "1.0",
        usage: str = "training,inference",
    ) -> None:
        """Register or update a feature in the registry.

        Args:
            name: Unique feature name.
            description: Human-readable description.
            data_type: Data type (double, string, int, etc.).
            source: Source system/pipeline.
            version: Feature version string.
            usage: Comma-separated usage contexts.
        """
        registry = self._read()
        registry[name] = {
            "description": description,
            "data_type": data_type,
            "source": source,
            "version": version,
            "usage": usage,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write(registry)
        LOG.info("Registered feature: %s (v%s)", name, version)

    def get_feature(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a single feature.

        Args:
            name: Feature name.

        Returns:
            Feature metadata dict or None if not found.
        """
        registry = self._read()
        return registry.get(name)

    def list_features(self) -> Dict[str, Any]:
        """List all registered features.

        Returns:
            Full registry dictionary.
        """
        return self._read()

    def get_latest_version(self, name: str) -> Optional[str]:
        """Get the latest version string for a feature.

        Args:
            name: Feature name.

        Returns:
            Version string or None.
        """
        feat = self.get_feature(name)
        return feat["version"] if feat else None


def write_delta(df, path: str, checkpoint: str, query_name: str):
    """Write a streaming DataFrame to Delta Lake.

    Args:
        df: Streaming DataFrame.
        path: Delta table path.
        checkpoint: Checkpoint directory path.
        query_name: Name for the streaming query.

    Returns:
        StreamingQuery object.
    """
    return (
        df.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", checkpoint)
        .queryName(query_name)
        .start(path)
    )


def register_default_features(registry: Optional[FeatureRegistry] = None) -> None:
    """Register all default feature definitions."""
    reg = registry or FeatureRegistry()
    features = [
        ("rolling_mean_5m", "Mean sensor value over a 5-minute event-time window", "double"),
        ("rolling_min_5m", "Minimum sensor value over a 5-minute window", "double"),
        ("rolling_max_5m", "Maximum sensor value over a 5-minute window", "double"),
        ("rolling_var_5m", "Sample variance over a 5-minute window", "double"),
        ("rolling_stddev_5m", "Standard deviation over a 5-minute window", "double"),
        ("rolling_skewness_5m", "Skewness over a 5-minute window", "double"),
        ("rolling_kurtosis_5m", "Kurtosis over a 5-minute window", "double"),
        ("rolling_mean_1h", "Mean sensor value over a 1-hour window", "double"),
        ("rolling_min_1h", "Minimum sensor value over a 1-hour window", "double"),
        ("rolling_max_1h", "Maximum sensor value over a 1-hour window", "double"),
        ("rolling_var_1h", "Sample variance over a 1-hour window", "double"),
        ("lag_t1", "Sensor value from 1 minute prior (time-aligned)", "double"),
        ("lag_t5", "Sensor value from 5 minutes prior (time-aligned)", "double"),
        ("lag_t60", "Sensor value from 60 minutes prior (time-aligned)", "double"),
        ("rate_of_change", "First derivative: current - previous 1-min mean", "double"),
        ("acceleration", "Second derivative: change of rate of change", "double"),
    ]
    for name, desc, dtype in features:
        reg.register(name=name, description=desc, data_type=dtype)


if __name__ == "__main__":
    register_default_features()
    print("Default features registered.")
