"""Avro schema validation and serialization for sensor events."""
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import fastavro

LOG = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.avsc"
_PARSED_SCHEMA: Optional[dict] = None


def _load_schema() -> dict:
    """Load and parse the Avro schema from disk (cached)."""
    global _PARSED_SCHEMA
    if _PARSED_SCHEMA is None:
        with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
            _PARSED_SCHEMA = fastavro.parse_schema(json.load(f))
    return _PARSED_SCHEMA


def validate_event(event: Dict[str, Any]) -> bool:
    """Validate a sensor event dict against the Avro schema.

    Args:
        event: Sensor event dictionary with keys: timestamp, sensor_id, value, unit, metadata.

    Returns:
        True if valid.

    Raises:
        ValueError: If the event fails schema validation.
    """
    schema = _load_schema()
    try:
        # Write to a buffer to validate
        buf = io.BytesIO()
        fastavro.schemaless_writer(buf, schema, event)
        return True
    except Exception as exc:
        raise ValueError(f"Schema validation failed: {exc}") from exc


def serialize_event(event: Dict[str, Any]) -> bytes:
    """Serialize a sensor event to Avro binary format.

    Args:
        event: Valid sensor event dictionary.

    Returns:
        Avro-serialized bytes.
    """
    schema = _load_schema()
    buf = io.BytesIO()
    fastavro.schemaless_writer(buf, schema, event)
    return buf.getvalue()


def deserialize_event(data: bytes) -> Dict[str, Any]:
    """Deserialize Avro binary data back to a sensor event dict.

    Args:
        data: Avro-serialized bytes.

    Returns:
        Deserialized sensor event dictionary.
    """
    schema = _load_schema()
    buf = io.BytesIO(data)
    return fastavro.schemaless_reader(buf, schema)
