"""MLflow registry helpers and A/B model routing.

Provides:
- MLflow configuration and experiment management
- Model registration, promotion, and loading
- A/B testing routing between model versions
"""
import logging
import os
import random
from typing import Any, Optional

import joblib
try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None
    MlflowClient = None

LOG = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_A_WEIGHT = float(os.getenv("MODEL_A_WEIGHT", "0.8"))
MODEL_B_WEIGHT = float(os.getenv("MODEL_B_WEIGHT", "0.2"))


def configure() -> None:
    """Configure MLflow tracking URI from environment."""
    mlflow.set_tracking_uri(MLFLOW_URI)


def register_model(model_uri: str, name: str) -> Any:
    """Register a model in the MLflow Model Registry.

    Args:
        model_uri: URI of the logged model (e.g., 'runs:/<run_id>/model').
        name: Registered model name.

    Returns:
        ModelVersion object.
    """
    configure()
    return mlflow.register_model(model_uri, name)


def transition_to_stage(
    name: str, version: int, stage: str = "Staging"
) -> Any:
    """Transition a model version to a new stage.

    Args:
        name: Registered model name.
        version: Model version number.
        stage: Target stage ('Staging', 'Production', 'Archived').

    Returns:
        Updated ModelVersion.
    """
    configure()
    client = MlflowClient()
    return client.transition_model_version_stage(
        name=name, version=str(version), stage=stage
    )


def set_model_alias(name: str, alias: str, version: int) -> None:
    """Set a model alias (MLflow 2.x+).

    Args:
        name: Registered model name.
        alias: Alias string (e.g., 'production', 'staging').
        version: Model version number.
    """
    configure()
    client = MlflowClient()
    client.set_registered_model_alias(name, alias, str(version))
    LOG.info("Set alias '%s' for model '%s' version %d", alias, name, version)


def load_production_model(
    name: str,
    fallback_path: Optional[str] = None,
) -> Any:
    """Load the production model from MLflow registry.

    Falls back to a local joblib file if MLflow is unavailable.

    Args:
        name: Registered model name.
        fallback_path: Local model file path for fallback.

    Returns:
        Loaded model object.
    """
    try:
        configure()
        model_uri = f"models:/{name}/Production"
        return mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        LOG.warning("Could not load from MLflow: %s", exc)
        if fallback_path and os.path.exists(fallback_path):
            LOG.info("Loading fallback model from %s", fallback_path)
            return joblib.load(fallback_path)
        return None


class ABRouter:
    """Route inference requests between two model versions.

    Configuration via environment variables:
        MODEL_A_WEIGHT: Weight for model A (default 0.8)
        MODEL_B_WEIGHT: Weight for model B (default 0.2)

    Usage:
        router = ABRouter(model_a, model_b)
        model, label = router.select()
        prediction = model.predict(X)
    """

    def __init__(
        self,
        model_a: Any,
        model_b: Any,
        weight_a: float = MODEL_A_WEIGHT,
        weight_b: float = MODEL_B_WEIGHT,
    ):
        self.model_a = model_a
        self.model_b = model_b
        total = weight_a + weight_b
        self.threshold = weight_a / total if total > 0 else 0.5

    def select(self) -> tuple:
        """Select a model based on configured weights.

        Returns:
            Tuple of (model, label_string).
        """
        if random.random() < self.threshold:
            return self.model_a, "model_a"
        return self.model_b, "model_b"
