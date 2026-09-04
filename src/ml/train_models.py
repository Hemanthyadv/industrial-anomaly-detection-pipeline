"""Train anomaly detection models on historical sensor data.

Models:
1. Isolation Forest — primary unsupervised anomaly detector
2. Local Outlier Factor (LOF) — density-based comparison model
3. Autoencoder — reconstruction-error based detector (sklearn MLP)

All models use StandardScaler preprocessing and are evaluated using
synthetic anomaly labels from the sensor simulator.
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
try:
    import mlflow
except ImportError:
    mlflow = None
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .model_evaluation import evaluate_anomaly

LOG = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

FEATURES = ["value", "pressure", "vibration"]
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DATA_PATH = os.getenv("TRAINING_DATA_PATH", "data/sample_sensor_data.csv")


def load_training_data(
    path: str = DATA_PATH,
    sample_frac: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load historical sensor data from CSV.

    The CSV must have columns: value, sensor_type, is_anomaly.
    Since sensor_type determines which reading (temp/pressure/vibration) each
    row is, we pivot to create a feature matrix with all three sensor types.

    For simplicity and to ensure all models train on the same features,
    we create synthetic multi-feature rows by grouping by timestamp.

    Args:
        path: Path to the CSV training data.
        sample_frac: Fraction of data to use (for faster iteration).
        seed: Random seed.

    Returns:
        Tuple of (X feature array, y label array, raw DataFrame).
    """
    LOG.info("Loading training data from %s", path)
    df = pd.read_csv(path)

    # Pivot sensor types into columns grouped by timestamp
    # Each timestamp has readings for multiple sensors; we group by timestamp
    # and take per-type means
    df["minute"] = pd.to_datetime(df["timestamp"]).dt.floor("min")

    pivoted = df.pivot_table(
        index="minute",
        columns="sensor_type",
        values="value",
        aggfunc="mean",
    ).reset_index()

    # Get anomaly labels per minute (any anomaly in that minute)
    anomaly_labels = df.groupby("minute")["is_anomaly"].max().reset_index()
    pivoted = pivoted.merge(anomaly_labels, on="minute", how="left")
    pivoted["is_anomaly"] = pivoted["is_anomaly"].fillna(0).astype(int)

    # Drop rows with missing sensor types
    feature_cols = ["temperature", "pressure", "vibration"]
    pivoted = pivoted.dropna(subset=feature_cols)

    if sample_frac < 1.0:
        pivoted = pivoted.sample(frac=sample_frac, random_state=seed)

    X = pivoted[feature_cols].values
    y = pivoted["is_anomaly"].values

    LOG.info("Loaded %d samples, anomaly rate: %.4f", len(X), y.mean())
    return X, y, pivoted


def make_dataset(n: int = 20000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data (fallback when CSV not available).

    Creates data in the sensor value domain (temperature ~70, pressure ~2, vibration ~0.05)
    with ~2% synthetic anomalies.

    Args:
        n: Number of samples.
        seed: Random seed.

    Returns:
        Tuple of (X, y) arrays.
    """
    rng = np.random.default_rng(seed)
    # Generate in realistic sensor domain
    temp = rng.normal(70, 3, size=n)
    pressure = rng.normal(2.0, 0.2, size=n)
    vibration = rng.normal(0.05, 0.015, size=n)
    x = np.column_stack([temp, pressure, vibration])

    anomalies = rng.random(n) < 0.02
    # Inject anomalies as large deviations
    x[anomalies, 0] += rng.normal(20, 5, size=anomalies.sum())  # temperature spike
    x[anomalies, 1] += rng.normal(1.0, 0.3, size=anomalies.sum())  # pressure spike
    x[anomalies, 2] += rng.normal(0.1, 0.03, size=anomalies.sum())  # vibration spike
    y = anomalies.astype(int)
    return x, y


def train_isolation_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    contamination: float = 0.02,
    max_samples: str = "auto",
    random_state: int = 42,
) -> Tuple[Pipeline, Dict]:
    """Train an Isolation Forest pipeline with StandardScaler.

    Args:
        X_train: Training features.
        y_train: Labels (for evaluation only, not used in fitting).
        n_estimators: Number of trees.
        contamination: Expected anomaly fraction.
        max_samples: Samples per tree.
        random_state: Random seed.

    Returns:
        Tuple of (fitted pipeline, evaluation metrics dict).
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
        )),
    ])
    pipeline.fit(X_train)

    predictions = (pipeline.predict(X_train) == -1).astype(int)
    scores = -pipeline.decision_function(X_train)  # higher = more anomalous
    metrics = evaluate_anomaly(y_train, predictions, scores=scores)
    metrics["false_positive_rate"] = float(
        ((predictions == 1) & (y_train == 0)).sum() / max((y_train == 0).sum(), 1)
    )

    return pipeline, metrics


def train_lof(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 20,
    contamination: float = 0.02,
) -> Tuple[Pipeline, Dict]:
    """Train a Local Outlier Factor pipeline.

    Uses novelty=True so the model can predict on new data.

    Args:
        X_train: Training features.
        y_train: Labels (for evaluation only).
        n_neighbors: Number of neighbors for LOF.
        contamination: Expected anomaly fraction.

    Returns:
        Tuple of (fitted pipeline, evaluation metrics dict).
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
        )),
    ])
    pipeline.fit(X_train)

    predictions = (pipeline.predict(X_train) == -1).astype(int)
    scores = -pipeline.decision_function(X_train)
    metrics = evaluate_anomaly(y_train, predictions, scores=scores)
    metrics["false_positive_rate"] = float(
        ((predictions == 1) & (y_train == 0)).sum() / max((y_train == 0).sum(), 1)
    )

    return pipeline, metrics


def train_autoencoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_layer_sizes: tuple = (32, 16, 8, 16, 32),
    max_iter: int = 200,
    threshold_percentile: float = 95.0,
    random_state: int = 42,
) -> Tuple[object, Dict]:
    """Train a reconstruction-error based autoencoder using sklearn MLPRegressor.

    The autoencoder learns to reconstruct normal sensor readings.
    High reconstruction error indicates anomalous readings.

    Args:
        X_train: Training features.
        y_train: Labels (for evaluation only).
        hidden_layer_sizes: MLP layer architecture.
        max_iter: Training epochs.
        threshold_percentile: Percentile of reconstruction error to use as threshold.
        random_state: Random seed.

    Returns:
        Tuple of (model dict with scaler/mlp/threshold, evaluation metrics dict).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Train autoencoder to reconstruct input
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
    )
    mlp.fit(X_scaled, X_scaled)  # reconstruct input

    # Compute reconstruction error
    X_reconstructed = mlp.predict(X_scaled)
    reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

    # Set threshold at specified percentile
    threshold = float(np.percentile(reconstruction_error, threshold_percentile))

    predictions = (reconstruction_error > threshold).astype(int)
    metrics = evaluate_anomaly(y_train, predictions, scores=reconstruction_error)
    metrics["false_positive_rate"] = float(
        ((predictions == 1) & (y_train == 0)).sum() / max((y_train == 0).sum(), 1)
    )
    metrics["reconstruction_threshold"] = threshold

    model_bundle = {
        "scaler": scaler,
        "mlp": mlp,
        "threshold": threshold,
        "type": "autoencoder",
    }
    return model_bundle, metrics


def train_all(
    output_dir: str = "models",
    data_path: Optional[str] = None,
) -> Dict[str, Dict]:
    """Train all anomaly detection models and log to MLflow.

    Args:
        output_dir: Directory to save model artifacts.
        data_path: Optional path to training CSV.

    Returns:
        Dictionary mapping model names to their evaluation metrics.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Load data
    dp = data_path or DATA_PATH
    try:
        X, y, _ = load_training_data(dp)
        LOG.info("Loaded CSV data from %s", dp)
    except (FileNotFoundError, KeyError) as exc:
        LOG.warning("Could not load CSV (%s), using synthetic data", exc)
        X, y = make_dataset()

    # Configure MLflow if available
    use_mlflow = False
    if mlflow is not None:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment("industrial-anomaly-detection")
            use_mlflow = True
        except Exception as exc:
            LOG.warning("MLflow not reachable: %s. Logging locally.", exc)

    results = {}

    # --- Isolation Forest ---
    LOG.info("Training Isolation Forest...")
    if_pipeline, if_metrics = train_isolation_forest(X, y)
    joblib.dump(if_pipeline, output / "isolation_forest.joblib")
    results["isolation_forest"] = if_metrics
    LOG.info("IF metrics: %s", {k: v for k, v in if_metrics.items() if isinstance(v, (int, float))})

    if use_mlflow:
        try:
            with mlflow.start_run(run_name="isolation-forest"):
                mlflow.log_params({"n_estimators": 200, "contamination": 0.02, "model_type": "IsolationForest"})
                mlflow.log_metrics({k: v for k, v in if_metrics.items() if isinstance(v, (int, float))})
                mlflow.sklearn.log_model(if_pipeline, "model")
        except Exception as e:
            LOG.warning("Failed to log IF to MLflow: %s", e)

    # --- Local Outlier Factor ---
    LOG.info("Training Local Outlier Factor...")
    lof_pipeline, lof_metrics = train_lof(X, y)
    joblib.dump(lof_pipeline, output / "lof.joblib")
    results["lof"] = lof_metrics
    LOG.info("LOF metrics: %s", {k: v for k, v in lof_metrics.items() if isinstance(v, (int, float))})

    if use_mlflow:
        try:
            with mlflow.start_run(run_name="local-outlier-factor"):
                mlflow.log_params({"n_neighbors": 20, "contamination": 0.02, "model_type": "LOF"})
                mlflow.log_metrics({k: v for k, v in lof_metrics.items() if isinstance(v, (int, float))})
                mlflow.sklearn.log_model(lof_pipeline, "model")
        except Exception as e:
            LOG.warning("Failed to log LOF to MLflow: %s", e)

    # --- Autoencoder ---
    LOG.info("Training Autoencoder...")
    ae_bundle, ae_metrics = train_autoencoder(X, y)
    joblib.dump(ae_bundle, output / "autoencoder.joblib")
    results["autoencoder"] = ae_metrics
    LOG.info("AE metrics: %s", {k: v for k, v in ae_metrics.items() if isinstance(v, (int, float))})

    if use_mlflow:
        try:
            with mlflow.start_run(run_name="autoencoder"):
                mlflow.log_params({
                    "hidden_layers": str((32, 16, 8, 16, 32)),
                    "max_iter": 200,
                    "threshold_percentile": 95.0,
                    "model_type": "Autoencoder",
                })
                mlflow.log_metrics({k: v for k, v in ae_metrics.items() if isinstance(v, (int, float))})
        except Exception as e:
            LOG.warning("Failed to log AE to MLflow: %s", e)

    # --- Select best model ---
    best_name = max(results, key=lambda k: results[k].get("f1", 0))
    LOG.info("Best model by F1: %s (F1=%.4f)", best_name, results[best_name]["f1"])

    # Save best model as default
    if best_name == "isolation_forest":
        joblib.dump(if_pipeline, output / "best_model.joblib")
    elif best_name == "lof":
        joblib.dump(lof_pipeline, output / "best_model.joblib")
    else:
        joblib.dump(ae_bundle, output / "best_model.joblib")

    return results


def train(output: str = "models/isolation_forest.joblib") -> str:
    """Legacy single-model training entrypoint."""
    train_all(output_dir=str(Path(output).parent))
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train anomaly detection models")
    p.add_argument("--output-dir", default="models", help="Model output directory")
    p.add_argument("--data", default=None, help="Training data CSV path")
    args = p.parse_args()
    results = train_all(output_dir=args.output_dir, data_path=args.data)
    print("\n=== Training Results ===")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
