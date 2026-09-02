"""Hyperparameter tuning for anomaly detection models.

Uses a custom scoring strategy suitable for unsupervised anomaly detection
with synthetic benchmark labels.

Design decision: We use a grid search with F1 score computed against synthetic
anomaly labels. This is NOT the same as supervised GridSearchCV — we fit each
model unsupervised and then evaluate against known injected anomalies.
"""
import logging
import os
from typing import Dict, List, Tuple

import mlflow
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def tune_isolation_forest(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[Dict, Dict]:
    """Grid search over Isolation Forest hyperparameters.

    Args:
        X: Feature matrix.
        y: Synthetic anomaly labels for evaluation.

    Returns:
        Tuple of (best_params, all_results).
    """
    param_grid = {
        "n_estimators": [100, 200, 300],
        "contamination": [0.01, 0.02, 0.05],
        "max_samples": [0.5, 0.8, 1.0],
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_f1 = -1.0
    best_params = {}
    all_results = []

    for n_est in param_grid["n_estimators"]:
        for contam in param_grid["contamination"]:
            for max_s in param_grid["max_samples"]:
                model = IsolationForest(
                    n_estimators=n_est,
                    contamination=contam,
                    max_samples=max_s,
                    random_state=42,
                )
                model.fit(X_scaled)
                preds = (model.predict(X_scaled) == -1).astype(int)
                f1 = float(f1_score(y, preds, zero_division=0))

                result = {
                    "n_estimators": n_est,
                    "contamination": contam,
                    "max_samples": max_s,
                    "f1": f1,
                }
                all_results.append(result)

                if f1 > best_f1:
                    best_f1 = f1
                    best_params = result.copy()

                LOG.debug(
                    "IF n_est=%d contam=%.3f max_s=%.1f -> F1=%.4f",
                    n_est, contam, max_s, f1,
                )

    LOG.info("Best IF params: %s (F1=%.4f)", best_params, best_f1)
    return best_params, {"all_results": all_results}


def tune_lof(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[Dict, Dict]:
    """Grid search over LOF hyperparameters.

    Args:
        X: Feature matrix.
        y: Synthetic anomaly labels.

    Returns:
        Tuple of (best_params, all_results).
    """
    param_grid = {
        "n_neighbors": [10, 20, 30, 50],
        "contamination": [0.01, 0.02, 0.05],
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_f1 = -1.0
    best_params = {}
    all_results = []

    for n_neigh in param_grid["n_neighbors"]:
        for contam in param_grid["contamination"]:
            model = LocalOutlierFactor(
                n_neighbors=n_neigh,
                contamination=contam,
                novelty=True,
            )
            model.fit(X_scaled)
            preds = (model.predict(X_scaled) == -1).astype(int)
            f1 = float(f1_score(y, preds, zero_division=0))

            result = {
                "n_neighbors": n_neigh,
                "contamination": contam,
                "f1": f1,
            }
            all_results.append(result)

            if f1 > best_f1:
                best_f1 = f1
                best_params = result.copy()

    LOG.info("Best LOF params: %s (F1=%.4f)", best_params, best_f1)
    return best_params, {"all_results": all_results}


def tune_autoencoder(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[Dict, Dict]:
    """Grid search over Autoencoder hyperparameters.

    Args:
        X: Feature matrix.
        y: Synthetic anomaly labels.

    Returns:
        Tuple of (best_params, all_results).
    """
    param_grid = {
        "hidden_layer_sizes": [
            (16, 8, 16),
            (32, 16, 8, 16, 32),
            (64, 32, 16, 32, 64),
        ],
        "max_iter": [100, 200],
        "threshold_percentile": [90.0, 95.0, 97.5],
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_f1 = -1.0
    best_params = {}
    all_results = []

    for layers in param_grid["hidden_layer_sizes"]:
        for max_it in param_grid["max_iter"]:
            mlp = MLPRegressor(
                hidden_layer_sizes=layers,
                activation="relu",
                max_iter=max_it,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            )
            mlp.fit(X_scaled, X_scaled)
            reconstructed = mlp.predict(X_scaled)
            recon_err = np.mean((X_scaled - reconstructed) ** 2, axis=1)

            for pct in param_grid["threshold_percentile"]:
                threshold = float(np.percentile(recon_err, pct))
                preds = (recon_err > threshold).astype(int)
                f1 = float(f1_score(y, preds, zero_division=0))

                result = {
                    "hidden_layer_sizes": str(layers),
                    "max_iter": max_it,
                    "threshold_percentile": pct,
                    "f1": f1,
                }
                all_results.append(result)

                if f1 > best_f1:
                    best_f1 = f1
                    best_params = result.copy()

    LOG.info("Best AE params: %s (F1=%.4f)", best_params, best_f1)
    return best_params, {"all_results": all_results}


if __name__ == "__main__":
    from .train_models import load_training_data, make_dataset

    try:
        X, y, _ = load_training_data()
    except Exception:
        X, y = make_dataset()

    print("\n=== Tuning Isolation Forest ===")
    best_if, _ = tune_isolation_forest(X, y)
    print(f"Best: {best_if}")

    print("\n=== Tuning LOF ===")
    best_lof, _ = tune_lof(X, y)
    print(f"Best: {best_lof}")

    print("\n=== Tuning Autoencoder ===")
    best_ae, _ = tune_autoencoder(X, y)
    print(f"Best: {best_ae}")
