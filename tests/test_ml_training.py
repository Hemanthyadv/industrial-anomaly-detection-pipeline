"""Tests for ML training components."""
import numpy as np
from src.ml.train_models import (
    make_dataset,
    train_isolation_forest,
    train_lof,
    train_autoencoder,
)


def test_make_dataset():
    X, y = make_dataset(n=500, seed=42)
    assert X.shape == (500, 3)
    assert y.shape == (500,)
    assert set(np.unique(y)).issubset({0, 1})


def test_train_isolation_forest():
    X, y = make_dataset(n=200, seed=42)
    pipeline, metrics = train_isolation_forest(X, y, n_estimators=10)
    assert pipeline is not None
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    preds = pipeline.predict(X[:5])
    assert len(preds) == 5


def test_train_lof():
    X, y = make_dataset(n=200, seed=42)
    pipeline, metrics = train_lof(X, y, n_neighbors=5)
    assert pipeline is not None
    assert "f1" in metrics
    preds = pipeline.predict(X[:5])
    assert len(preds) == 5


def test_train_autoencoder():
    X, y = make_dataset(n=200, seed=42)
    bundle, metrics = train_autoencoder(X, y, hidden_layer_sizes=(8, 4, 8), max_iter=20)
    assert bundle["type"] == "autoencoder"
    assert "scaler" in bundle
    assert "mlp" in bundle
    assert "threshold" in bundle
    assert "f1" in metrics
