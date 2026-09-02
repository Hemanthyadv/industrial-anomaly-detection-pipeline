"""Tests for model evaluation utilities."""
import numpy as np
import pytest

from src.ml.model_evaluation import evaluate_anomaly, evaluate_forecast


def test_anomaly_metrics_basic():
    """Basic anomaly metrics with known labels."""
    result = evaluate_anomaly([0, 1, 1, 0], [0, 1, 0, 0], scores=[0.1, 0.9, 0.2, 0.1])
    assert result["recall"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(1.0)
    assert "roc_auc" in result
    assert result["roc_auc"] is not None
    assert 0 <= result["roc_auc"] <= 1
    assert "f1" in result
    assert "confusion_matrix" in result
    assert "false_positive_rate" in result


def test_anomaly_metrics_without_scores():
    """Metrics without scores should omit roc_auc."""
    result = evaluate_anomaly([0, 1, 0], [0, 0, 0])
    assert "roc_auc" not in result
    assert result["recall"] == pytest.approx(0.0)


def test_anomaly_single_class():
    """Single-class y_true should not crash ROC-AUC."""
    result = evaluate_anomaly([0, 0, 0, 0], [0, 0, 1, 0], scores=[0.1, 0.1, 0.9, 0.1])
    assert result["roc_auc"] is None  # gracefully handled
    assert result["false_positive_rate"] > 0


def test_anomaly_perfect_score():
    """Perfect predictions should give F1 = 1.0."""
    result = evaluate_anomaly([0, 1, 0, 1], [0, 1, 0, 1], scores=[0.1, 0.9, 0.1, 0.9])
    assert result["f1"] == pytest.approx(1.0)
    assert result["false_positive_rate"] == pytest.approx(0.0)


def test_forecast_metrics():
    """Basic forecast metrics."""
    result = evaluate_forecast([1, 2, 3], [1, 3, 2])
    assert result["mae"] == pytest.approx(2 / 3)
    assert result["rmse"] == pytest.approx(np.sqrt(2 / 3))


def test_forecast_perfect():
    """Perfect forecast should give MAE = RMSE = 0."""
    result = evaluate_forecast([1, 2, 3], [1, 2, 3])
    assert result["mae"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)


def test_forecast_numpy_arrays():
    """Should work with numpy arrays."""
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([11.0, 19.0, 31.0])
    result = evaluate_forecast(actual, predicted)
    assert result["mae"] == pytest.approx(1.0)
