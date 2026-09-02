"""Evaluation utilities for anomaly detection and forecast models.

Provides standardized evaluation functions with proper edge-case handling.
"""
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

LOG = logging.getLogger(__name__)


def evaluate_anomaly(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    scores: Optional[Union[List, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Evaluate anomaly detection model performance.

    Args:
        y_true: Ground truth binary labels (0=normal, 1=anomaly).
        y_pred: Predicted binary labels.
        scores: Optional continuous anomaly scores for ROC-AUC.

    Returns:
        Dictionary containing precision, recall, f1, confusion_matrix,
        and optionally roc_auc.

    Note:
        These evaluations use synthetic anomaly labels injected by the
        sensor simulator. They serve as benchmark labels for model comparison,
        not real-world ground truth.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    result: Dict[str, Any] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # False positive rate
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result["false_positive_rate"] = float(fp / max(fp + tn, 1))
    result["true_positive_rate"] = float(tp / max(tp + fn, 1))

    if scores is not None:
        scores = np.asarray(scores)
        # Guard against single-class ROC-AUC (raises ValueError)
        unique_classes = np.unique(y_true)
        if len(unique_classes) >= 2:
            try:
                result["roc_auc"] = float(roc_auc_score(y_true, scores))
            except ValueError as exc:
                LOG.warning("ROC-AUC calculation failed: %s", exc)
                result["roc_auc"] = None
        else:
            LOG.warning("Single class in y_true, skipping ROC-AUC")
            result["roc_auc"] = None

    return result


def evaluate_forecast(
    actual: Union[List, np.ndarray],
    predicted: Union[List, np.ndarray],
) -> Dict[str, float]:
    """Evaluate time-series forecast performance.

    Args:
        actual: Ground truth values.
        predicted: Predicted values.

    Returns:
        Dictionary with MAE and RMSE.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }
