import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)


def _safe_metric(func, y_true, y_pred_or_prob, **kwargs):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(func(y_true, y_pred_or_prob, **kwargs))
    except Exception:
        return np.nan


def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= float(threshold)).astype(int)

    metrics = {
        "auroc": _safe_metric(roc_auc_score, y_true, y_prob),
        "auprc": _safe_metric(average_precision_score, y_true, y_prob),
        "f1": _safe_metric(f1_score, y_true, y_pred),
        "mcc": _safe_metric(matthews_corrcoef, y_true, y_pred),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_true, y_pred),
        "recall_export_0": np.nan,
        "recall_import_1": np.nan,
    }

    if len(np.unique(y_true)) >= 1:
        recalls = recall_score(
            y_true,
            y_pred,
            labels=[0, 1],
            average=None,
            zero_division=0,
        )
        if isinstance(recalls, np.ndarray) and recalls.size == 2:
            metrics["recall_export_0"] = float(recalls[0])
            metrics["recall_import_1"] = float(recalls[1])

    return metrics
