import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def _safe_auroc(y_true, y_prob):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def _safe_auprc(y_true, y_prob):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_prob))


def find_best_threshold(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(np.unique(y_true)) < 2:
        return 0.5

    thresholds = np.linspace(0.05, 0.95, 19)
    best_thr = 0.5
    best_score = -np.inf

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        try:
            score = matthews_corrcoef(y_true, y_pred)
        except Exception:
            score = -np.inf

        if score > best_score:
            best_score = score
            best_thr = float(thr)

    return best_thr


def compute_metrics(y_true, y_prob, thr):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= thr).astype(int)

    try:
        f1 = float(f1_score(y_true, y_pred))
    except Exception:
        f1 = np.nan

    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = np.nan

    return {
        "auroc": _safe_auroc(y_true, y_prob),
        "auprc": _safe_auprc(y_true, y_prob),
        "f1": f1,
        "mcc": mcc,
    }
