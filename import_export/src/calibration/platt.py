"""Platt scaling for post-hoc probability calibration (preserves AUROC)."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.train.single_run import _as_numpy_matrix


def fit_platt_calibrator(scores, labels):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)

    if len(np.unique(labels)) < 2:
        raise ValueError("Platt scaling requires both positive and negative labels.")

    model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=5000)
    model.fit(scores.reshape(-1, 1), labels)

    coef = float(model.coef_.reshape(-1)[0])
    intercept = float(model.intercept_.reshape(-1)[0])

    if coef <= 0:
        raise ValueError(
            f"Platt scaling requires a positive slope, got coef={coef}. "
            "Scores may be inverted or constant."
        )

    return {
        "method": "platt",
        "coef": coef,
        "intercept": intercept,
        "positive_class": "import",
    }


def apply_platt_calibrator(scores, calibrator):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    coef = float(calibrator["coef"])
    intercept = float(calibrator["intercept"])
    logits = coef * scores + intercept
    return 1.0 / (1.0 + np.exp(-logits))


def save_platt_calibrator(calibrator, path, extra_meta=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(calibrator)
    if extra_meta:
        payload.update(extra_meta)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_platt_calibrator(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_auroc(y_true, scores):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def collect_oof_test_decision_scores(run_dir, feature_set=None):
    run_dir = Path(run_dir)
    meta_paths = sorted(run_dir.glob("fold_artifacts/*/*/artifact_meta.json"))
    if not meta_paths:
        raise FileNotFoundError(f"No fold artifacts found under: {run_dir}")

    if feature_set is not None:
        meta_paths = [
            p
            for p in meta_paths
            if json.loads(p.read_text(encoding="utf-8")).get("feature_set") == feature_set
        ]
        if not meta_paths:
            raise FileNotFoundError(
                f"No fold artifacts for feature_set={feature_set!r} under: {run_dir}"
            )

    rows = []
    for meta_path in meta_paths:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fold_dir = meta_path.parent
        oof_csv = fold_dir / "test_predictions.csv"
        if not oof_csv.exists():
            raise FileNotFoundError(f"Missing test predictions: {oof_csv}")

        fold_df = pd.read_csv(oof_csv)
        pipeline = joblib.load(fold_dir / "pipeline.joblib")
        model = joblib.load(fold_dir / "model.joblib")
        input_keys = meta.get("input_keys")

        features, _ = pipeline.transform(fold_df)
        X = _as_numpy_matrix(features, input_keys=input_keys)
        if not hasattr(model, "decision_function"):
            raise ValueError(
                f"Model {meta.get('model_name')} does not expose decision_function; "
                "Platt scaling is not supported."
            )
        decision_scores = np.asarray(model.decision_function(X), dtype=np.float64)

        out = fold_df.copy()
        out["decision_score"] = decision_scores
        if "prob_import" in out.columns:
            out["prob_import_raw"] = out["prob_import"]
        rows.append(out)

    return pd.concat(rows, ignore_index=True)


def build_calibrated_oof_predictions(run_dir, feature_set=None, positive_class="import"):
    run_dir = Path(run_dir)
    oof_df = collect_oof_test_decision_scores(run_dir, feature_set=feature_set)

    if "LABEL" in oof_df.columns:
        labels = oof_df["LABEL"].astype(int).to_numpy()
    elif "Transport_Direction" in oof_df.columns:
        if str(positive_class).lower() == "import":
            labels = (oof_df["Transport_Direction"] == "Nuclear Import").astype(int).to_numpy()
        else:
            labels = (oof_df["Transport_Direction"] == "Nuclear Export").astype(int).to_numpy()
    else:
        raise ValueError("OOF table must contain LABEL or Transport_Direction.")

    scores = oof_df["decision_score"].to_numpy()
    calibrator = fit_platt_calibrator(scores, labels)
    prob_cal = apply_platt_calibrator(scores, calibrator)

    out = oof_df.copy()
    if "prob_import_raw" not in out.columns and "prob_import" in out.columns:
        out["prob_import_raw"] = out["prob_import"]
    out["prob_import"] = prob_cal
    out["prob_export"] = 1.0 - out["prob_import"]

    if "pred_label" in out.columns:
        threshold = float(out["threshold"].iloc[0]) if "threshold" in out.columns else 0.5
        out["pred_label"] = (out["prob_import"] >= threshold).astype(int)
        if str(positive_class).lower() == "import":
            out["pred_direction"] = np.where(
                out["pred_label"] == 1, "Nuclear Import", "Nuclear Export"
            )
        else:
            out["pred_direction"] = np.where(
                out["pred_label"] == 1, "Nuclear Export", "Nuclear Import"
            )

    meta = {
        "run_dir": str(run_dir),
        "feature_set": feature_set,
        "positive_class": positive_class,
        "n_fit": int(len(out)),
        "auroc_raw": _safe_auroc(labels, scores),
        "auroc_calibrated": _safe_auroc(labels, prob_cal),
        "prob_import_raw_min": float(out["prob_import_raw"].min()),
        "prob_import_raw_max": float(out["prob_import_raw"].max()),
        "prob_import_cal_min": float(out["prob_import"].min()),
        "prob_import_cal_max": float(out["prob_import"].max()),
    }
    calibrator["positive_class"] = positive_class
    return out, calibrator, meta


def sklearn_decision_scores(model, X):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=np.float64)
    raise ValueError("Model does not provide decision_function for Platt calibration.")
