"""Resolve per-fold class / sample weights for sklearn models."""

from __future__ import annotations

import numpy as np


def _count_binary_classes(y):
    y = np.asarray(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return n_neg, n_pos


def resolve_class_weight(y_train, model_cfg):
    """Map train_cfg weight fields to sklearn ``class_weight``.

    Supported ``model_cfg`` fields (priority: ``pos_weight`` > ``class_weight``):

    - ``pos_weight: auto``
        Per-fold ratio for positive class (Import, LABEL=1):
        ``{0: 1.0, 1: n_export / n_import}``.
    - ``pos_weight: <float>``
        Fixed ``{0: 1.0, 1: pos_weight}``; Export weight is 1.0.
    - ``class_weight: balanced``
        Pass through to sklearn (recomputed on each ``fit``).
    - ``class_weight: null`` / ``none``
        No weighting (``None``).
    - ``class_weight: auto``
        Per-fold sklearn-balanced dict:
        ``{j: n / (2 * n_j)}`` for ``j in {0, 1}``.
    - ``class_weight: {0: w0, 1: w1}``
        Explicit per-class weights.

    Returns:
        ``"balanced"``, ``None``, or ``dict`` with integer keys ``0`` and ``1``.
    """
    if model_cfg is None:
        return "balanced"

    if "pos_weight" in model_cfg and model_cfg.get("pos_weight") is not None:
        pos_weight = model_cfg["pos_weight"]
        if isinstance(pos_weight, str) and pos_weight.lower() == "auto":
            n_neg, n_pos = _count_binary_classes(y_train)
            if n_pos <= 0:
                raise ValueError("pos_weight=auto requires at least one LABEL=1 sample.")
            ratio = float(n_neg) / float(n_pos)
            return {0: 1.0, 1: ratio}
        return {0: 1.0, 1: float(pos_weight)}

    class_weight = model_cfg.get("class_weight", "balanced")
    if class_weight is None:
        return None
    if isinstance(class_weight, str):
        key = class_weight.lower()
        if key in {"none", "null"}:
            return None
        if key == "balanced":
            return "balanced"
        if key == "auto":
            y = np.asarray(y_train).astype(int)
            n = len(y)
            weights = {}
            for label in (0, 1):
                count = int((y == label).sum())
                if count <= 0:
                    raise ValueError(
                        f"class_weight=auto requires both classes in train fold; "
                        f"missing label={label}."
                    )
                weights[label] = float(n) / (2.0 * count)
            return weights
        raise ValueError(
            f"Unsupported class_weight string: {class_weight!r}. "
            "Use balanced, auto, none, or an explicit mapping."
        )

    if isinstance(class_weight, dict):
        return {int(k): float(v) for k, v in class_weight.items()}

    raise ValueError(f"Unsupported class_weight type: {type(class_weight)}")


def resolve_xgb_scale_pos_weight(y_train, model_cfg, class_weight):
    """Optional XGBoost ``scale_pos_weight`` from cfg or resolved class weights."""
    if model_cfg.get("scale_pos_weight") is not None:
        return float(model_cfg["scale_pos_weight"])

    if "pos_weight" in model_cfg and model_cfg.get("pos_weight") is not None:
        pos_weight = model_cfg["pos_weight"]
        if isinstance(pos_weight, str) and pos_weight.lower() == "auto":
            n_neg, n_pos = _count_binary_classes(y_train)
            if n_pos <= 0:
                return 1.0
            return float(n_neg) / float(n_pos)
        return float(pos_weight)

    if isinstance(class_weight, dict):
        w0 = float(class_weight.get(0, 1.0))
        w1 = float(class_weight.get(1, 1.0))
        if w0 <= 0:
            return 1.0
        return w1 / w0

    return None


def summarize_resolved_weights(class_weight):
    if class_weight is None:
        return "none"
    if class_weight == "balanced":
        return "balanced"
    if isinstance(class_weight, dict):
        return {
            "export_0": class_weight.get(0),
            "import_1": class_weight.get(1),
        }
    return class_weight
