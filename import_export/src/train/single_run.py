from math import prod

import numpy as np
import torch

from src.features.pipeline import FeaturePipeline
from src.models.build_model import build_model
from src.models.build_sklearn_model import build_sklearn_model
from src.models.class_weight_utils import (
    resolve_class_weight,
    resolve_xgb_scale_pos_weight,
    summarize_resolved_weights,
)
from src.train.trainer import Trainer
from src.evaluate.metrics import compute_metrics
from src.utils import setup_seed


def _get_feature_meta(x):
    if x is None:
        return None

    if hasattr(x, "shape"):
        return tuple(x.shape[1:])

    if isinstance(x, list):
        if len(x) == 0:
            return None

        first = x[0]

        if hasattr(first, "x") and hasattr(first.x, "shape"):
            return (int(first.x.shape[-1]),)

        if hasattr(first, "shape"):
            return tuple(first.shape)

    return None


def _infer_feature_dim(feature_dict, input_keys=None):
    keys = input_keys if input_keys is not None else list(feature_dict.keys())
    total_dim = 0

    for key in keys:
        x = feature_dict.get(key)
        meta = _get_feature_meta(x)
        if meta is None:
            continue
        total_dim += int(prod(meta))

    return total_dim


def _build_pos_weight(y_train, pos_weight_cfg=None, device="cuda"):
    y_train = np.asarray(y_train)
    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())

    if n_pos <= 0:
        return None

    if pos_weight_cfg is None or pos_weight_cfg == "auto":
        value = n_neg / n_pos
    else:
        value = float(pos_weight_cfg)

    return torch.tensor([value], dtype=torch.float32, device=device)


def _get_model_cfg(train_cfg):
    if "selected_model" in train_cfg and "models" in train_cfg:
        model_name = train_cfg["selected_model"]
        model_cfg = train_cfg["models"][model_name]
        return model_name, model_cfg

    model_name = train_cfg.get("model_name", "esm_cnn2d_site_gnn")
    return model_name, train_cfg


def _as_numpy_matrix(feature_dict, input_keys=None):
    keys = input_keys if input_keys is not None else list(feature_dict.keys())

    arrays = []
    for key in keys:
        if key not in feature_dict:
            raise KeyError(f"Feature key '{key}' is not found in feature_dict.")

        x = feature_dict[key]

        if isinstance(x, list):
            raise ValueError(
                f"Feature key '{key}' is a list object. "
                "Sklearn backend only supports numeric array features. "
                "Do not use graph features for sklearn models."
            )

        arr = np.asarray(x, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        arrays.append(arr)

    if len(arrays) == 0:
        raise ValueError("No valid numeric features were provided for sklearn backend.")

    return np.concatenate(arrays, axis=1)


def _predict_sklearn_proba(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba.reshape(-1)

    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-score))

    raise ValueError("The sklearn model has neither predict_proba nor decision_function.")


def _run_sklearn_train_test(
    train_features,
    test_features,
    y_train,
    y_test,
    model_name,
    model_cfg,
    seed,
):
    input_keys = model_cfg.get("input_keys", None)
    threshold = float(model_cfg.get("threshold", 0.5))

    X_train = _as_numpy_matrix(train_features, input_keys=input_keys)
    X_test = _as_numpy_matrix(test_features, input_keys=input_keys)

    fit_cfg = dict(model_cfg)
    resolved_class_weight = resolve_class_weight(y_train, model_cfg)
    fit_cfg["class_weight"] = resolved_class_weight
    if str(model_name).lower() == "xgb":
        scale_pos_weight = resolve_xgb_scale_pos_weight(
            y_train,
            model_cfg,
            resolved_class_weight,
        )
        if scale_pos_weight is not None:
            fit_cfg["scale_pos_weight"] = scale_pos_weight

    model = build_sklearn_model(
        model_name=model_name,
        model_cfg=fit_cfg,
        seed=seed,
    )

    model.fit(X_train, y_train)

    train_prob = _predict_sklearn_proba(model, X_train)
    test_prob = _predict_sklearn_proba(model, X_test)

    train_metrics = compute_metrics(y_train, train_prob, threshold)
    test_metrics = compute_metrics(y_test, test_prob, threshold)

    history = [
        {
            "epoch": 1,
            "train_loss": np.nan,
            "threshold": float(threshold),
        }
    ]

    return {
        "model": model,
        "history": history,
        "threshold": float(threshold),
        "pos_weight": None,
        "resolved_class_weight": summarize_resolved_weights(resolved_class_weight),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_prob": train_prob,
        "test_prob": test_prob,
        "x_train_dim": int(X_train.shape[1]),
        "input_dims": None,
    }


def _run_torch_train_test(
    train_features,
    test_features,
    y_train,
    y_test,
    model_name,
    model_cfg,
    seed,
    device,
):
    input_keys = model_cfg.get("input_keys", None)
    threshold = float(model_cfg.get("threshold", 0.5))

    input_dims = {}
    for k, v in train_features.items():
        meta = _get_feature_meta(v)
        if meta is not None:
            input_dims[k] = meta

    model = build_model(
        model_name=model_name,
        model_cfg=model_cfg,
        input_dims=input_dims,
    )

    pos_weight = _build_pos_weight(
        y_train=y_train,
        pos_weight_cfg=model_cfg.get("pos_weight", "auto"),
        device=device,
    )

    trainer = Trainer(
        model=model,
        input_keys=input_keys,
        device=device,
        lr=model_cfg.get("lr", 1e-3),
        weight_decay=model_cfg.get("weight_decay", 1e-4),
        batch_size=model_cfg.get("batch_size", 128),
        num_epochs=model_cfg.get("num_epochs", 50),
        pos_weight=pos_weight,
        seed=seed,
    )

    history = trainer.fit(
        train_features=train_features,
        y_train=y_train,
    )

    y_train_true, train_prob = trainer.predict_proba(train_features, y_train)
    y_test_true, test_prob = trainer.predict_proba(test_features, y_test)

    train_metrics = compute_metrics(y_train_true, train_prob, threshold)
    test_metrics = compute_metrics(y_test_true, test_prob, threshold)

    for h in history:
        h["threshold"] = float(threshold)

    return {
        "model": trainer.model,
        "history": history,
        "threshold": float(threshold),
        "pos_weight": None if pos_weight is None else float(pos_weight.item()),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_prob": train_prob,
        "test_prob": test_prob,
        "x_train_dim": int(_infer_feature_dim(train_features, input_keys=input_keys)),
        "input_dims": input_dims,
    }


def run_single_split(
    train_df,
    test_df,
    feature_cfg,
    train_cfg,
    seed,
    device="cuda",
    fixed_retained_columns_by_family=None,
):
    setup_seed(seed, deterministic=True)

    model_name, model_cfg = _get_model_cfg(train_cfg)
    model_cfg = dict(model_cfg)
    if feature_cfg.get("input_keys") is not None:
        model_cfg["input_keys"] = feature_cfg["input_keys"]

    blocks_cfg = feature_cfg.get("blocks", feature_cfg)
    pipeline = FeaturePipeline(
        feature_cfg={"blocks": blocks_cfg},
        fixed_retained_columns_by_family=fixed_retained_columns_by_family,
    )

    train_features, train_df = pipeline.fit_transform(train_df)
    test_features, test_df = pipeline.transform(test_df)

    y_train = train_df["LABEL"].values
    y_test = test_df["LABEL"].values

    backend = model_cfg.get("backend", "torch")

    if backend == "sklearn":
        result = _run_sklearn_train_test(
            train_features=train_features,
            test_features=test_features,
            y_train=y_train,
            y_test=y_test,
            model_name=model_name,
            model_cfg=model_cfg,
            seed=seed,
        )
    elif backend == "torch":
        result = _run_torch_train_test(
            train_features=train_features,
            test_features=test_features,
            y_train=y_train,
            y_test=y_test,
            model_name=model_name,
            model_cfg=model_cfg,
            seed=seed,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return {
        "model": result["model"],
        "pipeline": pipeline,
        "history": result["history"],
        "threshold": float(result["threshold"]),
        "pos_weight": result["pos_weight"],
        "train_metrics": result["train_metrics"],
        "test_metrics": result["test_metrics"],
        "train_prob": result["train_prob"],
        "test_prob": result["test_prob"],
        "feature_info": pipeline.feature_info,
        "feature_names": pipeline.get_feature_names_out(),
        "x_train_dim": int(result["x_train_dim"]),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "model_name": model_name,
        "backend": backend,
        "model_cfg": model_cfg,
        "input_keys": model_cfg.get("input_keys", None),
        "input_dims": result.get("input_dims"),
        "train_df": train_df.reset_index(drop=True),
        "test_df": test_df.reset_index(drop=True),
    }