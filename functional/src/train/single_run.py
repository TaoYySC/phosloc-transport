from math import prod
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import torch

from src.features.pipeline import FeaturePipeline
from src.models.build_model import build_model
from src.train.trainer import Trainer
from src.evaluate.metrics import compute_metrics
from src.utils import setup_seed, ensure_dir


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


def _build_class_prior(y_train, class_prior_cfg="auto"):
    y_train = np.asarray(y_train)
    n_pos = float((y_train == 1).sum())
    n_unlabeled = float((y_train == 0).sum())

    if n_pos <= 0:
        raise ValueError("PU loss requires at least one positive training sample.")

    if class_prior_cfg is None or class_prior_cfg == "auto":
        if n_pos + n_unlabeled <= 0:
            raise ValueError("Cannot estimate class prior from empty training set.")
        return float(n_pos / (n_pos + n_unlabeled))

    return float(class_prior_cfg)


def _resolve_loss_settings(model_cfg):
    loss_type = model_cfg.get("loss_type", "bce")
    class_prior_cfg = model_cfg.get("class_prior", "auto")
    return {
        "loss_type": str(loss_type),
        "class_prior_cfg": class_prior_cfg,
        "u_loss_weight": float(model_cfg.get("u_loss_weight", 1.0)),
        "non_negative": bool(model_cfg.get("non_negative", True)),
        "positive_loss_weight": float(model_cfg.get("positive_loss_weight", 1.0)),
    }


def _get_model_cfg(train_cfg):
    if "selected_model" in train_cfg and "models" in train_cfg:
        model_name = train_cfg["selected_model"]
        model_cfg = train_cfg["models"][model_name]
        return model_name, model_cfg

    model_name = train_cfg.get("model_name", "onehot_cnn")
    return model_name, train_cfg


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _save_json(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, indent=2, ensure_ascii=False)


def _save_pickle(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _save_prediction_table(df, y_true, prob, threshold, path):
    out = df.copy()
    prob = np.asarray(prob).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)

    out["true_label"] = y_true.astype(int)
    out["pred_prob"] = prob.astype(float)
    out["pred_label"] = (prob >= float(threshold)).astype(int)
    out["threshold"] = float(threshold)

    path = Path(path)
    ensure_dir(path.parent)
    out.to_csv(path, index=False)


def _save_split_samples(train_df, val_df, test_df, path):
    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()

    train_out["split"] = "train"
    val_out["split"] = "val"
    test_out["split"] = "test"

    out = pd.concat([train_out, val_out, test_out], axis=0, ignore_index=True)

    path = Path(path)
    ensure_dir(path.parent)
    out.to_csv(path, index=False)


def _save_artifacts(
    artifact_dir,
    model,
    pipeline,
    train_df,
    val_df,
    test_df,
    y_train_true,
    y_val_true,
    y_test_true,
    train_prob,
    val_prob,
    test_prob,
    history,
    best_threshold,
    seed,
    feature_set_name,
    feature_cfg,
    train_cfg,
    model_name,
    model_cfg,
    input_dims,
    input_keys,
    pos_weight,
    train_metrics,
    val_metrics,
    test_metrics,
    loss_info=None,
    save_train_predictions=True,
):
    artifact_dir = Path(artifact_dir)
    ensure_dir(artifact_dir)

    checkpoint = {
        "model_state_dict": {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
        },
        "model_name": model_name,
        "model_cfg": model_cfg,
        "input_dims": input_dims,
        "input_keys": input_keys,
        "best_threshold": float(best_threshold),
        "seed": int(seed),
        "feature_set_name": feature_set_name,
        "pos_weight": pos_weight,
        "loss_info": loss_info or {},
    }

    torch.save(checkpoint, artifact_dir / "model_checkpoint.pt")
    _save_pickle(pipeline, artifact_dir / "fitted_pipeline.pkl")

    metadata = {
        "seed": int(seed),
        "feature_set_name": feature_set_name,
        "model_name": model_name,
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "feature_cfg": feature_cfg,
        "input_dims": input_dims,
        "input_keys": input_keys,
        "best_threshold": float(best_threshold),
        "pos_weight": pos_weight,
        "loss_info": loss_info or {},
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_info": getattr(pipeline, "feature_info", {}),
        "feature_names": pipeline.get_feature_names_out(),
    }
    _save_json(metadata, artifact_dir / "metadata.json")

    with open(artifact_dir / "feature_names.txt", "w", encoding="utf-8") as f:
        for name in pipeline.get_feature_names_out():
            f.write(str(name) + "\n")

    pd.DataFrame(history).to_csv(artifact_dir / "history.csv", index=False)

    if save_train_predictions:
        _save_prediction_table(
            train_df,
            y_train_true,
            train_prob,
            best_threshold,
            artifact_dir / "train_predictions.csv",
        )
    _save_prediction_table(
        val_df,
        y_val_true,
        val_prob,
        best_threshold,
        artifact_dir / "val_predictions.csv",
    )
    _save_prediction_table(
        test_df,
        y_test_true,
        test_prob,
        best_threshold,
        artifact_dir / "test_predictions.csv",
    )
    _save_split_samples(
        train_df,
        val_df,
        test_df,
        artifact_dir / "split_samples.csv",
    )


def build_fold_eval_features(
    feature_cfg,
    val_df,
    test_df,
    fixed_retained_columns_by_family=None,
):
    """Load val/test features once per fold; safe when blocks are stateless (esm/graph)."""
    pipeline = FeaturePipeline(
        feature_cfg=feature_cfg,
        fixed_retained_columns_by_family=fixed_retained_columns_by_family,
    )
    val_features, val_df_out = pipeline.transform(val_df.copy())
    test_features, test_df_out = pipeline.transform(test_df.copy())
    return {
        "pipeline": pipeline,
        "val_features": val_features,
        "val_df": val_df_out,
        "test_features": test_features,
        "test_df": test_df_out,
    }


def run_single_split(
    train_df,
    val_df,
    test_df,
    feature_cfg,
    train_cfg,
    seed,
    device="cuda",
    fixed_retained_columns_by_family=None,
    artifact_dir=None,
    feature_set_name=None,
    fold_eval_features=None,
):
    setup_seed(seed, deterministic=True)

    if fold_eval_features is not None:
        pipeline = fold_eval_features["pipeline"]
        val_features = fold_eval_features["val_features"]
        val_df = fold_eval_features["val_df"]
        test_features = fold_eval_features["test_features"]
        test_df = fold_eval_features["test_df"]
        train_features, train_df = pipeline.fit_transform(train_df)
    else:
        pipeline = FeaturePipeline(
            feature_cfg=feature_cfg,
            fixed_retained_columns_by_family=fixed_retained_columns_by_family,
        )
        train_features, train_df = pipeline.fit_transform(train_df)
        val_features, val_df = pipeline.transform(val_df)
        test_features, test_df = pipeline.transform(test_df)

    y_train = train_df["LABEL"].values
    y_val = val_df["LABEL"].values
    y_test = test_df["LABEL"].values

    model_name, model_cfg = _get_model_cfg(train_cfg)
    if model_name == "xgboost":
        from src.train.xgb_single_run import run_xgb_single_split

        return run_xgb_single_split(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cfg=feature_cfg,
            train_cfg=train_cfg,
            seed=seed,
            device=device,
            fixed_retained_columns_by_family=fixed_retained_columns_by_family,
            artifact_dir=artifact_dir,
            feature_set_name=feature_set_name,
            fold_eval_features=fold_eval_features,
        )

    input_keys = model_cfg.get("input_keys", None)

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
    loss_settings = _resolve_loss_settings(model_cfg)
    loss_type = loss_settings["loss_type"]
    class_prior = _build_class_prior(
        y_train=y_train,
        class_prior_cfg=loss_settings["class_prior_cfg"],
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
        loss_type=loss_type,
        class_prior=class_prior,
        u_loss_weight=loss_settings["u_loss_weight"],
        non_negative=loss_settings["non_negative"],
        positive_loss_weight=loss_settings["positive_loss_weight"],
        early_stopping=model_cfg.get("early_stopping", True),
        early_stopping_metric=model_cfg.get("early_stopping_metric", "val_auprc"),
        early_stopping_patience=model_cfg.get("early_stopping_patience", 8),
        early_stopping_min_delta=model_cfg.get("early_stopping_min_delta", 0.0),
        val_eval_interval=model_cfg.get("val_eval_interval", 1),
        seed=seed,
    )

    history, best_thr = trainer.fit(
        train_features=train_features,
        y_train=y_train,
        val_features=val_features,
        y_val=y_val,
    )

    save_train_predictions = bool(model_cfg.get("save_train_predictions", True))

    if save_train_predictions:
        y_train_true, train_prob = trainer.predict_proba(
            feature_dict=train_features,
            y=y_train,
        )
        train_metrics = compute_metrics(y_train_true, train_prob, best_thr)
    else:
        y_train_true = y_train
        train_prob = np.zeros(len(y_train), dtype=np.float32)
        train_metrics = {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "f1": float("nan"),
            "mcc": float("nan"),
            "threshold": float(best_thr),
        }

    y_val_true, val_prob = trainer.predict_proba(
        feature_dict=val_features,
        y=y_val,
    )
    y_test_true, test_prob = trainer.predict_proba(
        feature_dict=test_features,
        y=y_test,
    )

    val_metrics = compute_metrics(y_val_true, val_prob, best_thr)
    test_metrics = compute_metrics(y_test_true, test_prob, best_thr)

    feature_dim = int(_infer_feature_dim(train_features, input_keys=input_keys))
    pos_weight_value = None if pos_weight is None else float(pos_weight.item())
    loss_info = {
        "loss_type": loss_type,
        "class_prior": float(class_prior),
        "class_prior_cfg": loss_settings["class_prior_cfg"],
        "u_loss_weight": float(loss_settings["u_loss_weight"]),
        "non_negative": bool(loss_settings["non_negative"]),
        "positive_loss_weight": float(loss_settings["positive_loss_weight"]),
        "pu_semantics": "LABEL=1 positive, LABEL=0 unlabeled",
    }

    if artifact_dir is not None:
        _save_artifacts(
            artifact_dir=artifact_dir,
            model=trainer.model,
            pipeline=pipeline,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            y_train_true=y_train_true,
            y_val_true=y_val_true,
            y_test_true=y_test_true,
            train_prob=train_prob,
            val_prob=val_prob,
            test_prob=test_prob,
            history=history,
            best_threshold=best_thr,
            seed=seed,
            feature_set_name=feature_set_name,
            feature_cfg=feature_cfg,
            train_cfg=train_cfg,
            model_name=model_name,
            model_cfg=model_cfg,
            input_dims=input_dims,
            input_keys=input_keys,
            pos_weight=pos_weight_value,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            loss_info=loss_info,
            save_train_predictions=save_train_predictions,
        )

    return {
        "model": trainer.model,
        "pipeline": pipeline,
        "history": history,
        "best_threshold": float(best_thr),
        "pos_weight": pos_weight_value,
        "loss_info": loss_info,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_prob": train_prob,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "feature_info": pipeline.feature_info,
        "feature_names": pipeline.get_feature_names_out(),
        "x_train_dim": feature_dim,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "model_name": model_name,
        "artifact_dir": None if artifact_dir is None else str(artifact_dir),
    }