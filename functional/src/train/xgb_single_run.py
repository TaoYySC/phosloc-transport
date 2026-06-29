import copy
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.evaluate.metrics import compute_metrics, find_best_threshold
from src.features.pipeline import FeaturePipeline
from src.train.single_run import _get_model_cfg, _save_json, _save_pickle, build_fold_eval_features
from src.utils import setup_seed


def _features_to_matrix(feature_dict, input_keys=None):
    keys = input_keys if input_keys is not None else list(feature_dict.keys())
    parts = []
    for key in keys:
        if key not in feature_dict:
            raise KeyError(
                f"Feature key '{key}' not found. Available keys: {list(feature_dict.keys())}"
            )
        x = np.asarray(feature_dict[key], dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        elif x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        parts.append(x)
    if len(parts) == 0:
        raise ValueError("No feature blocks available for XGBoost.")
    return np.concatenate(parts, axis=1)


def _build_scale_pos_weight(y_train, cfg):
    if cfg.get("scale_pos_weight") in (None, "auto"):
        y_train = np.asarray(y_train)
        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        if n_pos <= 0:
            return 1.0
        return float(n_neg / n_pos)
    return float(cfg["scale_pos_weight"])


def _build_xgb_model(model_cfg, scale_pos_weight, seed):
    params = {
        "n_estimators": int(model_cfg.get("n_estimators", 300)),
        "max_depth": int(model_cfg.get("max_depth", 4)),
        "learning_rate": float(model_cfg.get("learning_rate", 0.05)),
        "subsample": float(model_cfg.get("subsample", 0.8)),
        "colsample_bytree": float(model_cfg.get("colsample_bytree", 0.8)),
        "reg_lambda": float(model_cfg.get("reg_lambda", 1.0)),
        "reg_alpha": float(model_cfg.get("reg_alpha", 0.0)),
        "min_child_weight": float(model_cfg.get("min_child_weight", 1.0)),
        "gamma": float(model_cfg.get("gamma", 0.0)),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": int(seed),
        "n_jobs": int(model_cfg.get("n_jobs", 4)),
        "scale_pos_weight": float(scale_pos_weight),
    }
    return XGBClassifier(**params)


class ESMMeanXGBModel:
    """StandardScaler + PCA (optional) + XGBClassifier."""

    def __init__(self, model_cfg, seed):
        self.model_cfg = model_cfg
        self.seed = seed
        self.use_scaler = bool(model_cfg.get("use_scaler", True))
        pca_components = model_cfg.get("pca_components", 128)
        self.pca_components = None if pca_components in (None, 0, "none") else int(pca_components)
        self.scaler = StandardScaler() if self.use_scaler else None
        self.pca = None
        self.xgb = None

    def fit(self, x_train, y_train):
        x_train = np.asarray(x_train, dtype=np.float32)
        if self.use_scaler:
            x_train = self.scaler.fit_transform(x_train)
        if self.pca_components is not None:
            n_comp = min(self.pca_components, x_train.shape[0], x_train.shape[1])
            self.pca = PCA(n_components=n_comp, random_state=self.seed)
            x_train = self.pca.fit_transform(x_train)
        scale_pos_weight = _build_scale_pos_weight(y_train, self.model_cfg)
        self.xgb = _build_xgb_model(self.model_cfg, scale_pos_weight, self.seed)
        self.xgb.fit(x_train, np.asarray(y_train).astype(int))
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.use_scaler:
            x = self.scaler.transform(x)
        if self.pca is not None:
            x = self.pca.transform(x)
        return x

    def predict_proba(self, x):
        x = self.transform(x)
        prob = self.xgb.predict_proba(x)[:, 1]
        return prob.astype(np.float32)


def run_xgb_single_split(
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
    del device
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
    input_keys = model_cfg.get("input_keys", None)

    x_train = _features_to_matrix(train_features, input_keys=input_keys)
    x_val = _features_to_matrix(val_features, input_keys=input_keys)
    x_test = _features_to_matrix(test_features, input_keys=input_keys)

    model = ESMMeanXGBModel(model_cfg=model_cfg, seed=seed)
    model.fit(x_train, y_train)

    train_prob = model.predict_proba(x_train)
    val_prob = model.predict_proba(x_val)
    test_prob = model.predict_proba(x_test)

    best_thr = find_best_threshold(y_val, val_prob)

    save_train_predictions = bool(model_cfg.get("save_train_predictions", False))
    if save_train_predictions:
        train_metrics = compute_metrics(y_train, train_prob, best_thr)
    else:
        train_metrics = {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "f1": float("nan"),
            "mcc": float("nan"),
            "threshold": float(best_thr),
        }

    val_metrics = compute_metrics(y_val, val_prob, best_thr)
    test_metrics = compute_metrics(y_test, test_prob, best_thr)

    input_dims = {k: tuple(np.asarray(v).shape[1:]) for k, v in train_features.items()}
    scale_pos_weight = _build_scale_pos_weight(y_train, model_cfg)
    loss_info = {
        "loss_type": "xgboost_binary_logistic",
        "scale_pos_weight": float(scale_pos_weight),
    }

    history = [{
        "epoch": 1,
        "train_loss": float("nan"),
        "val_auroc": float(val_metrics["auroc"]),
        "val_auprc": float(val_metrics["auprc"]),
        "val_f1": float(val_metrics["f1"]),
        "val_mcc": float(val_metrics["mcc"]),
        "threshold": float(best_thr),
        "monitor_value": float(val_metrics["auprc"]),
        "best_epoch_so_far": 1,
        "no_improve_count": 0,
    }]

    if artifact_dir is not None:
        artifact_dir = Path(artifact_dir)
        _save_pickle(model, artifact_dir / "xgb_model.pkl")
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
            "best_threshold": float(best_thr),
            "scale_pos_weight": float(scale_pos_weight),
            "loss_info": loss_info,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "feature_info": getattr(pipeline, "feature_info", {}),
            "feature_names": pipeline.get_feature_names_out(),
            "raw_feature_dim": int(x_train.shape[1]),
            "pca_components": model.pca.n_components_ if model.pca is not None else None,
        }
        _save_json(metadata, artifact_dir / "metadata.json")
        from src.train.single_run import _save_prediction_table, _save_split_samples

        if save_train_predictions:
            _save_prediction_table(
                train_df, y_train, train_prob, best_thr, artifact_dir / "train_predictions.csv"
            )
        _save_prediction_table(
            val_df, y_val, val_prob, best_thr, artifact_dir / "val_predictions.csv"
        )
        _save_prediction_table(
            test_df, y_test, test_prob, best_thr, artifact_dir / "test_predictions.csv"
        )
        _save_split_samples(
            train_df, val_df, test_df, artifact_dir / "split_samples.csv"
        )

    return {
        "model": model,
        "pipeline": pipeline,
        "history": history,
        "best_threshold": float(best_thr),
        "pos_weight": float(scale_pos_weight),
        "loss_info": loss_info,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_prob": train_prob,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "feature_info": pipeline.feature_info,
        "feature_names": pipeline.get_feature_names_out(),
        "x_train_dim": int(x_train.shape[1]),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "model_name": model_name,
        "artifact_dir": None if artifact_dir is None else str(artifact_dir),
    }


__all__ = ["run_xgb_single_split", "build_fold_eval_features", "ESMMeanXGBModel"]
