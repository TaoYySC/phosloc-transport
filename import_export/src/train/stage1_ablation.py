import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.data.split import build_group_kfold_splits, subset_by_indices
from src.evaluate.metrics import find_best_threshold
from src.evaluate.prediction_eval import evaluate_predictions
from src.features.classifier_repr.loader import ClassifierReprLoader
from src.features.esm_window_site_pca.loader import ESMWindowSitePCALoader
from src.features.manual_tabular.block import ManualTabularBlock
from src.models.build_sklearn_model import build_sklearn_model
from src.models.class_weight_utils import resolve_class_weight, summarize_resolved_weights
from src.models.projection_mlp import ProjectionMLP
from src.train.stage1_projection import (
    build_inner_train_val_split,
    train_projection_model,
    transform_stage1_embedding_with_projection,
)
from src.utils import setup_seed


ABLATION_SPECS = {
    "baseline_stage1_pls": {
        "include_stage1_branch": True,
        "use_stage1_pls": True,
        "use_stage1_projection": False,
    },
    "no_stage1_branch": {
        "include_stage1_branch": False,
        "use_stage1_pls": False,
        "use_stage1_projection": False,
    },
    "stage1_projection_ce_only": {
        "include_stage1_branch": True,
        "use_stage1_pls": False,
        "use_stage1_projection": True,
        "projection_alpha": 0.0,
    },
    "stage1_projection_ce_supcon": {
        "include_stage1_branch": True,
        "use_stage1_pls": False,
        "use_stage1_projection": True,
        "projection_alpha": None,
    },
    "stage1_pls_plus_projection": {
        "include_stage1_branch": True,
        "use_stage1_pls": True,
        "use_stage1_projection": True,
        "projection_alpha": None,
    },
}

MODELS_NEED_FINAL_SCALER = {
    "logreg",
    "elasticnet",
    "elastic_net",
    "svm",
    "svm_linear",
    "linear_svm",
    "svm_rbf",
    "rbf_svm",
}


def _json_dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def _build_stage1_loader(block_cfg):
    cfg = dict(block_cfg)
    cfg.pop("type", None)
    return ClassifierReprLoader(**cfg)


def _build_esm_loader(block_cfg):
    cfg = dict(block_cfg)
    cfg.pop("type", None)
    return ESMWindowSitePCALoader(**cfg)


def _build_manual_block(block_cfg):
    cfg = dict(block_cfg)
    cfg.pop("type", None)
    return ManualTabularBlock(**cfg)


def _predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba.reshape(-1)
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-score))
    raise ValueError("Model has no predict_proba or decision_function.")


def _concat_blocks(block_arrays: Dict[str, np.ndarray], order: List[str]) -> np.ndarray:
    arrays = []
    for key in order:
        if key not in block_arrays:
            continue
        arr = np.asarray(block_arrays[key], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arrays.append(arr)
    if not arrays:
        raise ValueError("No feature blocks available for concatenation.")
    return np.concatenate(arrays, axis=1)


def _build_predictions_table(
    df,
    y_prob,
    threshold,
    seed,
    fold,
    ablation_name,
    model_name,
):
    out = pd.DataFrame()
    out["sample_id"] = df["INDEX"].astype(str) if "INDEX" in df.columns else df.index.astype(str)
    out["ACC_ID"] = df["ACC_ID"].astype(str) if "ACC_ID" in df.columns else ""
    out["POSITION"] = df["POSITION"].astype(int) if "POSITION" in df.columns else -1
    out["true_label"] = df["LABEL"].astype(int).values
    out["pred_score"] = np.asarray(y_prob, dtype=float)
    out["pred_label"] = (out["pred_score"] >= float(threshold)).astype(int)
    out["fold"] = int(fold)
    out["seed"] = int(seed)
    out["ablation_name"] = ablation_name
    out["model_name"] = model_name
    return out


def _maybe_save_embedding_plot(z, y, out_path, seed):
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        return

    if z.shape[0] < 5:
        return

    setup_seed(seed, deterministic=True)
    perplexity = min(30, max(2, z.shape[0] // 3))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    ).fit_transform(z)

    fig, ax = plt.subplots(figsize=(5, 4))
    for label, color, name in [(0, "#4C78A8", "Export"), (1, "#F58518", "Import")]:
        mask = np.asarray(y).astype(int) == label
        if mask.any():
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=28,
                alpha=0.85,
                c=color,
                label=name,
            )
    ax.set_title("Projected stage1 embedding (t-SNE)")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_fold_feature_matrices(
    train_df,
    test_df,
    blocks_cfg,
    ablation_name,
    projection_cfg,
    seed,
    fold,
    include_stage1_prob=False,
):
    spec = ABLATION_SPECS[ablation_name]
    y_train = train_df["LABEL"].values

    parts_train = {}
    parts_test = {}
    feature_config = {
        "ablation_name": ablation_name,
        "spec": spec,
        "blocks": {},
        "projection_meta": None,
    }
    projection_history = None
    projection_model = None
    stage1_scaler = None

    if spec["include_stage1_branch"]:
        stage1_loader = _build_stage1_loader(blocks_cfg["classifier_l2"])

        if spec["use_stage1_projection"]:
            alpha = projection_cfg.get("alpha", 0.05)
            if spec.get("projection_alpha") is not None:
                alpha = float(spec["projection_alpha"])

            x_train_raw = stage1_loader.fit_transform_raw(train_df)
            x_test_raw = stage1_loader.transform_raw(test_df)
            stage1_scaler = stage1_loader.scaler

            inner_train_idx, inner_val_idx = build_inner_train_val_split(
                train_df,
                group_col=projection_cfg.get("group_col", "Cluster_ID"),
                seed=seed + int(fold),
                val_fraction=projection_cfg.get("inner_val_fraction", 0.2),
            )
            projection_model, projection_history, projection_meta = train_projection_model(
                x_train=x_train_raw[inner_train_idx],
                y_train=y_train[inner_train_idx],
                x_val=x_train_raw[inner_val_idx],
                y_val=y_train[inner_val_idx],
                input_dim=x_train_raw.shape[1],
                hidden_dim=projection_cfg.get("hidden_dim", 64),
                projection_dim=projection_cfg.get("projection_dim", 16),
                dropout=projection_cfg.get("dropout", 0.3),
                alpha=alpha,
                temperature=projection_cfg.get("temperature", 0.1),
                batch_size=projection_cfg.get("batch_size", 32),
                lr=projection_cfg.get("lr", 3e-4),
                weight_decay=projection_cfg.get("weight_decay", 1e-3),
                max_epochs=projection_cfg.get("max_epochs", 200),
                early_stopping_patience=projection_cfg.get("early_stopping_patience", 20),
                selection_metric=projection_cfg.get("selection_metric", "val_auroc"),
                class_weight=projection_cfg.get("class_weight", "balanced"),
                seed=seed + int(fold),
            )
            z_train = transform_stage1_embedding_with_projection(
                projection_model,
                stage1_scaler,
                x_train_raw,
            )
            z_test = transform_stage1_embedding_with_projection(
                projection_model,
                stage1_scaler,
                x_test_raw,
            )
            parts_train["stage1_projection"] = z_train
            parts_test["stage1_projection"] = z_test
            feature_config["projection_meta"] = projection_meta
            feature_config["blocks"]["stage1_projection"] = {
                "dim": int(z_train.shape[1]),
                "alpha": float(alpha),
            }

            if include_stage1_prob:
                with torch.no_grad():
                    x_tensor = torch.tensor(
                        stage1_scaler.transform(x_train_raw),
                        dtype=torch.float32,
                    )
                    logits = projection_model.forward_logits(x_tensor)
                    prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                parts_train["stage1_prob"] = prob.reshape(-1, 1)
                x_test_tensor = torch.tensor(
                    stage1_scaler.transform(x_test_raw),
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    logits_test = projection_model.forward_logits(x_test_tensor)
                    prob_test = torch.softmax(logits_test, dim=1)[:, 1].cpu().numpy()
                parts_test["stage1_prob"] = prob_test.reshape(-1, 1)

        if spec["use_stage1_pls"]:
            pls_loader = _build_stage1_loader(blocks_cfg["classifier_l2"])
            parts_train["stage1_pls"] = pls_loader.fit_transform(train_df, y=y_train)
            parts_test["stage1_pls"] = pls_loader.transform(test_df)
            feature_config["blocks"]["stage1_pls"] = {
                "dim": int(parts_train["stage1_pls"].shape[1]),
            }

    esm_window_loader = _build_esm_loader(blocks_cfg["esm_window"])
    parts_train["esm_window"] = esm_window_loader.fit_transform(train_df, y=y_train)
    parts_test["esm_window"] = esm_window_loader.transform(test_df)
    feature_config["blocks"]["esm_window"] = {
        "dim": int(parts_train["esm_window"].shape[1]),
    }

    esm_site_loader = _build_esm_loader(blocks_cfg["esm_site"])
    parts_train["esm_site"] = esm_site_loader.fit_transform(train_df, y=y_train)
    parts_test["esm_site"] = esm_site_loader.transform(test_df)
    feature_config["blocks"]["esm_site"] = {
        "dim": int(parts_train["esm_site"].shape[1]),
    }

    manual_block = _build_manual_block(blocks_cfg["tabular"])
    train_df_manual = manual_block.attach_features(train_df)
    test_df_manual = manual_block.attach_features(test_df)
    parts_train["manual"] = manual_block.fit_transform(train_df_manual, y=y_train)
    parts_test["manual"] = manual_block.transform(test_df_manual)
    feature_config["blocks"]["manual"] = {
        "dim": int(parts_train["manual"].shape[1]),
    }

    concat_order = []
    if spec["include_stage1_branch"]:
        if spec["use_stage1_pls"]:
            concat_order.append("stage1_pls")
        if spec["use_stage1_projection"]:
            concat_order.append("stage1_projection")
            if include_stage1_prob:
                concat_order.append("stage1_prob")
    concat_order.extend(["esm_window", "esm_site", "manual"])

    x_train = _concat_blocks(parts_train, concat_order)
    x_test = _concat_blocks(parts_test, concat_order)
    feature_config["concat_order"] = concat_order
    feature_config["final_dim"] = int(x_train.shape[1])

    artifacts = {
        "projection_model": projection_model,
        "projection_history": projection_history,
        "stage1_scaler": stage1_scaler,
        "z_stage1_train": parts_train.get("stage1_projection"),
        "z_stage1_test": parts_test.get("stage1_projection"),
    }
    return x_train, x_test, feature_config, artifacts


def run_ablation_fold(
    train_df,
    test_df,
    blocks_cfg,
    ablation_name,
    train_cfg,
    projection_cfg,
    seed,
    fold,
    output_dir,
    include_stage1_prob=False,
    save_embedding_plot=False,
):
    setup_seed(seed, deterministic=True)

    model_name = train_cfg.get("selected_model", "logreg")
    model_cfg = dict(train_cfg.get("models", {}).get(model_name, {}))
    threshold = float(model_cfg.get("threshold", 0.5))

    x_train, x_test, feature_config, artifacts = build_fold_feature_matrices(
        train_df=train_df,
        test_df=test_df,
        blocks_cfg=blocks_cfg,
        ablation_name=ablation_name,
        projection_cfg=projection_cfg,
        seed=seed,
        fold=fold,
        include_stage1_prob=include_stage1_prob,
    )

    y_train = train_df["LABEL"].values
    y_test = test_df["LABEL"].values

    fit_cfg = dict(model_cfg)
    resolved_class_weight = resolve_class_weight(y_train, model_cfg)
    fit_cfg["class_weight"] = resolved_class_weight

    final_scaler = None
    x_train_fit = x_train
    x_test_fit = x_test
    if model_name in MODELS_NEED_FINAL_SCALER:
        final_scaler = StandardScaler()
        x_train_fit = final_scaler.fit_transform(x_train)
        x_test_fit = final_scaler.transform(x_test)

    model = build_sklearn_model(
        model_name=model_name,
        model_cfg=fit_cfg,
        seed=seed,
    )
    model.fit(x_train_fit, y_train)

    train_prob = _predict_proba(model, x_train_fit)
    test_prob = _predict_proba(model, x_test_fit)

    train_thr = find_best_threshold(y_train, train_prob)
    train_metrics = evaluate_predictions(y_train, train_prob, threshold=train_thr)
    test_metrics = evaluate_predictions(y_test, test_prob, threshold=train_thr)

    fold_dir = (
        Path(output_dir)
        / f"ablation={ablation_name}"
        / f"seed={seed}_fold={fold}"
    )
    fold_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(
        [
            {"split": "train", **train_metrics},
            {"split": "test", **test_metrics},
        ]
    )
    metrics_df.to_csv(fold_dir / "metrics.csv", index=False)

    pred_train = _build_predictions_table(
        train_df, train_prob, train_thr, seed, fold, ablation_name, model_name
    )
    pred_test = _build_predictions_table(
        test_df, test_prob, train_thr, seed, fold, ablation_name, model_name
    )
    predictions_df = pd.concat([pred_train, pred_test], ignore_index=True)
    predictions_df.to_csv(fold_dir / "predictions.csv", index=False)

    _json_dump(feature_config, fold_dir / "feature_config.json")
    _json_dump(
        {
            "selected_model": model_name,
            "model_cfg": model_cfg,
            "resolved_class_weight": summarize_resolved_weights(resolved_class_weight),
            "threshold": float(train_thr),
            "use_final_scaler": final_scaler is not None,
        },
        fold_dir / "final_model_config.json",
    )

    if artifacts.get("projection_history") is not None:
        artifacts["projection_history"].to_csv(
            fold_dir / "projection_training_log.csv",
            index=False,
        )

    joblib.dump(model, fold_dir / "final_model.joblib")
    if final_scaler is not None:
        joblib.dump(final_scaler, fold_dir / "final_scaler.joblib")
    if artifacts.get("projection_model") is not None:
        torch.save(
            {
                "encoder_state_dict": artifacts["projection_model"].encoder_state_dict(),
                "projection_meta": feature_config.get("projection_meta"),
            },
            fold_dir / "projection_encoder.pt",
        )

    if save_embedding_plot and artifacts.get("z_stage1_train") is not None:
        _maybe_save_embedding_plot(
            artifacts["z_stage1_train"],
            y_train,
            fold_dir / "z_stage1_tsne.png",
            seed=seed,
        )

    result = {
        "ablation_name": ablation_name,
        "seed": seed,
        "fold": fold,
        "model_name": model_name,
        "threshold": float(train_thr),
        "feature_config": feature_config,
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "output_dir": str(fold_dir),
    }
    return result


def run_ablation(
    df,
    split_cfg,
    blocks_cfg,
    ablation_names,
    train_cfg,
    projection_cfg,
    output_dir,
    include_stage1_prob=False,
    save_embedding_plot=False,
):
    seeds = split_cfg.get("seeds", [42])
    n_splits = int(split_cfg.get("n_splits", 5))
    group_col = split_cfg.get("group_col", "Cluster_ID")
    label_col = split_cfg.get("label_col", "LABEL")

    all_rows = []
    for ablation_name in ablation_names:
        if ablation_name not in ABLATION_SPECS:
            raise ValueError(f"Unknown ablation: {ablation_name}")

        for seed in seeds:
            outer_splits = build_group_kfold_splits(
                df=df,
                n_splits=n_splits,
                group_col=group_col,
                label_col=label_col,
                seed=seed,
                stratify=split_cfg.get("stratify_group_kfold", True),
            )
            for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
                train_df = subset_by_indices(df, train_idx)
                test_df = subset_by_indices(df, test_idx)
                print(
                    f"[ABLATION] {ablation_name} seed={seed} fold={fold}/{n_splits}"
                )
                row = run_ablation_fold(
                    train_df=train_df,
                    test_df=test_df,
                    blocks_cfg=blocks_cfg,
                    ablation_name=ablation_name,
                    train_cfg=train_cfg,
                    projection_cfg=projection_cfg,
                    seed=seed,
                    fold=fold,
                    output_dir=output_dir,
                    include_stage1_prob=include_stage1_prob,
                    save_embedding_plot=save_embedding_plot,
                )
                all_rows.append(row)

    summary_df = pd.DataFrame(all_rows)
    metric_cols = [
        c for c in summary_df.columns if c.startswith("test_")
    ]
    agg = (
        summary_df.groupby("ablation_name")[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_path = Path(output_dir) / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    agg_path = Path(output_dir) / "ablation_summary_mean_std.csv"
    agg.to_csv(agg_path, index=False)
    return summary_df, agg
