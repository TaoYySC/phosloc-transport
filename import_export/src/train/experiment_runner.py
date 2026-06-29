import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.data.split import build_group_kfold_splits, subset_by_indices
from src.train.single_run import run_single_split


def _prefix_metrics(metrics, prefix):
    return {
        f"{prefix}_auroc": metrics.get("auroc"),
        f"{prefix}_auprc": metrics.get("auprc"),
        f"{prefix}_f1": metrics.get("f1"),
        f"{prefix}_mcc": metrics.get("mcc"),
    }


def _label_count(df, label_col="LABEL"):
    if label_col not in df.columns:
        return {}

    counts = df[label_col].value_counts().to_dict()
    return {f"n_label_{k}": int(v) for k, v in counts.items()}


def _print_group_overlap(df_train, df_test, group_col, seed, fold):
    train_groups = set(df_train[group_col].astype(str).tolist())
    test_groups = set(df_test[group_col].astype(str).tolist())

    print(f"[INFO] seed={seed} fold={fold}")
    print(f"train groups: {len(train_groups)}")
    print(f"test groups: {len(test_groups)}")
    print(f"train ∩ test: {len(train_groups & test_groups)}")


def _safe_name(x):
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_.=]+", "_", x)
    return x.strip("_")


def _json_dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def _prediction_table(df, prob, seed, fold, feature_set_name, model_name, threshold, split_name):
    meta_cols = [
        "INDEX",
        "ACC_ID",
        "POSITION",
        "RESIDUE",
        "LABEL",
        "Transport_Direction",
        "Cluster_ID",
        "TASK",
    ]
    keep_cols = [c for c in meta_cols if c in df.columns]

    out = df[keep_cols].copy()
    out["seed"] = int(seed)
    out["fold"] = int(fold)
    out["feature_set"] = feature_set_name
    out["model_name"] = model_name
    out["split"] = split_name
    out["prob_import"] = np.asarray(prob, dtype=float)
    out["prob_export"] = 1.0 - out["prob_import"]
    out["threshold"] = float(threshold)
    out["pred_label"] = (out["prob_import"] >= float(threshold)).astype(int)
    out["pred_direction"] = np.where(
        out["pred_label"] == 1,
        "Nuclear Import",
        "Nuclear Export",
    )

    if "LABEL" in out.columns:
        out["correct"] = (out["pred_label"] == out["LABEL"].astype(int)).astype(int)

    return out


def _save_fold_artifacts(
    result,
    feature_cfg,
    train_cfg,
    feature_set_name,
    seed,
    fold,
    output_dir,
):
    fold_dir = (
        Path(output_dir)
        / "fold_artifacts"
        / f"feature_set={_safe_name(feature_set_name)}"
        / f"seed={seed}_fold={fold}"
    )
    fold_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = fold_dir / "pipeline.joblib"
    joblib.dump(result["pipeline"], pipeline_path)

    backend = result["backend"]
    model_name = result["model_name"]

    if backend == "sklearn":
        model_path = fold_dir / "model.joblib"
        joblib.dump(result["model"], model_path)

    elif backend == "torch":
        model_path = fold_dir / "model.pt"
        state_dict = {
            k: v.detach().cpu()
            for k, v in result["model"].state_dict().items()
        }
        torch.save(
            {
                "model_state_dict": state_dict,
                "model_name": model_name,
                "model_cfg": result["model_cfg"],
                "input_dims": result["input_dims"],
                "input_keys": result["input_keys"],
                "threshold": float(result["threshold"]),
            },
            model_path,
        )

    else:
        raise ValueError(f"Unsupported backend for saving: {backend}")

    feature_names = pd.DataFrame({"feature_name": list(result["feature_names"])})
    feature_names.to_csv(fold_dir / "feature_names.csv", index=False)

    train_pred_df = _prediction_table(
        df=result["train_df"],
        prob=result["train_prob"],
        seed=seed,
        fold=fold,
        feature_set_name=feature_set_name,
        model_name=model_name,
        threshold=result["threshold"],
        split_name="train",
    )

    test_pred_df = _prediction_table(
        df=result["test_df"],
        prob=result["test_prob"],
        seed=seed,
        fold=fold,
        feature_set_name=feature_set_name,
        model_name=model_name,
        threshold=result["threshold"],
        split_name="test",
    )

    train_pred_df.to_csv(fold_dir / "train_predictions.csv", index=False)
    test_pred_df.to_csv(fold_dir / "test_predictions.csv", index=False)

    meta = {
        "feature_set": feature_set_name,
        "seed": int(seed),
        "fold": int(fold),
        "backend": backend,
        "model_name": model_name,
        "threshold": float(result["threshold"]),
        "x_train_dim": int(result["x_train_dim"]),
        "n_train": int(result["n_train"]),
        "n_test": int(result["n_test"]),
        "pos_weight": result["pos_weight"],
        "model_path": str(model_path),
        "pipeline_path": str(pipeline_path),
        "feature_cfg": feature_cfg,
        "train_cfg": train_cfg,
        "model_cfg": result["model_cfg"],
        "input_keys": result["input_keys"],
        "input_dims": result["input_dims"],
        "feature_info": result["feature_info"],
    }

    _json_dump(meta, fold_dir / "artifact_meta.json")

    return train_pred_df, test_pred_df


def run_experiment_grid(
    df,
    split_cfg,
    feature_sets_cfg,
    train_cfg,
    task_name,
    device="cuda",
    output_dir=None,
    save_fold_artifacts=True,
):
    all_rows = []
    history_rows = []
    all_train_prediction_rows = []
    all_test_prediction_rows = []

    seeds = split_cfg.get("seeds", [42])
    n_splits = int(split_cfg.get("n_splits", 5))
    group_col = split_cfg.get("group_col", "Cluster_ID")
    label_col = split_cfg.get("label_col", "LABEL")
    stratify = bool(split_cfg.get("stratify_group_kfold", True))

    feature_sets = feature_sets_cfg.get("feature_sets", {})

    for feature_set_name, feature_cfg in feature_sets.items():
        for seed in seeds:
            outer_splits = build_group_kfold_splits(
                df=df,
                n_splits=n_splits,
                group_col=group_col,
                label_col=label_col,
                seed=seed,
                stratify=stratify,
            )

            for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
                df_train = subset_by_indices(df, train_idx)
                df_test = subset_by_indices(df, test_idx)

                _print_group_overlap(
                    df_train=df_train,
                    df_test=df_test,
                    group_col=group_col,
                    seed=seed,
                    fold=fold,
                )

                result = run_single_split(
                    train_df=df_train,
                    test_df=df_test,
                    feature_cfg=feature_cfg,
                    train_cfg=train_cfg,
                    seed=seed,
                    device=device,
                    fixed_retained_columns_by_family=None,
                )

                row = {
                    "seed": seed,
                    "fold": fold,
                    "feature_set": feature_set_name,
                    "model_name": result["model_name"],
                    "x_train_dim": result["x_train_dim"],
                    "n_train": result["n_train"],
                    "n_test": result["n_test"],
                    "threshold": result["threshold"],
                }

                row.update(_prefix_metrics(result["train_metrics"], "train"))
                row.update(_prefix_metrics(result["test_metrics"], "test"))

                train_counts = _label_count(df_train, label_col=label_col)
                test_counts = _label_count(df_test, label_col=label_col)

                for k, v in train_counts.items():
                    row[f"train_{k}"] = v
                for k, v in test_counts.items():
                    row[f"test_{k}"] = v

                all_rows.append(row)

                if save_fold_artifacts and output_dir is not None:
                    train_pred_df, test_pred_df = _save_fold_artifacts(
                        result=result,
                        feature_cfg=feature_cfg,
                        train_cfg=train_cfg,
                        feature_set_name=feature_set_name,
                        seed=seed,
                        fold=fold,
                        output_dir=output_dir,
                    )
                    all_train_prediction_rows.append(train_pred_df)
                    all_test_prediction_rows.append(test_pred_df)

                for h in result["history"]:
                    h_row = {
                        "seed": seed,
                        "fold": fold,
                        "feature_set": feature_set_name,
                        "task_name": task_name,
                    }
                    h_row.update(h)
                    history_rows.append(h_row)

    all_metrics_df = pd.DataFrame(all_rows)
    best_metrics_df = all_metrics_df.copy()

    metric_cols = [
        "train_auroc",
        "train_auprc",
        "train_f1",
        "train_mcc",
        "test_auroc",
        "test_auprc",
        "test_f1",
        "test_mcc",
    ]
    existing_metric_cols = [c for c in metric_cols if c in all_metrics_df.columns]

    feature_summary_df = (
        all_metrics_df
        .groupby(["feature_set", "model_name"])[existing_metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    history_df = pd.DataFrame(history_rows)

    if output_dir is not None and len(all_test_prediction_rows) > 0:
        all_train_predictions_df = pd.concat(all_train_prediction_rows, ignore_index=True)
        all_test_predictions_df = pd.concat(all_test_prediction_rows, ignore_index=True)

        all_train_predictions_df.to_csv(
            Path(output_dir) / "all_fold_train_predictions.csv",
            index=False,
        )
        all_test_predictions_df.to_csv(
            Path(output_dir) / "all_fold_test_predictions.csv",
            index=False,
        )

    run_meta = {
        "task_name": task_name,
        "n_rows": int(len(df)),
        "group_col": group_col,
        "label_col": label_col,
        "seeds": list(seeds),
        "n_splits": int(n_splits),
        "stratify_group_kfold": bool(stratify),
        "feature_sets": list(feature_sets.keys()),
        "selected_model": train_cfg.get("selected_model"),
        "validation_used": False,
        "fold_artifacts_saved": bool(save_fold_artifacts and output_dir is not None),
    }

    return all_metrics_df, best_metrics_df, feature_summary_df, history_df, run_meta