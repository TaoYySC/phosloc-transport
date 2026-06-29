from pathlib import Path
import copy
import re

import shutil
import pandas as pd
import numpy as np
from src.data.split import (
    build_fixed_test_split,
    build_single_train_val_split,
    iter_dev_train_val_splits,
    subset_by_indices,
)
from src.train.single_run import run_single_split


def _safe_name(name):
    name = str(name)
    name = re.sub(r"[^\w\-.]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def _build_metrics_row(fold, seed, feature_set_name, result):
    row = {
        "fold": int(fold),
        "seed": int(seed),
        "feature_set": feature_set_name,
        "x_train_dim": int(result["x_train_dim"]),
        "n_train": int(result["n_train"]),
        "n_val": int(result["n_val"]),
        "n_test": int(result["n_test"]),
        "best_threshold": float(result["best_threshold"]),
        "artifact_dir": result.get("artifact_dir", ""),
    }

    for split_name in ["train", "val", "test"]:
        metrics = result[f"{split_name}_metrics"]
        for k, v in metrics.items():
            row[f"{split_name}_{k}"] = float(v)

    return row


def _build_feature_summary_row(fold, seed, feature_set_name, result):
    feature_names = result.get("feature_names", [])
    feature_info = result.get("feature_info", {})

    row = {
        "fold": int(fold),
        "seed": int(seed),
        "feature_set": feature_set_name,
        "n_feature_names": int(len(feature_names)),
        "feature_names": "|".join(feature_names) if len(feature_names) > 0 else "",
        "feature_info": str(feature_info),
        "artifact_dir": result.get("artifact_dir", ""),
    }
    return row


def check_group_leakage(df_train, df_val, df_test, group_col="ACC_ID"):
    train_groups = set(df_train[group_col].astype(str).unique())
    val_groups = set(df_val[group_col].astype(str).unique())
    test_groups = set(df_test[group_col].astype(str).unique())

    inter_train_val = train_groups & val_groups
    inter_train_test = train_groups & test_groups
    inter_val_test = val_groups & test_groups

    print(f"train groups: {len(train_groups)}")
    print(f"val groups: {len(val_groups)}")
    print(f"test groups: {len(test_groups)}")
    print(f"train ∩ val: {len(inter_train_val)}")
    print(f"train ∩ test: {len(inter_train_test)}")
    print(f"val ∩ test: {len(inter_val_test)}")

    if len(inter_train_val) > 0 or len(inter_train_test) > 0 or len(inter_val_test) > 0:
        raise ValueError("Group leakage detected across splits.")


def _build_prediction_rows(
    df_split,
    prob,
    threshold,
    fold,
    seed,
    feature_set_name,
    prediction_source,
):
    out = df_split.copy()
    prob = np.asarray(prob).reshape(-1)

    out["true_label"] = out["LABEL"].astype(int).values
    out["pred_prob"] = prob.astype(float)
    out["pred_label"] = (out["pred_prob"] >= float(threshold)).astype(int)
    out["threshold"] = float(threshold)
    out["fold"] = int(fold)
    out["seed"] = int(seed)
    out["feature_set"] = feature_set_name
    out["prediction_source"] = prediction_source

    return out


def _summarize_cv_metrics(all_metrics_df):
    summary_rows = []

    if len(all_metrics_df) == 0:
        return pd.DataFrame(summary_rows)

    metric_cols = [
        c for c in all_metrics_df.columns
        if (
            c.startswith("val_") or c.startswith("test_")
        ) and pd.api.types.is_numeric_dtype(all_metrics_df[c])
    ]

    for feature_set_name, sub_df in all_metrics_df.groupby("feature_set"):
        row = {
            "feature_set": feature_set_name,
            "n_folds": int(sub_df["fold"].nunique()),
        }

        for col in metric_cols:
            row[f"{col}_mean"] = float(sub_df[col].mean())
            row[f"{col}_std"] = float(sub_df[col].std(ddof=1))

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def _get_selection_score(result, selection_metric):
    if selection_metric.startswith("val_"):
        metric_name = selection_metric.replace("val_", "", 1)
        return float(result["val_metrics"][metric_name])

    if selection_metric.startswith("test_"):
        metric_name = selection_metric.replace("test_", "", 1)
        return float(result["test_metrics"][metric_name])

    if selection_metric.startswith("train_"):
        metric_name = selection_metric.replace("train_", "", 1)
        return float(result["train_metrics"][metric_name])

    return float(result["val_metrics"][selection_metric])


def _copy_best_artifact(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if dst_dir.exists():
        shutil.rmtree(dst_dir)

    shutil.copytree(src_dir, dst_dir)

def run_experiment_grid(
    df,
    split_cfg,
    feature_sets_cfg,
    train_cfg,
    task_name,
    device="cuda",
    artifact_root=None,
):
    seed = int(split_cfg.get("seed", 42))
    n_splits = int(split_cfg.get("n_splits", 5))
    fixed_test_size = float(split_cfg.get("fixed_test_size", 0.2))
    fixed_test_seed = int(split_cfg.get("fixed_test_seed", seed))
    group_col = split_cfg.get("group_col", "Cluster_ID")
    label_col = split_cfg.get("label_col", "LABEL")
    stratify = bool(split_cfg.get("stratify", True))
    selection_metric = split_cfg.get("selection_metric", "val_auprc")

    feature_sets = feature_sets_cfg["feature_sets"]

    dev_idx, fixed_test_idx = build_fixed_test_split(
        df=df,
        seed=fixed_test_seed,
        test_size=fixed_test_size,
        group_col=group_col,
        label_col=label_col,
        stratify=stratify,
    )

    df_dev = subset_by_indices(df, dev_idx)
    df_fixed_test = subset_by_indices(df, fixed_test_idx)
    fixed_test_samples_df = df_fixed_test.copy()
    fixed_test_samples_df["split"] = "fixed_test"
    fixed_test_samples_df["fixed_test_seed"] = int(fixed_test_seed)
    fixed_test_samples_df["fixed_test_size"] = float(fixed_test_size)

    dev_samples_df = df_dev.copy()
    dev_samples_df["split"] = "development"
    dev_samples_df["fixed_test_seed"] = int(fixed_test_seed)
    dev_samples_df["fixed_test_size"] = float(fixed_test_size)

    all_rows = []
    feature_rows = []
    history_rows = []
    cv_val_rows = []
    fixed_test_rows = []

    best_by_feature_set = {}
    best_overall = None

    run_meta = {
        "task_name": task_name,
        "device": device,
        "cv_type": (
            "fixed_test_plus_single_split"
            if int(n_splits) == 1
            else (
                "fixed_test_plus_StratifiedGroupKFold"
                if stratify
                else "fixed_test_plus_GroupKFold"
            )
        ),
        "n_splits": int(n_splits),
        "seed": int(seed),
        "fixed_test_seed": int(fixed_test_seed),
        "fixed_test_size": float(fixed_test_size),
        "selection_metric": selection_metric,
        "best_model_selection": "validation_metric",
        "group_col": group_col,
        "label_col": label_col,
        "feature_sets": list(feature_sets.keys()),
        "n_samples_total": int(len(df)),
        "n_development": int(len(df_dev)),
        "n_fixed_test": int(len(df_fixed_test)),
        "n_positive_total": int((df[label_col] == 1).sum()),
        "n_negative_total": int((df[label_col] == 0).sum()),
        "n_positive_development": int((df_dev[label_col] == 1).sum()),
        "n_negative_development": int((df_dev[label_col] == 0).sum()),
        "n_positive_fixed_test": int((df_fixed_test[label_col] == 1).sum()),
        "n_negative_fixed_test": int((df_fixed_test[label_col] == 0).sum()),
        "artifact_root": "" if artifact_root is None else str(artifact_root),
    }

    for fold, train_idx, val_idx in iter_dev_train_val_splits(
        df=df_dev,
        n_splits=n_splits,
        seed=seed,
        group_col=group_col,
        label_col=label_col,
        stratify=stratify,
    ):
        df_train = subset_by_indices(df_dev, train_idx)
        df_val = subset_by_indices(df_dev, val_idx)
        df_test = df_fixed_test.copy()

        inner_seed = seed + fold

        check_group_leakage(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            group_col=group_col,
        )

        for feature_set_name, feature_cfg in feature_sets.items():
            feature_cfg_local = copy.deepcopy(feature_cfg)

            artifact_dir = None
            if artifact_root is not None:
                artifact_dir = (
                    Path(artifact_root)
                    / f"fold_{int(fold)}"
                    / _safe_name(feature_set_name)
                )

            result = run_single_split(
                train_df=df_train,
                val_df=df_val,
                test_df=df_test,
                feature_cfg=feature_cfg_local,
                train_cfg=copy.deepcopy(train_cfg),
                seed=inner_seed,
                device=device,
                fixed_retained_columns_by_family=None,
                artifact_dir=artifact_dir,
                feature_set_name=feature_set_name,
            )

            metrics_row = _build_metrics_row(
                fold=fold,
                seed=inner_seed,
                feature_set_name=feature_set_name,
                result=result,
            )
            all_rows.append(metrics_row)

            feature_row = _build_feature_summary_row(
                fold=fold,
                seed=inner_seed,
                feature_set_name=feature_set_name,
                result=result,
            )
            feature_rows.append(feature_row)

            cv_val_row = _build_prediction_rows(
                df_split=df_val,
                prob=result["val_prob"],
                threshold=result["best_threshold"],
                fold=fold,
                seed=inner_seed,
                feature_set_name=feature_set_name,
                prediction_source="cv_validation",
            )
            cv_val_rows.append(cv_val_row)

            fixed_test_row = _build_prediction_rows(
                df_split=df_test,
                prob=result["test_prob"],
                threshold=result["best_threshold"],
                fold=fold,
                seed=inner_seed,
                feature_set_name=feature_set_name,
                prediction_source="fixed_test",
            )
            fixed_test_rows.append(fixed_test_row)

            selection_score = _get_selection_score(
                result=result,
                selection_metric=selection_metric,
            )

            best_record = {
                "task_name": task_name,
                "fold": int(fold),
                "seed": int(inner_seed),
                "feature_set": feature_set_name,
                "selection_metric": selection_metric,
                "selection_score": float(selection_score),
                "best_threshold": float(result["best_threshold"]),
                "artifact_dir": result.get("artifact_dir", ""),
                "train_auroc": float(result["train_metrics"]["auroc"]),
                "train_auprc": float(result["train_metrics"]["auprc"]),
                "train_f1": float(result["train_metrics"]["f1"]),
                "train_mcc": float(result["train_metrics"]["mcc"]),
                "val_auroc": float(result["val_metrics"]["auroc"]),
                "val_auprc": float(result["val_metrics"]["auprc"]),
                "val_f1": float(result["val_metrics"]["f1"]),
                "val_mcc": float(result["val_metrics"]["mcc"]),
                "test_auroc": float(result["test_metrics"]["auroc"]),
                "test_auprc": float(result["test_metrics"]["auprc"]),
                "test_f1": float(result["test_metrics"]["f1"]),
                "test_mcc": float(result["test_metrics"]["mcc"]),
            }

            current_best_for_feature = best_by_feature_set.get(feature_set_name)
            if (
                current_best_for_feature is None
                or selection_score > current_best_for_feature["selection_score"]
            ):
                best_by_feature_set[feature_set_name] = best_record

            if best_overall is None or selection_score > best_overall["selection_score"]:
                best_overall = best_record

            history = result.get("history", [])
            for h in history:
                history_rows.append(
                    {
                        "fold": int(fold),
                        "seed": int(inner_seed),
                        "feature_set": feature_set_name,
                        "epoch": int(h["epoch"]),
                        "train_loss": float(h["train_loss"]),
                        "val_auroc": float(h["val_auroc"]),
                        "val_auprc": float(h["val_auprc"]),
                        "val_f1": float(h["val_f1"]),
                        "val_mcc": float(h["val_mcc"]),
                        "threshold": float(h["threshold"]),
                        "monitor_value": float(h.get("monitor_value", np.nan)),
                        "best_epoch_so_far": int(h.get("best_epoch_so_far", 0)),
                        "no_improve_count": int(h.get("no_improve_count", 0)),
                        "artifact_dir": result.get("artifact_dir", ""),
                    }
                )

    all_metrics_df = pd.DataFrame(all_rows)
    cv_summary_df = _summarize_cv_metrics(all_metrics_df)
    feature_summary_df = pd.DataFrame(feature_rows)
    history_df = pd.DataFrame(history_rows)

    if len(cv_val_rows) > 0:
        cv_val_predictions_df = pd.concat(cv_val_rows, axis=0, ignore_index=True)
    else:
        cv_val_predictions_df = pd.DataFrame()

    if len(fixed_test_rows) > 0:
        fixed_test_predictions_df = pd.concat(fixed_test_rows, axis=0, ignore_index=True)
    else:
        fixed_test_predictions_df = pd.DataFrame()

    best_model_rows = list(best_by_feature_set.values())
    if best_overall is not None:
        best_overall_row = copy.deepcopy(best_overall)
        best_overall_row["feature_set"] = "__overall_best__"
        best_model_rows.append(best_overall_row)

    best_model_summary_df = pd.DataFrame(best_model_rows)

    if artifact_root is not None:
        best_root = Path(artifact_root) / "best_models"
        by_feature_root = best_root / "by_feature_set"

        for feature_set_name, record in best_by_feature_set.items():
            src_dir = record.get("artifact_dir", "")
            if src_dir:
                dst_dir = by_feature_root / _safe_name(feature_set_name)
                _copy_best_artifact(src_dir, dst_dir)
                record["best_model_dir"] = str(dst_dir)

        if best_overall is not None and best_overall.get("artifact_dir", ""):
            dst_dir = best_root / "overall_best"
            _copy_best_artifact(best_overall["artifact_dir"], dst_dir)
            best_overall["best_model_dir"] = str(dst_dir)

    run_meta["best_by_feature_set"] = best_by_feature_set
    run_meta["best_overall"] = best_overall

    return (
        all_metrics_df,
        cv_summary_df,
        feature_summary_df,
        history_df,
        cv_val_predictions_df,
        fixed_test_predictions_df,
        best_model_summary_df,
        fixed_test_samples_df,
        dev_samples_df,
        run_meta,
    )