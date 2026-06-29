import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.split import build_group_kfold_splits, subset_by_indices
from src.train.stacking_plr import run_stacking_outer_fold
from src.utils import ensure_dir, save_csv, save_json


def _print_group_overlap(df_train, df_test, group_col, seed, fold, split_name="outer"):
    train_groups = set(df_train[group_col].astype(str).tolist())
    test_groups = set(df_test[group_col].astype(str).tolist())
    overlap = len(train_groups & test_groups)
    print(f"[INFO] {split_name} seed={seed} fold={fold}")
    print(f"  train groups: {len(train_groups)}, test groups: {len(test_groups)}, overlap: {overlap}")
    if overlap > 0:
        print(f"  [WARN] group leakage detected: {overlap}")


SUMMARY_MODELS = [
    "window_only",
    "site_only",
    "manual_only",
    "mean_fusion",
    "stacking_plr",
]


def _metrics_row(model_name, seed, fold_id, metrics):
    row = {"model": model_name, "seed": seed, "fold": fold_id}
    row.update(metrics)
    return row


def run_stacking_grid(
    df,
    split_cfg,
    feature_sets_cfg,
    train_cfg,
    feature_set_name,
    output_dir,
    id_col="INDEX",
):
    stacking_cfg = train_cfg.get("stacking", {})
    inner_cfg = stacking_cfg.get("inner_cv", {"n_splits": 5, "seed": 42, "stratify": True})
    branch_pca = stacking_cfg.get(
        "branch_pca",
        {"window": 48, "site": 24, "manual": None},
    )
    base_C = float(stacking_cfg.get("base_C", 0.1))
    final_C = float(stacking_cfg.get("final_C", 1.0))
    meta_input = str(stacking_cfg.get("meta_input", "prob")).lower()
    fusion_weights = stacking_cfg.get("fusion_weights", [1.0, 1.0, 1.0])
    threshold = float(stacking_cfg.get("threshold", 0.5))

    seeds = split_cfg.get("seeds", [42])
    n_splits = int(split_cfg.get("n_splits", 5))
    group_col = split_cfg.get("group_col", "Cluster_ID")
    label_col = split_cfg.get("label_col", "LABEL")
    stratify = bool(split_cfg.get("stratify_group_kfold", True))

    feature_cfg = feature_sets_cfg["feature_sets"][feature_set_name]
    blocks_cfg = feature_cfg["blocks"]

    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    all_metric_rows = []
    all_predictions = []

    for seed in seeds:
        outer_splits = build_group_kfold_splits(
            df=df,
            n_splits=n_splits,
            group_col=group_col,
            label_col=label_col,
            seed=seed,
            stratify=stratify,
        )

        for fold_id, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            train_df = subset_by_indices(df, train_idx)
            test_df = subset_by_indices(df, test_idx)
            y_train = train_df[label_col].to_numpy()
            y_test = test_df[label_col].to_numpy()

            _print_group_overlap(train_df, test_df, group_col, seed, fold_id, "outer")

            fold_result = run_stacking_outer_fold(
                train_df=train_df,
                test_df=test_df,
                blocks_cfg=blocks_cfg,
                y_train=y_train,
                y_test=y_test,
                inner_cfg=inner_cfg,
                group_col=group_col,
                label_col=label_col,
                branch_pca=branch_pca,
                base_C=base_C,
                final_C=final_C,
                meta_input=meta_input,
                fusion_weights=fusion_weights,
                random_state=seed,
                threshold=threshold,
            )

            fold_metrics = {
                "window_only": fold_result["branch_metrics"]["window"],
                "site_only": fold_result["branch_metrics"]["site"],
                "manual_only": fold_result["branch_metrics"]["manual"],
                "mean_fusion": fold_result["mean_metrics"],
                "weighted_mean_fusion": fold_result["weighted_metrics"],
                "stacking_plr": fold_result["stacking_metrics"],
            }
            for model_name, metrics in fold_metrics.items():
                all_metric_rows.append(_metrics_row(model_name, seed, fold_id, metrics))

            pred_df = pd.DataFrame(
                {
                    id_col: test_df[id_col].values,
                    "true_label": y_test,
                    "p_window": fold_result["test_probs"]["window"],
                    "p_site": fold_result["test_probs"]["site"],
                    "p_manual": fold_result["test_probs"]["manual"],
                    "p_mean_fusion": fold_result["mean_probs"],
                    "p_weighted_mean_fusion": fold_result["weighted_probs"],
                    "final_probability": fold_result["final_probs"],
                    "final_prediction": (fold_result["final_probs"] >= threshold).astype(int),
                    "seed": seed,
                    "outer_fold": fold_id,
                }
            )
            fold_pred_path = output_dir / f"seed{seed}_fold_{fold_id}_predictions.csv"
            save_csv(pred_df, fold_pred_path)
            all_predictions.append(pred_df)

            meta_path = output_dir / f"seed{seed}_fold_{fold_id}_meta_coef.json"
            save_json(fold_result["meta_coef"], meta_path)

            print(
                f"[INFO] seed={seed} fold={fold_id} stacking_plr "
                f"AUROC={fold_result['stacking_metrics']['auroc']:.4f} "
                f"AUPRC={fold_result['stacking_metrics']['auprc']:.4f}"
            )

    metrics_df = pd.DataFrame(all_metric_rows)
    save_csv(metrics_df, output_dir / "stacking_metrics_by_fold.csv")

    summary_rows = []
    metric_cols = [c for c in metrics_df.columns if c not in {"model", "seed", "fold"}]
    for model_name in SUMMARY_MODELS:
        group = metrics_df[metrics_df["model"] == model_name]
        if group.empty:
            continue
        row = {"model": model_name}
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=0))
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    save_csv(summary_df, output_dir / "stacking_summary.csv")

    all_pred_df = pd.concat(all_predictions, ignore_index=True)
    save_csv(all_pred_df, output_dir / "all_folds_predictions.csv")

    run_meta = {
        "feature_set": feature_set_name,
        "output_dir": str(output_dir),
        "base_C": base_C,
        "final_C": final_C,
        "meta_input": meta_input,
        "branch_pca": branch_pca,
        "inner_cv": inner_cfg,
        "fusion_weights": fusion_weights,
        "n_samples": int(len(df)),
        "n_folds": n_splits,
        "seeds": seeds,
    }
    save_json(run_meta, output_dir / "run_meta.json")

    return metrics_df, summary_df, all_pred_df, run_meta
