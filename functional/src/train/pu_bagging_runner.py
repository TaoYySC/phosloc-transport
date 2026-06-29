from pathlib import Path
import copy
import json

import numpy as np
import pandas as pd

from src.data.split import (
    build_fixed_test_split,
    iter_dev_train_val_splits,
    subset_by_indices,
)
from src.evaluate.metrics import compute_metrics, find_best_threshold
from src.train.experiment_runner import (
    _build_feature_summary_row,
    _build_prediction_rows,
    _copy_best_artifact,
    _get_selection_score,
    _safe_name,
    _summarize_cv_metrics,
    check_group_leakage,
)
from src.train.negative_chunk_utils import (
    build_fixed_ratio_bag_plan,
    compute_balanced_bag_count,
    sample_unlabeled_bags,
    split_negative_by_group,
)
from src.train.single_run import run_single_split, build_fold_eval_features


def _ensemble_prob(bag_results, split_name):
    prob_key = f"{split_name}_prob"
    prob_matrix = np.stack(
        [np.asarray(result[prob_key]).reshape(-1) for result in bag_results],
        axis=0,
    )
    return prob_matrix.mean(axis=0), prob_matrix.std(axis=0)


def _build_bag_metrics_row(
    fold,
    seed,
    bag_idx,
    feature_set_name,
    train_df_sub,
    u_train_all,
    result,
):
    row = {
        "fold": int(fold),
        "seed": int(seed),
        "bag_idx": int(bag_idx),
        "feature_set": feature_set_name,
        "n_train_sub": int(len(train_df_sub)),
        "n_train_pos": int((train_df_sub["LABEL"] == 1).sum()),
        "n_train_unlabeled": int((train_df_sub["LABEL"] == 0).sum()),
        "n_unlabeled_total_available": int(len(u_train_all)),
        "x_train_dim": int(result["x_train_dim"]),
        "n_train": int(result["n_train"]),
        "n_val": int(result["n_val"]),
        "n_test": int(result["n_test"]),
        "best_threshold": float(result["best_threshold"]),
        "artifact_dir": result.get("artifact_dir", ""),
    }

    for split_name in ["train", "val", "test"]:
        metrics = result[f"{split_name}_metrics"]
        for metric_name, metric_value in metrics.items():
            row[f"{split_name}_{metric_name}"] = float(metric_value)

    return row


def _build_ensemble_metrics_row(
    fold,
    seed,
    feature_set_name,
    n_bags,
    df_val,
    df_test,
    ensemble_threshold,
    val_prob,
    test_prob,
    bag_results,
    bag_threshold_mean,
    bag_threshold_std,
):
    y_val = df_val["LABEL"].astype(int).values
    y_test = df_test["LABEL"].astype(int).values

    val_metrics = compute_metrics(y_val, val_prob, ensemble_threshold)
    test_metrics = compute_metrics(y_test, test_prob, ensemble_threshold)

    train_metric_keys = ["auroc", "auprc", "f1", "mcc", "threshold"]
    train_metrics = {}
    for metric_key in train_metric_keys:
        train_metrics[metric_key] = float(
            np.mean([float(result["train_metrics"][metric_key]) for result in bag_results])
        )

    row = {
        "fold": int(fold),
        "seed": int(seed),
        "feature_set": feature_set_name,
        "n_bags": int(n_bags),
        "n_val": int(len(df_val)),
        "n_test": int(len(df_test)),
        "best_threshold": float(ensemble_threshold),
        "bag_threshold_mean": float(bag_threshold_mean),
        "bag_threshold_std": float(bag_threshold_std),
    }

    for split_name, metrics in [
        ("train", train_metrics),
        ("val", val_metrics),
        ("test", test_metrics),
    ]:
        for metric_name, metric_value in metrics.items():
            row[f"{split_name}_{metric_name}"] = float(metric_value)

    return row, {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def _save_ensemble_manifest(
    artifact_dir,
    fold,
    feature_set_name,
    n_bags,
    ensemble_threshold,
    bag_artifact_dirs,
    bag_thresholds,
    bag_plan=None,
):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "fold": int(fold),
        "feature_set": feature_set_name,
        "n_bags": int(n_bags),
        "ensemble_threshold": float(ensemble_threshold),
        "bag_thresholds": [float(x) for x in bag_thresholds],
        "bag_artifact_dirs": [str(x) for x in bag_artifact_dirs],
        "ensemble_method": "mean_prob",
        "threshold_selection": "ensemble_val_f1",
    }
    if bag_plan is not None:
        manifest["bag_plan"] = bag_plan
    with open(artifact_dir / "ensemble_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def run_pu_bagging_experiment(
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
    selection_metric = split_cfg.get("selection_metric", "val_auroc")
    bag_count_mode = split_cfg.get("bag_count_mode", "auto")
    n_bags_fixed = int(split_cfg.get("n_bags", 10))
    target_u_to_p_ratio = float(split_cfg.get("target_u_to_p_ratio", 2.0))
    min_bags = int(split_cfg.get("min_bags", 1))
    max_bags = int(split_cfg.get("max_bags", 50))
    bag_group_col = split_cfg.get("bag_group_col", "Cluster_ID")
    bag_seed_offset = int(split_cfg.get("bag_seed_offset", 0))
    u_sample_with_replacement = bool(split_cfg.get("u_sample_with_replacement", True))
    single_val_size = float(split_cfg.get("single_val_size", 0.2))

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

    fold_metrics_rows = []
    bag_metrics_rows = []
    feature_rows = []
    cv_val_rows = []
    fixed_test_rows = []
    best_by_feature_set = {}
    best_overall = None

    run_meta = {
        "task_name": task_name,
        "device": device,
        "training_mode": "pu_bagging",
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
        "single_val_size": float(single_val_size),
        "bag_count_mode": bag_count_mode,
        "n_bags_fixed": int(n_bags_fixed),
        "target_u_to_p_ratio": float(target_u_to_p_ratio),
        "min_bags": int(min_bags),
        "max_bags": int(max_bags),
        "u_sample_with_replacement": bool(u_sample_with_replacement),
        "seed": int(seed),
        "fixed_test_seed": int(fixed_test_seed),
        "fixed_test_size": float(fixed_test_size),
        "selection_metric": selection_metric,
        "best_model_selection": "validation_metric_on_ensemble",
        "group_col": group_col,
        "bag_group_col": bag_group_col,
        "label_col": label_col,
        "unlabeled_semantics": "LABEL=0 is unlabeled; training uses PU loss (nnPU by default)",
        "bag_partition": "Unlabeled sites partitioned by bag_group_col without replacement; each bag uses all P_train",
        "feature_sets": list(feature_sets.keys()),
        "n_samples_total": int(len(df)),
        "n_development": int(len(df_dev)),
        "n_fixed_test": int(len(df_fixed_test)),
        "n_positive_total": int((df[label_col] == 1).sum()),
        "n_unlabeled_total": int((df[label_col] == 0).sum()),
        "n_positive_development": int((df_dev[label_col] == 1).sum()),
        "n_unlabeled_development": int((df_dev[label_col] == 0).sum()),
        "n_positive_fixed_test": int((df_fixed_test[label_col] == 1).sum()),
        "n_unlabeled_fixed_test": int((df_fixed_test[label_col] == 0).sum()),
        "artifact_root": "" if artifact_root is None else str(artifact_root),
    }

    for fold, train_idx, val_idx in iter_dev_train_val_splits(
        df=df_dev,
        n_splits=n_splits,
        seed=seed,
        group_col=group_col,
        label_col=label_col,
        stratify=stratify,
        single_val_size=single_val_size,
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

        df_pos_train = df_train[df_train[label_col] == 1].copy().reset_index(drop=True)
        df_u_train = df_train[df_train[label_col] == 0].copy().reset_index(drop=True)
        if bag_group_col not in df_u_train.columns:
            raise ValueError(
                f"bag_group_col '{bag_group_col}' not found in unlabeled train dataframe"
            )
        df_u_train[bag_group_col] = df_u_train[bag_group_col].astype(str).str.strip()

        if bag_count_mode == "auto":
            bag_plan = compute_balanced_bag_count(
                n_pos=len(df_pos_train),
                n_u=len(df_u_train),
                target_u_to_p_ratio=target_u_to_p_ratio,
                min_bags=min_bags,
                max_bags=max_bags,
            )
            n_bags_fold = int(bag_plan["n_bags"])
            if bag_plan["capped_by_max_bags"]:
                print(
                    f"[WARN] fold={fold}: need {bag_plan['n_bags_needed_uncapped']} bags "
                    f"for target U:P={target_u_to_p_ratio}, capped at max_bags={max_bags}. "
                    f"expected U:P={bag_plan['expected_u_to_p_ratio']:.2f}"
                )
            u_bags = split_negative_by_group(
                df_neg=df_u_train,
                n_chunks=n_bags_fold,
                group_col=bag_group_col,
                seed=inner_seed + bag_seed_offset,
            )
        elif bag_count_mode == "fixed_ratio":
            n_bags_fold = int(n_bags_fixed)
            bag_plan = build_fixed_ratio_bag_plan(
                n_pos=len(df_pos_train),
                n_u=len(df_u_train),
                n_bags=n_bags_fold,
                target_u_to_p_ratio=target_u_to_p_ratio,
                with_replacement=u_sample_with_replacement,
            )
            u_bags = sample_unlabeled_bags(
                df_u=df_u_train,
                n_bags=n_bags_fold,
                u_per_bag=bag_plan["u_per_bag_target"],
                seed=inner_seed + bag_seed_offset,
                with_replacement=u_sample_with_replacement,
            )
        else:
            bag_plan = {
                "n_bags": n_bags_fixed,
                "u_per_bag_target": int(
                    round(len(df_pos_train) * target_u_to_p_ratio)
                ),
                "expected_u_per_bag": int(
                    np.ceil(len(df_u_train) / max(n_bags_fixed, 1))
                ),
                "expected_u_to_p_ratio": float(
                    np.ceil(len(df_u_train) / max(n_bags_fixed, 1))
                    / max(len(df_pos_train), 1)
                ),
                "u_coverage": 1.0,
                "capped_by_max_bags": False,
            }
            n_bags_fold = n_bags_fixed
            u_bags = split_negative_by_group(
                df_neg=df_u_train,
                n_chunks=n_bags_fold,
                group_col=bag_group_col,
                seed=inner_seed + bag_seed_offset,
            )

        print(
            f"[INFO] fold={fold}: P_train={len(df_pos_train)} U_train={len(df_u_train)} "
            f"n_bags={n_bags_fold} expected_U:P={bag_plan['expected_u_to_p_ratio']:.2f} "
            f"u_per_bag={bag_plan.get('expected_u_per_bag', bag_plan.get('u_per_bag_target'))} "
            f"u_coverage={bag_plan.get('u_coverage', 1.0):.3f} "
            f"bag_mode={bag_count_mode}"
        )

        for feature_set_name, feature_cfg in feature_sets.items():
            feature_cfg_local = copy.deepcopy(feature_cfg)
            bag_results = []
            bag_artifact_dirs = []
            bag_thresholds = []
            bag_feature_rows = []

            fold_feature_root = None
            if artifact_root is not None:
                fold_feature_root = (
                    Path(artifact_root)
                    / f"fold_{int(fold)}"
                    / _safe_name(feature_set_name)
                )

            print(
                f"[INFO] fold={fold} feature_set={feature_set_name}: "
                "precomputing val/test features (once per fold)"
            )
            fold_eval_features = build_fold_eval_features(
                feature_cfg=feature_cfg_local,
                val_df=df_val,
                test_df=df_test,
            )

            for bag_idx, u_chunk_df in enumerate(u_bags, start=1):
                train_df_sub = pd.concat(
                    [df_pos_train, u_chunk_df],
                    axis=0,
                    ignore_index=True,
                )

                bag_artifact_dir = None
                if fold_feature_root is not None:
                    bag_artifact_dir = fold_feature_root / f"bag_{int(bag_idx)}"

                bag_seed = inner_seed * 100 + bag_idx
                result = run_single_split(
                    train_df=train_df_sub,
                    val_df=df_val.copy(),
                    test_df=df_test.copy(),
                    feature_cfg=feature_cfg_local,
                    train_cfg=copy.deepcopy(train_cfg),
                    seed=bag_seed,
                    device=device,
                    fixed_retained_columns_by_family=None,
                    artifact_dir=bag_artifact_dir,
                    feature_set_name=feature_set_name,
                    fold_eval_features=fold_eval_features,
                )

                bag_results.append(result)
                bag_thresholds.append(float(result["best_threshold"]))
                if bag_artifact_dir is not None:
                    bag_artifact_dirs.append(str(bag_artifact_dir))

                bag_metrics_rows.append(
                    _build_bag_metrics_row(
                        fold=fold,
                        seed=bag_seed,
                        bag_idx=bag_idx,
                        feature_set_name=feature_set_name,
                        train_df_sub=train_df_sub,
                        u_train_all=df_u_train,
                        result=result,
                    )
                )

                feature_row = _build_feature_summary_row(
                    fold=fold,
                    seed=bag_seed,
                    feature_set_name=feature_set_name,
                    result=result,
                )
                feature_row["bag_idx"] = int(bag_idx)
                bag_feature_rows.append(feature_row)

            val_prob, val_prob_std = _ensemble_prob(bag_results, "val")
            test_prob, test_prob_std = _ensemble_prob(bag_results, "test")

            ensemble_threshold = find_best_threshold(
                df_val[label_col].astype(int).values,
                val_prob,
            )

            metrics_row, metrics_bundle = _build_ensemble_metrics_row(
                fold=fold,
                seed=inner_seed,
                feature_set_name=feature_set_name,
                n_bags=n_bags_fold,
                df_val=df_val,
                df_test=df_test,
                ensemble_threshold=ensemble_threshold,
                val_prob=val_prob,
                test_prob=test_prob,
                bag_results=bag_results,
                bag_threshold_mean=float(np.mean(bag_thresholds)),
                bag_threshold_std=float(np.std(bag_thresholds, ddof=1)) if len(bag_thresholds) > 1 else 0.0,
            )
            metrics_row["n_pos_train"] = int(len(df_pos_train))
            metrics_row["n_u_train"] = int(len(df_u_train))
            metrics_row["target_u_to_p_ratio"] = float(target_u_to_p_ratio)
            metrics_row["expected_u_to_p_ratio"] = float(bag_plan["expected_u_to_p_ratio"])
            metrics_row["expected_u_per_bag"] = int(bag_plan["expected_u_per_bag"])
            metrics_row["capped_by_max_bags"] = int(bool(bag_plan.get("capped_by_max_bags", False)))
            metrics_row["val_prob_std_mean"] = float(val_prob_std.mean())
            metrics_row["test_prob_std_mean"] = float(test_prob_std.mean())
            fold_metrics_rows.append(metrics_row)

            if fold_feature_root is not None:
                _save_ensemble_manifest(
                    artifact_dir=fold_feature_root,
                    fold=fold,
                    feature_set_name=feature_set_name,
                    n_bags=n_bags_fold,
                    ensemble_threshold=ensemble_threshold,
                    bag_artifact_dirs=bag_artifact_dirs,
                    bag_thresholds=bag_thresholds,
                    bag_plan=bag_plan,
                )

            feature_rows.extend(bag_feature_rows)

            cv_val_row = _build_prediction_rows(
                df_split=df_val,
                prob=val_prob,
                threshold=ensemble_threshold,
                fold=fold,
                seed=inner_seed,
                feature_set_name=feature_set_name,
                prediction_source="cv_validation_ensemble",
            )
            cv_val_row["prob_std"] = val_prob_std
            cv_val_rows.append(cv_val_row)

            fixed_test_row = _build_prediction_rows(
                df_split=df_test,
                prob=test_prob,
                threshold=ensemble_threshold,
                fold=fold,
                seed=inner_seed,
                feature_set_name=feature_set_name,
                prediction_source="fixed_test_ensemble",
            )
            fixed_test_row["prob_std"] = test_prob_std
            fixed_test_rows.append(fixed_test_row)

            ensemble_result = {
                "val_metrics": metrics_bundle["val_metrics"],
                "test_metrics": metrics_bundle["test_metrics"],
                "train_metrics": metrics_bundle["train_metrics"],
                "best_threshold": float(ensemble_threshold),
                "artifact_dir": "" if fold_feature_root is None else str(fold_feature_root),
            }
            selection_score = _get_selection_score(
                result=ensemble_result,
                selection_metric=selection_metric,
            )

            best_record = {
                "task_name": task_name,
                "fold": int(fold),
                "seed": int(inner_seed),
                "feature_set": feature_set_name,
                "selection_metric": selection_metric,
                "selection_score": float(selection_score),
                "best_threshold": float(ensemble_threshold),
                "n_bags": int(n_bags_fold),
                "expected_u_to_p_ratio": float(bag_plan["expected_u_to_p_ratio"]),
                "artifact_dir": ensemble_result["artifact_dir"],
                "val_auroc": float(metrics_bundle["val_metrics"]["auroc"]),
                "val_auprc": float(metrics_bundle["val_metrics"]["auprc"]),
                "val_f1": float(metrics_bundle["val_metrics"]["f1"]),
                "val_mcc": float(metrics_bundle["val_metrics"]["mcc"]),
                "test_auroc": float(metrics_bundle["test_metrics"]["auroc"]),
                "test_auprc": float(metrics_bundle["test_metrics"]["auprc"]),
                "test_f1": float(metrics_bundle["test_metrics"]["f1"]),
                "test_mcc": float(metrics_bundle["test_metrics"]["mcc"]),
            }

            current_best_for_feature = best_by_feature_set.get(feature_set_name)
            if (
                current_best_for_feature is None
                or selection_score > current_best_for_feature["selection_score"]
            ):
                best_by_feature_set[feature_set_name] = best_record

            if best_overall is None or selection_score > best_overall["selection_score"]:
                best_overall = best_record

    all_metrics_df = pd.DataFrame(fold_metrics_rows)
    bag_metrics_df = pd.DataFrame(bag_metrics_rows)
    cv_summary_df = _summarize_cv_metrics(all_metrics_df)
    feature_summary_df = pd.DataFrame(feature_rows)

    cv_val_predictions_df = (
        pd.concat(cv_val_rows, axis=0, ignore_index=True)
        if len(cv_val_rows) > 0
        else pd.DataFrame()
    )
    fixed_test_predictions_df = (
        pd.concat(fixed_test_rows, axis=0, ignore_index=True)
        if len(fixed_test_rows) > 0
        else pd.DataFrame()
    )

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
        bag_metrics_df,
        cv_summary_df,
        feature_summary_df,
        cv_val_predictions_df,
        fixed_test_predictions_df,
        best_model_summary_df,
        fixed_test_samples_df,
        dev_samples_df,
        run_meta,
    )
