import argparse
import copy
import re
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_yaml,
    ensure_dir,
    save_csv,
    save_json,
    get_timestamp,
    load_task_dataframe,
)
from src.train.experiment_runner import run_experiment_grid


DEFAULT_CLUSTER_DIR = Path(__file__).resolve().parents[1] / "data" / "cluster"


def resolve_device(device):
    device = str(device).strip().lower()
    if device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Resolved device=auto -> {resolved}")
        return resolved
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] Requested device={device}, but CUDA is unavailable. Falling back to cpu.")
        return "cpu"
    return device


def resolve_cluster_window_for_esm(window_size):
    """ESM window < 31 uses window31 CD-HIT clusters for GroupKFold."""
    window_size = int(window_size)
    return 31 if window_size < 31 else window_size


def resolve_cluster_file_tag(cluster_tag, window_size):
    """Map sweep tag to on-disk cluster suffix.

    Only window31 @ 80% identity reuses the existing legacy c08 file (10823 sites).
    All other sweeps use updated current-site cluster files (c50 / c80, 10611 sites).
    """
    cluster_tag = str(cluster_tag)
    window_size = int(window_size)
    if window_size == 31 and cluster_tag == "c80":
        return "c08"
    return cluster_tag


def resolve_cluster_csv_path(window_size, cluster_tag, cluster_dir=None):
    cluster_dir = Path(cluster_dir or DEFAULT_CLUSTER_DIR)
    cluster_window = resolve_cluster_window_for_esm(window_size)
    file_tag = resolve_cluster_file_tag(cluster_tag, window_size)
    cluster_path = cluster_dir / f"tf_phos_site_window{cluster_window}_{file_tag}_cluster.csv"
    if not cluster_path.exists():
        raise FileNotFoundError(f"cluster_csv does not exist: {cluster_path}")
    return cluster_path, cluster_window


TASK_CHOICES = [
    "Nuclear Import",
    "Nuclear Export",
    "Functional Transport",
    "all",
    "all_with_functional",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_cfg", type=str, required=True)
    parser.add_argument(
        "--train_cfg",
        type=str,
        default=None,
        help="Override train config path from experiment yaml.",
    )
    parser.add_argument(
        "--split_cfg",
        type=str,
        default=None,
        help="Override split config path from experiment yaml.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=None,
        help="Override CV folds on development set; use 1 for single train/val split.",
    )
    parser.add_argument(
        "--output_tag",
        type=str,
        default=None,
        help="Optional run name suffix, e.g. window41_c80 -> results/run_<ts>_window41_c80.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Override esm_graph.blocks.esm.window_size in feature_sets_cfg.",
    )
    parser.add_argument(
        "--cluster_csv",
        type=str,
        default=None,
        help="Override data.cluster_csv in experiment cfg.",
    )
    parser.add_argument(
        "--cluster_tag",
        type=str,
        default=None,
        help="Cluster suffix tag (e.g. c50, c80). For window_size < 31, uses window31 cluster.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override runtime device: auto, cuda, cuda:<id>, or cpu.",
    )
    return parser.parse_args()


def resolve_requested_tasks(task):
    if task == "all":
        return ["Nuclear Import", "Nuclear Export"]
    if task == "all_with_functional":
        return ["Nuclear Import", "Nuclear Export", "Functional Transport"]
    return [task]


def run_one_task(
    task_name,
    exp_cfg,
    split_cfg,
    feature_sets_cfg,
    train_cfg,
    base_output_dir,
    runtime_meta=None,
):
    data_cfg = exp_cfg["data"]
    runtime_cfg = exp_cfg.get("runtime", {})

    df = load_task_dataframe(
        positive_csv=data_cfg["positive_csv"],
        negative_csv=data_cfg["negative_csv"],
        fasta_path=data_cfg["fasta_path"],
        task_name=task_name,
        cluster_csv=data_cfg.get("cluster_csv"),
        acc_col=split_cfg.get("acc_col", "ACC_ID"),
        site_col=split_cfg.get("site_col", "POSITION"),
        cluster_key_col=split_cfg.get("cluster_key_col", "INDEX"),
        cluster_group_col=split_cfg.get("cluster_group_col", "Cluster_ID"),
        negative_min_distance=data_cfg.get("negative_min_distance", 15),
        unlabeled_mode=data_cfg.get("unlabeled_mode", "same_positive_tf_far"),
        pdb_dir=data_cfg.get("pdb_dir"),
        require_pdb=bool(data_cfg.get("require_pdb", False)),
    )

    task_output_dir = Path(base_output_dir) / task_name.replace(" ", "_")
    artifact_root = task_output_dir / "artifacts"

    ensure_dir(task_output_dir)
    ensure_dir(artifact_root)

    (
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
    ) = run_experiment_grid(
        df=df,
        split_cfg=split_cfg,
        feature_sets_cfg=feature_sets_cfg,
        train_cfg=train_cfg,
        task_name=task_name,
        device=runtime_cfg.get("device", "cuda"),
        artifact_root=artifact_root,
    )

    save_csv(all_metrics_df, task_output_dir / "metrics_all_folds.csv")
    save_csv(cv_summary_df, task_output_dir / "metrics_cv_summary.csv")
    save_csv(feature_summary_df, task_output_dir / "feature_summary.csv")
    save_csv(history_df, task_output_dir / "training_history.csv")
    save_csv(cv_val_predictions_df, task_output_dir / "cv_val_predictions.csv")
    save_csv(fixed_test_predictions_df, task_output_dir / "fixed_test_predictions_by_fold.csv")
    save_csv(best_model_summary_df, task_output_dir / "best_model_summary.csv")
    save_csv(fixed_test_samples_df, task_output_dir / "fixed_test_samples.csv")
    save_csv(dev_samples_df, task_output_dir / "development_samples.csv")

    if runtime_meta:
        run_meta.update(runtime_meta)

    save_json(run_meta, task_output_dir / "run_meta.json")

    print(f"[DONE] {task_name} results saved to: {task_output_dir}")
    print(f"[DONE] {task_name} artifacts saved to: {artifact_root}")
    print(f"[DONE] {task_name} CV validation predictions saved to: {task_output_dir / 'cv_val_predictions.csv'}")
    print(f"[DONE] {task_name} fixed test predictions saved to: {task_output_dir / 'fixed_test_predictions_by_fold.csv'}")
    print(f"[DONE] {task_name} best model summary saved to: {task_output_dir / 'best_model_summary.csv'}")
    print(f"[DONE] {task_name} best model artifacts saved to: {artifact_root / 'best_models'}")


def apply_runtime_overrides(exp_cfg, feature_sets_cfg, window_size, cluster_csv, cluster_tag):
    exp_cfg = copy.deepcopy(exp_cfg)
    feature_sets_cfg = copy.deepcopy(feature_sets_cfg)
    cluster_window_size = None

    if cluster_csv is not None:
        cluster_path = Path(cluster_csv)
        if not cluster_path.exists():
            raise FileNotFoundError(f"cluster_csv does not exist: {cluster_path}")
    elif cluster_tag is not None:
        if window_size is None:
            raise ValueError("--cluster_tag requires --window_size to resolve cluster path")
        cluster_path, cluster_window_size = resolve_cluster_csv_path(
            window_size=window_size,
            cluster_tag=cluster_tag,
        )
        cluster_csv = str(cluster_path)
    else:
        cluster_path = None

    if cluster_csv is not None:
        exp_cfg.setdefault("data", {})["cluster_csv"] = str(cluster_path)
        if cluster_window_size is None:
            match = re.search(r"window(\d+)_(c\d+)_cluster", str(cluster_path))
            cluster_window_size = int(match.group(1)) if match else None
        print(f"[INFO] Override cluster_csv: {cluster_path}")
        if window_size is not None and cluster_window_size is not None:
            file_tag = resolve_cluster_file_tag(cluster_tag or "", window_size) if cluster_tag else None
            if int(window_size) == 31 and cluster_tag == "c80" and file_tag == "c08":
                print(
                    f"[INFO] ESM window_size=31; reusing existing legacy window31_c08 cluster "
                    f"(sweep tag {cluster_tag}, no re-cluster)"
                )
            elif int(window_size) < 31 and cluster_window_size == 31:
                print(
                    f"[INFO] ESM window_size={window_size} < 31; "
                    f"using window31 cluster for CV grouping"
                )
            elif int(window_size) != int(cluster_window_size):
                print(
                    f"[INFO] ESM window_size={window_size}, "
                    f"cluster_window_size={cluster_window_size}"
                )

    if window_size is not None:
        esm_block = (
            feature_sets_cfg.get("feature_sets", {})
            .get("esm_graph", {})
            .get("blocks", {})
            .get("esm")
        )
        if esm_block is None:
            raise ValueError("feature_sets_cfg missing feature_sets.esm_graph.blocks.esm")
        esm_block["window_size"] = int(window_size)
        print(f"[INFO] Override esm window_size: {window_size}")

    return exp_cfg, feature_sets_cfg, cluster_window_size


def infer_output_tag(window_size, cluster_csv, cluster_tag, output_tag):
    if output_tag:
        return output_tag

    if window_size is None and cluster_csv is None and cluster_tag is None:
        return None

    tag_suffix = cluster_tag
    if tag_suffix is None and cluster_csv:
        match = re.search(r"window(\d+)_(c\d+)_cluster", str(cluster_csv))
        if match:
            tag_suffix = match.group(2)

    if window_size is not None and tag_suffix is not None:
        return f"window{window_size}_{tag_suffix}"

    if window_size is not None:
        return f"window{window_size}"

    if cluster_csv:
        return Path(cluster_csv).stem
    return None


def main():
    args = parse_args()

    exp_cfg = load_yaml(args.experiment_cfg)

    task = exp_cfg["task"]
    if task not in TASK_CHOICES:
        raise ValueError(f"Unsupported task: {task}")

    cfg_paths = exp_cfg["configs"]
    split_cfg = load_yaml(cfg_paths["split_cfg"])
    feature_sets_cfg = load_yaml(cfg_paths["feature_sets_cfg"])
    train_cfg = load_yaml(cfg_paths["train_cfg"])

    if args.split_cfg is not None:
        split_cfg = load_yaml(args.split_cfg)
    if args.train_cfg is not None:
        train_cfg = load_yaml(args.train_cfg)
    if args.n_splits is not None:
        split_cfg["n_splits"] = int(args.n_splits)

    exp_cfg, feature_sets_cfg, cluster_window_size = apply_runtime_overrides(
        exp_cfg=exp_cfg,
        feature_sets_cfg=feature_sets_cfg,
        window_size=args.window_size,
        cluster_csv=args.cluster_csv,
        cluster_tag=args.cluster_tag,
    )

    if args.device is not None:
        exp_cfg.setdefault("runtime", {})["device"] = args.device

    runtime_cfg = exp_cfg.get("runtime", {})
    runtime_cfg["device"] = resolve_device(runtime_cfg.get("device", "auto"))
    output_root = runtime_cfg.get("output_dir", "results")

    timestamp = get_timestamp()
    output_tag = infer_output_tag(
        args.window_size,
        args.cluster_csv or exp_cfg["data"].get("cluster_csv"),
        args.cluster_tag,
        args.output_tag,
    )
    if output_tag:
        base_output_dir = Path(output_root) / f"run_{timestamp}_{output_tag}"
    else:
        base_output_dir = Path(output_root) / f"run_{timestamp}"
    ensure_dir(base_output_dir)

    tasks = resolve_requested_tasks(task)

    runtime_meta = {
        "window_size": args.window_size,
        "cluster_csv": exp_cfg["data"].get("cluster_csv"),
        "cluster_window_size": cluster_window_size,
        "cluster_tag": args.cluster_tag,
        "output_tag": output_tag,
    }

    for task_name in tasks:
        run_one_task(
            task_name=task_name,
            exp_cfg=exp_cfg,
            split_cfg=split_cfg,
            feature_sets_cfg=feature_sets_cfg,
            train_cfg=train_cfg,
            base_output_dir=base_output_dir,
            runtime_meta=runtime_meta,
        )


if __name__ == "__main__":
    main()
