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
    load_import_export_binary_dataframe,
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

    - ESM window31: legacy c05/c08 files (same pool as run_150722_window21).
    - ESM window < 31 with c80: window31 legacy c08 for CV grouping.
    - Other windows: use c50/c80 files from current-site clustering.
    """
    cluster_tag = str(cluster_tag)
    window_size = int(window_size)
    if window_size == 31:
        if cluster_tag == "c80":
            return "c08"
        if cluster_tag == "c50":
            return "c05"
    elif window_size < 31 and cluster_tag == "c80":
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_cfg", type=str, required=True)
    parser.add_argument(
        "--output_tag",
        type=str,
        default=None,
        help="Run suffix, e.g. window41_c80 -> results/run_<ts>_window41_c80.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Override blocks.esm.window_size for the selected feature set.",
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
        "--feature_set",
        type=str,
        default="esm_plus_manual_all",
        help="Feature set name whose esm.window_size will be overridden.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override runtime device: auto, cuda, cuda:<id>, or cpu.",
    )
    return parser.parse_args()


def apply_runtime_overrides(
    exp_cfg,
    feature_sets_cfg,
    window_size,
    cluster_csv,
    cluster_tag,
    feature_set,
):
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
            if int(window_size) == 31 and cluster_tag in {"c50", "c80"} and file_tag in {"c05", "c08"}:
                print(
                    f"[INFO] ESM window_size=31; using legacy window31_{file_tag} cluster "
                    f"(sweep tag {cluster_tag})"
                )
            elif int(window_size) < 31 and cluster_window_size == 31:
                msg = (
                    f"[INFO] ESM window_size={window_size} < 31; "
                    f"using window31 cluster for CV grouping"
                )
                if cluster_tag == "c80" and file_tag == "c08":
                    msg += " (legacy c08 file for 80% identity)"
                print(msg)
            elif int(window_size) != int(cluster_window_size):
                print(
                    f"[INFO] ESM window_size={window_size}, "
                    f"cluster_window_size={cluster_window_size}"
                )

    if window_size is not None:
        feature_sets = feature_sets_cfg.get("feature_sets", {})
        targets = [feature_set] if feature_set in feature_sets else list(feature_sets.keys())
        updated = False
        for fs_name in targets:
            blocks = feature_sets.get(fs_name, {}).get("blocks", {})
            for block_name, block_cfg in blocks.items():
                if block_cfg.get("type") != "esm_window_site_pca":
                    continue
                block_cfg["window_size"] = int(window_size)
                print(
                    f"[INFO] Override {fs_name}.{block_name}.window_size: {window_size}"
                )
                updated = True
        if not updated:
            raise ValueError(
                f"No esm_window_site_pca blocks found to override window_size={window_size}"
            )

    return exp_cfg, feature_sets_cfg, cluster_window_size


def infer_output_tag(window_size, cluster_csv, cluster_tag, output_tag):
    if output_tag:
        return output_tag

    if window_size is None:
        return None

    tag_suffix = cluster_tag
    if tag_suffix is None and cluster_csv:
        match = re.search(r"window(\d+)_(c\d+)_cluster", str(cluster_csv))
        if match:
            tag_suffix = match.group(2)

    if tag_suffix is not None:
        return f"window{window_size}_{tag_suffix}"

    return f"window{window_size}"


def main():
    args = parse_args()
    exp_cfg = load_yaml(args.experiment_cfg)

    split_cfg = load_yaml(exp_cfg["configs"]["split_cfg"])
    feature_sets_cfg = load_yaml(exp_cfg["configs"]["feature_sets_cfg"])
    train_cfg = load_yaml(exp_cfg["configs"]["train_cfg"])

    exp_cfg, feature_sets_cfg, cluster_window_size = apply_runtime_overrides(
        exp_cfg=exp_cfg,
        feature_sets_cfg=feature_sets_cfg,
        window_size=args.window_size,
        cluster_csv=args.cluster_csv,
        cluster_tag=args.cluster_tag,
        feature_set=args.feature_set,
    )

    runtime_cfg = exp_cfg.get("runtime", {})
    if args.device is not None:
        runtime_cfg["device"] = args.device
    runtime_cfg["device"] = resolve_device(runtime_cfg.get("device", "auto"))
    output_root = runtime_cfg.get("output_dir", "results")

    data_cfg = exp_cfg.get("data", {})
    df = load_import_export_binary_dataframe(
        positive_csv=data_cfg["positive_csv"],
        fasta_path=data_cfg["fasta_path"],
        cluster_csv=data_cfg.get("cluster_csv"),
        acc_col=split_cfg.get("acc_col", "ACC_ID"),
        site_col=split_cfg.get("site_col", "POSITION"),
        cluster_key_col=split_cfg.get("cluster_key_col", "INDEX"),
        cluster_group_col=split_cfg.get("cluster_group_col", "Cluster_ID"),
        positive_class=data_cfg.get("positive_class", "import"),
    )

    timestamp = get_timestamp()
    output_tag = infer_output_tag(
        args.window_size,
        args.cluster_csv or exp_cfg["data"].get("cluster_csv"),
        args.cluster_tag,
        args.output_tag,
    )
    if output_tag:
        out_dir = Path(output_root) / f"run_{timestamp}_{output_tag}" / "Import_vs_Export"
    else:
        out_dir = Path(output_root) / f"run_{timestamp}" / "Import_vs_Export"
    ensure_dir(out_dir)

    all_metrics_df, best_metrics_df, feature_summary_df, history_df, run_meta = run_experiment_grid(
        df=df,
        split_cfg=split_cfg,
        feature_sets_cfg=feature_sets_cfg,
        train_cfg=train_cfg,
        task_name="Import vs Export",
        device=runtime_cfg.get("device", "cuda"),
        output_dir=out_dir,
        save_fold_artifacts=runtime_cfg.get("save_fold_artifacts", True),
    )

    run_meta["window_size"] = args.window_size
    run_meta["cluster_csv"] = exp_cfg["data"].get("cluster_csv")
    run_meta["cluster_window_size"] = cluster_window_size
    run_meta["cluster_tag"] = args.cluster_tag
    run_meta["output_tag"] = output_tag
    run_meta["feature_set"] = args.feature_set

    save_csv(all_metrics_df, out_dir / "metrics_all_runs.csv")
    save_csv(best_metrics_df, out_dir / "metrics_best_per_seed.csv")
    save_csv(feature_summary_df, out_dir / "feature_summary.csv")
    save_csv(history_df, out_dir / "training_history.csv")
    save_json(run_meta, out_dir / "run_meta.json")

    print(f"[DONE] Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
