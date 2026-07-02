#!/usr/bin/env python3
# Import vs export direction inference on new phosphosites (ensemble prediction from saved fold artifacts).

"""
Predict Import vs Export for new phosphosites using saved fold artifacts.

Default preset: ESM Window only + SupCon+CE, Import=LABEL 1
  run_dir: results/run_20260612_125646_esm_window_only_supcon_ce_import_pos/Import_vs_Export
  input:   functional/data/dataset_phos_site/tf_all_phos_site_for_prediction.csv

Each fold artifact contains pipeline.joblib (PLS feature transform) and model.joblib.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUNCTIONAL_ROOT = Path(__file__).resolve().parents[2] / "functional"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.platt import (
    apply_platt_calibrator,
    load_platt_calibrator,
    sklearn_decision_scores,
)
from src.data.torch_dataset import MultiInputDataset
from src.models.build_model import build_model
from src.train.single_run import _as_numpy_matrix, _predict_sklearn_proba
from src.train.trainer import multi_input_collate
from src.utils import (
    attach_sequence_and_cluster,
    drop_invalid_site_rows,
    parse_fasta,
    standardize_sample_table,
)

DEFAULT_INPUT_CSV = (
    str(FUNCTIONAL_ROOT / "data" / "dataset_phos_site" / "tf_all_phos_site_for_prediction.csv")
)
DEFAULT_FASTA = PROJECT_ROOT / "data/fasta/transcription_fasta.fasta"
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "data/model_artifacts/run_20260612_125646_esm_window_only_supcon_ce_import_pos/Import_vs_Export"
)
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT
    / "results/1_transport_classifier_results/esm_window_only_import_pos_predictions"
    / "tf_all_phos_site_predictions.csv"
)
DEFAULT_FEATURE_SET = "esm_window_only_supcon_ce"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ensemble predict Import vs Export on new sites. "
            "Default: ESM Window only (Import=LABEL 1)."
        )
    )
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--fasta_path", type=str, default=str(DEFAULT_FASTA))
    parser.add_argument("--run_dir", type=str, default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output_csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--cluster_csv", type=str, default=None)
    parser.add_argument(
        "--feature_set",
        type=str,
        default=None,
        help=(
            "Only use fold artifacts for this feature_set. "
            f"Default: auto from run_meta or {DEFAULT_FEATURE_SET!r}."
        ),
    )
    parser.add_argument(
        "--positive_class",
        type=str,
        default=None,
        choices=["import", "export"],
        help="Which class is LABEL=1 / prob_positive. Default: from run_meta.json.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for torch models: auto, cuda, cuda:<id>, or cpu. Default: auto.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--acc_col", type=str, default="ACC_ID")
    parser.add_argument("--site_col", type=str, default="POSITION")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold on mean prob_positive. Default: from artifact meta.",
    )
    parser.add_argument(
        "--save_dropped_csv",
        type=str,
        default=None,
        help="Optional path to save rows dropped during preprocessing.",
    )
    parser.add_argument(
        "--platt_calibrator",
        type=str,
        default=None,
        help="Path to platt_calibrator.json. Default: <run_dir>/platt_calibrator.json",
    )
    parser.add_argument(
        "--use_platt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply Platt scaling to sklearn decision_function scores.",
    )
    return parser.parse_args()


def load_meta(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_run_meta(run_dir):
    meta_path = Path(run_dir) / "run_meta.json"
    if meta_path.exists():
        return load_meta(meta_path)
    return {}


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


def resolve_feature_set(args, run_meta):
    if args.feature_set:
        return args.feature_set
    if run_meta.get("feature_set"):
        return run_meta["feature_set"]
    feature_sets = run_meta.get("feature_sets") or []
    if len(feature_sets) == 1:
        return feature_sets[0]
    return DEFAULT_FEATURE_SET


def resolve_positive_class(args, run_meta):
    if args.positive_class:
        return args.positive_class
    return str(run_meta.get("positive_class", "import")).lower()


def infer_n(feature_dict):
    for value in feature_dict.values():
        return len(value)
    raise ValueError("Empty feature_dict.")


def move_batch_to_device(batch, device):
    out = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


@torch.no_grad()
def predict_torch_proba(model, feature_dict, input_keys, device, batch_size):
    n = infer_n(feature_dict)
    y_dummy = np.zeros(n, dtype=np.float32)

    dataset = MultiInputDataset(feature_dict=feature_dict, y=y_dummy)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=multi_input_collate,
    )

    model.eval()
    probs = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        keys = input_keys
        if keys is None:
            keys = [k for k in batch.keys() if k != "y"]
        model_inputs = {k: batch[k] for k in keys if k in batch}
        logits = model(model_inputs)
        prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs.append(prob)

    return np.concatenate(probs)


def attach_sequence_drop_missing(df, fasta_path, acc_col):
    df = df.copy()
    seq_dict = parse_fasta(fasta_path)
    df["FULL_SEQUENCE"] = df[acc_col].astype(str).map(seq_dict)
    missing_mask = df["FULL_SEQUENCE"].isna()
    dropped_df = df.loc[missing_mask].copy()
    kept_df = df.loc[~missing_mask].copy()
    return kept_df.reset_index(drop=True), dropped_df.reset_index(drop=True)


def filter_esm_window_predictable(df, embedding_dir, acc_col, site_col):
    """Keep rows with ESM embedding file and in-range POSITION."""
    embedding_dir = Path(embedding_dir)
    keep_mask = np.zeros(len(df), dtype=bool)
    reasons = []

    for i, (_, row) in enumerate(df.iterrows()):
        acc = str(row[acc_col])
        pos = int(row[site_col])
        emb_path = embedding_dir / f"{acc}.pt"
        seq = row.get("FULL_SEQUENCE")

        if not emb_path.exists():
            keep_mask[i] = False
            reasons.append("missing_esm_embedding")
            continue

        if seq is None or (isinstance(seq, float) and np.isnan(seq)):
            keep_mask[i] = False
            reasons.append("missing_sequence")
            continue

        if pos < 1 or pos > len(seq):
            keep_mask[i] = False
            reasons.append("position_out_of_range")
            continue

        keep_mask[i] = True
        reasons.append("ok")

    out = df.copy()
    out["_predictable_reason"] = reasons
    kept_df = out.loc[keep_mask].drop(columns="_predictable_reason").reset_index(drop=True)
    dropped_df = out.loc[~keep_mask].reset_index(drop=True)
    return kept_df, dropped_df


def extract_embedding_dir_from_meta(meta):
    blocks = meta.get("feature_cfg", {}).get("blocks", {})
    for block_cfg in blocks.values():
        if block_cfg.get("type") == "esm_window_site_pca":
            return block_cfg.get("embedding_dir")
    return None


def prepare_new_dataframe(args, embedding_dir=None):
    raw_df = pd.read_csv(args.input_csv)
    n_input = len(raw_df)

    raw_df = standardize_sample_table(
        raw_df,
        acc_col=args.acc_col,
        site_col=args.site_col,
    )
    df = raw_df.copy()

    df, fasta_dropped_df = attach_sequence_drop_missing(
        df=df,
        fasta_path=args.fasta_path,
        acc_col=args.acc_col,
    )

    if args.cluster_csv:
        df = attach_sequence_and_cluster(
            df=df,
            fasta_path=args.fasta_path,
            cluster_csv=args.cluster_csv,
            acc_col=args.acc_col,
            cluster_key_col="INDEX",
            cluster_group_col="Cluster_ID",
        )

    before_invalid_df = df.copy()
    df = drop_invalid_site_rows(
        df=df,
        acc_col=args.acc_col,
        site_col=args.site_col,
        seq_col="FULL_SEQUENCE",
        index_col="INDEX",
        enforce_sty=True,
    )

    kept_index = set(df["INDEX"].astype(str))
    invalid_dropped_df = before_invalid_df.loc[
        ~before_invalid_df["INDEX"].astype(str).isin(kept_index)
    ].copy()

    esm_dropped_df = pd.DataFrame()
    if embedding_dir:
        df, esm_dropped_df = filter_esm_window_predictable(
            df=df,
            embedding_dir=embedding_dir,
            acc_col=args.acc_col,
            site_col=args.site_col,
        )

    dropped_parts = []
    for part, reason in (
        (fasta_dropped_df, "missing_fasta_sequence"),
        (invalid_dropped_df, "invalid_site_row"),
        (esm_dropped_df, "esm_window_not_predictable"),
    ):
        if len(part) > 0:
            part = part.copy()
            if "drop_reason" not in part.columns:
                if "_predictable_reason" in part.columns:
                    part["drop_reason"] = part["_predictable_reason"]
                else:
                    part["drop_reason"] = reason
            part = part.drop(columns=["_predictable_reason"], errors="ignore")
            dropped_parts.append(part)

    dropped_df = (
        pd.concat(dropped_parts, ignore_index=True)
        if dropped_parts
        else pd.DataFrame()
    )

    stats = {
        "n_input": int(n_input),
        "n_after_fasta": int(len(before_invalid_df)),
        "n_kept": int(len(df)),
        "n_dropped": int(len(dropped_df)),
    }
    return df.reset_index(drop=True), raw_df, dropped_df, stats


def clean_name(value):
    value = str(value)
    for ch in (" ", "/", "\\", ":"):
        value = value.replace(ch, "_")
    return value


def make_wide_per_fold_table(fold_pred_df, id_cols, prob_positive_col, prob_negative_col):
    df = fold_pred_df.copy()
    feature_sets = df["feature_set"].astype(str).unique()
    include_feature_set = len(feature_sets) > 1

    if include_feature_set:
        df["model_key"] = (
            df["feature_set"].map(clean_name)
            + "_seed"
            + df["seed"].astype(str)
            + "_fold"
            + df["fold"].astype(str)
        )
        sort_cols = ["feature_set", "seed", "fold"]
    else:
        df["model_key"] = (
            "seed"
            + df["seed"].astype(str)
            + "_fold"
            + df["fold"].astype(str)
        )
        sort_cols = ["seed", "fold"]

    model_order = (
        df[sort_cols + ["model_key"]]
        .drop_duplicates()
        .sort_values(sort_cols)["model_key"]
        .tolist()
    )

    wide = df.pivot_table(
        index=id_cols,
        columns="model_key",
        values=[prob_positive_col, prob_negative_col],
        aggfunc="mean",
    )
    wide.columns = [f"{metric}_{model_key}" for metric, model_key in wide.columns]
    wide = wide.reset_index()

    ordered_prob_cols = []
    for model_key in model_order:
        pos_col = f"{prob_positive_col}_{model_key}"
        neg_col = f"{prob_negative_col}_{model_key}"
        if pos_col in wide.columns:
            ordered_prob_cols.append(pos_col)
        if neg_col in wide.columns:
            ordered_prob_cols.append(neg_col)

    return wide[id_cols + ordered_prob_cols]


def collect_meta_paths(run_dir, feature_set=None):
    meta_paths = sorted(run_dir.glob("fold_artifacts/*/*/artifact_meta.json"))
    if len(meta_paths) == 0:
        raise FileNotFoundError(f"No artifact_meta.json files found under: {run_dir}")

    if feature_set is None:
        return meta_paths

    selected = []
    for meta_path in meta_paths:
        meta = load_meta(meta_path)
        if meta.get("feature_set") == feature_set:
            selected.append(meta_path)

    if len(selected) == 0:
        raise FileNotFoundError(
            f"No fold artifacts found for feature_set={feature_set!r} under: {run_dir}"
        )
    return selected


def attach_original_columns(summary_df, raw_df):
    extra_cols = [c for c in raw_df.columns if c not in summary_df.columns]
    if len(extra_cols) == 0:
        return summary_df

    meta = raw_df[["INDEX"] + extra_cols].copy()
    meta["INDEX"] = meta["INDEX"].astype(str).str.strip()
    out = summary_df.copy()
    out["INDEX"] = out["INDEX"].astype(str).str.strip()
    return out.merge(meta, on="INDEX", how="left")


def direction_labels(positive_class):
    positive_class = str(positive_class).lower()
    if positive_class == "import":
        return {
            "positive_direction": "Nuclear Import",
            "negative_direction": "Nuclear Export",
            "prob_positive_col": "prob_import",
            "prob_negative_col": "prob_export",
        }
    if positive_class == "export":
        return {
            "positive_direction": "Nuclear Export",
            "negative_direction": "Nuclear Import",
            "prob_positive_col": "prob_export",
            "prob_negative_col": "prob_import",
        }
    raise ValueError(f"Unsupported positive_class={positive_class!r}")


def resolve_platt_calibrator(args, run_dir):
    if not args.use_platt:
        return None
    calibrator_path = (
        Path(args.platt_calibrator)
        if args.platt_calibrator
        else run_dir / "platt_calibrator.json"
    )
    if not calibrator_path.exists():
        print(f"[WARN] Platt calibrator not found: {calibrator_path} (using raw probabilities)")
        return None
    calibrator = load_platt_calibrator(calibrator_path)
    print(f"[INFO] Using Platt calibrator: {calibrator_path}")
    return calibrator


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    run_dir = Path(args.run_dir)
    run_meta = load_run_meta(run_dir)
    feature_set = resolve_feature_set(args, run_meta)
    positive_class = resolve_positive_class(args, run_meta)
    label_info = direction_labels(positive_class)
    platt_calibrator = resolve_platt_calibrator(args, run_dir)

    meta_paths = collect_meta_paths(run_dir, feature_set=feature_set)
    first_meta = load_meta(meta_paths[0])
    embedding_dir = extract_embedding_dir_from_meta(first_meta)

    base_df, raw_df, dropped_df, prep_stats = prepare_new_dataframe(
        args,
        embedding_dir=embedding_dir,
    )

    print(
        f"[INFO] Model preset: feature_set={feature_set} positive_class={positive_class}"
    )
    print(
        f"[INFO] Preprocess: input={prep_stats['n_input']} "
        f"kept={prep_stats['n_kept']} dropped={prep_stats['n_dropped']}"
    )

    if args.save_dropped_csv and len(dropped_df) > 0:
        dropped_path = Path(args.save_dropped_csv)
        dropped_path.parent.mkdir(parents=True, exist_ok=True)
        dropped_df.to_csv(dropped_path, index=False)
        print(f"[INFO] Dropped rows saved to: {dropped_path}")

    all_fold_outputs = []
    thresholds = []

    for meta_path in meta_paths:
        meta = load_meta(meta_path)
        fold_dir = meta_path.parent

        pipeline = joblib.load(fold_dir / "pipeline.joblib")
        features, transformed_df = pipeline.transform(base_df.copy())

        backend = meta["backend"]
        input_keys = meta.get("input_keys")
        thresholds.append(float(meta.get("threshold", 0.5)))

        if backend == "sklearn":
            model = joblib.load(fold_dir / "model.joblib")
            X = _as_numpy_matrix(features, input_keys=input_keys)
            if platt_calibrator is not None:
                decision_scores = sklearn_decision_scores(model, X)
                prob_positive = apply_platt_calibrator(decision_scores, platt_calibrator)
            else:
                prob_positive = _predict_sklearn_proba(model, X)
        elif backend == "torch":
            checkpoint = torch.load(
                fold_dir / "model.pt",
                map_location=args.device,
            )
            model = build_model(
                model_name=checkpoint["model_name"],
                model_cfg=checkpoint["model_cfg"],
                input_dims=checkpoint["input_dims"],
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(args.device)
            prob_positive = predict_torch_proba(
                model=model,
                feature_dict=features,
                input_keys=checkpoint.get("input_keys"),
                device=args.device,
                batch_size=args.batch_size,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        fold_out = transformed_df[["INDEX", args.acc_col, args.site_col]].copy()
        fold_out["feature_set"] = meta["feature_set"]
        fold_out["seed"] = int(meta["seed"])
        fold_out["fold"] = int(meta["fold"])
        fold_out[label_info["prob_positive_col"]] = prob_positive
        fold_out[label_info["prob_negative_col"]] = 1.0 - prob_positive
        all_fold_outputs.append(fold_out)

        print(
            f"[INFO] Predicted fold seed={meta['seed']} fold={meta['fold']} "
            f"feature_set={meta['feature_set']} input_keys={input_keys}"
        )

    fold_pred_df = pd.concat(all_fold_outputs, ignore_index=True)
    id_cols = ["INDEX", args.acc_col, args.site_col]
    prob_positive_col = label_info["prob_positive_col"]
    prob_negative_col = label_info["prob_negative_col"]

    summary_df = (
        fold_pred_df
        .groupby(id_cols, as_index=False)
        .agg(
            mean_prob_positive=(prob_positive_col, "mean"),
            std_prob_positive=(prob_positive_col, "std"),
            mean_prob_negative=(prob_negative_col, "mean"),
            std_prob_negative=(prob_negative_col, "std"),
            n_models=(prob_positive_col, "size"),
        )
    )

    threshold = float(args.threshold) if args.threshold is not None else float(np.mean(thresholds))
    summary_df["threshold"] = threshold
    summary_df["positive_class"] = positive_class
    summary_df["pred_label"] = (summary_df["mean_prob_positive"] >= threshold).astype(int)
    summary_df["pred_direction"] = np.where(
        summary_df["pred_label"] == 1,
        label_info["positive_direction"],
        label_info["negative_direction"],
    )

    if positive_class == "import":
        summary_df["mean_prob_import"] = summary_df["mean_prob_positive"]
        summary_df["std_prob_import"] = summary_df["std_prob_positive"]
        summary_df["mean_prob_export"] = summary_df["mean_prob_negative"]
        summary_df["std_prob_export"] = summary_df["std_prob_negative"]
    else:
        summary_df["mean_prob_export"] = summary_df["mean_prob_positive"]
        summary_df["std_prob_export"] = summary_df["std_prob_positive"]
        summary_df["mean_prob_import"] = summary_df["mean_prob_negative"]
        summary_df["std_prob_import"] = summary_df["std_prob_negative"]

    summary_df["feature_set"] = feature_set
    summary_df["model_name"] = first_meta.get("model_name")
    summary_df = attach_original_columns(summary_df, raw_df)

    wide_fold_pred_df = make_wide_per_fold_table(
        fold_pred_df=fold_pred_df,
        id_cols=id_cols,
        prob_positive_col=prob_positive_col,
        prob_negative_col=prob_negative_col,
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)

    fold_output_csv = output_csv.with_name(output_csv.stem + "_per_fold.csv")
    wide_fold_pred_df.to_csv(fold_output_csv, index=False)

    run_info = {
        "input_csv": str(args.input_csv),
        "run_dir": str(run_dir),
        "feature_set": feature_set,
        "positive_class": positive_class,
        "platt_calibration": platt_calibrator is not None,
        "platt_calibrator": (
            str(args.platt_calibrator or run_dir / "platt_calibrator.json")
            if platt_calibrator is not None
            else None
        ),
        "model_name": first_meta.get("model_name"),
        "n_sites_predicted": int(len(summary_df)),
        "n_fold_models": int(
            fold_pred_df[["feature_set", "seed", "fold"]].drop_duplicates().shape[0]
        ),
        "threshold": threshold,
        "prep_stats": prep_stats,
    }
    save_meta_path = output_csv.with_name(output_csv.stem + "_run_meta.json")
    with open(save_meta_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Ensemble prediction saved to: {output_csv}")
    print(f"[DONE] Wide per-fold prediction saved to: {fold_output_csv}")
    print(f"[DONE] Run meta saved to: {save_meta_path}")
    print(f"[INFO] Number of sites predicted: {len(summary_df)}")
    print(
        f"[INFO] Number of fold models: "
        f"{fold_pred_df[['feature_set', 'seed', 'fold']].drop_duplicates().shape[0]}"
    )
    print(f"[INFO] Decision threshold: {threshold}")
    print(
        f"[INFO] Predicted Import: "
        f"{(summary_df['pred_direction'] == 'Nuclear Import').sum()} | "
        f"Export: {(summary_df['pred_direction'] == 'Nuclear Export').sum()}"
    )


if __name__ == "__main__":
    main()
