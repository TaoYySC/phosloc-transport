# Functional transport classifier inference on new phosphosites (ensemble prediction from saved fold artifacts).

# predicted with saved FuncTransport model artifacts

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.build_model import build_model
from src.train.trainer import Trainer
from src.utils import attach_sequence_and_cluster, drop_invalid_site_rows


FOLD_DIR_PATTERN = re.compile(r"fold_(\d+)", re.IGNORECASE)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "data/model_artifacts/run_20260610_204935_ESM Window+Site+PDB/Functional_Transport/artifacts"
)
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data/dataset_phos_site/tf_all_phos_site_for_prediction.csv"
DEFAULT_FASTA_PATH = PROJECT_ROOT / "data/fasta/transcription_fasta.fasta"
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT
    / "results/2_1_functional_classifier_results/predictions"
    / "esm_window_site_pdb_5_folds_ensemble_predictions.csv"
)
DEFAULT_EXPECTED_FOLDS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Predict with saved FuncTransport model artifacts using the five fold "
            "checkpoints under run_20260610_204935_ESM Window+Site+PDB by default."
        )
    )
    parser.add_argument(
        "--artifact_root",
        type=str,
        default=str(DEFAULT_ARTIFACT_ROOT),
        help=f"Root directory containing fold artifacts. Default: {DEFAULT_ARTIFACT_ROOT}",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=str(DEFAULT_INPUT_CSV),
        help=f"Input site table. Default: {DEFAULT_INPUT_CSV}",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"Output prediction CSV. Default: {DEFAULT_OUTPUT_CSV}",
    )
    parser.add_argument(
        "--fasta_path",
        type=str,
        default=str(DEFAULT_FASTA_PATH),
        help=f"FASTA used when FULL_SEQUENCE is missing. Default: {DEFAULT_FASTA_PATH}",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--threshold_mode", type=str, choices=["artifact", "mean"], default="artifact")
    parser.add_argument("--mean_threshold", type=float, default=0.5)
    parser.add_argument("--skip_pdb_position_filter", action="store_true")
    parser.add_argument(
        "--with-threshold",
        action="store_true",
        help="Write threshold columns and pred_label. Default: probabilities only.",
    )
    parser.add_argument(
        "--all-artifacts",
        action="store_true",
        help="Load all checkpoints under artifact_root, including best_models.",
    )
    parser.add_argument(
        "--expected-folds",
        type=int,
        default=DEFAULT_EXPECTED_FOLDS,
        help=f"Expected number of fold artifacts when folds-only mode is enabled. Default: {DEFAULT_EXPECTED_FOLDS}.",
    )
    return parser.parse_args()


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_prediction_dataframe(input_csv, fasta_path=None):
    df = pd.read_csv(input_csv)

    if fasta_path is not None and "FULL_SEQUENCE" not in df.columns:
        df = attach_sequence_and_cluster(
            df=df,
            fasta_path=fasta_path,
            cluster_csv=None,
            acc_col="ACC_ID",
        )

    if "FULL_SEQUENCE" in df.columns:
        df = drop_invalid_site_rows(
            df=df,
            acc_col="ACC_ID",
            site_col="POSITION",
            seq_col="FULL_SEQUENCE",
            index_col="INDEX",
            enforce_sty=True,
        )

    if "LABEL" not in df.columns:
        df["LABEL"] = 0

    df["POSITION"] = pd.to_numeric(df["POSITION"], errors="coerce")
    df = df.dropna(subset=["ACC_ID", "POSITION"]).copy()
    df["POSITION"] = df["POSITION"].astype(int)

    return df.reset_index(drop=True)


def _fold_number(artifact_dir):
    match = FOLD_DIR_PATTERN.search(str(artifact_dir))
    if match is None:
        return None
    return int(match.group(1))


def find_artifact_dirs(artifact_root, folds_only=False, expected_folds=None):
    artifact_root = Path(artifact_root)
    checkpoint_paths = sorted(artifact_root.rglob("model_checkpoint.pt"))
    artifact_dirs = [p.parent for p in checkpoint_paths]

    if folds_only:
        artifact_dirs = [d for d in artifact_dirs if _fold_number(d) is not None]
        artifact_dirs = sorted(artifact_dirs, key=_fold_number)

    if expected_folds is not None and len(artifact_dirs) != expected_folds:
        raise ValueError(
            f"Expected {expected_folds} fold artifacts under {artifact_root}, "
            f"found {len(artifact_dirs)}: "
            + ", ".join(str(d) for d in artifact_dirs)
        )

    return artifact_dirs


def _result_suffix(result, index):
    fold_num = _fold_number(result["artifact_dir"])
    if fold_num is not None:
        return f"fold_{fold_num}"
    return f"model_{index + 1}"


def find_pdb_file(pdb_dir, acc):
    pdb_dir = Path(pdb_dir)

    candidates = [
        pdb_dir / f"{acc}.pdb",
        pdb_dir / f"{acc}.PDB",
        pdb_dir / f"AF-{acc}-F1-model_v4.pdb",
        pdb_dir / f"AF-{acc}-F1-model_v3.pdb",
        pdb_dir / f"AF-{acc}-F1-model_v2.pdb",
        pdb_dir / f"AF-{acc}-F1-model_v1.pdb",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = sorted(pdb_dir.glob(f"*{acc}*.pdb"))
    if len(matches) > 0:
        return matches[0]

    matches = sorted(pdb_dir.glob(f"*{acc}*.PDB"))
    if len(matches) > 0:
        return matches[0]

    return None


def parse_pdb_residue_positions(pdb_path):
    positions = set()

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            resseq = line[22:26].strip()

            if not resseq:
                continue

            try:
                positions.add(int(resseq))
            except ValueError:
                continue

    return positions


def get_graph_blocks_from_pipeline(pipeline):
    graph_blocks = []

    if not hasattr(pipeline, "blocks"):
        return graph_blocks

    for block_name, block in pipeline.blocks.items():
        if hasattr(block, "pdb_dir"):
            graph_blocks.append((block_name, block))

    return graph_blocks


def filter_rows_with_available_pdb_sites(df, artifact_dirs, dropped_csv):
    keep_mask = pd.Series(True, index=df.index)
    dropped_records = []
    pdb_cache = {}

    for artifact_dir in artifact_dirs:
        pipeline_path = Path(artifact_dir) / "fitted_pipeline.pkl"
        pipeline = load_pickle(pipeline_path)
        graph_blocks = get_graph_blocks_from_pipeline(pipeline)

        for block_name, block in graph_blocks:
            pdb_dir = Path(getattr(block, "pdb_dir"))
            acc_col = getattr(block, "acc_col", "ACC_ID")
            pos_col = getattr(block, "pos_col", "POSITION")

            print(f"[INFO] Checking PDB sites for artifact: {artifact_dir}")
            print(f"[INFO] Graph block: {block_name}")
            print(f"[INFO] PDB directory: {pdb_dir}")

            for idx, row in df.iterrows():
                if not keep_mask.loc[idx]:
                    continue

                acc = str(row[acc_col]).strip()
                pos = int(row[pos_col])

                cache_key = (str(pdb_dir), acc)

                if cache_key not in pdb_cache:
                    pdb_path = find_pdb_file(pdb_dir, acc)

                    if pdb_path is None:
                        pdb_cache[cache_key] = {
                            "pdb_path": None,
                            "positions": None,
                        }
                    else:
                        pdb_cache[cache_key] = {
                            "pdb_path": pdb_path,
                            "positions": parse_pdb_residue_positions(pdb_path),
                        }

                cached = pdb_cache[cache_key]
                pdb_path = cached["pdb_path"]
                positions = cached["positions"]

                if pdb_path is None:
                    keep_mask.loc[idx] = False
                    dropped_records.append(
                        {
                            "INDEX": row.get("INDEX", ""),
                            "ACC_ID": acc,
                            "POSITION": pos,
                            "reason": "missing_pdb_file",
                            "pdb_dir": str(pdb_dir),
                            "pdb_path": "",
                            "artifact_dir": str(artifact_dir),
                            "block_name": block_name,
                        }
                    )
                    continue

                if positions is None or pos not in positions:
                    keep_mask.loc[idx] = False
                    dropped_records.append(
                        {
                            "INDEX": row.get("INDEX", ""),
                            "ACC_ID": acc,
                            "POSITION": pos,
                            "reason": "position_not_found_in_pdb",
                            "pdb_dir": str(pdb_dir),
                            "pdb_path": str(pdb_path),
                            "artifact_dir": str(artifact_dir),
                            "block_name": block_name,
                        }
                    )

    filtered_df = df[keep_mask].copy().reset_index(drop=True)
    dropped_df = pd.DataFrame(dropped_records).drop_duplicates()

    print(f"[INFO] Rows before PDB filtering: {len(df)}")
    print(f"[INFO] Rows after PDB filtering: {len(filtered_df)}")
    print(f"[INFO] Rows dropped by PDB filtering: {len(dropped_df)}")

    if len(dropped_df) > 0:
        dropped_csv = Path(dropped_csv)
        dropped_csv.parent.mkdir(parents=True, exist_ok=True)
        dropped_df.to_csv(dropped_csv, index=False)
        print(f"[INFO] Saved dropped PDB rows to: {dropped_csv}")
        print(dropped_df.head(20).to_string(index=False))

    if len(filtered_df) == 0:
        raise ValueError("No valid rows remain after sequence and PDB filtering.")

    return filtered_df


def predict_one_artifact(artifact_dir, df, device="cuda", batch_size=256):
    artifact_dir = Path(artifact_dir)
    checkpoint_path = artifact_dir / "model_checkpoint.pt"
    pipeline_path = artifact_dir / "fitted_pipeline.pkl"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    pipeline = load_pickle(pipeline_path)

    features, transformed_df = pipeline.transform(df)

    model = build_model(
        model_name=checkpoint["model_name"],
        model_cfg=checkpoint["model_cfg"],
        input_dims=checkpoint["input_dims"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    trainer = Trainer(
        model=model,
        input_keys=checkpoint.get("input_keys"),
        device=device,
        batch_size=batch_size,
        num_epochs=1,
    )

    y_dummy = np.zeros(len(transformed_df), dtype=np.float32)
    _, prob = trainer.predict_proba(
        feature_dict=features,
        y=y_dummy,
    )

    prob = np.asarray(prob).reshape(-1)

    return {
        "artifact_dir": str(artifact_dir),
        "seed": checkpoint.get("seed"),
        "feature_set_name": checkpoint.get("feature_set_name"),
        "threshold": float(checkpoint.get("best_threshold", 0.5)),
        "df": transformed_df,
        "prob": prob,
    }


def main():
    args = parse_args()
    folds_only = not args.all_artifacts
    no_threshold = not args.with_threshold
    expected_folds = args.expected_folds if folds_only else None

    artifact_dirs = find_artifact_dirs(
        args.artifact_root,
        folds_only=folds_only,
        expected_folds=expected_folds,
    )
    if len(artifact_dirs) == 0:
        raise FileNotFoundError(f"No model_checkpoint.pt found under {args.artifact_root}")

    print("[INFO] Artifact directories:")
    for artifact_dir in artifact_dirs:
        print(f"  - {artifact_dir}")

    df = prepare_prediction_dataframe(
        input_csv=args.input_csv,
        fasta_path=args.fasta_path,
    )

    if not args.skip_pdb_position_filter:
        dropped_csv = Path(args.output_csv).with_name(
            Path(args.output_csv).stem + "_dropped_pdb_rows.csv"
        )

        df = filter_rows_with_available_pdb_sites(
            df=df,
            artifact_dirs=artifact_dirs,
            dropped_csv=dropped_csv,
        )

    all_results = []
    for artifact_dir in artifact_dirs:
        result = predict_one_artifact(
            artifact_dir=artifact_dir,
            df=df,
            device=args.device,
            batch_size=args.batch_size,
        )
        all_results.append(result)

    base_df = all_results[0]["df"].copy()

    prob_cols = []

    for i, result in enumerate(all_results):
        suffix = _result_suffix(result, i)
        col = f"prob_{suffix}"
        base_df[col] = result["prob"]
        prob_cols.append(col)

        base_df[f"artifact_{suffix}"] = result["artifact_dir"]
        if result["seed"] is not None:
            base_df[f"seed_{suffix}"] = result["seed"]

        if not no_threshold:
            base_df[f"threshold_{suffix}"] = result["threshold"]

    base_df["mean_prob"] = base_df[prob_cols].mean(axis=1)
    base_df["std_prob"] = base_df[prob_cols].std(axis=1)

    if not no_threshold:
        threshold_values = [result["threshold"] for result in all_results]
        if args.threshold_mode == "artifact":
            final_threshold = float(np.mean(threshold_values))
        else:
            final_threshold = float(args.mean_threshold)

        base_df["final_threshold"] = final_threshold
        base_df["pred_label"] = (base_df["mean_prob"] >= final_threshold).astype(int)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    base_df.to_csv(output_csv, index=False)

    print(f"[INFO] Loaded artifacts: {len(artifact_dirs)}")
    print(f"[INFO] Saved prediction file: {output_csv}")


if __name__ == "__main__":
    main()