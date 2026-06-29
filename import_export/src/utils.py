import json
import os
import random
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_csv(df, path):
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_seed(seed, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def infer_expected_site_aa_from_index(index_value):
    m = re.search(r"_([A-Za-z])(\d+)$", str(index_value))
    if m is None:
        return None
    return m.group(1).upper()


def parse_fasta(fasta_path):
    seq_dict = {}
    current_id = None
    current_seq = []

    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id is not None:
                    seq_dict[current_id] = "".join(current_seq)

                header = line[1:]
                acc = header.split()[0]
                if "|" in acc:
                    parts = acc.split("|")
                    if len(parts) >= 2:
                        acc = parts[1]

                current_id = acc
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        seq_dict[current_id] = "".join(current_seq)

    return seq_dict


def standardize_sample_table(df, acc_col="ACC_ID", site_col="POSITION"):
    df = df.copy()

    if "INDEX" not in df.columns:
        if {"ACC_ID", "POSITION"}.issubset(df.columns):
            df["INDEX"] = df["ACC_ID"].astype(str) + "_" + df["POSITION"].astype(str)
        else:
            raise ValueError("Input table must contain INDEX or enough columns to build it.")

    if acc_col not in df.columns:
        raise ValueError(f"Missing required acc column: {acc_col}")

    if site_col not in df.columns:
        raise ValueError(f"Missing required site column: {site_col}")

    df["INDEX"] = df["INDEX"].astype(str).str.strip()
    df[acc_col] = df[acc_col].astype(str).str.strip()
    df[site_col] = pd.to_numeric(df[site_col], errors="coerce")

    bad_mask = df[site_col].isna()
    if bad_mask.any():
        examples = df.loc[bad_mask, "INDEX"].head(10).tolist()
        raise ValueError(f"Found rows with invalid {site_col}. Examples: {examples}")

    df[site_col] = df[site_col].astype(int)
    return df


def load_cluster_map(cluster_csv, cluster_key_col="INDEX", cluster_group_col="Cluster_ID"):
    cluster_df = pd.read_csv(cluster_csv)

    if cluster_key_col not in cluster_df.columns:
        raise ValueError(f"cluster_key_col '{cluster_key_col}' not found in cluster_csv")

    if cluster_group_col not in cluster_df.columns:
        raise ValueError(f"cluster_group_col '{cluster_group_col}' not found in cluster_csv")

    cluster_df = cluster_df[[cluster_key_col, cluster_group_col]].drop_duplicates().copy()
    cluster_df[cluster_key_col] = cluster_df[cluster_key_col].astype(str).str.strip()
    cluster_df[cluster_group_col] = cluster_df[cluster_group_col].astype(str).str.strip()

    return dict(zip(cluster_df[cluster_key_col], cluster_df[cluster_group_col]))


def attach_sequence_and_cluster(
    df,
    fasta_path,
    cluster_csv=None,
    acc_col="ACC_ID",
    cluster_key_col="INDEX",
    cluster_group_col="Cluster_ID",
):
    df = df.copy()

    seq_dict = parse_fasta(fasta_path)
    df["FULL_SEQUENCE"] = df[acc_col].astype(str).map(seq_dict)

    missing_seq = df["FULL_SEQUENCE"].isna().sum()
    if missing_seq > 0:
        missing_acc = df.loc[df["FULL_SEQUENCE"].isna(), acc_col].drop_duplicates().tolist()[:20]
        raise ValueError(
            f"{missing_seq} rows cannot find sequence in FASTA. Example missing ACC_IDs: {missing_acc}"
        )

    if cluster_csv is None:
        return df

    if cluster_key_col not in df.columns:
        raise ValueError(f"cluster_key_col '{cluster_key_col}' not found in dataframe")

    cluster_map = load_cluster_map(
        cluster_csv=cluster_csv,
        cluster_key_col=cluster_key_col,
        cluster_group_col=cluster_group_col,
    )

    df[cluster_group_col] = df[cluster_key_col].astype(str).str.strip().map(cluster_map)

    missing_mask = df[cluster_group_col].isna()
    if int(missing_mask.sum()) > 0:
        df.loc[missing_mask, cluster_group_col] = (
            "UNCLUSTERED_" + df.loc[missing_mask, cluster_key_col].astype(str).str.strip()
        )

    return df


def drop_invalid_site_rows(
    df,
    acc_col="ACC_ID",
    site_col="POSITION",
    seq_col="FULL_SEQUENCE",
    index_col="INDEX",
    enforce_sty=True,
):
    df = df.copy()

    df[site_col] = pd.to_numeric(df[site_col], errors="coerce")

    invalid_pos_mask = (
        df[site_col].isna()
        | (df[site_col] < 1)
        | df[seq_col].isna()
        | (df[site_col] > df[seq_col].astype(str).str.len())
    )

    df = df.loc[~invalid_pos_mask].copy().reset_index(drop=True)
    df[site_col] = df[site_col].astype(int)

    df["site_residue_from_seq"] = [
        seq[pos - 1].upper()
        for seq, pos in zip(df[seq_col].astype(str), df[site_col].tolist())
    ]

    df["expected_site_aa"] = df[index_col].astype(str).map(infer_expected_site_aa_from_index)

    aa_invalid_mask = pd.Series(False, index=df.index)
    has_expected_mask = df["expected_site_aa"].notna()
    aa_invalid_mask = has_expected_mask & (df["site_residue_from_seq"] != df["expected_site_aa"])

    if enforce_sty:
        sty_invalid_mask = ~df["site_residue_from_seq"].isin(["S", "T", "Y"])
        aa_invalid_mask = aa_invalid_mask | sty_invalid_mask

    df = df.loc[~aa_invalid_mask].copy().reset_index(drop=True)
    df = df.drop(columns=["site_residue_from_seq", "expected_site_aa"], errors="ignore")
    return df


def normalize_transport_direction(df):
    if "Transport_Direction" in df.columns:
        df = df.copy()
        df["Transport_Direction"] = df["Transport_Direction"].astype(str).str.strip()
        return df

    if "LABEL" not in df.columns:
        raise ValueError("Input CSV must contain either Transport_Direction or LABEL.")

    direction_map = {
        "promote_import": "Nuclear Import",
        "inhibit_export": "Nuclear Import",
        "promote_export": "Nuclear Export",
        "inhibit_import": "Nuclear Export",
        "Nuclear Import": "Nuclear Import",
        "Nuclear Export": "Nuclear Export",
    }

    df = df.copy()
    df["LABEL"] = df["LABEL"].astype(str).str.strip()
    df["Transport_Direction"] = df["LABEL"].map(direction_map)

    if df["Transport_Direction"].isna().any():
        unknown = df.loc[df["Transport_Direction"].isna(), "LABEL"].dropna().unique().tolist()
        raise ValueError(f"Unknown LABEL values found: {unknown}")

    return df


def load_import_export_binary_dataframe(
    positive_csv,
    fasta_path,
    cluster_csv=None,
    acc_col="ACC_ID",
    site_col="POSITION",
    cluster_key_col="INDEX",
    cluster_group_col="Cluster_ID",
    positive_class="import",
):
    df = pd.read_csv(positive_csv)
    df = normalize_transport_direction(df)
    df = standardize_sample_table(df, acc_col=acc_col, site_col=site_col)

    df["Transport_Direction"] = df["Transport_Direction"].astype(str).str.strip()
    df = df[df["Transport_Direction"].isin(["Nuclear Import", "Nuclear Export"])].copy()

    conflict_counts = df.groupby("INDEX")["Transport_Direction"].nunique()
    conflict_index = conflict_counts[conflict_counts > 1].index.tolist()
    if len(conflict_index) > 0:
        df = df[~df["INDEX"].isin(conflict_index)].copy()

    positive_class = str(positive_class).strip().lower()
    if positive_class in {"import", "nuclear import"}:
        label_map = {"Nuclear Import": 1, "Nuclear Export": 0}
        task_name = "Import vs Export"
    elif positive_class in {"export", "nuclear export"}:
        label_map = {"Nuclear Export": 1, "Nuclear Import": 0}
        task_name = "Export vs Import"
    else:
        raise ValueError(
            f"Unsupported positive_class='{positive_class}'. "
            "Use 'import' or 'export'."
        )

    df["LABEL"] = df["Transport_Direction"].map(label_map)
    df["LABEL"] = df["LABEL"].astype(int)
    df["TASK"] = task_name

    df = attach_sequence_and_cluster(
        df=df,
        fasta_path=fasta_path,
        cluster_csv=cluster_csv,
        acc_col=acc_col,
        cluster_key_col=cluster_key_col,
        cluster_group_col=cluster_group_col,
    )

    df = drop_invalid_site_rows(
        df=df,
        acc_col=acc_col,
        site_col=site_col,
        seq_col="FULL_SEQUENCE",
        index_col="INDEX",
        enforce_sty=True,
    )

    df = df.sort_values("INDEX").reset_index(drop=True)

    n_import = int((df["Transport_Direction"] == "Nuclear Import").sum())
    n_export = int((df["Transport_Direction"] == "Nuclear Export").sum())
    n_label_1 = int((df["LABEL"] == 1).sum())
    n_label_0 = int((df["LABEL"] == 0).sum())
    print(
        f"[INFO] {task_name} dataset: import={n_import} export={n_export} "
        f"| LABEL=1 -> {positive_class} ({n_label_1}), LABEL=0 ({n_label_0})"
    )

    return df
