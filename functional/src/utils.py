import os
import re
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from src.data.pdb_filter import filter_dataframe_by_pdb


VALID_TASKS = {
    "Nuclear Import",
    "Nuclear Export",
    "Functional Transport",
}


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


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def infer_expected_site_aa_from_index(index_value: str):
    m = re.search(r"_([A-Za-z])(\d+)$", str(index_value))
    if m is None:
        return None
    return m.group(1).upper()


def _min_distance_to_sorted_sites(site, sorted_sites):
    if len(sorted_sites) == 0:
        return np.inf

    idx = np.searchsorted(sorted_sites, site)

    left_dist = np.inf
    right_dist = np.inf

    if idx > 0:
        left_dist = abs(site - sorted_sites[idx - 1])

    if idx < len(sorted_sites):
        right_dist = abs(site - sorted_sites[idx])

    return min(left_dist, right_dist)


def filter_negative_by_positive_proteins(
    positive_df,
    negative_df,
    protein_id_col="ACC_ID",
    site_col="POSITION",
    index_col="INDEX",
    min_distance=15,
):
    pos_df = positive_df.copy()
    neg_df = negative_df.copy()

    pos_df[protein_id_col] = pos_df[protein_id_col].astype(str).str.strip()
    neg_df[protein_id_col] = neg_df[protein_id_col].astype(str).str.strip()

    pos_df[index_col] = pos_df[index_col].astype(str).str.strip()
    neg_df[index_col] = neg_df[index_col].astype(str).str.strip()

    pos_df[site_col] = pd.to_numeric(pos_df[site_col], errors="coerce")
    neg_df[site_col] = pd.to_numeric(neg_df[site_col], errors="coerce")

    pos_df = pos_df.dropna(subset=[site_col]).copy()
    neg_df = neg_df.dropna(subset=[site_col]).copy()

    pos_df[site_col] = pos_df[site_col].astype(int)
    neg_df[site_col] = neg_df[site_col].astype(int)

    positive_proteins = set(pos_df[protein_id_col].unique())
    positive_index = set(pos_df[index_col].unique())

    before_n = len(neg_df)

    filtered = neg_df[
        neg_df[protein_id_col].isin(positive_proteins)
    ].copy()

    filtered = filtered[
        ~filtered[index_col].isin(positive_index)
    ].copy()

    pos_sites_map = (
        pos_df.groupby(protein_id_col)[site_col]
        .apply(lambda s: np.sort(s.astype(int).unique()))
        .to_dict()
    )

    filtered["_min_positive_distance"] = [
        _min_distance_to_sorted_sites(
            site=int(site),
            sorted_sites=pos_sites_map.get(acc, np.array([], dtype=int)),
        )
        for acc, site in zip(filtered[protein_id_col], filtered[site_col])
    ]

    filtered = filtered[
        filtered["_min_positive_distance"] > min_distance
    ].copy()

    after_n = len(filtered)

    print(
        f"[INFO] Negative filtering by positive proteins and distance > {min_distance}: "
        f"before={before_n} after={after_n} removed={before_n - after_n}"
    )

    return filtered.drop(columns=["_min_positive_distance"])


def drop_invalid_site_rows(
    df: pd.DataFrame,
    acc_col: str = "ACC_ID",
    site_col: str = "POSITION",
    seq_col: str = "FULL_SEQUENCE",
    index_col: str = "INDEX",
    enforce_sty: bool = True,
):
    df = df.copy()

    df[site_col] = pd.to_numeric(df[site_col], errors="coerce")

    invalid_pos_mask = (
        df[site_col].isna()
        | (df[site_col] < 1)
        | df[seq_col].isna()
        | (df[site_col] > df[seq_col].astype(str).str.len())
    )

    if invalid_pos_mask.any():
        bad_df = df.loc[invalid_pos_mask, [index_col, acc_col, site_col]].copy()
        bad_df["seq_len"] = df.loc[invalid_pos_mask, seq_col].astype(str).str.len().values

        ensure_dir("results")
        bad_df.to_csv("results/invalid_site_rows.csv", index=False)

        print(
            "[WARN] Dropping %d rows with invalid site positions. Example rows:\n%s"
            % (int(invalid_pos_mask.sum()), bad_df.head(20).to_string(index=False))
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

    if aa_invalid_mask.any():
        bad_df = df.loc[
            aa_invalid_mask,
            [index_col, acc_col, site_col, "expected_site_aa", "site_residue_from_seq"]
        ].copy()
        print(
            "[WARN] Dropping %d rows with invalid site residue. Example rows:\n%s"
            % (int(aa_invalid_mask.sum()), bad_df.head(20).to_string(index=False))
        )
        df = df.loc[~aa_invalid_mask].copy().reset_index(drop=True)

    df = df.drop(columns=["site_residue_from_seq", "expected_site_aa"], errors="ignore")
    return df


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


def load_cluster_map(cluster_csv, cluster_key_col="ACC_ID", cluster_group_col="Cluster_ID"):
    cluster_df = pd.read_csv(cluster_csv)

    if cluster_key_col not in cluster_df.columns:
        raise ValueError(f"cluster_key_col '{cluster_key_col}' not found in cluster_csv")

    if cluster_group_col not in cluster_df.columns:
        raise ValueError(f"cluster_group_col '{cluster_group_col}' not found in cluster_csv")

    cluster_df = cluster_df[[cluster_key_col, cluster_group_col]].drop_duplicates().copy()
    cluster_df[cluster_key_col] = cluster_df[cluster_key_col].astype(str).str.strip()
    cluster_df[cluster_group_col] = cluster_df[cluster_group_col].astype(str).str.strip()

    return dict(zip(cluster_df[cluster_key_col], cluster_df[cluster_group_col]))


def normalize_positive_direction_column(df):
    if "Transport_Direction" in df.columns:
        df = df.copy()
        df["Transport_Direction"] = df["Transport_Direction"].astype(str).str.strip()
        return df, "Transport_Direction"

    if "LABEL" not in df.columns:
        raise ValueError("Positive CSV must contain either 'Transport_Direction' or 'LABEL'.")

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

    unknown = df.loc[df["Transport_Direction"].isna(), "LABEL"].dropna().unique().tolist()
    if unknown:
        raise ValueError(f"Unknown positive LABEL values found: {unknown}")

    return df, "Transport_Direction"


def standardize_sample_table(df, acc_col="ACC_ID", site_col="POSITION"):
    df = df.copy()

    if "INDEX" not in df.columns:
        if {"ACC_ID", "POSITION"}.issubset(df.columns):
            df["INDEX"] = df["ACC_ID"].astype(str) + "_" + df["POSITION"].astype(str)
        else:
            raise ValueError("Input table must contain 'INDEX' or enough columns to build it.")

    if acc_col not in df.columns:
        raise ValueError(f"Missing required acc column: {acc_col}")

    if site_col not in df.columns:
        raise ValueError(f"Missing required site column: {site_col}")

    df["INDEX"] = df["INDEX"].astype(str).str.strip()
    df[acc_col] = df[acc_col].astype(str).str.strip()
    df[site_col] = pd.to_numeric(df[site_col], errors="coerce")

    missing_site_mask = df[site_col].isna()
    if missing_site_mask.any():
        raise ValueError(
            f"Found rows with invalid {site_col}. Example INDEX values: "
            f"{df.loc[missing_site_mask, 'INDEX'].head(10).tolist()}"
        )

    df[site_col] = df[site_col].astype(int)

    return df


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
        raise ValueError(f"cluster_key_col '{cluster_key_col}' not found in merged dataframe")

    cluster_map = load_cluster_map(
        cluster_csv=cluster_csv,
        cluster_key_col=cluster_key_col,
        cluster_group_col=cluster_group_col,
    )

    df[cluster_group_col] = df[cluster_key_col].astype(str).str.strip().map(cluster_map)

    missing_mask = df[cluster_group_col].isna()
    missing_cluster = int(missing_mask.sum())

    if missing_cluster > 0:
        print(f"[WARN] {missing_cluster} rows cannot find cluster in cluster_csv. Assigning unique fallback clusters.")
        df.loc[missing_mask, cluster_group_col] = (
            "UNCLUSTERED_" + df.loc[missing_mask, cluster_key_col].astype(str).str.strip()
        )

    return df


def select_positive_rows_by_task(pos_df, direction_col, task_name):
    if task_name not in VALID_TASKS:
        raise ValueError(f"Unsupported task_name: {task_name}. Valid tasks: {sorted(VALID_TASKS)}")

    if task_name == "Functional Transport":
        pos_task_df = pos_df[
            pos_df[direction_col].isin(["Nuclear Import", "Nuclear Export"])
        ].copy()
    else:
        pos_task_df = pos_df[pos_df[direction_col] == task_name].copy()

    if len(pos_task_df) == 0:
        direction_counts = pos_df[direction_col].value_counts(dropna=False).to_dict()
        raise ValueError(
            f"No positive samples found for task: {task_name}. "
            f"Observed direction counts: {direction_counts}"
        )

    return pos_task_df

def filter_negative_sites(
    positive_df,
    negative_df,
    protein_id_col="ACC_ID",
    site_col="POSITION",
    index_col="INDEX",
    min_distance=15,
    restrict_to_positive_proteins=False,
):
    pos_df = positive_df.copy()
    neg_df = negative_df.copy()

    pos_df[protein_id_col] = pos_df[protein_id_col].astype(str).str.strip()
    neg_df[protein_id_col] = neg_df[protein_id_col].astype(str).str.strip()

    pos_df[index_col] = pos_df[index_col].astype(str).str.strip()
    neg_df[index_col] = neg_df[index_col].astype(str).str.strip()

    pos_df[site_col] = pd.to_numeric(pos_df[site_col], errors="coerce")
    neg_df[site_col] = pd.to_numeric(neg_df[site_col], errors="coerce")

    pos_df = pos_df.dropna(subset=[site_col]).copy()
    neg_df = neg_df.dropna(subset=[site_col]).copy()

    pos_df[site_col] = pos_df[site_col].astype(int)
    neg_df[site_col] = neg_df[site_col].astype(int)

    positive_proteins = set(pos_df[protein_id_col].unique())
    positive_index = set(pos_df[index_col].unique())

    before_n = len(neg_df)

    if restrict_to_positive_proteins:
        filtered = neg_df[
            neg_df[protein_id_col].isin(positive_proteins)
        ].copy()
    else:
        filtered = neg_df.copy()

    filtered = filtered[
        ~filtered[index_col].isin(positive_index)
    ].copy()

    pos_sites_map = (
        pos_df.groupby(protein_id_col)[site_col]
        .apply(lambda s: np.sort(s.astype(int).unique()))
        .to_dict()
    )

    def _min_distance_to_sorted_sites(site, sorted_sites):
        if len(sorted_sites) == 0:
            return np.inf

        idx = np.searchsorted(sorted_sites, site)

        left_dist = np.inf
        right_dist = np.inf

        if idx > 0:
            left_dist = abs(site - sorted_sites[idx - 1])

        if idx < len(sorted_sites):
            right_dist = abs(site - sorted_sites[idx])

        return min(left_dist, right_dist)

    filtered["_min_positive_distance"] = [
        _min_distance_to_sorted_sites(
            site=int(site),
            sorted_sites=pos_sites_map.get(acc, np.array([], dtype=int)),
        )
        for acc, site in zip(filtered[protein_id_col], filtered[site_col])
    ]

    filtered = filtered[
        filtered["_min_positive_distance"] > min_distance
    ].copy()

    after_n = len(filtered)

    print(
        f"[INFO] Negative filtering with min_distance > {min_distance}: "
        f"before={before_n} after={after_n} removed={before_n - after_n}"
    )

    return filtered.drop(columns=["_min_positive_distance"])


def filter_other_sites_on_positive_proteins(
    positive_df,
    candidate_df,
    protein_id_col="ACC_ID",
    site_col="POSITION",
    index_col="INDEX",
    min_distance=0,
):
    """Other phospho sites on proteins that carry at least one positive site."""
    pos_df = positive_df.copy()
    cand_df = candidate_df.copy()

    pos_df[protein_id_col] = pos_df[protein_id_col].astype(str).str.strip()
    cand_df[protein_id_col] = cand_df[protein_id_col].astype(str).str.strip()

    pos_df[index_col] = pos_df[index_col].astype(str).str.strip()
    cand_df[index_col] = cand_df[index_col].astype(str).str.strip()

    pos_df[site_col] = pd.to_numeric(pos_df[site_col], errors="coerce")
    cand_df[site_col] = pd.to_numeric(cand_df[site_col], errors="coerce")

    pos_df = pos_df.dropna(subset=[site_col]).copy()
    cand_df = cand_df.dropna(subset=[site_col]).copy()

    pos_df[site_col] = pos_df[site_col].astype(int)
    cand_df[site_col] = cand_df[site_col].astype(int)

    positive_proteins = set(pos_df[protein_id_col].unique())
    positive_index = set(pos_df[index_col].unique())
    before_n = len(cand_df)

    filtered = cand_df[cand_df[protein_id_col].isin(positive_proteins)].copy()
    filtered = filtered[~filtered[index_col].isin(positive_index)].copy()

    min_distance = int(min_distance)
    if min_distance > 0:
        pos_sites_map = (
            pos_df.groupby(protein_id_col)[site_col]
            .apply(lambda s: np.sort(s.astype(int).unique()))
            .to_dict()
        )
        filtered["_min_positive_distance"] = [
            _min_distance_to_sorted_sites(
                site=int(site),
                sorted_sites=pos_sites_map.get(acc, np.array([], dtype=int)),
            )
            for acc, site in zip(filtered[protein_id_col], filtered[site_col])
        ]
        filtered = filtered[filtered["_min_positive_distance"] > min_distance].copy()
        filtered = filtered.drop(columns=["_min_positive_distance"])

    after_n = len(filtered)
    print(
        f"[INFO] Negative mode=same_positive_tf_other: "
        f"proteins_with_positive={len(positive_proteins)} "
        f"before={before_n} after={after_n} removed={before_n - after_n}"
        + (f" min_distance>{min_distance}" if min_distance > 0 else "")
    )
    return filtered


VALID_UNLABELED_MODES = {
    "same_positive_tf_far",
    "same_positive_tf_other",
    "no_positive_tf",
}


def select_unlabeled_pool(
    positive_df,
    negative_df,
    mode="same_positive_tf_far",
    protein_id_col="ACC_ID",
    site_col="POSITION",
    index_col="INDEX",
    min_distance=30,
):
    if mode not in VALID_UNLABELED_MODES:
        raise ValueError(
            f"Unsupported unlabeled mode: {mode}. "
            f"Expected one of {sorted(VALID_UNLABELED_MODES)}"
        )

    if mode == "same_positive_tf_far":
        return filter_negative_by_positive_proteins(
            positive_df=positive_df,
            negative_df=negative_df,
            protein_id_col=protein_id_col,
            site_col=site_col,
            index_col=index_col,
            min_distance=min_distance,
        )

    if mode == "same_positive_tf_other":
        return filter_other_sites_on_positive_proteins(
            positive_df=positive_df,
            candidate_df=negative_df,
            protein_id_col=protein_id_col,
            site_col=site_col,
            index_col=index_col,
            min_distance=min_distance,
        )

    pos_df = positive_df.copy()
    neg_df = negative_df.copy()

    pos_df[protein_id_col] = pos_df[protein_id_col].astype(str).str.strip()
    neg_df[protein_id_col] = neg_df[protein_id_col].astype(str).str.strip()
    pos_df[index_col] = pos_df[index_col].astype(str).str.strip()
    neg_df[index_col] = neg_df[index_col].astype(str).str.strip()

    positive_proteins = set(pos_df[protein_id_col].unique())
    positive_index = set(pos_df[index_col].unique())
    before_n = len(neg_df)

    filtered = neg_df[~neg_df[protein_id_col].isin(positive_proteins)].copy()
    filtered = filtered[~filtered[index_col].isin(positive_index)].copy()
    after_n = len(filtered)

    print(
        f"[INFO] Unlabeled mode=no_positive_tf: "
        f"before={before_n} after={after_n} removed={before_n - after_n} "
        f"(proteins_with_positive={len(positive_proteins)})"
    )
    return filtered


def load_task_dataframe(
    positive_csv,
    negative_csv,
    fasta_path,
    task_name,
    cluster_csv=None,
    acc_col="ACC_ID",
    site_col="POSITION",
    cluster_key_col="INDEX",
    cluster_group_col="Cluster_ID",
    negative_min_distance=15,
    unlabeled_mode="same_positive_tf_far",
    pdb_dir=None,
    require_pdb=False,
):
    pos_df = pd.read_csv(positive_csv)
    neg_df = pd.read_csv(negative_csv)

    pos_df, direction_col = normalize_positive_direction_column(pos_df)

    pos_df = standardize_sample_table(pos_df, acc_col=acc_col, site_col=site_col)
    neg_df = standardize_sample_table(neg_df, acc_col=acc_col, site_col=site_col)

    pos_task_df = select_positive_rows_by_task(
        pos_df=pos_df,
        direction_col=direction_col,
        task_name=task_name,
    )

    neg_df = select_unlabeled_pool(
        positive_df=pos_task_df,
        negative_df=neg_df,
        mode=unlabeled_mode,
        protein_id_col="ACC_ID",
        site_col="POSITION",
        index_col="INDEX",
        min_distance=negative_min_distance,
    )

    pos_task_df["LABEL"] = 1
    neg_df["LABEL"] = 0

    pos_task_df["TASK"] = task_name
    neg_df["TASK"] = task_name

    merged_df = pd.concat([pos_task_df, neg_df], axis=0, ignore_index=True)

    merged_df = attach_sequence_and_cluster(
        df=merged_df,
        fasta_path=fasta_path,
        cluster_csv=cluster_csv,
        acc_col=acc_col,
        cluster_key_col=cluster_key_col,
        cluster_group_col=cluster_group_col,
    )

    merged_df = drop_invalid_site_rows(
        df=merged_df,
        acc_col=acc_col,
        site_col=site_col,
        seq_col="FULL_SEQUENCE",
        index_col="INDEX",
        enforce_sty=True,
    )

    if require_pdb or pdb_dir is not None:
        if pdb_dir is None:
            raise ValueError("require_pdb=True but pdb_dir is not provided")
        merged_df = filter_dataframe_by_pdb(
            df=merged_df,
            pdb_dir=pdb_dir,
            acc_col=acc_col,
            label_col="LABEL",
        )

    n_pos = int((merged_df["LABEL"] == 1).sum())
    n_neg = int((merged_df["LABEL"] == 0).sum())

    if task_name == "Functional Transport":
        pos_direction_counts = pos_task_df[direction_col].value_counts().to_dict()
        print(
            f"[INFO] Functional Transport positives: {pos_direction_counts} | "
            f"total_pos={n_pos} total_neg={n_neg}"
        )
    else:
        print(f"[INFO] task={task_name} total_pos={n_pos} total_neg={n_neg}")

    return merged_df