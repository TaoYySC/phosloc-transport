import math
import random

import numpy as np
import pandas as pd


def compute_balanced_bag_count(
    n_pos,
    n_u,
    target_u_to_p_ratio=2.0,
    min_bags=1,
    max_bags=50,
):
    n_pos = int(n_pos)
    n_u = int(n_u)
    target_u_to_p_ratio = float(target_u_to_p_ratio)
    min_bags = int(min_bags)
    max_bags = int(max_bags)

    if n_u <= 0:
        return {
            "n_bags": max(min_bags, 1),
            "u_per_bag_target": 0,
            "expected_u_per_bag": 0,
            "expected_u_to_p_ratio": 0.0,
            "u_coverage": 0.0,
            "capped_by_max_bags": False,
        }

    if n_pos <= 0:
        n_bags = max(min_bags, 1)
        expected_u_per_bag = int(math.ceil(n_u / n_bags))
        return {
            "n_bags": n_bags,
            "u_per_bag_target": expected_u_per_bag,
            "expected_u_per_bag": expected_u_per_bag,
            "expected_u_to_p_ratio": float("inf"),
            "u_coverage": 1.0,
            "capped_by_max_bags": False,
        }

    u_per_bag_target = max(1, int(round(n_pos * target_u_to_p_ratio)))
    n_bags_needed = int(math.ceil(n_u / u_per_bag_target))
    capped_by_max_bags = n_bags_needed > max_bags
    n_bags = max(min_bags, min(max_bags, n_bags_needed))
    expected_u_per_bag = int(math.ceil(n_u / n_bags))

    return {
        "n_bags": int(n_bags),
        "u_per_bag_target": int(u_per_bag_target),
        "expected_u_per_bag": int(expected_u_per_bag),
        "expected_u_to_p_ratio": float(expected_u_per_bag / n_pos),
        "u_coverage": 1.0,
        "capped_by_max_bags": bool(capped_by_max_bags),
        "n_bags_needed_uncapped": int(n_bags_needed),
    }


def split_negative_by_group(df_neg, n_chunks=10, group_col="ACC_ID", seed=42):
    df_neg = df_neg.copy()
    df_neg[group_col] = df_neg[group_col].astype(str).str.strip()

    group_sizes = (
        df_neg.groupby(group_col)
        .size()
        .reset_index(name="n_rows")
    )

    rng = random.Random(seed)
    rows = group_sizes.to_dict("records")
    rng.shuffle(rows)
    rows = sorted(rows, key=lambda x: x["n_rows"], reverse=True)

    chunks = [[] for _ in range(n_chunks)]
    chunk_sizes = [0] * n_chunks

    for row in rows:
        smallest_idx = int(np.argmin(chunk_sizes))
        chunks[smallest_idx].append(row[group_col])
        chunk_sizes[smallest_idx] += int(row["n_rows"])

    out_dfs = []
    for i in range(n_chunks):
        acc_ids = set(chunks[i])
        chunk_df = df_neg[df_neg[group_col].isin(acc_ids)].copy().reset_index(drop=True)
        out_dfs.append(chunk_df)

    return out_dfs


def sample_unlabeled_bags(
    df_u,
    n_bags,
    u_per_bag,
    seed=42,
    with_replacement=True,
):
    """Sample U for each bag; does not require full U pool coverage."""
    df_u = df_u.copy().reset_index(drop=True)
    n_bags = int(n_bags)
    u_per_bag = int(u_per_bag)
    if n_bags <= 0:
        raise ValueError("n_bags must be positive.")
    if u_per_bag <= 0:
        raise ValueError("u_per_bag must be positive.")
    if len(df_u) == 0:
        raise ValueError("Cannot sample unlabeled bags from empty U pool.")

    rng = np.random.default_rng(int(seed))
    n_u = len(df_u)
    bags = []
    for _ in range(n_bags):
        if with_replacement or u_per_bag > n_u:
            idx = rng.integers(0, n_u, size=u_per_bag)
        else:
            idx = rng.choice(n_u, size=u_per_bag, replace=False)
        bags.append(df_u.iloc[idx].reset_index(drop=True))
    return bags


def build_fixed_ratio_bag_plan(
    n_pos,
    n_u,
    n_bags,
    target_u_to_p_ratio=1.0,
    with_replacement=True,
):
    n_pos = int(n_pos)
    n_u = int(n_u)
    n_bags = int(n_bags)
    target_u_to_p_ratio = float(target_u_to_p_ratio)

    if n_pos <= 0:
        raise ValueError("fixed_ratio bagging requires at least one positive training sample.")

    u_per_bag = max(1, int(round(n_pos * target_u_to_p_ratio)))
    expected_u_to_p_ratio = float(u_per_bag / n_pos)
    if with_replacement:
        u_coverage = min(1.0, float(u_per_bag / max(n_u, 1)))
    else:
        u_coverage = min(1.0, float(n_bags * u_per_bag / max(n_u, 1)))

    return {
        "n_bags": n_bags,
        "u_per_bag_target": int(u_per_bag),
        "expected_u_per_bag": int(u_per_bag),
        "expected_u_to_p_ratio": expected_u_to_p_ratio,
        "u_coverage": float(u_coverage),
        "capped_by_max_bags": False,
        "u_sample_with_replacement": bool(with_replacement),
        "u_full_coverage": False,
    }