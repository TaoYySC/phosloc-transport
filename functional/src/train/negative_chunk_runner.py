import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.train.single_run import run_single_split
from src.train.negative_chunk_utils import split_negative_by_group


def _validate_ratio_cfg(ratio_cfg, name):
    required = ["train", "val", "test"]
    for key in required:
        if key not in ratio_cfg:
            raise ValueError(f"{name} is missing key: {key}")

    total = float(ratio_cfg["train"]) + float(ratio_cfg["val"]) + float(ratio_cfg["test"])
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"{name} must sum to 1.0, got {total}")


def _split_df_by_group_ratios(df, ratio_cfg, group_col="ACC_ID", seed=42):
    _validate_ratio_cfg(ratio_cfg, "ratio_cfg")

    if len(df) == 0:
        empty = df.copy().reset_index(drop=True)
        return empty, empty, empty

    train_ratio = float(ratio_cfg["train"])
    val_ratio = float(ratio_cfg["val"])
    test_ratio = float(ratio_cfg["test"])

    if test_ratio <= 0 or val_ratio <= 0 or train_ratio <= 0:
        raise ValueError("train, val, test ratios must all be > 0")

    groups = df[group_col].astype(str).values
    idx = np.arange(len(df))

    outer_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_ratio,
        random_state=seed,
    )
    trainval_idx, test_idx = next(outer_splitter.split(idx, groups=groups))

    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    inner_val_size = val_ratio / (train_ratio + val_ratio)

    groups_tv = df_trainval[group_col].astype(str).values
    idx_tv = np.arange(len(df_trainval))

    inner_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=inner_val_size,
        random_state=seed + 1,
    )
    train_idx, val_idx = next(inner_splitter.split(idx_tv, groups=groups_tv))

    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

    return df_train, df_val, df_test


def _assert_no_group_overlap(df_train, df_val, df_test, group_col="ACC_ID", tag="split"):
    train_groups = set(df_train[group_col].astype(str).unique())
    val_groups = set(df_val[group_col].astype(str).unique())
    test_groups = set(df_test[group_col].astype(str).unique())

    if train_groups & val_groups:
        raise ValueError(f"{tag}: overlap between train and val groups")
    if train_groups & test_groups:
        raise ValueError(f"{tag}: overlap between train and test groups")
    if val_groups & test_groups:
        raise ValueError(f"{tag}: overlap between val and test groups")


def build_metrics_row(
    seed,
    model_idx,
    feature_set_name,
    train_df_sub,
    train_neg_all,
    val_df,
    test_df,
    result,
):
    row = {
        "seed": int(seed),
        "model_idx": int(model_idx),
        "feature_set": feature_set_name,
        "n_train_sub": int(len(train_df_sub)),
        "n_train_pos": int((train_df_sub["LABEL"] == 1).sum()),
        "n_train_neg": int((train_df_sub["LABEL"] == 0).sum()),
        "n_train_neg_total_available": int(len(train_neg_all)),
        "n_val_pos": int((val_df["LABEL"] == 1).sum()),
        "n_val_neg": int((val_df["LABEL"] == 0).sum()),
        "n_test_pos": int((test_df["LABEL"] == 1).sum()),
        "n_test_neg": int((test_df["LABEL"] == 0).sum()),
        "x_train_dim": int(result["x_train_dim"]),
        "n_train": int(result["n_train"]),
        "n_val": int(result["n_val"]),
        "n_test": int(result["n_test"]),
        "best_threshold": float(result["best_threshold"]),
    }

    for split_name in ["train", "val", "test"]:
        metrics = result[f"{split_name}_metrics"]
        for k, v in metrics.items():
            row[f"{split_name}_{k}"] = float(v)

    return row


def run_negative_chunk_ensemble(
    df,
    split_cfg,
    feature_cfg,
    train_cfg,
    feature_set_name,
    n_negative_chunks=10,
    device="cuda",
):
    seeds = split_cfg.get("seeds", [42])
    group_col = split_cfg.get("group_col", "ACC_ID")

    positive_split = split_cfg.get(
        "positive_split",
        {"train": 0.8, "val": 0.1, "test": 0.1},
    )
    negative_split = split_cfg.get(
        "negative_split",
        {"train": 0.98, "val": 0.01, "test": 0.01},
    )

    _validate_ratio_cfg(positive_split, "positive_split")
    _validate_ratio_cfg(negative_split, "negative_split")

    all_rows = []

    for seed in seeds:
        df_pos = df[df["LABEL"] == 1].copy().reset_index(drop=True)
        df_neg = df[df["LABEL"] == 0].copy().reset_index(drop=True)

        df_pos_train, df_pos_val, df_pos_test = _split_df_by_group_ratios(
            df=df_pos,
            ratio_cfg=positive_split,
            group_col=group_col,
            seed=seed,
        )

        pos_train_groups = set(df_pos_train[group_col].astype(str).unique())
        pos_val_groups = set(df_pos_val[group_col].astype(str).unique())
        pos_test_groups = set(df_pos_test[group_col].astype(str).unique())
        all_pos_groups = pos_train_groups | pos_val_groups | pos_test_groups

        df_neg[group_col] = df_neg[group_col].astype(str)

        df_neg_shared = df_neg[df_neg[group_col].isin(all_pos_groups)].copy().reset_index(drop=True)
        df_neg_extra = df_neg[~df_neg[group_col].isin(all_pos_groups)].copy().reset_index(drop=True)

        df_neg_shared_train = df_neg_shared[df_neg_shared[group_col].isin(pos_train_groups)].copy().reset_index(drop=True)
        df_neg_shared_val = df_neg_shared[df_neg_shared[group_col].isin(pos_val_groups)].copy().reset_index(drop=True)
        df_neg_shared_test = df_neg_shared[df_neg_shared[group_col].isin(pos_test_groups)].copy().reset_index(drop=True)

        df_neg_extra_train, df_neg_extra_val, df_neg_extra_test = _split_df_by_group_ratios(
            df=df_neg_extra,
            ratio_cfg=negative_split,
            group_col=group_col,
            seed=seed,
        )

        df_train_neg_all = pd.concat(
            [df_neg_shared_train, df_neg_extra_train],
            axis=0,
            ignore_index=True,
        )
        df_val = pd.concat(
            [df_pos_val, df_neg_shared_val, df_neg_extra_val],
            axis=0,
            ignore_index=True,
        )
        df_test = pd.concat(
            [df_pos_test, df_neg_shared_test, df_neg_extra_test],
            axis=0,
            ignore_index=True,
        )

        _assert_no_group_overlap(
            df_train=df_pos_train.assign(_tmp=1).rename(columns={group_col: group_col}),
            df_val=df_val,
            df_test=df_test,
            group_col=group_col,
            tag=f"seed_{seed}_combined_split",
        )

        neg_chunks = split_negative_by_group(
            df_neg=df_train_neg_all,
            n_chunks=n_negative_chunks,
            group_col=group_col,
            seed=seed,
        )

        for model_idx, neg_chunk_df in enumerate(neg_chunks, start=1):
            train_df_sub = pd.concat(
                [df_pos_train, neg_chunk_df],
                axis=0,
                ignore_index=True,
            )

            result = run_single_split(
                train_df=train_df_sub,
                val_df=df_val,
                test_df=df_test,
                feature_cfg=copy.deepcopy(feature_cfg),
                train_cfg=copy.deepcopy(train_cfg),
                seed=seed * 100 + model_idx,
                device=device,
                fixed_retained_columns_by_family=None,
            )

            row = build_metrics_row(
                seed=seed,
                model_idx=model_idx,
                feature_set_name=feature_set_name,
                train_df_sub=train_df_sub,
                train_neg_all=df_train_neg_all,
                val_df=df_val,
                test_df=df_test,
                result=result,
            )
            all_rows.append(row)

    return pd.DataFrame(all_rows)