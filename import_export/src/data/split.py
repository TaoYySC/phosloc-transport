# import numpy as np
# from sklearn.model_selection import GroupShuffleSplit


# def build_outer_split(df, seed, test_size, group_col="ACC_ID"):
#     groups = df[group_col].astype(str).values
#     idx = np.arange(len(df))

#     splitter = GroupShuffleSplit(
#         n_splits=1,
#         test_size=test_size,
#         random_state=seed,
#     )
#     trainval_idx, test_idx = next(splitter.split(idx, groups=groups))
#     return trainval_idx, test_idx


# def build_single_train_val_split(df_trainval, seed, val_size=0.2, group_col="ACC_ID"):
#     groups = df_trainval[group_col].astype(str).values
#     idx = np.arange(len(df_trainval))

#     splitter = GroupShuffleSplit(
#         n_splits=1,
#         test_size=val_size,
#         random_state=seed,
#     )
#     train_idx, val_idx = next(splitter.split(idx, groups=groups))
#     return train_idx, val_idx


# def subset_by_indices(df, idx):
#     return df.iloc[idx].reset_index(drop=True)


import numpy as np
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None


def build_group_kfold_splits(
    df,
    n_splits=5,
    group_col="Cluster_ID",
    label_col="LABEL",
    seed=42,
    stratify=True,
):
    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' is not found in dataframe.")

    groups = df[group_col].astype(str).values
    idx = np.arange(len(df))
    unique_groups = np.unique(groups)

    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Number of unique groups is {len(unique_groups)}, "
            f"which is smaller than n_splits={n_splits}."
        )

    if stratify and StratifiedGroupKFold is not None and label_col in df.columns:
        y = df[label_col].values
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        return list(splitter.split(idx, y, groups))

    splitter = GroupKFold(n_splits=n_splits)
    y_dummy = np.zeros(len(df), dtype=np.int64)
    return list(splitter.split(idx, y_dummy, groups))


def subset_by_indices(df, idx):
    return df.iloc[idx].reset_index(drop=True)