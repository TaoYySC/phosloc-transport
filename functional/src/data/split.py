import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, StratifiedGroupKFold


def build_outer_split(df, seed, test_size, group_col="ACC_ID"):
    groups = df[group_col].astype(str).values
    idx = np.arange(len(df))

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    trainval_idx, test_idx = next(splitter.split(idx, groups=groups))
    return trainval_idx, test_idx


def build_single_train_val_split(df_trainval, seed, val_size=0.2, group_col="ACC_ID"):
    groups = df_trainval[group_col].astype(str).values
    idx = np.arange(len(df_trainval))

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=seed,
    )
    train_idx, val_idx = next(splitter.split(idx, groups=groups))
    return train_idx, val_idx


def build_outer_cv_splits(
    df,
    n_splits=5,
    seed=42,
    group_col="ACC_ID",
    label_col="LABEL",
    stratify=True,
):
    idx = np.arange(len(df))
    groups = df[group_col].astype(str).values
    y = df[label_col].astype(int).values

    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(
            f"Number of groups ({n_groups}) is smaller than n_splits ({n_splits})."
        )

    if stratify:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        split_iter = splitter.split(idx, y, groups)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(idx, y, groups)

    for fold_id, (trainval_idx, test_idx) in enumerate(split_iter, start=1):
        yield fold_id, trainval_idx, test_idx


def iter_dev_train_val_splits(
    df,
    n_splits=5,
    seed=42,
    group_col="ACC_ID",
    label_col="LABEL",
    stratify=True,
    single_val_size=0.2,
):
    """Yield (fold_id, train_idx, val_idx) on a development dataframe.

    When n_splits == 1, perform one group-aware train/val split on df.
    With stratify=True, the split matches fold 1 of StratifiedGroupKFold
    where n_folds ~= round(1 / single_val_size).
    """
    if int(n_splits) != 1:
        yield from build_outer_cv_splits(
            df=df,
            n_splits=n_splits,
            seed=seed,
            group_col=group_col,
            label_col=label_col,
            stratify=stratify,
        )
        return

    idx = np.arange(len(df))
    groups = df[group_col].astype(str).values
    y = df[label_col].astype(int).values
    val_size = float(single_val_size)

    if stratify:
        n_folds = max(2, int(round(1.0 / val_size)))
        n_groups = len(np.unique(groups))
        if n_groups < n_folds:
            raise ValueError(
                f"Number of groups ({n_groups}) is smaller than "
                f"single-split folds ({n_folds})."
            )
        splitter = StratifiedGroupKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        )
        train_idx, val_idx = next(splitter.split(idx, y, groups))
    else:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=val_size,
            random_state=seed,
        )
        train_idx, val_idx = next(splitter.split(idx, groups=groups))

    yield 1, train_idx, val_idx


def build_fixed_test_split(
    df,
    seed,
    test_size=0.2,
    group_col="ACC_ID",
    label_col="LABEL",
    stratify=True,
):
    idx = np.arange(len(df))
    groups = df[group_col].astype(str).values
    y = df[label_col].astype(int).values

    if stratify:
        n_holdout_splits = int(round(1.0 / float(test_size)))
        n_groups = len(np.unique(groups))

        if n_groups < n_holdout_splits:
            raise ValueError(
                f"Number of groups ({n_groups}) is smaller than "
                f"n_holdout_splits ({n_holdout_splits})."
            )

        splitter = StratifiedGroupKFold(
            n_splits=n_holdout_splits,
            shuffle=True,
            random_state=seed,
        )

        trainval_idx, test_idx = next(splitter.split(idx, y, groups))
    else:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed,
        )

        trainval_idx, test_idx = next(splitter.split(idx, groups=groups))

    return trainval_idx, test_idx

def subset_by_indices(df, idx):
    return df.iloc[idx].reset_index(drop=True)