import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.data.split import build_group_kfold_splits, subset_by_indices
from src.evaluate.metrics import compute_metrics
from src.train.single_run import run_single_split


def _safe_auroc(y_true, probs):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, probs))


def normalize_weights(weights, n_branches=None):
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    total = weights.sum()
    if total <= 0:
        n_branches = n_branches or len(weights)
        return np.ones(n_branches, dtype=np.float64) / n_branches
    return weights / total


def weighted_fusion_from_branch_probs(branch_probs, branch_order, weights):
    arrays = [
        np.asarray(branch_probs[branch], dtype=np.float64)
        for branch in branch_order
    ]
    probs = np.column_stack(arrays)
    weights = normalize_weights(weights, n_branches=len(branch_order))
    return probs.dot(weights)


def weighted_fusion_probs(p_window, p_site, p_manual, weights):
    return weighted_fusion_from_branch_probs(
        {"window": p_window, "site": p_site, "manual": p_manual},
        ["window", "site", "manual"],
        weights,
    )


def compute_branch_oof_aurocs(oof_probs, branch_order, y_train, floor=0.5):
    scores = {}
    for branch in branch_order:
        auroc = _safe_auroc(y_train, oof_probs[branch])
        if np.isnan(auroc):
            scores[branch] = float(floor)
        else:
            scores[branch] = float(max(auroc, floor))
    return scores


def compute_oof_adaptive_weights(
    oof_probs,
    branch_order,
    y_train,
    power=2.0,
    floor=0.5,
):
    """Per-fold weights from inner OOF branch AUROC: w_b ∝ max(AUROC_b, floor)^power."""
    branch_scores = compute_branch_oof_aurocs(
        oof_probs,
        branch_order,
        y_train,
        floor=floor,
    )
    raw_weights = np.array(
        [branch_scores[branch] ** float(power) for branch in branch_order],
        dtype=np.float64,
    )
    weights = normalize_weights(raw_weights, n_branches=len(branch_order))
    return weights, branch_scores


def search_fusion_weights_grid(branch_oof_probs, branch_order, y_train, step=0.1):
    n_branches = len(branch_order)
    best_weights = np.ones(n_branches, dtype=np.float64) / n_branches
    best_auroc = -np.inf

    if n_branches == 2:
        grid = np.arange(0.0, 1.0 + step / 2, step)
        for w0 in grid:
            weights = normalize_weights([w0, 1.0 - w0], n_branches=2)
            fused = weighted_fusion_from_branch_probs(branch_oof_probs, branch_order, weights)
            auroc = _safe_auroc(y_train, fused)
            if np.isnan(auroc):
                continue
            if auroc > best_auroc:
                best_auroc = auroc
                best_weights = weights.copy()
        return best_weights, float(best_auroc)

    if n_branches == 3:
        grid = np.arange(0.0, 1.0 + step / 2, step)
        for w_window in grid:
            for w_site in grid:
                w_manual = 1.0 - w_window - w_site
                if w_manual < -1e-9:
                    continue
                weights = normalize_weights(
                    [w_window, w_site, max(w_manual, 0.0)],
                    n_branches=3,
                )
                fused = weighted_fusion_from_branch_probs(
                    branch_oof_probs, branch_order, weights
                )
                auroc = _safe_auroc(y_train, fused)
                if np.isnan(auroc):
                    continue
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_weights = weights.copy()
        return best_weights, float(best_auroc)

    if n_branches == 4:
        grid = np.arange(0.0, 1.0 + step / 2, step)
        for w0 in grid:
            for w1 in grid:
                for w2 in grid:
                    w3 = 1.0 - w0 - w1 - w2
                    if w3 < -1e-9:
                        continue
                    weights = normalize_weights(
                        [w0, w1, w2, max(w3, 0.0)],
                        n_branches=4,
                    )
                    fused = weighted_fusion_from_branch_probs(
                        branch_oof_probs, branch_order, weights
                    )
                    auroc = _safe_auroc(y_train, fused)
                    if np.isnan(auroc):
                        continue
                    if auroc > best_auroc:
                        best_auroc = auroc
                        best_weights = weights.copy()
        return best_weights, float(best_auroc)

    raise ValueError(f"Grid search supports 2, 3, or 4 branches, got {n_branches}.")


def prob_to_logit(probs, eps=1e-6):
    probs = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    return np.log(probs / (1.0 - probs))


def stack_meta_features_from_branch_probs(branch_probs, branch_order, mode="prob"):
    arrays = [np.asarray(branch_probs[branch], dtype=np.float64) for branch in branch_order]
    if mode == "prob":
        return np.column_stack(arrays).astype(np.float32)
    if mode == "logit":
        return np.column_stack([prob_to_logit(arr) for arr in arrays]).astype(np.float32)
    raise ValueError(f"Unsupported meta input mode: {mode}")


def _fit_meta_logreg(X, y, C, random_state):
    model = LogisticRegression(
        C=float(C),
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=int(random_state),
    )
    model.fit(X, y)
    return model


def _predict_positive_proba(model, X):
    proba = model.predict_proba(X)
    pos_idx = int(np.where(model.classes_ == 1)[0][0])
    return proba[:, pos_idx]


def fit_oof_meta_logreg(
    oof_probs,
    test_probs,
    branch_order,
    y_train,
    final_C=1.0,
    meta_input="prob",
    random_state=42,
):
    meta_train = stack_meta_features_from_branch_probs(
        oof_probs, branch_order, mode=meta_input
    )
    meta_model = _fit_meta_logreg(meta_train, y_train, final_C, random_state)
    meta_test = stack_meta_features_from_branch_probs(
        test_probs, branch_order, mode=meta_input
    )
    final_probs = _predict_positive_proba(meta_model, meta_test)
    coef = meta_model.coef_.reshape(-1)
    meta_coef = {
        "intercept": float(meta_model.intercept_[0]),
        "meta_input": meta_input,
        "final_C": float(final_C),
    }
    for branch, value in zip(branch_order, coef):
        meta_coef[f"coef_{branch}"] = float(value)
    oof_train_probs = _predict_positive_proba(meta_model, meta_train)
    return final_probs, meta_model, meta_coef, oof_train_probs


def build_inner_splits(train_df, inner_cfg, group_col, label_col):
    return build_group_kfold_splits(
        train_df,
        n_splits=int(inner_cfg["n_splits"]),
        group_col=group_col,
        label_col=label_col,
        seed=int(inner_cfg.get("seed", 42)),
        stratify=bool(inner_cfg.get("stratify", True)),
    )


def _run_branch_test_probs(
    train_df,
    test_df,
    feature_cfg,
    train_cfg,
    seed,
    device,
):
    result = run_single_split(
        train_df=train_df,
        test_df=test_df,
        feature_cfg=feature_cfg,
        train_cfg=train_cfg,
        seed=seed,
        device=device,
    )
    return np.asarray(result["test_prob"], dtype=np.float64)


def _run_branch_oof_probs(
    train_df,
    inner_splits,
    feature_cfg,
    train_cfg,
    seed,
    device,
):
    y_len = len(train_df)
    oof_probs = np.full(y_len, np.nan, dtype=np.float64)

    for inner_train_idx, inner_val_idx in inner_splits:
        inner_train_df = subset_by_indices(train_df, inner_train_idx)
        inner_val_df = subset_by_indices(train_df, inner_val_idx)
        val_probs = _run_branch_test_probs(
            train_df=inner_train_df,
            test_df=inner_val_df,
            feature_cfg=feature_cfg,
            train_cfg=train_cfg,
            seed=seed,
            device=device,
        )
        oof_probs[inner_val_idx] = val_probs

    if np.isnan(oof_probs).any():
        raise RuntimeError("Branch OOF probabilities are incomplete.")
    return oof_probs


def run_weighted_fusion_outer_fold(
    train_df,
    test_df,
    branch_feature_cfgs,
    train_cfg,
    inner_cfg,
    group_col,
    label_col,
    weight_schemes,
    seed,
    device,
    threshold=0.5,
    grid_step=0.1,
    branch_order=None,
    enable_oof_tune=True,
    enable_oof_adaptive=True,
    adaptive_power=2.0,
    adaptive_floor=0.5,
    enable_oof_meta_logreg=True,
    meta_final_C=1.0,
    meta_input="prob",
):
    branch_order = branch_order or list(branch_feature_cfgs.keys())
    y_train = train_df["LABEL"].to_numpy()
    y_test = test_df["LABEL"].to_numpy()

    inner_splits = build_inner_splits(train_df, inner_cfg, group_col, label_col)

    oof_probs = {}
    test_probs = {}
    for branch in branch_order:
        feature_cfg = branch_feature_cfgs[branch]
        oof_probs[branch] = _run_branch_oof_probs(
            train_df=train_df,
            inner_splits=inner_splits,
            feature_cfg=feature_cfg,
            train_cfg=train_cfg,
            seed=seed,
            device=device,
        )
        test_probs[branch] = _run_branch_test_probs(
            train_df=train_df,
            test_df=test_df,
            feature_cfg=feature_cfg,
            train_cfg=train_cfg,
            seed=seed,
            device=device,
        )

    branch_metrics = {}
    for branch in branch_order:
        branch_metrics[branch] = compute_metrics(
            y_test, test_probs[branch], threshold
        )

    fusion_results = {}
    for scheme_name, weights in weight_schemes.items():
        if len(weights) != len(branch_order):
            raise ValueError(
                f"Scheme '{scheme_name}' has {len(weights)} weights "
                f"but {len(branch_order)} branches."
            )
        fused = weighted_fusion_from_branch_probs(test_probs, branch_order, weights)
        fusion_results[scheme_name] = {
            "weights": normalize_weights(weights, n_branches=len(branch_order)).tolist(),
            "metrics": compute_metrics(y_test, fused, threshold),
            "probs": fused,
        }

    if enable_oof_adaptive:
        adaptive_weights, branch_oof_aurocs = compute_oof_adaptive_weights(
            oof_probs,
            branch_order,
            y_train,
            power=adaptive_power,
            floor=adaptive_floor,
        )
        adaptive_train_probs = weighted_fusion_from_branch_probs(
            oof_probs, branch_order, adaptive_weights
        )
        adaptive_test_probs = weighted_fusion_from_branch_probs(
            test_probs, branch_order, adaptive_weights
        )
        fusion_results["oof_auroc_squared_adaptive"] = {
            "weights": adaptive_weights.tolist(),
            "branch_oof_aurocs": branch_oof_aurocs,
            "oof_auroc": _safe_auroc(y_train, adaptive_train_probs),
            "metrics": compute_metrics(y_test, adaptive_test_probs, threshold),
            "probs": adaptive_test_probs,
        }

    if enable_oof_tune:
        tuned_weights, tuned_oof_auroc = search_fusion_weights_grid(
            oof_probs,
            branch_order,
            y_train,
            step=grid_step,
        )
        tuned_probs = weighted_fusion_from_branch_probs(
            test_probs, branch_order, tuned_weights
        )
        fusion_results["oof_grid_tuned"] = {
            "weights": tuned_weights.tolist(),
            "oof_auroc": tuned_oof_auroc,
            "metrics": compute_metrics(y_test, tuned_probs, threshold),
            "probs": tuned_probs,
        }

    if enable_oof_meta_logreg:
        meta_probs, meta_model, meta_coef, meta_oof_probs = fit_oof_meta_logreg(
            oof_probs=oof_probs,
            test_probs=test_probs,
            branch_order=branch_order,
            y_train=y_train,
            final_C=meta_final_C,
            meta_input=meta_input,
            random_state=seed,
        )
        fusion_results["oof_meta_logreg"] = {
            "meta_coef": meta_coef,
            "oof_auroc": _safe_auroc(y_train, meta_oof_probs),
            "metrics": compute_metrics(y_test, meta_probs, threshold),
            "probs": meta_probs,
            "meta_model": meta_model,
        }

    return {
        "oof_probs": oof_probs,
        "test_probs": test_probs,
        "branch_metrics": branch_metrics,
        "fusion_results": fusion_results,
        "y_test": y_test,
        "branch_order": branch_order,
    }
