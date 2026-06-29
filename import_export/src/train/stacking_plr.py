import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.split import build_group_kfold_splits
from src.evaluate.metrics import compute_metrics
from src.features.three_branch_extractor import ThreeBranchExtractor


def _resolve_pca_components(n_requested, n_samples, n_features):
    if n_requested is None:
        return None
    n_requested = int(n_requested)
    if n_requested <= 0:
        return None
    return max(1, min(n_requested, n_samples, n_features))


def prob_to_logit(probs, eps=1e-6):
    probs = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    return np.log(probs / (1.0 - probs))


def stack_meta_features(p_window, p_site, p_manual, mode="prob"):
    cols = [np.asarray(p_window), np.asarray(p_site), np.asarray(p_manual)]
    if mode == "prob":
        return np.column_stack(cols).astype(np.float32)
    if mode == "logit":
        return np.column_stack([prob_to_logit(c) for c in cols]).astype(np.float32)
    raise ValueError(f"Unsupported meta input mode: {mode}")


class BranchPreprocessor:
    def __init__(self, branch_type, pca_components=None, random_state=0):
        self.branch_type = str(branch_type)
        self.pca_components = pca_components
        self.random_state = int(random_state)
        self.imputer = SimpleImputer(strategy="median") if self.branch_type == "manual" else None
        self.scaler = StandardScaler()
        self.pca = None
        self._fitted = False

    def fit_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.imputer is not None:
            X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)
        n_comp = _resolve_pca_components(self.pca_components, X.shape[0], X.shape[1])
        if n_comp is not None and self.branch_type in {"window", "site"}:
            self.pca = PCA(n_components=n_comp, random_state=self.random_state)
            X = self.pca.fit_transform(X)
        self._fitted = True
        return X.astype(np.float32)

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("BranchPreprocessor must be fit before transform.")
        X = np.asarray(X, dtype=np.float64)
        if self.imputer is not None:
            X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        if self.pca is not None:
            X = self.pca.transform(X)
        return X.astype(np.float32)


def _fit_logreg(X, y, C, random_state):
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


def build_inner_splits(train_df, inner_cfg, group_col, label_col):
    return build_group_kfold_splits(
        train_df,
        n_splits=int(inner_cfg["n_splits"]),
        group_col=group_col,
        label_col=label_col,
        seed=int(inner_cfg.get("seed", inner_cfg.get("random_state", 42))),
        stratify=bool(inner_cfg.get("stratify", inner_cfg.get("stratify_group_kfold", True))),
    )


def _fit_branch_model_on_arrays(X_train, y_train, X_eval, branch_type, pca_components, base_C, random_state):
    prep = BranchPreprocessor(
        branch_type=branch_type,
        pca_components=pca_components,
        random_state=random_state,
    )
    X_train_prep = prep.fit_transform(X_train)
    X_eval_prep = prep.transform(X_eval)
    model = _fit_logreg(X_train_prep, y_train, base_C, random_state)
    eval_probs = _predict_positive_proba(model, X_eval_prep)
    return eval_probs, prep, model


def generate_branch_oof_probs(
    train_df,
    y_train,
    inner_splits,
    blocks_cfg,
    branch_type,
    pca_components,
    base_C,
    random_state,
):
    oof_probs = np.full(len(y_train), np.nan, dtype=np.float64)

    for inner_train_idx, inner_val_idx in inner_splits:
        inner_train_df = train_df.iloc[inner_train_idx].reset_index(drop=True)
        inner_val_df = train_df.iloc[inner_val_idx].reset_index(drop=True)

        extractor = ThreeBranchExtractor(blocks_cfg)
        inner_train_features, _ = extractor.fit_transform(
            inner_train_df,
            y=y_train[inner_train_idx],
        )
        inner_val_features, _ = extractor.transform(inner_val_df)

        val_probs, _, _ = _fit_branch_model_on_arrays(
            inner_train_features[branch_type],
            y_train[inner_train_idx],
            inner_val_features[branch_type],
            branch_type=branch_type,
            pca_components=pca_components,
            base_C=base_C,
            random_state=random_state,
        )
        oof_probs[inner_val_idx] = val_probs

    if np.isnan(oof_probs).any():
        raise RuntimeError(f"OOF probabilities incomplete for branch={branch_type}.")

    return oof_probs


def generate_branch_test_probs(
    train_df,
    test_df,
    y_train,
    blocks_cfg,
    branch_type,
    pca_components,
    base_C,
    random_state,
):
    extractor = ThreeBranchExtractor(blocks_cfg)
    train_features, _ = extractor.fit_transform(train_df, y=y_train)
    test_features, _ = extractor.transform(test_df)

    _, prep_full, model_full = _fit_branch_model_on_arrays(
        train_features[branch_type],
        y_train,
        test_features[branch_type],
        branch_type=branch_type,
        pca_components=pca_components,
        base_C=base_C,
        random_state=random_state,
    )
    test_probs = _predict_positive_proba(
        model_full,
        prep_full.transform(test_features[branch_type]),
    )
    return test_probs, prep_full, model_full


def fit_meta_plr(oof_probs_dict, y_train, test_probs_dict, final_C, meta_input, random_state):
    meta_train = stack_meta_features(
        oof_probs_dict["window"],
        oof_probs_dict["site"],
        oof_probs_dict["manual"],
        mode=meta_input,
    )
    meta_model = _fit_logreg(meta_train, y_train, final_C, random_state)
    meta_test = stack_meta_features(
        test_probs_dict["window"],
        test_probs_dict["site"],
        test_probs_dict["manual"],
        mode=meta_input,
    )
    final_probs = _predict_positive_proba(meta_model, meta_test)
    coef = meta_model.coef_.reshape(-1)
    meta_coef = {
        "coef_window": float(coef[0]),
        "coef_site": float(coef[1]),
        "coef_manual": float(coef[2]),
        "intercept": float(meta_model.intercept_[0]),
        "meta_input": meta_input,
    }
    return final_probs, meta_model, meta_coef


def mean_fusion_probs(test_probs_dict, weights=None):
    probs = np.column_stack(
        [
            test_probs_dict["window"],
            test_probs_dict["site"],
            test_probs_dict["manual"],
        ]
    )
    if weights is None:
        return probs.mean(axis=1)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    return probs.dot(weights)


def evaluate_prob_predictions(y_true, probs, threshold=0.5):
    return compute_metrics(np.asarray(y_true), np.asarray(probs), float(threshold))


def run_stacking_outer_fold(
    train_df,
    test_df,
    blocks_cfg,
    y_train,
    y_test,
    inner_cfg,
    group_col,
    label_col,
    branch_pca,
    base_C,
    final_C,
    meta_input,
    fusion_weights,
    random_state,
    threshold=0.5,
):
    inner_splits = build_inner_splits(train_df, inner_cfg, group_col, label_col)

    oof_probs = {}
    test_probs = {}
    base_models = {}

    for branch in ("window", "site", "manual"):
        pca_n = branch_pca.get(branch)
        oof_probs[branch] = generate_branch_oof_probs(
            train_df=train_df,
            y_train=y_train,
            inner_splits=inner_splits,
            blocks_cfg=blocks_cfg,
            branch_type=branch,
            pca_components=pca_n,
            base_C=base_C,
            random_state=random_state,
        )
        test_p, prep_full, model_full = generate_branch_test_probs(
            train_df=train_df,
            test_df=test_df,
            y_train=y_train,
            blocks_cfg=blocks_cfg,
            branch_type=branch,
            pca_components=pca_n,
            base_C=base_C,
            random_state=random_state,
        )
        test_probs[branch] = test_p
        base_models[branch] = {"preprocessor": prep_full, "model": model_full}

    branch_metrics = {}
    for branch in ("window", "site", "manual"):
        branch_metrics[branch] = evaluate_prob_predictions(y_test, test_probs[branch], threshold)

    final_probs, meta_model, meta_coef = fit_meta_plr(
        oof_probs,
        y_train,
        test_probs,
        final_C=final_C,
        meta_input=meta_input,
        random_state=random_state,
    )
    stacking_metrics = evaluate_prob_predictions(y_test, final_probs, threshold)

    mean_probs = mean_fusion_probs(test_probs, weights=None)
    mean_metrics = evaluate_prob_predictions(y_test, mean_probs, threshold)

    weighted_probs = mean_fusion_probs(test_probs, weights=fusion_weights)
    weighted_metrics = evaluate_prob_predictions(y_test, weighted_probs, threshold)

    return {
        "oof_probs": oof_probs,
        "test_probs": test_probs,
        "final_probs": final_probs,
        "mean_probs": mean_probs,
        "weighted_probs": weighted_probs,
        "branch_metrics": branch_metrics,
        "stacking_metrics": stacking_metrics,
        "mean_metrics": mean_metrics,
        "weighted_metrics": weighted_metrics,
        "meta_coef": meta_coef,
        "meta_model": meta_model,
        "base_models": base_models,
    }
