# src/models/build_sklearn_model.py

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from src.models.pairwise_ranking_classifier import PairwiseRankingClassifier
from src.models.supcon_ce_classifier import SupConCEClassifier


def build_sklearn_model(model_name, model_cfg, seed=42):
    name = str(model_name).lower()
    class_weight = model_cfg.get("class_weight", "balanced")

    if name in {"supcon_ce", "supcon", "ce_supcon"}:
        return SupConCEClassifier(
            C=model_cfg.get("C", 1.0),
            alpha=model_cfg.get("alpha", 1.0),
            temperature=model_cfg.get("temperature", 0.07),
            embed_dim=model_cfg.get("embed_dim"),
            max_iter=model_cfg.get("max_iter", 5000),
            lr=model_cfg.get("lr", 0.05),
            tol=model_cfg.get("tol", 1e-5),
            class_weight=class_weight,
            classification_loss=model_cfg.get("classification_loss", "ce"),
            focal_gamma=model_cfg.get("focal_gamma", 2.0),
            focal_alpha=model_cfg.get("focal_alpha", 0.25),
            random_state=seed,
        )

    if name in {"supcon_focal", "focal_supcon"}:
        return SupConCEClassifier(
            C=model_cfg.get("C", 1.0),
            alpha=model_cfg.get("alpha", 1.0),
            temperature=model_cfg.get("temperature", 0.07),
            embed_dim=model_cfg.get("embed_dim"),
            max_iter=model_cfg.get("max_iter", 5000),
            lr=model_cfg.get("lr", 0.05),
            tol=model_cfg.get("tol", 1e-5),
            class_weight=class_weight,
            classification_loss="focal",
            focal_gamma=model_cfg.get("focal_gamma", 2.0),
            focal_alpha=model_cfg.get("focal_alpha", 0.25),
            random_state=seed,
        )

    if name in {"pairwise_ranking", "pairwise_ranking_logreg", "ranking"}:
        return PairwiseRankingClassifier(
            C=model_cfg.get("C", 1.0),
            max_iter=model_cfg.get("max_iter", 5000),
            lr=model_cfg.get("lr", 0.05),
            tol=model_cfg.get("tol", 1e-5),
            class_weight=class_weight,
            random_state=seed,
        )

    if name == "logreg":
        return LogisticRegression(
            C=model_cfg.get("C", 1.0),
            penalty=model_cfg.get("penalty", "l2"),
            solver=model_cfg.get("solver", "liblinear"),
            class_weight=class_weight,
            max_iter=model_cfg.get("max_iter", 20000),
            random_state=seed,
        )

    if name in {"elasticnet", "elastic_net"}:
        return LogisticRegression(
            C=model_cfg.get("C", 1.0),
            penalty="elasticnet",
            solver="saga",
            l1_ratio=float(model_cfg.get("l1_ratio", 0.5)),
            class_weight=class_weight,
            max_iter=model_cfg.get("max_iter", 20000),
            random_state=seed,
        )

    if name in {"lda", "shrinkage_lda"}:
        shrinkage = model_cfg.get("shrinkage", "auto")
        solver = model_cfg.get("solver", "lsqr")
        if shrinkage is not None and solver not in {"lsqr", "eigen"}:
            solver = "lsqr"
        return LinearDiscriminantAnalysis(
            solver=solver,
            shrinkage=shrinkage,
            n_components=model_cfg.get("n_components"),
            priors=model_cfg.get("priors"),
            store_covariance=model_cfg.get("store_covariance", False),
            tol=model_cfg.get("tol", 1e-4),
        )

    if name == "ridge":
        return RidgeClassifier(
            alpha=model_cfg.get("alpha", 1.0),
            fit_intercept=model_cfg.get("fit_intercept", True),
            class_weight=class_weight,
            solver=model_cfg.get("solver", "auto"),
            max_iter=model_cfg.get("max_iter", 5000),
            tol=model_cfg.get("tol", 1e-4),
            random_state=seed,
        )

    if name in {"svm_linear", "linear_svm"}:
        return SVC(
            C=model_cfg.get("C", 1.0),
            kernel="linear",
            gamma=model_cfg.get("gamma", "scale"),
            class_weight=class_weight,
            probability=model_cfg.get("probability", True),
            random_state=seed,
        )

    if name in {"svm_rbf", "rbf_svm"}:
        return SVC(
            C=model_cfg.get("C", 1.0),
            kernel="rbf",
            gamma=model_cfg.get("gamma", "scale"),
            class_weight=class_weight,
            probability=model_cfg.get("probability", True),
            random_state=seed,
        )

    if name == "svm":
        return SVC(
            C=model_cfg.get("C", 1.0),
            kernel=model_cfg.get("kernel", "linear"),
            gamma=model_cfg.get("gamma", "scale"),
            class_weight=class_weight,
            probability=model_cfg.get("probability", True),
            random_state=seed,
        )

    if name == "knn":
        return KNeighborsClassifier(
            n_neighbors=model_cfg.get("n_neighbors", 25),
            weights=model_cfg.get("weights", "distance"),
            metric=model_cfg.get("metric", "minkowski"),
            p=model_cfg.get("p", 2),
        )

    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=tuple(model_cfg.get("hidden_layer_sizes", [128, 64])),
            activation=model_cfg.get("activation", "relu"),
            alpha=model_cfg.get("alpha", 0.01),
            batch_size=model_cfg.get("batch_size", 32),
            learning_rate_init=model_cfg.get("learning_rate_init", 5e-4),
            max_iter=model_cfg.get("max_iter", 1000),
            early_stopping=model_cfg.get("early_stopping", True),
            validation_fraction=model_cfg.get("validation_fraction", 0.2),
            n_iter_no_change=model_cfg.get("n_iter_no_change", 30),
            random_state=seed,
        )

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=model_cfg.get("n_estimators", 800),
            max_depth=model_cfg.get("max_depth", 4),
            min_samples_split=model_cfg.get("min_samples_split", 8),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 4),
            max_features=model_cfg.get("max_features", "sqrt"),
            class_weight=model_cfg.get("class_weight", "balanced_subsample"),
            n_jobs=model_cfg.get("n_jobs", -1),
            random_state=seed,
        )

    if name == "gbdt":
        return GradientBoostingClassifier(
            n_estimators=model_cfg.get("n_estimators", 500),
            learning_rate=model_cfg.get("learning_rate", 0.02),
            max_depth=model_cfg.get("max_depth", 2),
            subsample=model_cfg.get("subsample", 0.8),
            min_samples_split=model_cfg.get("min_samples_split", 8),
            min_samples_leaf=model_cfg.get("min_samples_leaf", 4),
            max_features=model_cfg.get("max_features", 0.7),
            random_state=seed,
        )

    if name == "adaboost":
        return AdaBoostClassifier(
            n_estimators=model_cfg.get("n_estimators", 500),
            learning_rate=model_cfg.get("learning_rate", 0.02),
            random_state=seed,
        )

    if name in {"lightgbm", "lgbm"}:
        if not HAS_LIGHTGBM:
            raise ImportError(
                "lightgbm is not installed. Install it with: pip install lightgbm"
            )
        return LGBMClassifier(
            n_estimators=model_cfg.get("n_estimators", 500),
            learning_rate=model_cfg.get("learning_rate", 0.02),
            max_depth=model_cfg.get("max_depth", 3),
            num_leaves=model_cfg.get("num_leaves", 15),
            subsample=model_cfg.get("subsample", 0.85),
            colsample_bytree=model_cfg.get("colsample_bytree", 0.75),
            reg_alpha=model_cfg.get("reg_alpha", 0.1),
            reg_lambda=model_cfg.get("reg_lambda", 1.0),
            class_weight=class_weight if class_weight != "balanced" else "balanced",
            random_state=seed,
            n_jobs=model_cfg.get("n_jobs", -1),
            verbose=-1,
        )

    if name == "xgb":
        if not HAS_XGBOOST:
            raise ImportError(
                "xgboost is not installed. Install it with: pip install xgboost"
            )

        xgb_kwargs = dict(
            n_estimators=model_cfg.get("n_estimators", 500),
            max_depth=model_cfg.get("max_depth", 2),
            learning_rate=model_cfg.get("learning_rate", 0.02),
            subsample=model_cfg.get("subsample", 0.85),
            colsample_bytree=model_cfg.get("colsample_bytree", 0.75),
            reg_alpha=model_cfg.get("reg_alpha", 1.5),
            reg_lambda=model_cfg.get("reg_lambda", 5.0),
            min_child_weight=model_cfg.get("min_child_weight", 5),
            gamma=model_cfg.get("gamma", 0.5),
            objective=model_cfg.get("objective", "binary:logistic"),
            eval_metric=model_cfg.get("eval_metric", "logloss"),
            n_jobs=model_cfg.get("n_jobs", -1),
            random_state=seed,
        )
        scale_pos_weight = model_cfg.get("scale_pos_weight")
        if scale_pos_weight is not None:
            xgb_kwargs["scale_pos_weight"] = float(scale_pos_weight)
        return XGBClassifier(**xgb_kwargs)

    raise ValueError(f"Unsupported sklearn model: {model_name}")