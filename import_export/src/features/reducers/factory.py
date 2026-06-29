import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from src.features.reducers.kernel_pls import KernelPLS

KERNEL_PLS_ALIASES = {"kernel_pls", "kpls", "kernel-pls"}


def normalize_reducer_name(reducer):
    return str(reducer).lower().replace("-", "_")


def build_reducer_model(reducer, n_components, kernel="rbf", gamma=None):
    reducer = normalize_reducer_name(reducer)

    if reducer == "pca":
        return PCA(n_components=n_components, random_state=0)

    if reducer == "pls":
        return PLSRegression(n_components=n_components, scale=False)

    if reducer in KERNEL_PLS_ALIASES:
        return KernelPLS(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
        )

    raise ValueError(f"Unsupported reducer: {reducer}")


def fit_reducer(model, reducer, x_scaled, y):
    reducer = normalize_reducer_name(reducer)

    if reducer == "pca":
        return model.fit_transform(x_scaled)

    if reducer == "pls":
        if y is None:
            raise ValueError("PLS reduction requires labels in fit_transform.")
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        model.fit(x_scaled, y_arr)
        return model.transform(x_scaled)

    if reducer in KERNEL_PLS_ALIASES:
        if y is None:
            raise ValueError("Kernel PLS reduction requires labels in fit_transform.")
        model.fit(x_scaled, y)
        return model.transform(x_scaled)

    raise ValueError(f"Unsupported reducer: {reducer}")


def transform_reducer(model, reducer, x_scaled):
    reducer = normalize_reducer_name(reducer)

    if reducer in {"pca", "pls", *KERNEL_PLS_ALIASES}:
        return model.transform(x_scaled)

    raise ValueError(f"Unsupported reducer: {reducer}")


def reducer_feature_tag(reducer):
    reducer = normalize_reducer_name(reducer)
    if reducer == "pls":
        return "PLS"
    if reducer == "pca":
        return "PCA"
    if reducer in KERNEL_PLS_ALIASES:
        return "KPLS"
    return str(reducer).upper()
