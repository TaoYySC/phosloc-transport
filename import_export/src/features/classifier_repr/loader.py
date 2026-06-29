from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.reducers.factory import (
    build_reducer_model,
    fit_reducer,
    normalize_reducer_name,
    reducer_feature_tag,
    transform_reducer,
)


class ClassifierReprLoader:
    """Load precomputed classifier layer embeddings aligned by INDEX."""

    LAYER_KEYS = {
        "l1": "classifier_l1",
        "l2": "classifier_l2",
        "classifier_l1": "classifier_l1",
        "classifier_l2": "classifier_l2",
    }

    def __init__(
        self,
        npz_path,
        index_path,
        layer="l1",
        index_col="INDEX",
        group_filter="positive",
        reducer="pls",
        pls_components=None,
        pca_components=None,
        n_components=None,
        standardize_before_reduction=True,
        standardize_after_reduction=False,
        kernel="rbf",
        kernel_gamma=None,
    ):
        self.npz_path = Path(npz_path)
        self.index_path = Path(index_path)
        self.index_col = index_col
        self.group_filter = group_filter
        self.layer = str(layer).lower()
        self.standardize_before_reduction = bool(standardize_before_reduction)
        self.standardize_after_reduction = bool(standardize_after_reduction)

        if self.layer not in self.LAYER_KEYS:
            raise ValueError(
                f"Unsupported layer '{layer}'. Expected one of: {list(self.LAYER_KEYS)}"
            )
        self.array_key = self.LAYER_KEYS[self.layer]

        if reducer is None:
            reducer = "pls" if pls_components is not None else "pca"
        self.reducer = normalize_reducer_name(reducer)
        if self.reducer in {"none", "null", "raw"}:
            self.reducer = None
        self.n_components = int(
            n_components or pls_components or pca_components or 16
        )
        self.kernel = kernel
        self.kernel_gamma = kernel_gamma

        self._lookup = None
        self.scaler = None
        self.post_reducer_scaler = None
        self.reducer_model = None
        self.raw_dim = None
        self.output_dim = None

        self._load_reference_table()

    def _load_reference_table(self):
        data = np.load(self.npz_path)
        if self.array_key not in data:
            raise KeyError(
                f"Key '{self.array_key}' not found in {self.npz_path}. "
                f"Available: {list(data.files)}"
            )

        embeddings = np.asarray(data[self.array_key], dtype=np.float32)
        index_df = pd.read_csv(self.index_path)
        if self.group_filter is not None:
            index_df = index_df[
                index_df["GROUP"].astype(str) == str(self.group_filter)
            ].copy()

        if len(index_df) != embeddings.shape[0]:
            raise ValueError(
                f"Index rows ({len(index_df)}) != embedding rows ({embeddings.shape[0]})."
            )

        lookup = {}
        for i, idx in enumerate(index_df[self.index_col].astype(str).tolist()):
            lookup[idx] = embeddings[i]
        self._lookup = lookup
        self.raw_dim = int(embeddings.shape[1])

    def _resolve_n_components(self, requested, n_samples, n_features):
        max_components = min(int(requested), n_samples, n_features)
        if max_components < 1:
            raise ValueError("Dimensionality reduction requires at least one component.")
        return max_components

    def _build_matrix(self, df):
        rows = []
        missing = []
        for idx in df[self.index_col].astype(str).tolist():
            vec = self._lookup.get(idx)
            if vec is None:
                missing.append(idx)
                rows.append(np.full(self.raw_dim, np.nan, dtype=np.float32))
            else:
                rows.append(vec)

        if missing:
            raise KeyError(
                f"{len(missing)} INDEX values missing from classifier repr: "
                f"{missing[:5]}"
            )
        return np.asarray(rows, dtype=np.float32)

    def _scale_fit_transform(self, x_raw):
        if self.standardize_before_reduction:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(x_raw)
        self.scaler = None
        return x_raw

    def _scale_transform(self, x_raw):
        if self.scaler is not None:
            return self.scaler.transform(x_raw)
        return x_raw

    def _fit_reducer(self, x_scaled, y):
        n_components = self._resolve_n_components(
            self.n_components,
            x_scaled.shape[0],
            x_scaled.shape[1],
        )
        self.reducer_model = build_reducer_model(
            self.reducer,
            n_components,
            kernel=self.kernel,
            gamma=self.kernel_gamma,
        )
        return fit_reducer(self.reducer_model, self.reducer, x_scaled, y)

    def _transform_reducer(self, x_scaled):
        if self.reducer_model is None:
            raise RuntimeError("Classifier repr reducer has not been fitted.")
        return transform_reducer(self.reducer_model, self.reducer, x_scaled)

    def fit_transform_raw(self, df):
        x_raw = self._build_matrix(df)
        x_scaled = self._scale_fit_transform(x_raw)
        self.output_dim = int(x_scaled.shape[1])
        return x_scaled.astype(np.float32)

    def transform_raw(self, df):
        x_raw = self._build_matrix(df)
        x_scaled = self._scale_transform(x_raw)
        return x_scaled.astype(np.float32)

    def _apply_reduction(self, x_raw, y=None, fit=False):
        if self.reducer is None:
            if fit:
                x_scaled = self._scale_fit_transform(x_raw)
            else:
                x_scaled = self._scale_transform(x_raw)
            self.output_dim = int(x_scaled.shape[1])
            return x_scaled.astype(np.float32)

        if fit:
            x_scaled = self._scale_fit_transform(x_raw)
            x_reduced = self._fit_reducer(x_scaled, y=y)
            if self.standardize_after_reduction:
                self.post_reducer_scaler = StandardScaler()
                x_reduced = self.post_reducer_scaler.fit_transform(x_reduced)
            else:
                self.post_reducer_scaler = None
        else:
            x_scaled = self._scale_transform(x_raw)
            x_reduced = self._transform_reducer(x_scaled)
            if self.post_reducer_scaler is not None:
                x_reduced = self.post_reducer_scaler.transform(x_reduced)

        self.output_dim = int(x_reduced.shape[1])
        return x_reduced.astype(np.float32)

    def fit_transform(self, df, y=None):
        x_raw = self._build_matrix(df)
        return self._apply_reduction(x_raw, y=y, fit=True)

    def transform(self, df):
        x_raw = self._build_matrix(df)
        return self._apply_reduction(x_raw, fit=False)

    def _reducer_prefix(self):
        layer_tag = self.layer.upper().replace("CLASSIFIER_", "")
        if self.reducer is None:
            return f"CLF_{layer_tag}_RAW"
        return f"CLF_{layer_tag}_{reducer_feature_tag(self.reducer)}"

    def get_feature_names_out(self):
        if self.output_dim is None:
            return [f"{self._reducer_prefix()}_{i + 1}" for i in range(self.raw_dim)]
        return [f"{self._reducer_prefix()}_{i + 1}" for i in range(self.output_dim)]

    @property
    def feature_info(self):
        return {
            "npz_path": str(self.npz_path),
            "index_path": str(self.index_path),
            "layer": self.layer,
            "array_key": self.array_key,
            "reducer": self.reducer,
            "kernel": self.kernel,
            "kernel_gamma": self.kernel_gamma,
            "n_components": self.n_components,
            "raw_dim": self.raw_dim,
            "output_dim": self.output_dim,
            "standardize_before_reduction": self.standardize_before_reduction,
            "standardize_after_reduction": self.standardize_after_reduction,
        }
