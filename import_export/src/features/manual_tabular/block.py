import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.manual_tabular.loader import ManualTabularLoader
from src.features.reducers.factory import (
    build_reducer_model,
    fit_reducer,
    normalize_reducer_name,
    reducer_feature_tag,
    transform_reducer,
)
from src.features.manual_tabular.assembler import TabularFeatureAssembler


class ManualTabularBlock:
    def __init__(
        self,
        families,
        index_col="INDEX",
        return_dataframe=False,
        reducer=None,
        n_components=None,
        pca_components=None,
        pls_components=None,
        standardize_before_reduction=True,
        standardize_before_pca=None,
        standardize_after_reduction=False,
        kernel="rbf",
        kernel_gamma=None,
        **assembler_kwargs,
    ):
        self.loader = ManualTabularLoader(index_col=index_col)
        self.assembler = TabularFeatureAssembler(
            feature_families=families,
            return_dataframe=return_dataframe,
            **assembler_kwargs,
        )

        if standardize_before_pca is not None:
            standardize_before_reduction = standardize_before_pca

        if reducer is None:
            if pls_components is not None:
                reducer = "pls"
                n_components = pls_components
            elif pca_components is not None:
                reducer = "pca"
                n_components = pca_components
            elif n_components is not None:
                reducer = "pca"
        else:
            reducer = normalize_reducer_name(reducer)
            if n_components is None:
                n_components = pls_components or pca_components

        self.reducer = reducer
        self.n_components = int(n_components) if n_components is not None else None
        self.kernel = kernel
        self.kernel_gamma = kernel_gamma
        self.standardize_before_reduction = bool(standardize_before_reduction)
        self.standardize_after_reduction = bool(standardize_after_reduction)
        self.scaler = None
        self.post_reducer_scaler = None
        self.reducer_model = None
        self.raw_dim = None
        self.output_dim = None

    def attach_features(self, sample_df: pd.DataFrame) -> pd.DataFrame:
        return self.loader.merge_all(sample_df)

    def _resolve_n_components(self, requested, n_samples, n_features):
        max_components = min(int(requested), n_samples, n_features)
        if max_components < 1:
            raise ValueError("Dimensionality reduction requires at least one component.")
        return max_components

    def _scale_fit_transform(self, x_raw):
        x_raw = np.asarray(x_raw, dtype=np.float32)
        if self.standardize_before_reduction:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(x_raw)
        self.scaler = None
        return x_raw

    def _scale_transform(self, x_raw):
        x_raw = np.asarray(x_raw, dtype=np.float32)
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
            raise RuntimeError("Manual tabular reducer has not been fitted.")
        return transform_reducer(self.reducer_model, self.reducer, x_scaled)

    def _apply_reduction(self, x_raw, y=None, fit=False):
        x_raw = np.asarray(x_raw, dtype=np.float32)
        self.raw_dim = int(x_raw.shape[1])

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

    def fit_transform(self, train_df: pd.DataFrame, y=None):
        x_raw = self.assembler.fit_transform(train_df, y=y)
        if self.reducer is None:
            self.raw_dim = int(np.asarray(x_raw).shape[1])
            self.output_dim = self.raw_dim
            return x_raw
        return self._apply_reduction(x_raw, y=y, fit=True)

    def transform(self, df: pd.DataFrame):
        x_raw = self.assembler.transform(df)
        if self.reducer is None:
            return x_raw
        return self._apply_reduction(x_raw, fit=False)

    def _reducer_prefix(self):
        if self.reducer is None:
            return "MANUAL"
        return f"MANUAL_{reducer_feature_tag(self.reducer)}"

    def get_feature_names_out(self):
        if self.reducer is not None and self.output_dim is not None:
            prefix = self._reducer_prefix()
            return [f"{prefix}_{i + 1}" for i in range(self.output_dim)]
        return self.assembler.get_feature_names_out()

    @property
    def feature_info(self):
        info = dict(self.assembler.feature_info)
        info["reducer"] = self.reducer
        info["kernel"] = self.kernel
        info["kernel_gamma"] = self.kernel_gamma
        info["n_components"] = self.n_components
        info["raw_dim"] = self.raw_dim
        info["output_dim"] = self.output_dim
        info["standardize_before_reduction"] = self.standardize_before_reduction
        info["standardize_after_reduction"] = self.standardize_after_reduction
        return info
