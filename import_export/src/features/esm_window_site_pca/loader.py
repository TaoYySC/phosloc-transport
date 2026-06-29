from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.features.reducers.factory import (
    build_reducer_model,
    fit_reducer,
    normalize_reducer_name,
    reducer_feature_tag,
    transform_reducer,
)


class ESMWindowSitePCALoader:
    def __init__(
        self,
        embedding_dir,
        acc_col="ACC_ID",
        pos_col="POSITION",
        window_size=31,
        embedding_dim=1280,
        window_pooling="mean",
        use_window_embedding=True,
        use_site_embedding=True,
        reducer=None,
        pca_components=50,
        pca_components_window=None,
        pca_components_site=None,
        n_components_window=None,
        n_components_site=None,
        pls_components_window=None,
        pls_components_site=None,
        separate_pca_per_embedding=False,
        separate_reduction_per_embedding=None,
        standardize_before_pca=True,
        standardize_before_reduction=None,
        standardize_after_reduction=False,
        fill_missing=False,
        kernel="rbf",
        kernel_gamma=None,
    ):
        self.embedding_dir = Path(embedding_dir)
        self.acc_col = acc_col
        self.pos_col = pos_col
        self.window_size = int(window_size)
        self.embedding_dim = int(embedding_dim)
        self.window_pooling = window_pooling
        self.use_window_embedding = bool(use_window_embedding)
        self.use_site_embedding = bool(use_site_embedding)

        if standardize_before_reduction is not None:
            standardize_before_pca = standardize_before_reduction

        self.kernel = kernel
        self.kernel_gamma = kernel_gamma

        if reducer is not None:
            reducer = normalize_reducer_name(reducer)
            if reducer in {"none", "null", "raw"}:
                self.reducer = None
            else:
                self.reducer = reducer
        elif pls_components_window is not None or pls_components_site is not None:
            self.reducer = "pls"
        else:
            self.reducer = "pca"

        self.n_components_window = int(
            n_components_window
            or pls_components_window
            or pca_components_window
            or pca_components
        )
        self.n_components_site = int(
            n_components_site
            or pls_components_site
            or pca_components_site
            or pca_components
        )
        self.pca_components = int(pca_components)
        self.pca_components_window = self.n_components_window
        self.pca_components_site = self.n_components_site

        if separate_reduction_per_embedding is not None:
            separate_pca_per_embedding = separate_reduction_per_embedding
        if separate_pca_per_embedding is None:
            separate_pca_per_embedding = False
        self.separate_pca_per_embedding = bool(separate_pca_per_embedding)
        self.standardize_before_pca = bool(standardize_before_pca)
        self.standardize_after_reduction = bool(standardize_after_reduction)
        self.fill_missing = bool(fill_missing)

        if self.window_size % 2 != 1:
            raise ValueError("window_size must be an odd number.")

        if not self.use_window_embedding and not self.use_site_embedding:
            raise ValueError(
                "At least one of use_window_embedding or use_site_embedding must be true."
            )

        if self.window_pooling not in ["mean", "flatten"]:
            raise ValueError("window_pooling must be either mean or flatten.")

        if (
            self.separate_pca_per_embedding
            and self.use_window_embedding
            and self.use_site_embedding
        ):
            self.reduction_mode = "separate"
        else:
            self.reduction_mode = "joint"

        # Backward-compatible attribute names used by transform paths.
        self.pca_mode = self.reduction_mode

        self.scaler = None
        self.pca = None
        self.window_scaler = None
        self.window_pca = None
        self.site_scaler = None
        self.site_pca = None
        self.output_dim = None
        self.raw_dim = None
        self.embedding_cache = {}

    def _load_embedding(self, acc):
        acc = str(acc)
        if acc in self.embedding_cache:
            return self.embedding_cache[acc]

        path = self.embedding_dir / f"{acc}.pt"
        if not path.exists():
            path = self.embedding_dir / f"{acc}.PT"

        if not path.exists():
            raise FileNotFoundError(f"Cannot find ESM embedding file for {acc}: {path}")

        obj = torch.load(path, map_location="cpu")

        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()
        elif isinstance(obj, np.ndarray):
            arr = obj
        elif isinstance(obj, dict):
            arr = self._extract_array_from_dict(obj, acc)
        else:
            raise TypeError(f"Unsupported embedding object type for {acc}: {type(obj)}")

        arr = np.asarray(arr, dtype=np.float32)

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        if arr.ndim != 2:
            raise ValueError(f"Embedding for {acc} must be 2D, got shape {arr.shape}")

        if arr.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch for {acc}. "
                f"Expected {self.embedding_dim}, got {arr.shape[1]}"
            )

        self.embedding_cache[acc] = arr
        return arr

    def _extract_array_from_dict(self, obj, acc):
        candidate_keys = [
            "embedding",
            "embeddings",
            "representations",
            "x",
            "tensor",
            "per_residue",
        ]

        for key in candidate_keys:
            if key in obj:
                value = obj[key]

                if isinstance(value, dict):
                    if len(value) == 0:
                        continue
                    layer_key = sorted(value.keys())[-1]
                    value = value[layer_key]

                if isinstance(value, torch.Tensor):
                    return value.detach().cpu().numpy()

                if isinstance(value, np.ndarray):
                    return value

        tensor_values = []
        for value in obj.values():
            if isinstance(value, torch.Tensor):
                tensor_values.append(value)
            elif isinstance(value, np.ndarray):
                tensor_values.append(value)

        if len(tensor_values) == 1:
            value = tensor_values[0]
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return value

        raise ValueError(f"Cannot extract embedding array from dict file for {acc}")

    def _extract_window_feature(self, emb, center):
        half = self.window_size // 2
        start = center - half
        end = center + half + 1
        protein_len = emb.shape[0]

        left_pad = max(0, -start)
        right_pad = max(0, end - protein_len)

        clipped_start = max(0, start)
        clipped_end = min(protein_len, end)

        window = emb[clipped_start:clipped_end]

        if left_pad > 0 or right_pad > 0:
            window = np.pad(
                window,
                pad_width=((left_pad, right_pad), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        if window.shape[0] != self.window_size:
            raise ValueError(
                f"Window extraction failed at center={center}. "
                f"Expected {self.window_size}, got {window.shape[0]}"
            )

        if self.window_pooling == "mean":
            return window.mean(axis=0).astype(np.float32)

        return window.reshape(-1).astype(np.float32)

    def _extract_site_parts(self, acc, pos):
        emb = self._load_embedding(acc)
        protein_len = emb.shape[0]

        pos = int(pos)
        center = pos - 1

        if center < 0 or center >= protein_len:
            raise ValueError(
                f"POSITION out of range for {acc}. "
                f"POSITION={pos}, protein length={protein_len}"
            )

        window_feature = None
        site_feature = None

        if self.use_window_embedding:
            window_feature = self._extract_window_feature(emb, center)

        if self.use_site_embedding:
            site_feature = emb[center].astype(np.float32)

        return window_feature, site_feature

    def _window_raw_dim(self):
        if self.window_pooling == "mean":
            return self.embedding_dim
        return self.window_size * self.embedding_dim

    def _expected_raw_dim(self):
        dim = 0

        if self.use_window_embedding:
            dim += self._window_raw_dim()

        if self.use_site_embedding:
            dim += self.embedding_dim

        return dim

    def _zero_vector(self, dim):
        return np.zeros(dim, dtype=np.float32)

    def _build_separate_matrices(self, df):
        window_rows = []
        site_rows = []

        for _, row in df.iterrows():
            acc = row[self.acc_col]
            pos = row[self.pos_col]

            try:
                window_feature, site_feature = self._extract_site_parts(acc, pos)
            except Exception:
                if not self.fill_missing:
                    raise
                window_feature = None
                site_feature = None

            if self.use_window_embedding:
                if window_feature is None:
                    window_feature = self._zero_vector(self._window_raw_dim())
                window_rows.append(window_feature)

            if self.use_site_embedding:
                if site_feature is None:
                    site_feature = self._zero_vector(self.embedding_dim)
                site_rows.append(site_feature)

        x_window = np.vstack(window_rows).astype(np.float32) if window_rows else None
        x_site = np.vstack(site_rows).astype(np.float32) if site_rows else None
        return x_window, x_site

    def _build_raw_matrix(self, df):
        rows = []

        for _, row in df.iterrows():
            acc = row[self.acc_col]
            pos = row[self.pos_col]

            try:
                window_feature, site_feature = self._extract_site_parts(acc, pos)
                parts = []
                if window_feature is not None:
                    parts.append(window_feature)
                if site_feature is not None:
                    parts.append(site_feature)
                feature = np.concatenate(parts).astype(np.float32)
            except Exception:
                if not self.fill_missing:
                    raise
                feature = self._zero_vector(self._expected_raw_dim())

            rows.append(feature)

        return np.vstack(rows).astype(np.float32)

    def _resolve_n_components(self, requested, n_samples, n_features):
        max_components = min(int(requested), n_samples, n_features)
        if max_components < 1:
            raise ValueError("Dimensionality reduction requires at least one component.")
        return max_components

    def _fit_reducer(self, x_scaled, n_components, y=None):
        n_components = self._resolve_n_components(
            n_components,
            x_scaled.shape[0],
            x_scaled.shape[1],
        )
        model = build_reducer_model(
            self.reducer,
            n_components,
            kernel=self.kernel,
            gamma=self.kernel_gamma,
        )
        x_reduced = fit_reducer(model, self.reducer, x_scaled, y)
        return model, x_reduced

    def _transform_reducer(self, model, x_scaled):
        return transform_reducer(model, self.reducer, x_scaled)

    def _fit_transform_block(
        self,
        x_raw,
        n_components,
        scaler_attr,
        model_attr,
        post_scaler_attr=None,
        y=None,
    ):
        x_raw = np.asarray(x_raw, dtype=np.float32)

        if self.standardize_before_pca:
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x_raw)
        else:
            scaler = None
            x_scaled = x_raw

        model, x_reduced = self._fit_reducer(x_scaled, n_components, y=y)

        if self.standardize_after_reduction:
            if post_scaler_attr is None:
                raise ValueError("post_scaler_attr is required when standardize_after_reduction is true.")
            post_scaler = StandardScaler()
            x_reduced = post_scaler.fit_transform(x_reduced)
            setattr(self, post_scaler_attr, post_scaler)
        elif post_scaler_attr is not None:
            setattr(self, post_scaler_attr, None)

        setattr(self, scaler_attr, scaler)
        setattr(self, model_attr, model)
        return x_reduced.astype(np.float32)

    def _transform_block(self, x_raw, scaler_attr, model_attr, post_scaler_attr=None):
        model = getattr(self, model_attr)
        if model is None:
            raise RuntimeError(f"{model_attr} has not been fitted.")

        x_raw = np.asarray(x_raw, dtype=np.float32)
        scaler = getattr(self, scaler_attr)
        if scaler is not None:
            x_scaled = scaler.transform(x_raw)
        else:
            x_scaled = x_raw

        x_reduced = self._transform_reducer(model, x_scaled)
        if post_scaler_attr is not None:
            post_scaler = getattr(self, post_scaler_attr)
            if post_scaler is not None:
                x_reduced = post_scaler.transform(x_reduced)
        return x_reduced.astype(np.float32)

    def _fit_transform_separate(self, df, y=None):
        x_window, x_site = self._build_separate_matrices(df)
        parts = []

        if x_window is not None:
            parts.append(
                self._fit_transform_block(
                    x_window,
                    self.n_components_window,
                    "window_scaler",
                    "window_pca",
                    "window_post_scaler",
                    y=y,
                )
            )

        if x_site is not None:
            parts.append(
                self._fit_transform_block(
                    x_site,
                    self.n_components_site,
                    "site_scaler",
                    "site_pca",
                    "site_post_scaler",
                    y=y,
                )
            )

        x_reduced = np.concatenate(parts, axis=1)
        self.raw_dim = int((x_window.shape[1] if x_window is not None else 0) + (
            x_site.shape[1] if x_site is not None else 0
        ))
        self.output_dim = int(x_reduced.shape[1])
        return x_reduced

    def _transform_separate(self, df):
        x_window, x_site = self._build_separate_matrices(df)
        parts = []

        if x_window is not None:
            parts.append(
                self._transform_block(
                    x_window, "window_scaler", "window_pca", "window_post_scaler"
                )
            )

        if x_site is not None:
            parts.append(
                self._transform_block(x_site, "site_scaler", "site_pca", "site_post_scaler")
            )

        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit_transform(self, df, y=None):
        if self.reduction_mode == "separate":
            return self._fit_transform_separate(df, y=y)

        x_raw = self._build_raw_matrix(df)
        self.raw_dim = int(x_raw.shape[1])

        x_reduced = self._fit_transform_block(
            x_raw,
            self.pca_components,
            "scaler",
            "pca",
            "post_scaler",
            y=y,
        )
        self.output_dim = int(x_reduced.shape[1])
        return x_reduced

    def transform(self, df):
        if self.reduction_mode == "separate":
            if self.window_pca is None and self.site_pca is None:
                raise RuntimeError(
                    "Separate reduction has not been fitted. Call fit_transform first."
                )
            return self._transform_separate(df)

        if self.pca is None:
            raise RuntimeError("Reducer has not been fitted. Call fit_transform first.")

        x_raw = self._build_raw_matrix(df)
        return self._transform_block(x_raw, "scaler", "pca", "post_scaler")

    def load(self, df):
        return self.transform(df)

    def _model_n_components(self, model):
        if self.reducer == "pca":
            return model.n_components_
        if hasattr(model, "pls_"):
            return model.pls_.n_components
        return model.n_components

    def _feature_prefix(self, part):
        tag = reducer_feature_tag(self.reducer)
        return f"ESM_{part.upper()}_{tag}"

    def get_feature_names_out(self):
        if self.output_dim is None:
            return []

        names = []

        if self.reduction_mode == "separate":
            if self.window_pca is not None:
                prefix = self._feature_prefix("window")
                n = self._model_n_components(self.window_pca)
                names.extend([f"{prefix}_{i + 1}" for i in range(n)])
            if self.site_pca is not None:
                prefix = self._feature_prefix("site")
                n = self._model_n_components(self.site_pca)
                names.extend([f"{prefix}_{i + 1}" for i in range(n)])
            return names

        prefix = f"ESM_{self.reducer.upper()}"
        return [f"{prefix}_{i + 1}" for i in range(self.output_dim)]
