from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


HOTSPOT_PREFIX = "SEQ_Hotspot_"


class HotspotPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe: bool = False,
        add_missing_indicator: bool = False,
        log_transform_dist: bool = True,
        clip_fraction_range: bool = True,
        clip_count_nonnegative: bool = True,
        all_nan_fill_value: float = 0.0,
    ):
        self.return_dataframe = return_dataframe
        self.add_missing_indicator = add_missing_indicator
        self.log_transform_dist = log_transform_dist
        self.clip_fraction_range = clip_fraction_range
        self.clip_count_nonnegative = clip_count_nonnegative
        self.all_nan_fill_value = all_nan_fill_value

        self._feature_cols: Optional[List[str]] = None
        self._fill_values: Optional[Dict[str, float]] = None
        self._missing_rate_: Optional[Dict[str, float]] = None
        self._output_feature_names: Optional[List[str]] = None
        self._transformed_distance_cols: Optional[List[str]] = None
        self._signed_distance_cols: Optional[List[str]] = None
        self._all_nan_cols: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X).copy()
        X = self._drop_index_column(X)

        self._feature_cols = list(X.columns)
        self._fill_values = {}
        self._missing_rate_ = {}
        self._all_nan_cols = []

        if len(self._feature_cols) == 0:
            self._output_feature_names = []
            return self

        X_prepared = self._prepare_frame(X, fit_mode=True)

        for col in self._feature_cols:
            s = X_prepared[col]
            self._missing_rate_[col] = float(s.isna().mean())
            values = s.to_numpy(dtype=float)
            values = values[np.isfinite(values)]

            if values.size == 0:
                self._fill_values[col] = float(self.all_nan_fill_value)
                self._all_nan_cols.append(col)
            else:
                self._fill_values[col] = float(np.median(values))

        self._output_feature_names = list(self._feature_cols)
        if self.add_missing_indicator:
            self._output_feature_names.extend([f"{c}_missing" for c in self._feature_cols])

        return self

    def transform(self, X: pd.DataFrame):
        if self._feature_cols is None or self._fill_values is None:
            raise RuntimeError("HotspotPreprocessor must be fitted before transform.")

        X = self._check_input(X).copy()
        X = self._drop_index_column(X)

        if len(self._feature_cols) == 0:
            out = pd.DataFrame(index=X.index)
            return out if self.return_dataframe else out.to_numpy(dtype=float)

        for col in self._feature_cols:
            if col not in X.columns:
                X[col] = np.nan

        X = X[self._feature_cols].copy()
        X_prepared = self._prepare_frame(X, fit_mode=False)
        X_missing = X_prepared.isna()

        X_out = X_prepared.copy()
        for col in self._feature_cols:
            X_out[col] = X_out[col].fillna(self._fill_values[col])

        if self.add_missing_indicator:
            for col in self._feature_cols:
                X_out[f"{col}_missing"] = X_missing[col].astype(float)

        X_out = X_out.astype(float)

        if self.return_dataframe:
            return X_out

        return X_out.to_numpy(dtype=float)

    def get_feature_names_out(self):
        if self._output_feature_names is None:
            return None
        return list(self._output_feature_names)

    def get_impute_summary(self) -> pd.DataFrame:
        if self._feature_cols is None or self._fill_values is None or self._missing_rate_ is None:
            raise RuntimeError("HotspotPreprocessor must be fitted before calling get_impute_summary().")

        rows = []
        all_nan = set(self._all_nan_cols or [])
        transformed_dist = set(self._transformed_distance_cols or [])
        signed_dist = set(self._signed_distance_cols or [])

        for col in self._feature_cols:
            rows.append(
                {
                    "feature_name": col,
                    "fill_value": self._fill_values[col],
                    "missing_rate": self._missing_rate_[col],
                    "all_nan_in_fit": col in all_nan,
                    "distance_log_transformed": col in transformed_dist,
                    "signed_distance_log_transformed": col in signed_dist,
                }
            )

        return pd.DataFrame(rows)

    def _prepare_frame(self, X: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        X = X.copy()

        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        if self.log_transform_dist:
            signed_cols, unsigned_cols = self._identify_distance_columns(X.columns)

            if fit_mode:
                self._signed_distance_cols = list(signed_cols)
                self._transformed_distance_cols = list(signed_cols) + list(unsigned_cols)
            else:
                signed_cols = self._signed_distance_cols or []
                unsigned_cols = [
                    c for c in (self._transformed_distance_cols or [])
                    if c not in set(signed_cols)
                ]

            for col in signed_cols:
                if col in X.columns:
                    values = X[col].to_numpy(dtype=float)
                    X[col] = np.sign(values) * np.log1p(np.abs(values))

            for col in unsigned_cols:
                if col in X.columns:
                    values = X[col].to_numpy(dtype=float)
                    X[col] = np.log1p(np.clip(values, 0.0, None))

        if self.clip_fraction_range:
            for col in self._fraction_like_columns(X.columns):
                X[col] = X[col].clip(lower=0.0, upper=1.0)

        if self.clip_count_nonnegative:
            for col in self._count_like_columns(X.columns):
                X[col] = X[col].clip(lower=0.0)

        return X

    def _identify_distance_columns(self, cols) -> Tuple[List[str], List[str]]:
        signed_cols = []
        unsigned_cols = []

        for col in cols:
            low = col.lower()
            is_distance = (
                "distance" in low
                or low.endswith("_dist")
                or "_dist_" in low
                or low.endswith("dist")
            )
            if not is_distance:
                continue

            if "signed" in low or "centersigned" in low or "center_signed" in low:
                signed_cols.append(col)
            else:
                unsigned_cols.append(col)

        return signed_cols, unsigned_cols

    def _fraction_like_columns(self, cols) -> List[str]:
        out = []
        for col in cols:
            low = col.lower()
            if "fraction" in low or low.endswith("_frac") or "density" in low:
                out.append(col)
        return out

    def _count_like_columns(self, cols) -> List[str]:
        out = []
        for col in cols:
            low = col.lower()
            if "count" in low or low.endswith("_n") or "number" in low:
                out.append(col)
        return out

    def _drop_index_column(self, X: pd.DataFrame) -> pd.DataFrame:
        if "INDEX" in X.columns:
            return X.drop(columns=["INDEX"])
        return X

    def _check_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HotspotPreprocessor expects a DataFrame")

        X = X.copy()
        bad_cols = [c for c in X.columns if c != "INDEX" and not c.startswith(HOTSPOT_PREFIX)]
        if bad_cols:
            raise ValueError(f"Non-{HOTSPOT_PREFIX} columns detected: {bad_cols}")

        return X
