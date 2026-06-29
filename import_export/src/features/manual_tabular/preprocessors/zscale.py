"""
Stable preprocessing for sequence Z-scale features.

This transformer is intended for fold-safe model training. It fixes the
output schema, computes imputation values on the training fold only, and can
optionally expose missingness as explicit binary indicators.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


ZSCALE_PREFIX = "SEQ_ZScale_"

ZSCALE_MEAN_COLS = [f"{ZSCALE_PREFIX}Window_Z{i}_Mean" for i in range(1, 6)]
ZSCALE_STD_COLS = [f"{ZSCALE_PREFIX}Window_Z{i}_Std" for i in range(1, 6)]
ZSCALE_COLS = ZSCALE_MEAN_COLS + ZSCALE_STD_COLS


class ZScalePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe: bool = False,
        strict: bool = True,
        keep_extra_zscale_cols: bool = False,
        impute_strategy: str = "median",
        all_nan_fill_value: float = 0.0,
        add_missing_indicator: bool = True,
        clip_std_nonnegative: bool = True,
        standardize: bool = False,
    ):
        self.return_dataframe = return_dataframe
        self.strict = strict
        self.keep_extra_zscale_cols = keep_extra_zscale_cols
        self.impute_strategy = impute_strategy
        self.all_nan_fill_value = all_nan_fill_value
        self.add_missing_indicator = add_missing_indicator
        self.clip_std_nonnegative = clip_std_nonnegative
        self.standardize = standardize

        self._feature_cols: Optional[List[str]] = None
        self._fill_values: Optional[Dict[str, float]] = None
        self._missing_rate: Optional[Dict[str, float]] = None
        self._all_nan_cols: Optional[Dict[str, bool]] = None
        self._scaler: Optional[StandardScaler] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X).copy()
        X = self._select_feature_columns(X)
        X = self._prepare_numeric_frame(X)

        self._feature_cols = list(X.columns)
        self._fill_values = {}
        self._missing_rate = {}
        self._all_nan_cols = {}

        for col in self._feature_cols:
            s = X[col]
            self._missing_rate[col] = float(s.isna().mean())
            self._all_nan_cols[col] = bool(s.dropna().empty)
            self._fill_values[col] = self._compute_fill_value(s)

        X_imp = self._apply_imputation(X)

        if self.standardize and len(self._feature_cols) > 0:
            self._scaler = StandardScaler()
            self._scaler.fit(X_imp[self._feature_cols])
        else:
            self._scaler = None

        self.feature_names_ = list(self._feature_cols)
        if self.add_missing_indicator:
            self.feature_names_.extend([f"{c}__missing" for c in self._feature_cols])

        return self

    def transform(self, X: pd.DataFrame):
        if self._feature_cols is None or self._fill_values is None:
            raise RuntimeError("ZScalePreprocessor must be fitted before transform.")

        X = self._check_input(X).copy()

        for col in self._feature_cols:
            if col not in X.columns:
                X[col] = np.nan

        X = X[self._feature_cols].copy()
        X = self._prepare_numeric_frame(X)
        missing_indicators = X.isna().astype(float)
        X = self._apply_imputation(X)

        if self._scaler is not None:
            X[self._feature_cols] = self._scaler.transform(X[self._feature_cols])

        X = X.astype(float)

        if self.add_missing_indicator:
            for col in self._feature_cols:
                X[f"{col}__missing"] = missing_indicators[col].astype(float)

        if self.return_dataframe:
            return X

        return X.to_numpy(dtype=float)

    def get_feature_names_out(self):
        if self.feature_names_ is None:
            return None
        return list(self.feature_names_)

    def get_impute_summary(self) -> pd.DataFrame:
        if (
            self._feature_cols is None
            or self._fill_values is None
            or self._missing_rate is None
            or self._all_nan_cols is None
        ):
            raise RuntimeError("ZScalePreprocessor must be fitted before calling get_impute_summary().")

        rows = []
        for col in self._feature_cols:
            rows.append(
                {
                    "feature_name": col,
                    "impute_strategy": self.impute_strategy,
                    "fill_value": self._fill_values[col],
                    "missing_rate": self._missing_rate[col],
                    "all_nan_in_fit": self._all_nan_cols[col],
                    "standardized": self.standardize,
                }
            )
        return pd.DataFrame(rows)

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("ZScalePreprocessor expects a pandas DataFrame.")

        X = X.copy()
        bad_cols = [c for c in X.columns if c != "INDEX" and not c.startswith(ZSCALE_PREFIX)]
        if bad_cols:
            raise ValueError(f"Non-{ZSCALE_PREFIX} columns found: {bad_cols}")

        if "INDEX" in X.columns:
            X = X.drop(columns=["INDEX"])

        return X

    def _select_feature_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        zscale_cols = [c for c in X.columns if c.startswith(ZSCALE_PREFIX)]

        if self.strict:
            for col in ZSCALE_COLS:
                if col not in X.columns:
                    X[col] = np.nan

            ordered_cols = list(ZSCALE_COLS)

            if self.keep_extra_zscale_cols:
                extra_cols = sorted([c for c in zscale_cols if c not in ZSCALE_COLS])
                ordered_cols.extend(extra_cols)

            return X[ordered_cols].copy()

        return X[zscale_cols].copy()

    def _prepare_numeric_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)

        if self.clip_std_nonnegative:
            for col in X.columns:
                if col in ZSCALE_STD_COLS or col.endswith("_Std"):
                    X[col] = X[col].clip(lower=0.0)

        return X

    def _compute_fill_value(self, s: pd.Series) -> float:
        valid = pd.to_numeric(s, errors="coerce").dropna()
        valid = valid[np.isfinite(valid)]

        if valid.empty:
            return float(self.all_nan_fill_value)

        if self.impute_strategy == "median":
            return float(valid.median())

        if self.impute_strategy == "mean":
            return float(valid.mean())

        if self.impute_strategy == "zero":
            return 0.0

        raise ValueError(f"Unsupported impute_strategy: {self.impute_strategy}")

    def _apply_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._fill_values is None:
            raise RuntimeError("ZScalePreprocessor must be fitted before imputation.")

        X = X.copy()
        for col in X.columns:
            fill_value = self._fill_values.get(col, float(self.all_nan_fill_value))
            X[col] = X[col].fillna(fill_value)

        return X
