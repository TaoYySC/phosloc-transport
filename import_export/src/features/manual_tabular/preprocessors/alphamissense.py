from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


ALPHAMISSENSE_PREFIX = "FUNC_AlphaMissense_"
ALPHAMISSENSE_HAS_COL = "FUNC_AlphaMissense_HasAnyVariant"


class AlphaMissensePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe: bool = False,
        strategy: str = "median",
        add_missing_indicator: bool = True,
        add_has_gates: bool = True,
        all_nan_fill_value: float = 0.0,
        clip_score_like_cols: bool = True,
        keep_empty_features: bool = True,
    ):
        self.return_dataframe = return_dataframe
        self.strategy = strategy
        self.add_missing_indicator = add_missing_indicator
        self.add_has_gates = add_has_gates
        self.all_nan_fill_value = all_nan_fill_value
        self.clip_score_like_cols = clip_score_like_cols
        self.keep_empty_features = keep_empty_features

        self._cols: Optional[List[str]] = None
        self._fill_values: Optional[Dict[str, float]] = None
        self._missing_rate: Optional[Dict[str, float]] = None
        self._all_nan_cols: Optional[List[str]] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X)
        X_num = self._select_and_prepare_columns(X)

        self._cols = list(X_num.columns)
        self._fill_values = {}
        self._missing_rate = {}
        self._all_nan_cols = []

        for col in self._cols:
            s = X_num[col]
            self._missing_rate[col] = float(s.isna().mean())
            valid = s.dropna()

            if len(valid) == 0:
                self._fill_values[col] = float(self.all_nan_fill_value)
                self._all_nan_cols.append(col)
                continue

            if col == ALPHAMISSENSE_HAS_COL:
                self._fill_values[col] = 0.0
            elif self.strategy == "median":
                self._fill_values[col] = float(valid.median())
            elif self.strategy == "mean":
                self._fill_values[col] = float(valid.mean())
            elif self.strategy == "zero":
                self._fill_values[col] = 0.0
            else:
                raise ValueError(f"Unsupported strategy: {self.strategy}")

        if not self.keep_empty_features:
            self._cols = [c for c in self._cols if c not in set(self._all_nan_cols)]

        self.feature_names_ = list(self._cols)
        if self.add_missing_indicator:
            self.feature_names_.extend([f"{c}_missing" for c in self._cols])
        if self.add_has_gates:
            self.feature_names_.extend([f"{c}__has" for c in self._cols if c != ALPHAMISSENSE_HAS_COL])

        return self

    def transform(self, X: pd.DataFrame):
        if self._cols is None or self._fill_values is None:
            raise RuntimeError("AlphaMissensePreprocessor must be fitted before transform.")

        X = self._check_input(X)
        X_num = self._select_and_prepare_columns(X)

        for col in self._cols:
            if col not in X_num.columns:
                X_num[col] = np.nan

        X_num = X_num[self._cols].copy()
        missing_mask = X_num.isna()

        out = X_num.copy()
        for col in self._cols:
            fill = self._fill_values.get(col, float(self.all_nan_fill_value))
            out[col] = out[col].fillna(fill)

        if ALPHAMISSENSE_HAS_COL in out.columns:
            out[ALPHAMISSENSE_HAS_COL] = (out[ALPHAMISSENSE_HAS_COL] > 0).astype(float)

        if self.add_missing_indicator:
            for col in self._cols:
                out[f"{col}_missing"] = missing_mask[col].astype(float)

        if self.add_has_gates:
            for col in self._cols:
                if col == ALPHAMISSENSE_HAS_COL:
                    continue
                out[f"{col}__has"] = (~missing_mask[col]).astype(float)

        if self.feature_names_ is not None:
            out = out[self.feature_names_]

        out = out.astype(float)

        if self.return_dataframe:
            return out

        return out.to_numpy(dtype=float)

    def get_feature_names_out(self):
        if self.feature_names_ is None:
            return None
        return list(self.feature_names_)

    def get_impute_summary(self) -> pd.DataFrame:
        if self._cols is None or self._fill_values is None or self._missing_rate is None:
            raise RuntimeError("AlphaMissensePreprocessor must be fitted before calling get_impute_summary().")

        rows = []
        all_nan = set(self._all_nan_cols or [])
        for col in self._cols:
            rows.append(
                {
                    "feature_name": col,
                    "impute_strategy": "zero" if col == ALPHAMISSENSE_HAS_COL else self.strategy,
                    "fill_value": self._fill_values.get(col, float(self.all_nan_fill_value)),
                    "missing_rate": self._missing_rate.get(col, np.nan),
                    "all_nan_in_fit": col in all_nan,
                    "score_like_clipped": self._is_score_like_col(col),
                }
            )
        return pd.DataFrame(rows)

    def _select_and_prepare_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in X.columns if str(c).startswith(ALPHAMISSENSE_PREFIX)]
        X_num = X[cols].copy() if cols else pd.DataFrame(index=X.index)

        if ALPHAMISSENSE_HAS_COL in X_num.columns:
            X_num[ALPHAMISSENSE_HAS_COL] = self._to_bool01(X_num[ALPHAMISSENSE_HAS_COL])

        for col in X_num.columns:
            if col == ALPHAMISSENSE_HAS_COL:
                continue
            X_num[col] = pd.to_numeric(X_num[col], errors="coerce")
            if self.clip_score_like_cols and self._is_score_like_col(col):
                X_num[col] = X_num[col].clip(lower=0.0, upper=1.0)

        return X_num

    def _to_bool01(self, values) -> pd.Series:
        s = pd.Series(values)

        if s.dtype == bool:
            return s.astype(float)

        if s.dtype == object:
            s_str = s.astype(str).str.strip().str.lower()
            mapped = s_str.map(
                {
                    "true": 1.0,
                    "false": 0.0,
                    "yes": 1.0,
                    "no": 0.0,
                    "y": 1.0,
                    "n": 0.0,
                    "1": 1.0,
                    "0": 0.0,
                }
            )
            if mapped.notna().any():
                numeric_fallback = pd.to_numeric(s, errors="coerce")
                return mapped.where(mapped.notna(), numeric_fallback).fillna(0.0).astype(float)

        x = pd.to_numeric(s, errors="coerce").fillna(0.0)
        return (x > 0).astype(float)

    def _is_score_like_col(self, col: str) -> bool:
        if col == ALPHAMISSENSE_HAS_COL:
            return False

        name = col.lower()
        keywords = [
            "score",
            "prob",
            "probability",
            "pathogenicity",
            "benign",
            "ambiguous",
            "pathogenic",
            "fraction",
            "ratio",
            "mean",
            "median",
            "max",
            "min",
        ]
        return any(k in name for k in keywords)

    def _check_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("AlphaMissensePreprocessor expects a pandas DataFrame.")

        X = X.copy()
        bad_cols = [
            c for c in X.columns
            if c != "INDEX" and not str(c).startswith(ALPHAMISSENSE_PREFIX)
        ]
        if bad_cols:
            raise ValueError(f"Non-{ALPHAMISSENSE_PREFIX} columns detected: {bad_cols}")

        if "INDEX" in X.columns:
            X = X.drop(columns=["INDEX"])

        return X
