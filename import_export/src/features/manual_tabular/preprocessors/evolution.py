from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


EVOLUTION_PREFIX = "FUNC_Evolution_"

PROPORTION_KEYWORDS = (
    "fraction",
    "frequency",
    "prob",
    "probability",
    "percent",
    "percentage",
)


class EvolutionPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe: bool = False,
        strategy: str = "median",
        add_missing_indicator: bool = True,
        add_has_gates: bool = False,
        all_nan_fill_value: float = 0.0,
        clip_proportion_like_cols: bool = True,
    ):
        self.return_dataframe = return_dataframe
        self.strategy = strategy
        self.add_missing_indicator = add_missing_indicator
        self.add_has_gates = add_has_gates
        self.all_nan_fill_value = all_nan_fill_value
        self.clip_proportion_like_cols = clip_proportion_like_cols

        self._cols: Optional[List[str]] = None
        self._fill_values: Optional[Dict[str, float]] = None
        self._missing_rate: Optional[Dict[str, float]] = None
        self._all_nan_cols: Optional[List[str]] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X)
        X = self._drop_index(X)

        self._cols = [c for c in X.columns if c.startswith(EVOLUTION_PREFIX)]
        self._fill_values = {}
        self._missing_rate = {}
        self._all_nan_cols = []

        if len(self._cols) == 0:
            self.feature_names_ = []
            return self

        X_num = self._prepare_numeric_frame(X, self._cols)

        for col in self._cols:
            values = X_num[col]
            self._missing_rate[col] = float(values.isna().mean())
            fill_value, is_all_nan = self._compute_fill_value(values)
            self._fill_values[col] = fill_value
            if is_all_nan:
                self._all_nan_cols.append(col)

        self.feature_names_ = self._build_output_feature_names()
        return self

    def transform(self, X: pd.DataFrame):
        if self._cols is None or self._fill_values is None:
            raise RuntimeError("EvolutionPreprocessor must be fitted before transform.")

        X = self._check_input(X)
        X = self._drop_index(X)

        if len(self._cols) == 0:
            out = pd.DataFrame(index=X.index)
            return out if self.return_dataframe else out.to_numpy(dtype=float)

        X_num = self._prepare_numeric_frame(X, self._cols)
        missing_mask = X_num.isna()
        X_out = X_num.copy()

        for col in self._cols:
            X_out[col] = X_out[col].fillna(self._fill_values[col])

        if self.clip_proportion_like_cols:
            for col in self._cols:
                if self._is_proportion_like(col):
                    X_out[col] = X_out[col].clip(lower=0.0, upper=1.0)

        if self.add_missing_indicator:
            for col in self._cols:
                X_out[f"{col}__missing"] = missing_mask[col].astype(float)

        if self.add_has_gates:
            for col in self._cols:
                X_out[f"{col}__has"] = (~missing_mask[col]).astype(float)

        X_out = X_out[self.feature_names_].astype(float)

        if self.return_dataframe:
            return X_out

        return X_out.to_numpy(dtype=float)

    def get_feature_names_out(self):
        if self.feature_names_ is None:
            return None
        return list(self.feature_names_)

    def get_impute_summary(self) -> pd.DataFrame:
        if self._cols is None or self._fill_values is None or self._missing_rate is None:
            raise RuntimeError("EvolutionPreprocessor must be fitted before calling get_impute_summary().")

        all_nan = set(self._all_nan_cols or [])
        rows = []
        for col in self._cols:
            rows.append(
                {
                    "feature_name": col,
                    "impute_strategy": self.strategy,
                    "fill_value": self._fill_values[col],
                    "missing_rate": self._missing_rate[col],
                    "all_nan_in_fit": col in all_nan,
                    "clipped_to_0_1": self.clip_proportion_like_cols and self._is_proportion_like(col),
                }
            )
        return pd.DataFrame(rows)

    def _prepare_numeric_frame(self, X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        X_num = X.copy()

        for col in cols:
            if col not in X_num.columns:
                X_num[col] = np.nan

        X_num = X_num[cols].copy()
        X_num = X_num.apply(pd.to_numeric, errors="coerce")
        X_num = X_num.replace([np.inf, -np.inf], np.nan)
        return X_num

    def _compute_fill_value(self, values: pd.Series):
        valid = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        valid = valid[np.isfinite(valid)]

        if valid.size == 0:
            return float(self.all_nan_fill_value), True

        if self.strategy == "median":
            return float(np.median(valid)), False
        if self.strategy == "mean":
            return float(np.mean(valid)), False
        if self.strategy == "zero":
            return 0.0, False
        if self.strategy == "constant":
            return float(self.all_nan_fill_value), False

        raise ValueError(f"Unknown imputation strategy: {self.strategy}")

    def _build_output_feature_names(self) -> List[str]:
        names = list(self._cols or [])

        if self.add_missing_indicator:
            names.extend([f"{col}__missing" for col in (self._cols or [])])

        if self.add_has_gates:
            names.extend([f"{col}__has" for col in (self._cols or [])])

        return names

    def _is_proportion_like(self, col: str) -> bool:
        lower = col.lower()
        return any(keyword in lower for keyword in PROPORTION_KEYWORDS)

    def _drop_index(self, X: pd.DataFrame) -> pd.DataFrame:
        if "INDEX" in X.columns:
            return X.drop(columns=["INDEX"])
        return X.copy()

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("EvolutionPreprocessor expects a pandas DataFrame.")

        X = X.copy()
        bad_cols = [c for c in X.columns if c != "INDEX" and not c.startswith(EVOLUTION_PREFIX)]
        if bad_cols:
            raise ValueError(f"Non-{EVOLUTION_PREFIX} columns detected: {bad_cols}")

        return X
