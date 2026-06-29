from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


SEQ_PREFIX = "SEQ_"

SEQ_RATIO_COLS = [
    f"{SEQ_PREFIX}ratio_R_over_K",
    f"{SEQ_PREFIX}ratio_E_over_D",
]

SEQ_HYDRO_COL = f"{SEQ_PREFIX}mean_hydropathy_kd"

DEFAULT_EXCLUDED_PREFIXES: List[str] = []


class SequencePhysicochemPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe: bool = False,
        impute_strategy: str = "median",
        all_nan_fill_value: float = 0.0,
        add_missing_indicator: bool = False,
        standardize: bool = False,
        clip_physical_ranges: bool = True,
        clip_fraction_range: bool = True,
        clip_ratio_range: bool = True,
        ratio_clip_max: float = 20.0,
        log_transform_ratios: bool = False,
        excluded_prefixes: Optional[Sequence[str]] = None,
    ):
        self.return_dataframe = return_dataframe
        self.impute_strategy = impute_strategy
        self.all_nan_fill_value = all_nan_fill_value
        self.add_missing_indicator = add_missing_indicator
        self.standardize = standardize
        self.clip_physical_ranges = clip_physical_ranges
        self.clip_fraction_range = clip_fraction_range
        self.clip_ratio_range = clip_ratio_range
        self.ratio_clip_max = ratio_clip_max
        self.log_transform_ratios = log_transform_ratios
        self.excluded_prefixes = excluded_prefixes

        self._feature_cols: Optional[List[str]] = None
        self._fill_values: Optional[Dict[str, float]] = None
        self._missing_rate_: Optional[Dict[str, float]] = None
        self._all_nan_cols_: Optional[Dict[str, bool]] = None
        self._scaler: Optional[StandardScaler] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X)
        X = self._select_feature_columns(X)
        X = self._prepare_numeric_frame(X)
        X = self._apply_feature_transforms(X)

        self._feature_cols = list(X.columns)
        self._fill_values = {}
        self._missing_rate_ = {}
        self._all_nan_cols_ = {}

        for col in self._feature_cols:
            s = X[col]
            self._missing_rate_[col] = float(s.isna().mean())
            valid = s.dropna()
            valid = valid[np.isfinite(valid.to_numpy(dtype=float))]
            is_all_nan = len(valid) == 0
            self._all_nan_cols_[col] = bool(is_all_nan)
            self._fill_values[col] = self._compute_fill_value(col, valid)

        X_imp = self._fill_missing(X)

        if self.standardize and self._feature_cols:
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
            raise RuntimeError("SequencePhysicochemPreprocessor must be fitted before transform.")

        X = self._check_input(X)

        for col in self._feature_cols:
            if col not in X.columns:
                X[col] = np.nan

        X = X[self._feature_cols].copy()
        X = self._prepare_numeric_frame(X)
        X = self._apply_feature_transforms(X)
        missing_mask = X.isna()
        X_out = self._fill_missing(X)

        if self._scaler is not None:
            X_out[self._feature_cols] = self._scaler.transform(X_out[self._feature_cols])

        if self.add_missing_indicator:
            for col in self._feature_cols:
                X_out[f"{col}__missing"] = missing_mask[col].astype(float)

        X_out = X_out[self.feature_names_]
        non_numeric = X_out.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise ValueError(f"Non numeric columns remain after SequencePhysicochemPreprocessor: {non_numeric}")

        if self.return_dataframe:
            return X_out

        return X_out.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_ is None:
            return None
        return list(self.feature_names_)

    def get_impute_summary(self) -> pd.DataFrame:
        if self._feature_cols is None or self._fill_values is None:
            raise RuntimeError("SequencePhysicochemPreprocessor must be fitted before calling get_impute_summary().")

        rows = []
        for col in self._feature_cols:
            rows.append(
                {
                    "feature_name": col,
                    "impute_strategy": self._strategy_for_col(col),
                    "fill_value": self._fill_values[col],
                    "missing_rate": self._missing_rate_.get(col, np.nan) if self._missing_rate_ else np.nan,
                    "all_nan_in_fit": self._all_nan_cols_.get(col, False) if self._all_nan_cols_ else False,
                    "is_ratio_feature": self._is_ratio_col(col),
                    "is_fraction_feature": self._is_fraction_col(col),
                    "is_hydropathy_feature": self._is_hydropathy_col(col),
                }
            )
        return pd.DataFrame(rows)

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("SequencePhysicochemPreprocessor expects a pandas DataFrame.")

        X = X.copy()
        bad = [c for c in X.columns if c != "INDEX" and not c.startswith(SEQ_PREFIX)]
        if bad:
            raise ValueError(f"Non {SEQ_PREFIX} columns found: {bad}")

        if "INDEX" in X.columns:
            X = X.drop(columns=["INDEX"])

        return X

    def _select_feature_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        excluded = tuple(self.excluded_prefixes) if self.excluded_prefixes is not None else tuple(DEFAULT_EXCLUDED_PREFIXES)
        cols = []
        for col in X.columns:
            if not col.startswith(SEQ_PREFIX):
                continue
            if excluded and col.startswith(excluded):
                continue
            cols.append(col)
        return X[cols].copy()

    def _prepare_numeric_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        return X

    def _apply_feature_transforms(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in X.columns:
            if self.clip_physical_ranges and self._is_hydropathy_col(col):
                X[col] = X[col].clip(lower=-4.5, upper=4.5)

            if self.clip_fraction_range and self._is_fraction_col(col):
                X[col] = X[col].clip(lower=0.0, upper=1.0)

            if self.clip_ratio_range and self._is_ratio_col(col):
                X[col] = X[col].clip(lower=0.0, upper=float(self.ratio_clip_max))

            if self.log_transform_ratios and self._is_ratio_col(col):
                X[col] = np.log1p(np.clip(X[col].to_numpy(dtype=float), 0.0, None))

            if self._is_nonnegative_col(col):
                X[col] = X[col].clip(lower=0.0)

        X = X.replace([np.inf, -np.inf], np.nan)
        return X

    def _fill_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._fill_values is None:
            raise RuntimeError("Fill values are not available. Fit the preprocessor first.")

        X_out = X.copy()
        for col in self._feature_cols or []:
            X_out[col] = X_out[col].fillna(self._fill_values[col])
        return X_out.astype(float)

    def _compute_fill_value(self, col: str, valid: pd.Series) -> float:
        if len(valid) == 0:
            return float(self._default_fill_for_col(col))

        strategy = self._strategy_for_col(col)
        if strategy == "mean":
            return float(valid.mean())
        if strategy == "median":
            return float(valid.median())
        if strategy == "zero":
            return 0.0
        raise ValueError(f"Unsupported impute_strategy: {self.impute_strategy}")

    def _strategy_for_col(self, col: str) -> str:
        if self._is_ratio_col(col):
            return "zero"
        if self.impute_strategy in {"mean", "median", "zero"}:
            return self.impute_strategy
        raise ValueError(f"Unsupported impute_strategy: {self.impute_strategy}")

    def _default_fill_for_col(self, col: str) -> float:
        if self._is_ratio_col(col):
            return 0.0
        if self._is_fraction_col(col):
            return 0.0
        if self._is_nonnegative_col(col):
            return 0.0
        if self._is_hydropathy_col(col):
            return 0.0
        return float(self.all_nan_fill_value)

    def _is_ratio_col(self, col: str) -> bool:
        if col in SEQ_RATIO_COLS:
            return True
        low = col.lower()
        if "ratio" not in low:
            return False
        return not self._is_fraction_col(col)

    def _is_fraction_col(self, col: str) -> bool:
        low = col.lower()
        tokens = ["fraction", "frac", "frequency", "freq", "percentage", "percent", "composition", "proportion"]
        return any(t in low for t in tokens)

    def _is_hydropathy_col(self, col: str) -> bool:
        low = col.lower()
        return col == SEQ_HYDRO_COL or "hydropathy" in low or "hydrophobicity" in low

    def _is_nonnegative_col(self, col: str) -> bool:
        low = col.lower()
        if self._is_ratio_col(col) or self._is_fraction_col(col):
            return False
        tokens = ["count", "length", "window_size", "num_", "number"]
        return any(t in low for t in tokens)
