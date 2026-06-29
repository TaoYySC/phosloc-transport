from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


CHARGE_COLUMNS = [
    "SEQ_Charge_Window_PositiveCount",
    "SEQ_Charge_Window_NegativeCount",
    "SEQ_Charge_Window_ChargeSum",
    "SEQ_Charge_Window_PositiveFraction",
    "SEQ_Charge_Window_NegativeFraction",
    "SEQ_Charge_Site_PreWindow_NetCharge",
    "SEQ_Charge_Site_PostWindow_NetCharge",
    "SEQ_Charge_Site_Delta_NetCharge",
    "SEQ_Charge_Window_KRCluster_Score",
    "SEQ_Charge_Window_KRCluster_Density",
]

CHARGE_PREFIX = "SEQ_Charge_"

CHARGE_FRACTION_COLS = [
    "SEQ_Charge_Window_PositiveFraction",
    "SEQ_Charge_Window_NegativeFraction",
    "SEQ_Charge_Window_KRCluster_Density",
]

CHARGE_NONNEGATIVE_COLS = [
    "SEQ_Charge_Window_PositiveCount",
    "SEQ_Charge_Window_NegativeCount",
    "SEQ_Charge_Window_KRCluster_Score",
]

SUPPORTED_IMPUTE_STRATEGIES = {"median", "mean", "zero"}


class ChargeFeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe: bool = False,
        impute_strategy: str = "median",
        standardize: bool = False,
        clip_physical_ranges: bool = True,
        add_missing_indicator: bool = False,
        strict: bool = True,
        keep_extra_charge_cols: bool = False,
        all_nan_fill_value: float = 0.0,
    ):
        self.return_dataframe = return_dataframe
        self.impute_strategy = impute_strategy
        self.standardize = standardize
        self.clip_physical_ranges = clip_physical_ranges
        self.add_missing_indicator = add_missing_indicator
        self.strict = strict
        self.keep_extra_charge_cols = keep_extra_charge_cols
        self.all_nan_fill_value = all_nan_fill_value

        self._feature_cols: Optional[List[str]] = None
        self._fill_values: Optional[Dict[str, float]] = None
        self._missing_rate_: Optional[Dict[str, float]] = None
        self._all_nan_cols_: Optional[List[str]] = None
        self._scaler: Optional[StandardScaler] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.impute_strategy not in SUPPORTED_IMPUTE_STRATEGIES:
            raise ValueError(
                f"Unsupported impute_strategy: {self.impute_strategy}. "
                f"Supported values are {sorted(SUPPORTED_IMPUTE_STRATEGIES)}."
            )

        X = self._check_input(X).copy()
        X = self._select_feature_columns(X)
        X = self._prepare_numeric_frame(X)

        self._feature_cols = list(X.columns)
        self._fill_values = {}
        self._missing_rate_ = {}
        self._all_nan_cols_ = []

        for col in self._feature_cols:
            s = X[col]
            self._missing_rate_[col] = float(s.isna().mean())
            fill_value, is_all_nan = self._compute_fill_value(s)
            self._fill_values[col] = fill_value
            if is_all_nan:
                self._all_nan_cols_.append(col)

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
            raise RuntimeError("ChargeFeaturePreprocessor must be fitted before transform.")

        X = self._check_input(X).copy()

        for col in self._feature_cols:
            if col not in X.columns:
                X[col] = np.nan

        X = X[self._feature_cols].copy()
        X = self._prepare_numeric_frame(X)

        missing_indicators = None
        if self.add_missing_indicator:
            missing_indicators = pd.DataFrame(index=X.index)
            for col in self._feature_cols:
                missing_indicators[f"{col}__missing"] = X[col].isna().astype(float)

        X_out = self._apply_imputation(X)

        if self._scaler is not None:
            X_out = pd.DataFrame(
                self._scaler.transform(X_out[self._feature_cols]),
                columns=self._feature_cols,
                index=X.index,
            )

        if missing_indicators is not None:
            X_out = pd.concat([X_out, missing_indicators], axis=1)

        X_out = X_out.astype(float)

        if self.return_dataframe:
            return X_out

        return X_out.to_numpy(dtype=float)

    def get_feature_names_out(self):
        if self.feature_names_ is None:
            return None
        return list(self.feature_names_)

    def get_impute_summary(self) -> pd.DataFrame:
        if self._feature_cols is None or self._fill_values is None or self._missing_rate_ is None:
            raise RuntimeError("ChargeFeaturePreprocessor must be fitted before calling get_impute_summary().")

        all_nan = set(self._all_nan_cols_ or [])
        rows = []
        for col in self._feature_cols:
            rows.append(
                {
                    "feature_name": col,
                    "impute_strategy": self.impute_strategy,
                    "fill_value": self._fill_values[col],
                    "missing_rate": self._missing_rate_[col],
                    "all_nan_in_fit": col in all_nan,
                    "clip_min": self._clip_bounds(col)[0],
                    "clip_max": self._clip_bounds(col)[1],
                    "standardized": bool(self.standardize),
                }
            )
        return pd.DataFrame(rows)

    def _select_feature_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.strict:
            for col in CHARGE_COLUMNS:
                if col not in X.columns:
                    X[col] = np.nan

            ordered_cols = list(CHARGE_COLUMNS)
            if self.keep_extra_charge_cols:
                extra_cols = sorted(
                    c for c in X.columns
                    if c.startswith(CHARGE_PREFIX) and c not in CHARGE_COLUMNS
                )
                ordered_cols.extend(extra_cols)
            return X[ordered_cols].copy()

        cols = [c for c in X.columns if c.startswith(CHARGE_PREFIX)]
        return X[cols].copy()

    def _prepare_numeric_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)

        if self.clip_physical_ranges:
            for col in X.columns:
                lower, upper = self._clip_bounds(col)
                if lower is not None or upper is not None:
                    X[col] = X[col].clip(lower=lower, upper=upper)

        return X

    def _apply_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self._feature_cols or []:
            X_out[col] = X_out[col].fillna(self._fill_values[col])
        return X_out

    def _compute_fill_value(self, s: pd.Series) -> tuple[float, bool]:
        valid = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

        if len(valid) == 0:
            return float(self.all_nan_fill_value), True

        if self.impute_strategy == "zero":
            return 0.0, False

        if self.impute_strategy == "mean":
            return float(valid.mean()), False

        if self.impute_strategy == "median":
            return float(valid.median()), False

        raise ValueError(f"Unsupported impute_strategy: {self.impute_strategy}")

    def _clip_bounds(self, col: str) -> tuple[Optional[float], Optional[float]]:
        if col in CHARGE_FRACTION_COLS or "fraction" in col.lower() or "density" in col.lower():
            return 0.0, 1.0

        if col in CHARGE_NONNEGATIVE_COLS or col.lower().endswith("count"):
            return 0.0, None

        return None, None

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("ChargeFeaturePreprocessor expects a pandas DataFrame.")

        X = X.copy()
        bad = [c for c in X.columns if c != "INDEX" and not c.startswith(CHARGE_PREFIX)]
        if bad:
            raise ValueError(f"Non-{CHARGE_PREFIX} columns found: {bad}")

        if "INDEX" in X.columns:
            X = X.drop(columns=["INDEX"])

        return X
