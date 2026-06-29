from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


STRUCT_PREFIX = "STRUCT_"

STRICT_STRUCTURE_FEATURES = [
    "STRUCT_Window_ContactCount_Mean",
    "STRUCT_Site_ContactCount",
    "STRUCT_Window_pLDDT_Mean",
    "STRUCT_Site_pLDDT",
    "STRUCT_Site_ASA",
    "STRUCT_Site_RSA",
    "STRUCT_Site_HSEUp",
    "STRUCT_Site_HSEDown",
    "STRUCT_Site_PhiSin",
    "STRUCT_Site_PhiCos",
    "STRUCT_Site_PsiSin",
    "STRUCT_Site_PsiCos",
    "STRUCT_Window_RSA_Mean",
    "STRUCT_Window_SS3_HelixRatio",
    "STRUCT_Window_SS3_SheetRatio",
    "STRUCT_Window_SS3_CoilRatio",
    "STRUCT_Site_SS3_Helix_Flag",
    "STRUCT_Site_SS3_Sheet_Flag",
    "STRUCT_Site_SS3_Coil_Flag",
]

STRUCT_SS3_RATIO_COLS = [
    "STRUCT_Window_SS3_HelixRatio",
    "STRUCT_Window_SS3_SheetRatio",
    "STRUCT_Window_SS3_CoilRatio",
]

STRUCT_SS3_FLAG_COLS = [
    "STRUCT_Site_SS3_Helix_Flag",
    "STRUCT_Site_SS3_Sheet_Flag",
    "STRUCT_Site_SS3_Coil_Flag",
]

IMPUTE_STRATEGY_MAP = {
    "STRUCT_Window_ContactCount_Mean": "median",
    "STRUCT_Site_ContactCount": "median",
    "STRUCT_Window_pLDDT_Mean": "mean",
    "STRUCT_Site_pLDDT": "mean",
    "STRUCT_Site_ASA": "median",
    "STRUCT_Site_RSA": "median",
    "STRUCT_Site_HSEUp": "median",
    "STRUCT_Site_HSEDown": "median",
    "STRUCT_Site_PhiSin": "mean",
    "STRUCT_Site_PhiCos": "mean",
    "STRUCT_Site_PsiSin": "mean",
    "STRUCT_Site_PsiCos": "mean",
    "STRUCT_Window_RSA_Mean": "median",
    "STRUCT_Window_SS3_HelixRatio": "median",
    "STRUCT_Window_SS3_SheetRatio": "median",
    "STRUCT_Window_SS3_CoilRatio": "median",
    "STRUCT_Site_SS3_Helix_Flag": "zero",
    "STRUCT_Site_SS3_Sheet_Flag": "zero",
    "STRUCT_Site_SS3_Coil_Flag": "zero",
}

DEFAULT_FILL_VALUE_MAP = {
    "STRUCT_Window_ContactCount_Mean": 0.0,
    "STRUCT_Site_ContactCount": 0.0,
    "STRUCT_Window_pLDDT_Mean": 50.0,
    "STRUCT_Site_pLDDT": 50.0,
    "STRUCT_Site_ASA": 0.0,
    "STRUCT_Site_RSA": 0.5,
    "STRUCT_Site_HSEUp": 0.0,
    "STRUCT_Site_HSEDown": 0.0,
    "STRUCT_Site_PhiSin": 0.0,
    "STRUCT_Site_PhiCos": 0.0,
    "STRUCT_Site_PsiSin": 0.0,
    "STRUCT_Site_PsiCos": 0.0,
    "STRUCT_Window_RSA_Mean": 0.5,
    "STRUCT_Window_SS3_HelixRatio": 1.0 / 3.0,
    "STRUCT_Window_SS3_SheetRatio": 1.0 / 3.0,
    "STRUCT_Window_SS3_CoilRatio": 1.0 / 3.0,
    "STRUCT_Site_SS3_Helix_Flag": 0.0,
    "STRUCT_Site_SS3_Sheet_Flag": 0.0,
    "STRUCT_Site_SS3_Coil_Flag": 0.0,
}

CLIP_RANGE_MAP = {
    "STRUCT_Window_ContactCount_Mean": (0.0, None),
    "STRUCT_Site_ContactCount": (0.0, None),
    "STRUCT_Window_pLDDT_Mean": (0.0, 100.0),
    "STRUCT_Site_pLDDT": (0.0, 100.0),
    "STRUCT_Site_ASA": (0.0, None),
    "STRUCT_Site_RSA": (0.0, 1.0),
    "STRUCT_Site_HSEUp": (0.0, None),
    "STRUCT_Site_HSEDown": (0.0, None),
    "STRUCT_Site_PhiSin": (-1.0, 1.0),
    "STRUCT_Site_PhiCos": (-1.0, 1.0),
    "STRUCT_Site_PsiSin": (-1.0, 1.0),
    "STRUCT_Site_PsiCos": (-1.0, 1.0),
    "STRUCT_Window_RSA_Mean": (0.0, 1.0),
    "STRUCT_Window_SS3_HelixRatio": (0.0, 1.0),
    "STRUCT_Window_SS3_SheetRatio": (0.0, 1.0),
    "STRUCT_Window_SS3_CoilRatio": (0.0, 1.0),
    "STRUCT_Site_SS3_Helix_Flag": (0.0, 1.0),
    "STRUCT_Site_SS3_Sheet_Flag": (0.0, 1.0),
    "STRUCT_Site_SS3_Coil_Flag": (0.0, 1.0),
}

GLOBAL_MISSING_FEATURES = [
    "STRUCT_HasAnyObserved",
    "STRUCT_MissingFraction",
]


class StructurePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe: bool = False,
        add_missing_indicator: bool = True,
        add_global_missing_features: bool = True,
        strict: bool = True,
        keep_extra_struct_cols: bool = False,
        clip_physical_ranges: bool = True,
        normalize_ss3_ratios: bool = True,
        enforce_ss3_site_onehot: bool = True,
        all_nan_fill_value: float = 0.0,
    ):
        self.return_dataframe = return_dataframe
        self.add_missing_indicator = add_missing_indicator
        self.add_global_missing_features = add_global_missing_features
        self.strict = strict
        self.keep_extra_struct_cols = keep_extra_struct_cols
        self.clip_physical_ranges = clip_physical_ranges
        self.normalize_ss3_ratios = normalize_ss3_ratios
        self.enforce_ss3_site_onehot = enforce_ss3_site_onehot
        self.all_nan_fill_value = all_nan_fill_value

        self._feature_cols: Optional[List[str]] = None
        self._fill_values: Optional[Dict[str, float]] = None
        self._output_feature_names: Optional[List[str]] = None
        self._missing_rate_: Optional[Dict[str, float]] = None
        self._all_nan_cols_: Optional[Dict[str, bool]] = None
        self._impute_strategy_: Optional[Dict[str, str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X).copy()
        X = self._select_feature_columns(X)
        X = self._prepare_numeric_frame(X)

        self._feature_cols = list(X.columns)
        self._fill_values = {}
        self._missing_rate_ = {}
        self._all_nan_cols_ = {}
        self._impute_strategy_ = {}

        for col in self._feature_cols:
            strategy = self._get_strategy(col)
            self._impute_strategy_[col] = strategy
            self._fill_values[col] = self._compute_fill_value(X[col], col, strategy)
            self._missing_rate_[col] = float(X[col].isna().mean())
            self._all_nan_cols_[col] = bool(X[col].isna().all())

        output_names = list(self._feature_cols)
        if self.add_missing_indicator:
            output_names.extend([f"{c}__missing" for c in self._feature_cols])
        if self.add_global_missing_features and len(self._feature_cols) > 0:
            output_names.extend(GLOBAL_MISSING_FEATURES)
        self._output_feature_names = output_names

        return self

    def transform(self, X: pd.DataFrame):
        if self._feature_cols is None or self._fill_values is None:
            raise RuntimeError("StructurePreprocessor must be fitted before transform.")

        X = self._check_input(X).copy()

        for col in self._feature_cols:
            if col not in X.columns:
                X[col] = np.nan

        X = X[self._feature_cols].copy()
        X = self._prepare_numeric_frame(X)
        missing_mask = X.isna()

        X_out = X.copy()
        for col in self._feature_cols:
            X_out[col] = X_out[col].fillna(self._fill_values[col])

        X_out = self._postprocess_after_imputation(X_out)

        if self.add_missing_indicator:
            for col in self._feature_cols:
                X_out[f"{col}__missing"] = missing_mask[col].astype(float)

        if self.add_global_missing_features and len(self._feature_cols) > 0:
            observed_count = (~missing_mask).sum(axis=1).astype(float)
            X_out["STRUCT_HasAnyObserved"] = (observed_count > 0).astype(float)
            X_out["STRUCT_MissingFraction"] = missing_mask.mean(axis=1).astype(float)

        if self._output_feature_names is not None:
            for col in self._output_feature_names:
                if col not in X_out.columns:
                    X_out[col] = 0.0
            X_out = X_out[self._output_feature_names]

        non_numeric = X_out.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise ValueError(f"Non-numeric columns remain after StructurePreprocessor: {non_numeric}")

        X_out = X_out.astype(float)

        if self.return_dataframe:
            return X_out

        return X_out.to_numpy(dtype=float)

    def get_feature_names_out(self):
        if self._output_feature_names is None:
            return None
        return list(self._output_feature_names)

    def get_impute_summary(self) -> pd.DataFrame:
        if (
            self._feature_cols is None
            or self._fill_values is None
            or self._missing_rate_ is None
            or self._all_nan_cols_ is None
            or self._impute_strategy_ is None
        ):
            raise RuntimeError("StructurePreprocessor must be fitted before calling get_impute_summary().")

        rows = []
        for col in self._feature_cols:
            lower, upper = self._get_clip_range(col)
            rows.append(
                {
                    "feature_name": col,
                    "impute_strategy": self._impute_strategy_[col],
                    "fill_value": self._fill_values[col],
                    "missing_rate": self._missing_rate_[col],
                    "all_nan_in_fit": self._all_nan_cols_[col],
                    "clip_min": lower,
                    "clip_max": upper,
                }
            )
        return pd.DataFrame(rows)

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("StructurePreprocessor expects a pandas DataFrame.")

        X = X.copy()
        bad_cols = [c for c in X.columns if c != "INDEX" and not c.startswith(STRUCT_PREFIX)]
        if bad_cols:
            raise ValueError(f"Non-STRUCT columns detected: {bad_cols}")

        if "INDEX" in X.columns:
            X = X.drop(columns=["INDEX"])

        return X

    def _select_feature_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        struct_cols = [c for c in X.columns if c.startswith(STRUCT_PREFIX)]

        if self.strict:
            for col in STRICT_STRUCTURE_FEATURES:
                if col not in X.columns:
                    X[col] = np.nan

            ordered_cols = list(STRICT_STRUCTURE_FEATURES)
            if self.keep_extra_struct_cols:
                extra_cols = [c for c in struct_cols if c not in STRICT_STRUCTURE_FEATURES]
                ordered_cols.extend(sorted(extra_cols))
            return X[ordered_cols].copy()

        return X[sorted(struct_cols)].copy()

    def _prepare_numeric_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)

        if self.clip_physical_ranges:
            X = self._clip_columns(X)

        return X

    def _postprocess_after_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if self.clip_physical_ranges:
            X = self._clip_columns(X)

        if self.normalize_ss3_ratios:
            X = self._normalize_ratio_group(X, STRUCT_SS3_RATIO_COLS)

        if self.enforce_ss3_site_onehot:
            X = self._enforce_onehot_group(X, STRUCT_SS3_FLAG_COLS)

        if self.clip_physical_ranges:
            X = self._clip_columns(X)

        return X

    def _clip_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            lower, upper = self._get_clip_range(col)
            if lower is not None or upper is not None:
                X[col] = X[col].clip(lower=lower, upper=upper)
        return X

    def _get_clip_range(self, col: str) -> Tuple[Optional[float], Optional[float]]:
        if col in CLIP_RANGE_MAP:
            return CLIP_RANGE_MAP[col]

        name = col.lower()
        if "plddt" in name:
            return 0.0, 100.0
        if "rsa" in name or "ratio" in name or "fraction" in name or name.endswith("_flag"):
            return 0.0, 1.0
        if "sin" in name or "cos" in name:
            return -1.0, 1.0
        if "count" in name or "asa" in name or "hse" in name or "length" in name:
            return 0.0, None
        return None, None

    def _get_strategy(self, col: str) -> str:
        if col in IMPUTE_STRATEGY_MAP:
            return IMPUTE_STRATEGY_MAP[col]

        name = col.lower()
        if name.endswith("_flag") or "flag" in name:
            return "zero"
        if "plddt" in name or "sin" in name or "cos" in name:
            return "mean"
        return "median"

    def _compute_fill_value(self, s: pd.Series, col: str, strategy: str) -> float:
        valid = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

        if valid.empty:
            return self._default_fill(col, strategy)

        if strategy == "mean":
            value = float(valid.mean())
        elif strategy == "median":
            value = float(valid.median())
        elif strategy == "zero":
            value = 0.0
        else:
            raise ValueError(f"Unsupported imputation strategy: {strategy}")

        lower, upper = self._get_clip_range(col)
        if lower is not None:
            value = max(value, lower)
        if upper is not None:
            value = min(value, upper)
        return float(value)

    def _default_fill(self, col: str, strategy: str) -> float:
        if col in DEFAULT_FILL_VALUE_MAP:
            return float(DEFAULT_FILL_VALUE_MAP[col])
        if strategy == "zero":
            return 0.0
        return float(self.all_nan_fill_value)

    def _normalize_ratio_group(self, X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        present = [c for c in cols if c in X.columns]
        if len(present) != len(cols):
            return X

        values = X[present].to_numpy(dtype=float)
        values = np.clip(values, 0.0, 1.0)
        row_sum = values.sum(axis=1)
        valid = row_sum > 0
        values[valid] = values[valid] / row_sum[valid, None]
        X[present] = values
        return X

    def _enforce_onehot_group(self, X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        present = [c for c in cols if c in X.columns]
        if len(present) != len(cols):
            return X

        values = X[present].to_numpy(dtype=float)
        values = np.clip(values, 0.0, 1.0)
        row_sum = values.sum(axis=1)
        conflict = row_sum > 1.0

        if np.any(conflict):
            selected = np.argmax(values[conflict], axis=1)
            new_values = np.zeros_like(values[conflict])
            new_values[np.arange(new_values.shape[0]), selected] = 1.0
            values[conflict] = new_values

        X[present] = values
        return X
