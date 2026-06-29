from __future__ import annotations

from typing import Dict, List, Optional

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


class StructurePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe: bool = False,
        add_missing_indicator: bool = False,
        strict: bool = True,
        keep_extra_struct_cols: bool = False,
        clip_physical_ranges: bool = True,
    ):
        self.return_dataframe = return_dataframe
        self.add_missing_indicator = add_missing_indicator
        self.strict = strict
        self.keep_extra_struct_cols = keep_extra_struct_cols
        self.clip_physical_ranges = clip_physical_ranges

        self._feature_cols: Optional[List[str]] = None
        self._fill_values: Optional[Dict[str, float]] = None
        self._output_feature_names: Optional[List[str]] = None
        self._missing_rate_: Optional[Dict[str, float]] = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X).copy()
        X = self._select_feature_columns(X)
        X = self._coerce_numeric(X)

        if self.clip_physical_ranges:
            X = self._clip_columns(X)

        self._feature_cols = list(X.columns)
        self._fill_values = {}
        self._missing_rate_ = {}

        for col in self._feature_cols:
            strategy = self._get_strategy(col)
            self._fill_values[col] = self._compute_fill_value(X[col], strategy)
            self._missing_rate_[col] = float(X[col].isna().mean())

        self._output_feature_names = list(self._feature_cols)
        if self.add_missing_indicator:
            self._output_feature_names.extend([f"{c}_missing" for c in self._feature_cols])

        return self

    def transform(self, X: pd.DataFrame):
        if self._feature_cols is None or self._fill_values is None:
            raise RuntimeError("StructurePreprocessor must be fitted before transform.")

        X = self._check_input(X).copy()

        for col in self._feature_cols:
            if col not in X.columns:
                X[col] = np.nan

        X = X[self._feature_cols].copy()
        X = self._coerce_numeric(X)

        if self.clip_physical_ranges:
            X = self._clip_columns(X)

        X_out = X.copy()

        for col in self._feature_cols:
            X_out[col] = X_out[col].fillna(self._fill_values[col])

        if self.add_missing_indicator:
            for col in self._feature_cols:
                X_out[f"{col}_missing"] = X[col].isna().astype(int)

        if self.return_dataframe:
            return X_out

        return X_out.to_numpy(dtype=float)

    def get_feature_names_out(self):
        if self._output_feature_names is None:
            return None
        return list(self._output_feature_names)

    def get_impute_summary(self) -> pd.DataFrame:
        if self._feature_cols is None or self._fill_values is None or self._missing_rate_ is None:
            raise RuntimeError("StructurePreprocessor must be fitted before calling get_impute_summary().")

        rows = []
        for col in self._feature_cols:
            rows.append(
                {
                    "feature_name": col,
                    "impute_strategy": self._get_strategy(col),
                    "fill_value": self._fill_values[col],
                    "missing_rate": self._missing_rate_[col],
                    "clip_min": CLIP_RANGE_MAP.get(col, (None, None))[0],
                    "clip_max": CLIP_RANGE_MAP.get(col, (None, None))[1],
                }
            )
        return pd.DataFrame(rows)

    def _check_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("StructurePreprocessor expects a DataFrame")

        bad_cols = [c for c in X.columns if c != "INDEX" and not c.startswith(STRUCT_PREFIX)]
        if bad_cols:
            raise ValueError(f"Non-STRUCT columns detected: {bad_cols}")

        return X.copy()

    def _select_feature_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        struct_cols = [c for c in X.columns if c.startswith(STRUCT_PREFIX)]

        if self.strict:
            missing_expected = [c for c in STRICT_STRUCTURE_FEATURES if c not in struct_cols]
            extra_found = [c for c in struct_cols if c not in STRICT_STRUCTURE_FEATURES]

            if missing_expected:
                for col in missing_expected:
                    X[col] = np.nan

            if extra_found and not self.keep_extra_struct_cols:
                struct_cols = [c for c in struct_cols if c in STRICT_STRUCTURE_FEATURES]

            ordered_cols = [c for c in STRICT_STRUCTURE_FEATURES if c in X.columns]
            if self.keep_extra_struct_cols:
                extra_cols = [c for c in struct_cols if c not in STRICT_STRUCTURE_FEATURES]
                ordered_cols.extend(sorted(extra_cols))
            return X[ordered_cols].copy()

        return X[struct_cols].copy()

    def _coerce_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        return X

    def _clip_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            lower, upper = CLIP_RANGE_MAP.get(col, (None, None))
            if lower is not None or upper is not None:
                X[col] = X[col].clip(lower=lower, upper=upper)
        return X

    def _get_strategy(self, col: str) -> str:
        if col in IMPUTE_STRATEGY_MAP:
            return IMPUTE_STRATEGY_MAP[col]

        if self.strict:
            raise ValueError(f"No imputation strategy defined for feature: {col}")

        return "median"

    def _compute_fill_value(self, s: pd.Series, strategy: str) -> float:
        valid = s.dropna()

        if len(valid) == 0:
            return self._default_fill(strategy)

        if strategy == "mean":
            return float(valid.mean())

        if strategy == "median":
            return float(valid.median())

        if strategy == "zero":
            return 0.0

        raise ValueError(f"Unsupported imputation strategy: {strategy}")

    def _default_fill(self, strategy: str) -> float:
        if strategy == "zero":
            return 0.0
        if strategy == "mean":
            return 0.0
        if strategy == "median":
            return 0.0
        raise ValueError(f"Unsupported imputation strategy: {strategy}")