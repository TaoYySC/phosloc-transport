from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


DOMAIN_FAMILY_PREFIX = "MOTIF_Domain"

DOMAIN_PREFIXES = [
    "DBD",
    "TAD",
    "Linker",
    "Boundary",
    "FunctionalDomain",
]

DOMAIN_BOOL_SUFFIXES = [
    "HasSegment_Flag",
    "Inside_Flag",
    "Within20AA_Flag",
    "Within50AA_Flag",
]

DOMAIN_BOUNDARY_SUFFIXES = [
    "Nearest_Start",
    "Nearest_End",
]

DOMAIN_DISTANCE_SUFFIXES = [
    "Nearest_Distance",
    "Nearest_DistanceNorm",
    "Nearest_CenterSignedDistance",
]

DOMAIN_INSIDE_ONLY_SUFFIXES = [
    "Inside_RelativePosition",
    "Inside_SegmentLength",
    "Inside_SegmentLengthNorm",
]

DOMAIN_OTHER_FLOAT_SUFFIXES = [
    "Protein_SegmentCount",
    "Protein_CoverageFraction",
]

SIGNED_DISTANCE_SUFFIXES = {
    "Nearest_CenterSignedDistance",
}

FRACTION_SUFFIXES = {
    "Protein_CoverageFraction",
    "Nearest_DistanceNorm",
    "Inside_RelativePosition",
    "Inside_SegmentLengthNorm",
}

NON_NEGATIVE_SUFFIXES = {
    "Nearest_Start",
    "Nearest_End",
    "Nearest_Distance",
    "Nearest_DistanceNorm",
    "Inside_SegmentLength",
    "Inside_SegmentLengthNorm",
    "Protein_SegmentCount",
    "Protein_CoverageFraction",
}


class DomainPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe: bool = False,
        drop_weak_cols: bool = False,
        drop_boundary_cols: bool = True,
        log_transform_dist: bool = True,
        large_value_mode: str = "p99_plus_1",
        add_missing_indicator: bool = False,
        all_nan_fill_value: float = 0.0,
    ):
        self.return_dataframe = return_dataframe
        self.drop_weak_cols = drop_weak_cols
        self.drop_boundary_cols = drop_boundary_cols
        self.log_transform_dist = log_transform_dist
        self.large_value_mode = large_value_mode
        self.add_missing_indicator = add_missing_indicator
        self.all_nan_fill_value = all_nan_fill_value

        self._all_cols: Optional[List[str]] = None
        self._keep_cols: Optional[List[str]] = None
        self._output_feature_names: Optional[List[str]] = None
        self._dist_fill_values: Optional[Dict[str, float]] = None
        self._signed_fill_values: Optional[Dict[str, float]] = None
        self._boundary_fill_values: Optional[Dict[str, float]] = None
        self._missing_rate: Optional[Dict[str, float]] = None
        self._all_nan_cols: Optional[Dict[str, bool]] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X).copy()
        X = self._coerce_and_normalize_flags(X)

        self._dist_fill_values = {}
        self._signed_fill_values = {}
        self._boundary_fill_values = {}
        self._missing_rate = {}
        self._all_nan_cols = {}

        for prefix in DOMAIN_PREFIXES:
            has_col = self._col(prefix, "HasSegment_Flag")
            has = X[has_col].to_numpy(dtype=int)

            for col in self._boundary_cols(prefix):
                values = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
                values_has = values[has == 1]
                values_has = values_has[np.isfinite(values_has)]
                self._boundary_fill_values[col] = self._compute_median_fill(values_has)
                self._missing_rate[col] = float(pd.isna(X[col]).mean())
                self._all_nan_cols[col] = bool(values_has.size == 0)

            for col in self._distance_cols(prefix):
                suffix = self._suffix_from_col(col, prefix)
                values = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
                values_has = values[has == 1]
                values_has = values_has[np.isfinite(values_has)]

                if suffix in SIGNED_DISTANCE_SUFFIXES:
                    self._signed_fill_values[col] = self._compute_median_fill(values_has)
                else:
                    self._dist_fill_values[col] = self._compute_large_fill(values_has)

                self._missing_rate[col] = float(pd.isna(X[col]).mean())
                self._all_nan_cols[col] = bool(values_has.size == 0)

            for col in self._inside_only_cols(prefix) + self._other_float_cols(prefix) + self._bool_cols(prefix):
                if col in X.columns:
                    self._missing_rate[col] = float(pd.isna(X[col]).mean())
                    numeric = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
                    self._all_nan_cols[col] = bool(np.isfinite(numeric).sum() == 0)

        self._all_cols = self._canonical_cols()
        self._keep_cols = self._build_keep_cols(self._all_cols)
        self._output_feature_names = list(self._keep_cols)

        if self.add_missing_indicator:
            self._output_feature_names.extend([f"{c}__missing" for c in self._keep_cols])

        self.feature_names_ = list(self._output_feature_names)
        return self

    def transform(self, X: pd.DataFrame):
        if self._keep_cols is None:
            raise RuntimeError("DomainPreprocessor must be fitted before transform.")

        X = self._check_input(X).copy()
        X = self._coerce_and_normalize_flags(X)
        missing_indicators = pd.DataFrame(index=X.index)

        for prefix in DOMAIN_PREFIXES:
            has_col = self._col(prefix, "HasSegment_Flag")
            inside_col = self._col(prefix, "Inside_Flag")

            has = X[has_col].to_numpy(dtype=int)
            inside = X[inside_col].to_numpy(dtype=int)

            for col in self._boundary_cols(prefix):
                X[col] = pd.to_numeric(X[col], errors="coerce")
                if self.add_missing_indicator and col in self._keep_cols:
                    missing_indicators[f"{col}__missing"] = X[col].isna().astype(float)
                fill = self._boundary_fill_values.get(col, self.all_nan_fill_value)
                X.loc[has == 0, col] = 0.0
                X[col] = X[col].fillna(fill)
                X[col] = np.clip(X[col].to_numpy(dtype=float), 0.0, None)

            for col in self._distance_cols(prefix):
                suffix = self._suffix_from_col(col, prefix)
                X[col] = pd.to_numeric(X[col], errors="coerce")
                if self.add_missing_indicator and col in self._keep_cols:
                    missing_indicators[f"{col}__missing"] = X[col].isna().astype(float)

                if suffix in SIGNED_DISTANCE_SUFFIXES:
                    fill = self._signed_fill_values.get(col, self.all_nan_fill_value)
                    X.loc[has == 0, col] = 0.0
                    X[col] = X[col].fillna(fill)
                    values = X[col].to_numpy(dtype=float)
                    if self.log_transform_dist:
                        values = np.sign(values) * np.log1p(np.abs(values))
                    X[col] = values
                else:
                    fill = self._dist_fill_values.get(col, 1.0)
                    X.loc[has == 0, col] = fill
                    X[col] = X[col].fillna(fill)
                    values = np.clip(X[col].to_numpy(dtype=float), 0.0, None)
                    if self.log_transform_dist:
                        values = np.log1p(values)
                    X[col] = values

            for col in self._other_float_cols(prefix):
                suffix = self._suffix_from_col(col, prefix)
                X[col] = pd.to_numeric(X[col], errors="coerce")
                if self.add_missing_indicator and col in self._keep_cols:
                    missing_indicators[f"{col}__missing"] = X[col].isna().astype(float)
                X.loc[has == 0, col] = 0.0
                X[col] = X[col].fillna(0.0)
                X[col] = self._clip_by_suffix(X[col].to_numpy(dtype=float), suffix)

            for col in self._inside_only_cols(prefix):
                suffix = self._suffix_from_col(col, prefix)
                X[col] = pd.to_numeric(X[col], errors="coerce")
                if self.add_missing_indicator and col in self._keep_cols:
                    missing_indicators[f"{col}__missing"] = X[col].isna().astype(float)

                if suffix == "Inside_RelativePosition":
                    X.loc[inside == 0, col] = -1.0
                    X[col] = X[col].fillna(-1.0)
                    values = X[col].to_numpy(dtype=float)
                    values = np.where(values < 0.0, -1.0, np.clip(values, 0.0, 1.0))
                    X[col] = values
                else:
                    X.loc[inside == 0, col] = 0.0
                    X[col] = X[col].fillna(0.0)
                    X[col] = self._clip_by_suffix(X[col].to_numpy(dtype=float), suffix)

        missing_cols = [c for c in self._keep_cols if c not in X.columns]
        for col in missing_cols:
            X[col] = 0.0

        X_out = X[self._keep_cols].copy()

        for col in X_out.columns:
            if X_out[col].dtype == bool:
                X_out[col] = X_out[col].astype(float)
            else:
                X_out[col] = pd.to_numeric(X_out[col], errors="coerce").fillna(0.0).astype(float)

        if self.add_missing_indicator:
            for col in self._keep_cols:
                ind_col = f"{col}__missing"
                if ind_col not in missing_indicators.columns:
                    missing_indicators[ind_col] = 0.0
            X_out = pd.concat([X_out, missing_indicators[[f"{c}__missing" for c in self._keep_cols]]], axis=1)

        if self.return_dataframe:
            return X_out

        return X_out.to_numpy(dtype=float)

    def get_feature_names_out(self):
        if self._output_feature_names is None:
            return None
        return list(self._output_feature_names)

    def get_impute_summary(self) -> pd.DataFrame:
        if self._keep_cols is None:
            raise RuntimeError("DomainPreprocessor must be fitted before calling get_impute_summary().")

        rows = []
        dist_fill = self._dist_fill_values or {}
        signed_fill = self._signed_fill_values or {}
        boundary_fill = self._boundary_fill_values or {}
        missing_rate = self._missing_rate or {}
        all_nan_cols = self._all_nan_cols or {}

        for col in self._keep_cols:
            if col in dist_fill:
                strategy = self.large_value_mode
                fill = dist_fill[col]
            elif col in signed_fill:
                strategy = "median_signed"
                fill = signed_fill[col]
            elif col in boundary_fill:
                strategy = "median_boundary"
                fill = boundary_fill[col]
            elif col.endswith("Inside_RelativePosition"):
                strategy = "outside_minus_one"
                fill = -1.0
            else:
                strategy = "zero"
                fill = 0.0

            rows.append(
                {
                    "feature_name": col,
                    "impute_strategy": strategy,
                    "fill_value": float(fill),
                    "missing_rate": float(missing_rate.get(col, 0.0)),
                    "all_nan_in_fit": bool(all_nan_cols.get(col, False)),
                }
            )

        return pd.DataFrame(rows)

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("DomainPreprocessor expects a DataFrame.")

        X = X.copy()
        if "INDEX" in X.columns:
            X = X.drop(columns=["INDEX"])

        expected_cols = set(self._canonical_cols())

        for col in expected_cols:
            if col not in X.columns:
                X[col] = np.nan

        bad = [c for c in X.columns if c not in expected_cols]
        if bad:
            raise ValueError(f"Non-domain feature columns detected: {bad}")

        return X[self._canonical_cols()].copy()

    def _coerce_and_normalize_flags(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for prefix in DOMAIN_PREFIXES:
            for col in self._bool_cols(prefix):
                X[col] = self._to_bool01(X[col].values).astype(float)
        return X

    def _canonical_cols(self) -> List[str]:
        cols = []
        for prefix in DOMAIN_PREFIXES:
            cols.extend(self._bool_cols(prefix))
            cols.extend(self._boundary_cols(prefix))
            cols.extend(self._distance_cols(prefix))
            cols.extend(self._inside_only_cols(prefix))
            cols.extend(self._other_float_cols(prefix))
        return cols

    def _build_keep_cols(self, cols: List[str]) -> List[str]:
        keep = list(cols)

        if self.drop_boundary_cols:
            boundary_cols = set()
            for prefix in DOMAIN_PREFIXES:
                boundary_cols.update(self._boundary_cols(prefix))
            keep = [c for c in keep if c not in boundary_cols]

        if self.drop_weak_cols:
            weak = set()
            for prefix in DOMAIN_PREFIXES:
                weak.add(self._col(prefix, "Protein_CoverageFraction"))
            keep = [c for c in keep if c not in weak]

        return keep

    def _col(self, prefix: str, suffix: str) -> str:
        return f"{DOMAIN_FAMILY_PREFIX}_{prefix}_{suffix}"

    def _suffix_from_col(self, col: str, prefix: str) -> str:
        prefix_text = f"{DOMAIN_FAMILY_PREFIX}_{prefix}_"
        return col.replace(prefix_text, "", 1)

    def _bool_cols(self, prefix: str) -> List[str]:
        return [self._col(prefix, suffix) for suffix in DOMAIN_BOOL_SUFFIXES]

    def _boundary_cols(self, prefix: str) -> List[str]:
        return [self._col(prefix, suffix) for suffix in DOMAIN_BOUNDARY_SUFFIXES]

    def _distance_cols(self, prefix: str) -> List[str]:
        return [self._col(prefix, suffix) for suffix in DOMAIN_DISTANCE_SUFFIXES]

    def _dist_cols(self, prefix: str) -> List[str]:
        return self._boundary_cols(prefix) + self._distance_cols(prefix)

    def _inside_only_cols(self, prefix: str) -> List[str]:
        return [self._col(prefix, suffix) for suffix in DOMAIN_INSIDE_ONLY_SUFFIXES]

    def _other_float_cols(self, prefix: str) -> List[str]:
        return [self._col(prefix, suffix) for suffix in DOMAIN_OTHER_FLOAT_SUFFIXES]

    def _to_bool01(self, values):
        s = pd.Series(values)

        if s.dtype == bool:
            return s.astype(int).to_numpy()

        if s.dtype == object:
            s_clean = s.astype(str).str.strip().str.lower()
            mapped = s_clean.map(
                {
                    "true": 1,
                    "false": 0,
                    "yes": 1,
                    "no": 0,
                    "y": 1,
                    "n": 0,
                    "1": 1,
                    "0": 0,
                    "nan": 0,
                    "none": 0,
                    "": 0,
                }
            )
            if mapped.notna().any():
                return mapped.fillna(0).astype(int).to_numpy()

        numeric = pd.to_numeric(s, errors="coerce").fillna(0.0)
        return (numeric > 0).astype(int).to_numpy()

    def _compute_large_fill(self, values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        values = values[values >= 0.0]

        if values.size == 0:
            return float(self.all_nan_fill_value if self.all_nan_fill_value > 0 else 1.0)

        if self.large_value_mode == "max_plus_1":
            return float(np.max(values) + 1.0)
        if self.large_value_mode == "p99_plus_1":
            return float(np.quantile(values, 0.99) + 1.0)
        if self.large_value_mode == "p95_times_2":
            return float(np.quantile(values, 0.95) * 2.0)

        raise ValueError(f"Unknown large_value_mode: {self.large_value_mode}")

    def _compute_median_fill(self, values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float(self.all_nan_fill_value)
        return float(np.median(values))

    def _clip_by_suffix(self, values: np.ndarray, suffix: str) -> np.ndarray:
        values = np.asarray(values, dtype=float)

        if suffix in FRACTION_SUFFIXES:
            return np.clip(values, 0.0, 1.0)

        if suffix in NON_NEGATIVE_SUFFIXES:
            return np.clip(values, 0.0, None)

        return values
