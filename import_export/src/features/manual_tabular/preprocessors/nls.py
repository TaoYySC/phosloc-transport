"""
File location:
src/preprocess/nls.py
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


NLS_PREFIX = "MOTIF_NLS_"

NLS_BOOL_COLS = [
    "MOTIF_NLS_HasSegment_Flag",
    "MOTIF_NLS_Inside_Flag",
    "MOTIF_NLS_Within20AA_Flag",
    "MOTIF_NLS_Within50AA_Flag",
]

NLS_BOUNDARY_COLS = [
    "MOTIF_NLS_Nearest_Start",
    "MOTIF_NLS_Nearest_End",
]

NLS_DISTANCE_COLS = [
    "MOTIF_NLS_Nearest_Distance",
    "MOTIF_NLS_Nearest_DistanceNorm",
]

NLS_SIGNED_DISTANCE_COLS = [
    "MOTIF_NLS_Nearest_CenterSignedDistance",
]

NLS_DIST_COLS = NLS_BOUNDARY_COLS + NLS_DISTANCE_COLS + NLS_SIGNED_DISTANCE_COLS

NLS_INSIDE_ONLY_COLS = [
    "MOTIF_NLS_Inside_RelativePosition",
    "MOTIF_NLS_Inside_SegmentLength",
    "MOTIF_NLS_Inside_SegmentLengthNorm",
]

NLS_PROTEIN_LEVEL_COLS = [
    "MOTIF_NLS_Protein_SegmentCount",
    "MOTIF_NLS_Protein_CoverageFraction",
]

NLS_SCORE_COLS = [
    "MOTIF_NLS_Nearest_Score",
]

NLS_REQUIRED_COLS = (
    NLS_BOOL_COLS
    + NLS_DIST_COLS
    + NLS_INSIDE_ONLY_COLS
    + NLS_PROTEIN_LEVEL_COLS
    + NLS_SCORE_COLS
)


class NLSFeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe: bool = False,
        drop_weak_cols: bool = True,
        drop_boundary_cols: bool = True,
        log_transform_dist: bool = True,
        large_value_mode: str = "p99_plus_1",
        add_missing_indicator: bool = False,
    ):
        self.return_dataframe = return_dataframe
        self.drop_weak_cols = drop_weak_cols
        self.drop_boundary_cols = drop_boundary_cols
        self.log_transform_dist = log_transform_dist
        self.large_value_mode = large_value_mode
        self.add_missing_indicator = add_missing_indicator

        self._all_cols = None
        self._keep_cols = None
        self._dist_fill_values = None
        self._score_fill_values = None
        self._missing_indicator_cols = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X).copy()
        X = self._drop_index_column(X)
        X = self._ensure_required_columns(X)
        X = self._coerce_core_columns(X)

        has = self._get_has_segment(X)

        self._dist_fill_values = {}
        for col in NLS_DIST_COLS:
            if col not in X.columns:
                continue
            values = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
            values_has = values[has == 1]
            values_has = values_has[np.isfinite(values_has)]
            self._dist_fill_values[col] = self._compute_large_fill(values_has)

        self._score_fill_values = {}
        for col in NLS_SCORE_COLS:
            if col not in X.columns:
                continue
            values = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
            values_has = values[has == 1]
            values_has = values_has[np.isfinite(values_has)]
            self._score_fill_values[col] = float(np.median(values_has)) if values_has.size else 0.0

        self._all_cols = list(X.columns)
        self._keep_cols = self._build_keep_cols(self._all_cols)

        self._missing_indicator_cols = []
        if self.add_missing_indicator:
            self._missing_indicator_cols = [
                f"{col}_missing" for col in self._keep_cols if col not in NLS_BOOL_COLS
            ]

        self.feature_names_ = list(self._keep_cols) + list(self._missing_indicator_cols)
        return self

    def transform(self, X: pd.DataFrame):
        check_is_fitted(self, ["_keep_cols", "_dist_fill_values", "_score_fill_values"])

        X = self._check_input(X).copy()
        X = self._drop_index_column(X)
        X = self._ensure_required_columns(X)
        X = self._coerce_core_columns(X)

        missing_mask = X.isna().copy()

        has = self._get_has_segment(X)
        inside = self._get_inside_segment(X)

        for col in NLS_BOOL_COLS:
            if col in X.columns:
                X[col] = self._to_bool01(X[col].values).astype(float)

        for col in NLS_BOUNDARY_COLS:
            if col in X.columns:
                fill = self._dist_fill_values.get(col, 1.0)
                X.loc[has == 0, col] = fill
                X[col] = X[col].fillna(fill)
                if self.log_transform_dist:
                    X[col] = self._log1p_nonnegative(X[col])

        for col in NLS_DISTANCE_COLS:
            if col in X.columns:
                fill = self._dist_fill_values.get(col, 1.0)
                X.loc[has == 0, col] = fill
                X[col] = X[col].fillna(fill)
                X[col] = X[col].clip(lower=0.0)
                if self.log_transform_dist:
                    X[col] = self._log1p_nonnegative(X[col])

        for col in NLS_SIGNED_DISTANCE_COLS:
            if col in X.columns:
                fill = self._dist_fill_values.get(col, 0.0)
                X.loc[has == 0, col] = fill
                X[col] = X[col].fillna(fill)
                if self.log_transform_dist:
                    X[col] = self._signed_log1p(X[col])

        if "MOTIF_NLS_Inside_RelativePosition" in X.columns:
            col = "MOTIF_NLS_Inside_RelativePosition"
            X.loc[inside == 0, col] = -1.0
            X[col] = X[col].fillna(-1.0)
            X[col] = X[col].clip(lower=-1.0, upper=1.0)

        for col in ["MOTIF_NLS_Inside_SegmentLength", "MOTIF_NLS_Inside_SegmentLengthNorm"]:
            if col in X.columns:
                X.loc[inside == 0, col] = 0.0
                X[col] = X[col].fillna(0.0)
                X[col] = X[col].clip(lower=0.0)

        for col in NLS_PROTEIN_LEVEL_COLS:
            if col in X.columns:
                X.loc[has == 0, col] = 0.0
                X[col] = X[col].fillna(0.0)
                X[col] = X[col].clip(lower=0.0)

        if "MOTIF_NLS_Protein_CoverageFraction" in X.columns:
            X["MOTIF_NLS_Protein_CoverageFraction"] = X[
                "MOTIF_NLS_Protein_CoverageFraction"
            ].clip(lower=0.0, upper=1.0)

        for col in NLS_SCORE_COLS:
            if col in X.columns:
                fill = self._score_fill_values.get(col, 0.0)
                X.loc[has == 0, col] = 0.0
                X[col] = X[col].fillna(fill)
                X[col] = X[col].clip(lower=0.0)

        for col in X.columns:
            if col.startswith(NLS_PREFIX):
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0).astype(float)

        if self._keep_cols is not None:
            for col in self._keep_cols:
                if col not in X.columns:
                    X[col] = 0.0
            X_out = X[self._keep_cols].copy()
        else:
            X_out = X.copy()

        if self.add_missing_indicator:
            for col in self._keep_cols:
                if col in NLS_BOOL_COLS:
                    continue
                X_out[f"{col}_missing"] = missing_mask[col].astype(float) if col in missing_mask else 1.0

        if self.return_dataframe:
            return X_out

        return X_out.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_ is None:
            return None
        return np.asarray(self.feature_names_, dtype=object)

    def _build_keep_cols(self, cols):
        cols = [col for col in cols if col.startswith(NLS_PREFIX)]

        if self.drop_boundary_cols:
            cols = [col for col in cols if col not in NLS_BOUNDARY_COLS]

        if self.drop_weak_cols:
            weak_cols = {
                "MOTIF_NLS_Protein_CoverageFraction",
            }
            cols = [col for col in cols if col not in weak_cols]

        ordered = [col for col in NLS_REQUIRED_COLS if col in cols]
        extra = sorted([col for col in cols if col not in ordered])
        return ordered + extra

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("NLSFeaturePreprocessor expects a pandas DataFrame.")

        X = X.copy()
        bad = [col for col in X.columns if not (col.startswith(NLS_PREFIX) or col == "INDEX")]
        if bad:
            raise ValueError(f"Non-NLS feature columns detected: {bad}")
        return X

    def _drop_index_column(self, X: pd.DataFrame) -> pd.DataFrame:
        if "INDEX" in X.columns:
            return X.drop(columns=["INDEX"])
        return X

    def _ensure_required_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in NLS_BOOL_COLS:
            if col not in X.columns:
                X[col] = 0.0
        for col in NLS_DIST_COLS + NLS_INSIDE_ONLY_COLS + NLS_PROTEIN_LEVEL_COLS + NLS_SCORE_COLS:
            if col not in X.columns:
                X[col] = np.nan
        return X

    def _coerce_core_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in NLS_BOOL_COLS:
            if col in X.columns:
                X[col] = self._to_bool01(X[col].values).astype(float)
        for col in X.columns:
            if col.startswith(NLS_PREFIX) and col not in NLS_BOOL_COLS:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        return X

    def _get_has_segment(self, X: pd.DataFrame) -> np.ndarray:
        if "MOTIF_NLS_HasSegment_Flag" in X.columns:
            return self._to_bool01(X["MOTIF_NLS_HasSegment_Flag"].values).astype(int)
        return np.zeros(len(X), dtype=int)

    def _get_inside_segment(self, X: pd.DataFrame) -> np.ndarray:
        if "MOTIF_NLS_Inside_Flag" in X.columns:
            return self._to_bool01(X["MOTIF_NLS_Inside_Flag"].values).astype(int)
        return np.zeros(len(X), dtype=int)

    def _compute_large_fill(self, values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 1.0

        if self.large_value_mode == "max_plus_1":
            return float(np.max(values) + 1.0)
        if self.large_value_mode == "p99_plus_1":
            return float(np.quantile(values, 0.99) + 1.0)
        if self.large_value_mode == "p95_times_2":
            return float(np.quantile(values, 0.95) * 2.0)

        raise ValueError(f"Unknown large_value_mode: {self.large_value_mode}")

    def _to_bool01(self, values):
        series = pd.Series(values)
        if series.dtype == bool:
            return series.astype(int).to_numpy()

        if series.dtype == object:
            normalized = series.astype(str).str.strip().str.lower()
            mapped = normalized.map(
                {
                    "true": 1,
                    "false": 0,
                    "yes": 1,
                    "no": 0,
                    "y": 1,
                    "n": 0,
                    "1": 1,
                    "0": 0,
                }
            )
            if mapped.notna().any():
                return mapped.fillna(0).astype(int).to_numpy()

        numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return (numeric > 0).astype(int).to_numpy()

    def _log1p_nonnegative(self, values: pd.Series) -> np.ndarray:
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        return np.log1p(np.clip(arr, 0.0, None))

    def _signed_log1p(self, values: pd.Series) -> np.ndarray:
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        return np.sign(arr) * np.log1p(np.abs(arr))
