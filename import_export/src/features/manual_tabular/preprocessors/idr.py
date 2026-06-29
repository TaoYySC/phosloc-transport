"""
File location:
src/preprocess/idr.py
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


IDR_PREFIX = "MOTIF_IDR_"

IDR_NUMERIC_COLS = [
    "MOTIF_IDR_Site_DisorderScore",
    "MOTIF_IDR_Window7_Disorder_Mean",
    "MOTIF_IDR_Window7_Disorder_Max",
    "MOTIF_IDR_Window7_Disorder_Min",
    "MOTIF_IDR_Window7_Disorder_Std",
    "MOTIF_IDR_Window7_Disorder_FractionGTThreshold",
    "MOTIF_IDR_Window15_Disorder_Mean",
    "MOTIF_IDR_Window15_Disorder_Max",
    "MOTIF_IDR_Window15_Disorder_Min",
    "MOTIF_IDR_Window15_Disorder_Std",
    "MOTIF_IDR_Window15_Disorder_FractionGTThreshold",
    "MOTIF_IDR_Protein_RegionCount",
    "MOTIF_IDR_Nearest_Start",
    "MOTIF_IDR_Nearest_End",
    "MOTIF_IDR_Nearest_Distance",
    "MOTIF_IDR_Nearest_DistanceNorm",
    "MOTIF_IDR_Nearest_CenterSignedDistance",
    "MOTIF_IDR_Inside_RelativePosition",
    "MOTIF_IDR_Inside_RegionLength",
    "MOTIF_IDR_Inside_RegionLengthNorm",
    "MOTIF_IDR_Protein_CoverageFraction",
    "MOTIF_IDR_NearestRegion_DisorderMean",
    "MOTIF_IDR_NearestRegion_DisorderMax",
]

IDR_BOOL_COLS = [
    "MOTIF_IDR_HasRegion_Flag",
    "MOTIF_IDR_Inside_Flag",
    "MOTIF_IDR_Within20AA_Flag",
    "MOTIF_IDR_Within50AA_Flag",
]

IDR_BOUNDARY_COLS = [
    "MOTIF_IDR_Nearest_Start",
    "MOTIF_IDR_Nearest_End",
]

IDR_UNSIGNED_DIST_COLS = [
    "MOTIF_IDR_Nearest_Start",
    "MOTIF_IDR_Nearest_End",
    "MOTIF_IDR_Nearest_Distance",
    "MOTIF_IDR_Nearest_DistanceNorm",
]

IDR_SIGNED_DIST_COLS = [
    "MOTIF_IDR_Nearest_CenterSignedDistance",
]

IDR_INSIDE_ONLY_FILL_MAP = {
    "MOTIF_IDR_Inside_RelativePosition": -1.0,
    "MOTIF_IDR_Inside_RegionLength": 0.0,
    "MOTIF_IDR_Inside_RegionLengthNorm": 0.0,
}

IDR_ZERO_FILL_COLS = [
    "MOTIF_IDR_Protein_RegionCount",
    "MOTIF_IDR_Protein_CoverageFraction",
]

IDR_SITE_WINDOW_COLS = [
    "MOTIF_IDR_Site_DisorderScore",
    "MOTIF_IDR_Window7_Disorder_Mean",
    "MOTIF_IDR_Window7_Disorder_Max",
    "MOTIF_IDR_Window7_Disorder_Min",
    "MOTIF_IDR_Window7_Disorder_Std",
    "MOTIF_IDR_Window7_Disorder_FractionGTThreshold",
    "MOTIF_IDR_Window15_Disorder_Mean",
    "MOTIF_IDR_Window15_Disorder_Max",
    "MOTIF_IDR_Window15_Disorder_Min",
    "MOTIF_IDR_Window15_Disorder_Std",
    "MOTIF_IDR_Window15_Disorder_FractionGTThreshold",
]

IDR_REGION_SCORE_COLS = [
    "MOTIF_IDR_NearestRegion_DisorderMean",
    "MOTIF_IDR_NearestRegion_DisorderMax",
]

IDR_FRACTION_COLS = [
    "MOTIF_IDR_Window7_Disorder_FractionGTThreshold",
    "MOTIF_IDR_Window15_Disorder_FractionGTThreshold",
    "MOTIF_IDR_Protein_CoverageFraction",
    "MOTIF_IDR_Inside_RelativePosition",
    "MOTIF_IDR_Inside_RegionLengthNorm",
]

IDR_DISORDER_SCORE_COLS = [
    "MOTIF_IDR_Site_DisorderScore",
    "MOTIF_IDR_Window7_Disorder_Mean",
    "MOTIF_IDR_Window7_Disorder_Max",
    "MOTIF_IDR_Window7_Disorder_Min",
    "MOTIF_IDR_Window15_Disorder_Mean",
    "MOTIF_IDR_Window15_Disorder_Max",
    "MOTIF_IDR_Window15_Disorder_Min",
    "MOTIF_IDR_NearestRegion_DisorderMean",
    "MOTIF_IDR_NearestRegion_DisorderMax",
]

IDR_NONNEGATIVE_COLS = [
    "MOTIF_IDR_Window7_Disorder_Std",
    "MOTIF_IDR_Window15_Disorder_Std",
    "MOTIF_IDR_Protein_RegionCount",
    "MOTIF_IDR_Inside_RegionLength",
    "MOTIF_IDR_Nearest_Start",
    "MOTIF_IDR_Nearest_End",
    "MOTIF_IDR_Nearest_Distance",
    "MOTIF_IDR_Nearest_DistanceNorm",
]


class IDRRegionPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe: bool = False,
        log_transform_dist: bool = True,
        large_value_mode: str = "p99_plus_1",
        add_missing_indicator: bool = False,
        drop_boundary_cols: bool = True,
    ):
        self.return_dataframe = return_dataframe
        self.log_transform_dist = log_transform_dist
        self.large_value_mode = large_value_mode
        self.add_missing_indicator = add_missing_indicator
        self.drop_boundary_cols = drop_boundary_cols

        self._keep_cols: Optional[List[str]] = None
        self._dist_fill_values: Optional[Dict[str, float]] = None
        self._median_fill_values: Optional[Dict[str, float]] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X).copy()
        X = self._drop_index_column(X)
        X = self._coerce_known_columns(X)

        has_region = self._get_flag_array(X, "MOTIF_IDR_HasRegion_Flag", default=1)

        self._dist_fill_values = {}
        for col in IDR_UNSIGNED_DIST_COLS:
            if col in X.columns:
                self._dist_fill_values[col] = self._compute_large_fill(X[col], has_region)

        self._median_fill_values = {}
        for col in IDR_SITE_WINDOW_COLS:
            if col in X.columns:
                self._median_fill_values[col] = self._compute_median_fill(X[col])

        for col in IDR_REGION_SCORE_COLS:
            if col in X.columns:
                self._median_fill_values[col] = self._compute_median_fill(X.loc[has_region == 1, col])

        X_trans = self._transform_internal(X.copy())
        self._keep_cols = self._build_keep_cols(list(X_trans.columns))
        self.feature_names_ = list(self._keep_cols)
        return self

    def transform(self, X: pd.DataFrame):
        if self._keep_cols is None:
            raise RuntimeError("IDRRegionPreprocessor must be fitted before transform.")

        X = self._check_input(X).copy()
        X = self._drop_index_column(X)
        X = self._coerce_known_columns(X)
        X = self._transform_internal(X)

        for col in self._keep_cols:
            if col not in X.columns:
                X[col] = 0.0

        X = X[self._keep_cols]

        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise ValueError(f"Non-numeric columns remain after IDRRegionPreprocessor: {non_numeric}")

        X = X.astype(float)

        if self.return_dataframe:
            return X

        return X.to_numpy(dtype=float)

    def get_feature_names_out(self):
        if self.feature_names_ is None:
            return None
        return list(self.feature_names_)

    def get_impute_summary(self) -> pd.DataFrame:
        if self._dist_fill_values is None or self._median_fill_values is None:
            raise RuntimeError("IDRRegionPreprocessor must be fitted before calling get_impute_summary().")

        rows = []
        for col, value in self._dist_fill_values.items():
            rows.append({"feature_name": col, "fill_type": "large_distance", "fill_value": value})
        for col, value in self._median_fill_values.items():
            rows.append({"feature_name": col, "fill_type": "training_median", "fill_value": value})
        for col, value in IDR_INSIDE_ONLY_FILL_MAP.items():
            rows.append({"feature_name": col, "fill_type": "outside_region_default", "fill_value": value})
        for col in IDR_ZERO_FILL_COLS:
            rows.append({"feature_name": col, "fill_type": "zero", "fill_value": 0.0})
        return pd.DataFrame(rows)

    def _transform_internal(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in IDR_BOOL_COLS:
            if col in X.columns:
                X[col] = self._to_bool01(X[col].values).astype(float)

        has_region = self._get_flag_array(X, "MOTIF_IDR_HasRegion_Flag", default=1)
        inside = self._get_flag_array(X, "MOTIF_IDR_Inside_Flag", default=0)

        missing_source = X[IDR_NUMERIC_COLS].isna().copy()

        for col in IDR_ZERO_FILL_COLS:
            if col in X.columns:
                X[col] = X[col].fillna(0.0)
                X.loc[has_region == 0, col] = 0.0

        for col, fill_value in IDR_INSIDE_ONLY_FILL_MAP.items():
            if col in X.columns:
                X[col] = X[col].fillna(fill_value)
                X.loc[inside == 0, col] = fill_value

        for col in IDR_SITE_WINDOW_COLS:
            if col in X.columns:
                fill = self._get_median_fill(col)
                X[col] = X[col].fillna(fill)

        for col in IDR_REGION_SCORE_COLS:
            if col in X.columns:
                fill = self._get_median_fill(col)
                X[col] = X[col].fillna(fill)
                X.loc[has_region == 0, col] = 0.0

        for col in IDR_UNSIGNED_DIST_COLS:
            if col in X.columns:
                fill = self._get_distance_fill(col)
                X[col] = X[col].fillna(fill)
                X.loc[has_region == 0, col] = fill
                X[col] = np.clip(X[col].to_numpy(dtype=float), 0.0, None)
                if self.log_transform_dist:
                    X[col] = np.log1p(X[col].to_numpy(dtype=float))

        for col in IDR_SIGNED_DIST_COLS:
            if col in X.columns:
                X[col] = X[col].fillna(0.0)
                X.loc[has_region == 0, col] = 0.0
                if self.log_transform_dist:
                    X[col] = self._signed_log1p(X[col].to_numpy(dtype=float))

        self._clip_physical_ranges(X)

        if self.add_missing_indicator:
            for col in IDR_NUMERIC_COLS:
                if col in missing_source.columns:
                    X[f"{col}_missing"] = missing_source[col].astype(float)

        for col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(float)

        leftover = X.columns[X.isna().any()].tolist()
        if leftover:
            X[leftover] = X[leftover].fillna(0.0)

        return X

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("IDRRegionPreprocessor expects a pandas DataFrame.")

        X = X.copy()

        unexpected = [
            col
            for col in X.columns
            if col != "INDEX" and col not in IDR_NUMERIC_COLS and col not in IDR_BOOL_COLS
        ]
        if unexpected:
            raise ValueError(f"Non-IDR feature columns detected: {unexpected}")

        for col in IDR_NUMERIC_COLS:
            if col not in X.columns:
                X[col] = np.nan

        for col in IDR_BOOL_COLS:
            if col not in X.columns:
                X[col] = 0.0

        return X

    def _drop_index_column(self, X: pd.DataFrame) -> pd.DataFrame:
        if "INDEX" in X.columns:
            return X.drop(columns=["INDEX"])
        return X

    def _coerce_known_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in IDR_BOOL_COLS:
            if col in X.columns:
                X[col] = self._to_bool01(X[col].values).astype(float)
        for col in IDR_NUMERIC_COLS:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        return X

    def _build_keep_cols(self, cols: List[str]) -> List[str]:
        cols = list(cols)

        if self.drop_boundary_cols:
            drop = set(IDR_BOUNDARY_COLS)
            drop.update(f"{col}_missing" for col in IDR_BOUNDARY_COLS)
            cols = [col for col in cols if col not in drop]

        return cols

    def _get_flag_array(self, X: pd.DataFrame, col: str, default: int) -> np.ndarray:
        if col in X.columns:
            return self._to_bool01(X[col].values).astype(int)
        return np.full(len(X), int(default), dtype=int)

    def _compute_large_fill(self, s: pd.Series, has_region: np.ndarray) -> float:
        v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        v = v[has_region == 1]
        v = v[np.isfinite(v)]

        if v.size == 0:
            return 1.0

        if self.large_value_mode == "max_plus_1":
            return float(np.max(v) + 1.0)
        if self.large_value_mode == "p99_plus_1":
            return float(np.quantile(v, 0.99) + 1.0)
        if self.large_value_mode == "p95_times_2":
            return float(np.quantile(v, 0.95) * 2.0)

        raise ValueError(f"Unknown large_value_mode: {self.large_value_mode}")

    def _compute_median_fill(self, s: pd.Series) -> float:
        v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return 0.0
        return float(np.median(v))

    def _get_distance_fill(self, col: str) -> float:
        if self._dist_fill_values is None:
            return 1.0
        return float(self._dist_fill_values.get(col, 1.0))

    def _get_median_fill(self, col: str) -> float:
        if self._median_fill_values is None:
            return 0.0
        return float(self._median_fill_values.get(col, 0.0))

    def _clip_physical_ranges(self, X: pd.DataFrame) -> None:
        for col in IDR_FRACTION_COLS:
            if col in X.columns:
                if col == "MOTIF_IDR_Inside_RelativePosition":
                    X[col] = X[col].clip(lower=-1.0, upper=1.0)
                else:
                    X[col] = X[col].clip(lower=0.0, upper=1.0)

        for col in IDR_DISORDER_SCORE_COLS:
            if col in X.columns:
                X[col] = X[col].clip(lower=0.0, upper=1.0)

        for col in IDR_NONNEGATIVE_COLS:
            if col in X.columns:
                X[col] = X[col].clip(lower=0.0)

    def _signed_log1p(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return np.sign(values) * np.log1p(np.abs(values))

    def _to_bool01(self, values) -> np.ndarray:
        s = pd.Series(values)

        if s.dtype == bool:
            return s.astype(int).to_numpy()

        if s.dtype == object:
            s_norm = s.astype(str).str.strip().str.lower()
            mapped = s_norm.map(
                {
                    "true": 1,
                    "false": 0,
                    "1": 1,
                    "0": 0,
                    "yes": 1,
                    "no": 0,
                    "y": 1,
                    "n": 0,
                }
            )
            if mapped.notna().any():
                return mapped.fillna(0).astype(int).to_numpy()

        x = pd.to_numeric(s, errors="coerce").fillna(0.0)
        return (x > 0).astype(int).to_numpy()
