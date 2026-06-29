"""
File location:
/mnt/Storage2/home/yangtao/phosloc-PLR/src/preprocess/idr.py
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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

IDR_DIST_COLS = [
    "MOTIF_IDR_Nearest_Start",
    "MOTIF_IDR_Nearest_End",
    "MOTIF_IDR_Nearest_Distance",
    "MOTIF_IDR_Nearest_DistanceNorm",
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

IDR_MEDIAN_FILL_COLS = [
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
    "MOTIF_IDR_NearestRegion_DisorderMean",
    "MOTIF_IDR_NearestRegion_DisorderMax",
]


class IDRRegionPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe: bool = False,
        log_transform_dist: bool = True,
        large_value_mode: str = "p99_plus_1",
        add_missing_indicator: bool = False,
    ):
        self.return_dataframe = return_dataframe
        self.log_transform_dist = log_transform_dist
        self.large_value_mode = large_value_mode
        self.add_missing_indicator = add_missing_indicator

        self._all_cols = None
        self._dist_fill_values = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X).copy()

        if "INDEX" in X.columns:
            X = X.drop(columns=["INDEX"])

        for c in IDR_BOOL_COLS:
            if c in X.columns:
                X[c] = self._to_bool01(X[c].values)

        if "MOTIF_IDR_HasRegion_Flag" in X.columns:
            has_region = X["MOTIF_IDR_HasRegion_Flag"].to_numpy(dtype=int)
        else:
            has_region = np.ones(len(X), dtype=int)

        self._dist_fill_values = {}
        for c in IDR_DIST_COLS:
            if c not in X.columns:
                continue

            v = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
            v_has = v[has_region == 1]
            v_has = v_has[np.isfinite(v_has)]

            if v_has.size == 0:
                fill = 1.0
            else:
                if self.large_value_mode == "max_plus_1":
                    fill = float(np.max(v_has) + 1.0)
                elif self.large_value_mode == "p99_plus_1":
                    fill = float(np.quantile(v_has, 0.99) + 1.0)
                elif self.large_value_mode == "p95_times_2":
                    fill = float(np.quantile(v_has, 0.95) * 2.0)
                else:
                    raise ValueError(f"Unknown large_value_mode: {self.large_value_mode}")

            self._dist_fill_values[c] = fill

        X_trans = self._transform_internal(X.copy())
        self._all_cols = list(X_trans.columns)
        return self

    def transform(self, X: pd.DataFrame):
        X = self._check_input(X).copy()

        if "INDEX" in X.columns:
            X = X.drop(columns=["INDEX"])

        X = self._transform_internal(X)

        if self._all_cols is not None:
            missing_cols = [c for c in self._all_cols if c not in X.columns]
            for c in missing_cols:
                X[c] = 0.0
            X = X[self._all_cols]

        if self.return_dataframe:
            return X

        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise ValueError(f"Non-numeric columns remain after IDRRegionPreprocessor: {non_numeric}")

        return X.to_numpy(dtype=float)

    def _transform_internal(self, X: pd.DataFrame) -> pd.DataFrame:
        for c in IDR_BOOL_COLS:
            if c in X.columns:
                X[c] = self._to_bool01(X[c].values)

        if "MOTIF_IDR_HasRegion_Flag" in X.columns:
            has_region = X["MOTIF_IDR_HasRegion_Flag"].to_numpy(dtype=int)
        else:
            has_region = np.ones(len(X), dtype=int)

        for c in IDR_NUMERIC_COLS:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce")

        if self.add_missing_indicator:
            for c in IDR_NUMERIC_COLS:
                if c in X.columns:
                    X[f"{c}_missing"] = X[c].isna().astype(float)

        for col, fill_value in IDR_INSIDE_ONLY_FILL_MAP.items():
            if col in X.columns:
                X[col] = X[col].fillna(fill_value)

        for col in IDR_ZERO_FILL_COLS:
            if col in X.columns:
                X[col] = X[col].fillna(0.0)

        for col in IDR_MEDIAN_FILL_COLS:
            if col in X.columns:
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                X[col] = X[col].fillna(float(median_val))

        for c in IDR_DIST_COLS:
            if c not in X.columns:
                continue
            fill = self._dist_fill_values.get(c, 1.0)
            X.loc[has_region == 0, c] = fill
            X[c] = X[c].fillna(fill)
            if self.log_transform_dist:
                X[c] = np.log1p(np.clip(X[c].to_numpy(dtype=float), 0.0, None))

        for col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(float)

        leftover = X.columns[X.isna().any()].tolist()
        if leftover:
            X[leftover] = X[leftover].fillna(0.0)

        return X

    def _to_bool01(self, v):
        s = pd.Series(v)
        if s.dtype == bool:
            return s.astype(int).to_numpy()

        if s.dtype == object:
            s = s.astype(str).str.strip().str.lower()
            mapped = s.map({
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
            })
            if mapped.notna().any():
                return mapped.fillna(0).astype(int).to_numpy()

        x = pd.to_numeric(s, errors="coerce").fillna(0.0)
        return (x > 0).astype(int).to_numpy()

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("IDRRegionPreprocessor expects a pandas DataFrame.")

        X = X.copy()

        for c in IDR_NUMERIC_COLS:
            if c not in X.columns:
                X[c] = np.nan

        for c in IDR_BOOL_COLS:
            if c not in X.columns:
                X[c] = 0.0

        unexpected = [c for c in X.columns if c != "INDEX" and c not in IDR_NUMERIC_COLS and c not in IDR_BOOL_COLS]
        if unexpected:
            raise ValueError(f"Non-IDR feature columns detected: {unexpected}")

        return X