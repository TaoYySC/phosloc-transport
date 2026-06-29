"""
File location:
/mnt/Storage2/home/yangtao/phosloc-PLR/src/preprocess/nls.py
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NLS_PREFIX = "MOTIF_NLS_"

NLS_BOOL_COLS = [
    "MOTIF_NLS_HasSegment_Flag",
    "MOTIF_NLS_Inside_Flag",
    "MOTIF_NLS_Within20AA_Flag",
    "MOTIF_NLS_Within50AA_Flag",
]

NLS_DIST_COLS = [
    "MOTIF_NLS_Nearest_Start",
    "MOTIF_NLS_Nearest_End",
    "MOTIF_NLS_Nearest_Distance",
    "MOTIF_NLS_Nearest_DistanceNorm",
    "MOTIF_NLS_Nearest_CenterSignedDistance",
]

NLS_INSIDE_ONLY_COLS = [
    "MOTIF_NLS_Inside_RelativePosition",
    "MOTIF_NLS_Inside_SegmentLength",
    "MOTIF_NLS_Inside_SegmentLengthNorm",
]

NLS_OTHER_FLOAT_COLS = [
    "MOTIF_NLS_Protein_SegmentCount",
    "MOTIF_NLS_Nearest_Score",
    "MOTIF_NLS_Protein_CoverageFraction",
]


class NLSFeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe: bool = False,
        drop_weak_cols: bool = True,
        log_transform_dist: bool = True,
        large_value_mode: str = "p99_plus_1",
    ):
        self.return_dataframe = return_dataframe
        self.drop_weak_cols = drop_weak_cols
        self.log_transform_dist = log_transform_dist
        self.large_value_mode = large_value_mode

        self._all_cols = None
        self._keep_cols = None
        self._dist_fill_values = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X).copy()

        for c in NLS_BOOL_COLS:
            if c in X.columns:
                X[c] = self._to_bool01(X[c].values)

        has = (
            X["MOTIF_NLS_HasSegment_Flag"].to_numpy(dtype=int)
            if "MOTIF_NLS_HasSegment_Flag" in X.columns
            else np.ones(len(X), dtype=int)
        )

        self._dist_fill_values = {}
        for c in NLS_DIST_COLS:
            if c not in X.columns:
                continue

            v = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
            v_has = v[has == 1]
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

        self._all_cols = list(X.columns)
        self._keep_cols = self._build_keep_cols(self._all_cols)
        self.feature_names_ = list(self._keep_cols)
        return self

    def transform(self, X: pd.DataFrame):
        X = self._check_input(X).copy()

        for c in NLS_BOOL_COLS:
            if c in X.columns:
                X[c] = self._to_bool01(X[c].values)

        has = (
            X["MOTIF_NLS_HasSegment_Flag"].to_numpy(dtype=int)
            if "MOTIF_NLS_HasSegment_Flag" in X.columns
            else np.ones(len(X), dtype=int)
        )
        inside = (
            X["MOTIF_NLS_Inside_Flag"].to_numpy(dtype=int)
            if "MOTIF_NLS_Inside_Flag" in X.columns
            else np.zeros(len(X), dtype=int)
        )

        for c in NLS_DIST_COLS:
            if c not in X.columns:
                continue
            X[c] = pd.to_numeric(X[c], errors="coerce")

            fill = self._dist_fill_values.get(c, 1.0)
            X.loc[has == 0, c] = fill
            X[c] = X[c].fillna(fill)

            if self.log_transform_dist:
                X[c] = np.log1p(np.clip(X[c].to_numpy(dtype=float), 0.0, None))

        for c in NLS_INSIDE_ONLY_COLS + NLS_OTHER_FLOAT_COLS:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce")
                if c in ["MOTIF_NLS_Inside_RelativePosition"]:
                    X.loc[inside == 0, c] = -1.0
                else:
                    X.loc[inside == 0, c] = 0.0
                X[c] = X[c].fillna(0.0)

        if self._keep_cols is not None:
            missing_cols = [c for c in self._keep_cols if c not in X.columns]
            for c in missing_cols:
                X[c] = 0.0
            X = X[self._keep_cols]

        if self.return_dataframe:
            return X

        return X.to_numpy(dtype=float)

    def get_feature_names_out(self):
        if self.feature_names_ is None:
            return None
        return list(self.feature_names_)

    def _build_keep_cols(self, cols):
        if not self.drop_weak_cols:
            return list(cols)

        weak = {
            "MOTIF_NLS_Nearest_Score",
            "MOTIF_NLS_Protein_CoverageFraction",
        }
        return [c for c in cols if c not in weak]

    def _to_bool01(self, v):
        s = pd.Series(v)
        if s.dtype == bool:
            return s.astype(int).to_numpy()
        x = pd.to_numeric(s, errors="coerce").fillna(0.0)
        return (x > 0).astype(int).to_numpy()

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("NLSFeaturePreprocessor expects a DataFrame.")

        X = X.copy()

        for c in NLS_BOOL_COLS:
            if c not in X.columns:
                X[c] = 0.0

        for c in NLS_DIST_COLS + NLS_INSIDE_ONLY_COLS + NLS_OTHER_FLOAT_COLS:
            if c not in X.columns:
                X[c] = np.nan

        bad = [c for c in X.columns if not c.startswith(NLS_PREFIX)]
        if bad:
            raise ValueError(f"Non-NLS_ columns detected: {bad}")

        return X