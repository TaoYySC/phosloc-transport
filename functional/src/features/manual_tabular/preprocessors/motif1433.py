import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


MOTIF_1433_PREFIX = "MOTIF_1433_"

MOTIF_1433_BOOL_COLS = [
    "MOTIF_1433_HasMotif_Flag",
    "MOTIF_1433_Inside_Flag",
    "MOTIF_1433_Within20AA_Flag",
    "MOTIF_1433_Within50AA_Flag",
    "MOTIF_1433_IsPredictedSite_Flag",
    "MOTIF_1433_Nearest_IsHighConfidence_Flag",
    "MOTIF_1433_Inside_HighConfidence_Flag",
]

MOTIF_1433_COUNT_COLS = [
    "MOTIF_1433_Protein_MotifCount",
    "MOTIF_1433_Inside_OverlapCount",
    "MOTIF_1433_Protein_HighConfidenceSegmentCount",
    "MOTIF_1433_Protein_PredictedSiteCount",
]

MOTIF_1433_FRACTION_COLS = [
    "MOTIF_1433_Protein_CoverageFraction",
    "MOTIF_1433_Protein_HighConfidenceCoverageFraction",
]

MOTIF_1433_SCORE_COLS = [
    "MOTIF_1433_Nearest_ConsensusScore",
]

MOTIF_1433_DIST_COLS = [
    "MOTIF_1433_Nearest_Distance",
    "MOTIF_1433_Nearest_DistanceNorm",
    "MOTIF_1433_Nearest_CenterSignedDistance",
]


class Motif1433Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe: bool = False,
        drop_dead_cols: bool = True,
        log_transform_dist: bool = True,
        large_value_mode: str = "p99_plus_1",
    ):
        self.return_dataframe = return_dataframe
        self.drop_dead_cols = drop_dead_cols
        self.log_transform_dist = log_transform_dist
        self.large_value_mode = large_value_mode

        self._all_cols = None
        self._keep_cols = None
        self._dist_fill_values = None
        self._score_fill_values = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X).copy()

        X["MOTIF_1433_HasMotif_Flag"] = self._to_bool01(X["MOTIF_1433_HasMotif_Flag"].values)
        has = X["MOTIF_1433_HasMotif_Flag"].to_numpy(dtype=int)

        self._dist_fill_values = {}
        self._score_fill_values = {}

        for c in MOTIF_1433_DIST_COLS:
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

        for c in MOTIF_1433_SCORE_COLS:
            if c not in X.columns:
                continue

            v = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
            v_has = v[has == 1]
            v_has = v_has[np.isfinite(v_has)]
            self._score_fill_values[c] = float(np.median(v_has)) if v_has.size else 0.0

        self._all_cols = list(X.columns)
        self._keep_cols = self._build_keep_cols(self._all_cols)
        self.feature_names_ = list(self._keep_cols)
        return self

    def transform(self, X: pd.DataFrame):
        X = self._check_input(X).copy()

        X["MOTIF_1433_HasMotif_Flag"] = self._to_bool01(X["MOTIF_1433_HasMotif_Flag"].values)

        for c in MOTIF_1433_BOOL_COLS:
            if c in X.columns:
                X[c] = self._to_bool01(X[c].values)

        has = X["MOTIF_1433_HasMotif_Flag"].to_numpy(dtype=int)

        for c in MOTIF_1433_COUNT_COLS:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce")
                X.loc[has == 0, c] = 0.0
                X[c] = X[c].fillna(0.0)

        for c in MOTIF_1433_FRACTION_COLS:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce")
                X.loc[has == 0, c] = 0.0
                X[c] = X[c].fillna(0.0)

        for c in MOTIF_1433_SCORE_COLS:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce")
                X.loc[has == 0, c] = 0.0
                X[c] = X[c].fillna(self._score_fill_values.get(c, 0.0))

        for c in MOTIF_1433_DIST_COLS:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce")
                fill = self._dist_fill_values.get(c, 1.0)
                X.loc[has == 0, c] = fill
                X[c] = X[c].fillna(fill)

                if self.log_transform_dist:
                    X[c] = np.log1p(np.clip(X[c].to_numpy(dtype=float), 0.0, None))

        if self._keep_cols is not None:
            missing_cols = [c for c in self._keep_cols if c not in X.columns]
            for c in missing_cols:
                if c.startswith(MOTIF_1433_PREFIX):
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
        dead_cols = {
            "MOTIF_1433_Inside_RelativePosition",
            "MOTIF_1433_Inside_SegmentLength",
            "MOTIF_1433_Inside_SegmentLengthNorm",
            "MOTIF_1433_Protein_CoverageFraction",
            "MOTIF_1433_Nearest_Start",
            "MOTIF_1433_Nearest_End",
        }

        if not self.drop_dead_cols:
            return list(cols)

        return [c for c in cols if c not in dead_cols]

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

    def _check_input(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Motif1433Preprocessor expects a pandas DataFrame.")

        X = X.copy()

        bad = [c for c in X.columns if not str(c).startswith(MOTIF_1433_PREFIX)]
        if bad:
            raise ValueError(f"Non-MOTIF_1433_ columns detected: {bad}")

        for c in MOTIF_1433_BOOL_COLS:
            if c not in X.columns:
                X[c] = 0.0

        for c in MOTIF_1433_COUNT_COLS:
            if c not in X.columns:
                X[c] = 0.0

        for c in MOTIF_1433_FRACTION_COLS:
            if c not in X.columns:
                X[c] = 0.0

        for c in MOTIF_1433_SCORE_COLS:
            if c not in X.columns:
                X[c] = np.nan

        for c in MOTIF_1433_DIST_COLS:
            if c not in X.columns:
                X[c] = np.nan

        return X