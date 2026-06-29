"""
File location:
/mnt/Storage2/home/yangtao/phosloc-PLR/src/preprocess/kinase.py
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


KINASE_COL_CANDIDATES = {
    "FUNC_Kinase_PWM_MaxMSS": [
        "FUNC_Kinase_PWM_MaxMSS",
        "KINASE_pwm_match__PWM_max_mss",
        "KINASE_PWM_max_mss",
    ],
    "FUNC_Kinase_PWM_Top005KinaseCount": [
        "FUNC_Kinase_PWM_Top005KinaseCount",
        "KINASE_pwm_match__PWM_nkinTop005",
        "KINASE_PWM_nkinTop005",
    ],
    "FUNC_Kinase_NetPhorest_MaxAll": [
        "FUNC_Kinase_NetPhorest_MaxAll",
        "KINASE_netphorest__netpho_max_all",
        "KINASE_netpho_max_all",
    ],
    "FUNC_Kinase_NetPhorest_MaxKinase": [
        "FUNC_Kinase_NetPhorest_MaxKinase",
        "KINASE_netphorest__netpho_max_KIN",
        "KINASE_netpho_max_KIN",
    ],
    "FUNC_Kinase_NetPhorest_MaxSTDomain": [
        "FUNC_Kinase_NetPhorest_MaxSTDomain",
        "KINASE_netphorest__netpho_max_STdomain",
        "KINASE_netpho_max_STdomain",
    ],
}


class KinasePriorPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, return_dataframe: bool = False, add_has_gates: bool = True):
        self.return_dataframe = return_dataframe
        self.add_has_gates = add_has_gates
        self._keep_cols = None
        self._median_by_col = None
        self._active_map = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X).copy()

        if "INDEX" in X.columns:
            X = X.drop(columns=["INDEX"])

        self._active_map = {}
        for canonical_col, candidates in KINASE_COL_CANDIDATES.items():
            found = None
            for c in candidates:
                if c in X.columns:
                    found = c
                    break
            if found is not None:
                self._active_map[canonical_col] = found

        if not self._active_map:
            fallback_col = "FUNC_Kinase_PWM_MaxMSS"
            X[fallback_col] = np.nan
            self._active_map = {fallback_col: fallback_col}

        self._median_by_col = {}

        for canonical_col, source_col in self._active_map.items():
            has = self._has_value(X[source_col])
            v = pd.to_numeric(X.loc[has, source_col], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            self._median_by_col[canonical_col] = float(np.median(v)) if v.size else 0.0

        X_trans = self._transform_internal(X.copy())
        self._keep_cols = list(X_trans.columns)
        self.feature_names_ = list(self._keep_cols)
        return self

    def transform(self, X: pd.DataFrame):
        X = self._check_input(X).copy()

        if "INDEX" in X.columns:
            X = X.drop(columns=["INDEX"])

        for canonical_col, source_col in self._active_map.items():
            if source_col not in X.columns and canonical_col not in X.columns:
                X[source_col] = np.nan

        X = self._transform_internal(X)

        if self._keep_cols is not None:
            missing_cols = [c for c in self._keep_cols if c not in X.columns]
            for c in missing_cols:
                X[c] = 0.0
            X = X[self._keep_cols]

        if self.return_dataframe:
            return X

        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise ValueError(f"Non-numeric columns remain after KinasePriorPreprocessor: {non_numeric}")

        return X.to_numpy(dtype=float)

    def get_feature_names_out(self):
        if self.feature_names_ is None:
            return None
        return list(self.feature_names_)

    def _transform_internal(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)

        for canonical_col, source_col in self._active_map.items():
            use_col = source_col if source_col in X.columns else canonical_col
            s = pd.to_numeric(X[use_col], errors="coerce")
            has = self._has_value(s).astype(int)

            if self.add_has_gates:
                out[f"{canonical_col}__has"] = has.astype(float)

            fill = self._median_by_col.get(canonical_col, 0.0)
            s = s.copy()
            s.loc[has == 0] = 0.0
            s.loc[has == 1] = s.loc[has == 1].fillna(fill)
            out[canonical_col] = s.astype(float)

        return out

    def _has_value(self, s: pd.Series) -> np.ndarray:
        x = pd.to_numeric(s, errors="coerce")
        return x.notna().to_numpy()

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("KinasePriorPreprocessor expects a pandas DataFrame.")
        return X.copy()