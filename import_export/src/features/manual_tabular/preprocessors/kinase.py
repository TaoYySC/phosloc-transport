import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


KINASE_PREFIX = "FUNC_Kinase_"

KINASE_COLUMNS = [
    "FUNC_Kinase_PWM_MaxMSS",
    "FUNC_Kinase_PWM_Top005KinaseCount",
    "FUNC_Kinase_PWM_MSS_AGC_Max",
    "FUNC_Kinase_PWM_MSS_CAMK_Max",
    "FUNC_Kinase_PWM_MSS_CK1_Max",
    "FUNC_Kinase_PWM_MSS_CMGC_Max",
    "FUNC_Kinase_PWM_MSS_STE_Max",
    "FUNC_Kinase_PWM_MSS_TK_Max",
    "FUNC_Kinase_PWM_MSS_TKL_Max",
    "FUNC_Kinase_PWM_MSS_RGC_Max",
    "FUNC_Kinase_PWM_MSS_Other_Max",
    "FUNC_Kinase_NetPhorest_MaxAll",
    "FUNC_Kinase_NetPhorest_MaxKinase",
    "FUNC_Kinase_NetPhorest_MaxSTDomain",
]

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
    def __init__(
        self,
        return_dataframe: bool = False,
        add_has_gates: bool = True,
        add_missing_indicator: bool = False,
        use_fixed_schema: bool = True,
        keep_extra_kinase_cols: bool = False,
        impute_strategy: str = "median",
        all_nan_fill_value: float = 0.0,
        clip_scores: bool = True,
        clip_counts: bool = True,
    ):
        self.return_dataframe = return_dataframe
        self.add_has_gates = add_has_gates
        self.add_missing_indicator = add_missing_indicator
        self.use_fixed_schema = use_fixed_schema
        self.keep_extra_kinase_cols = keep_extra_kinase_cols
        self.impute_strategy = impute_strategy
        self.all_nan_fill_value = all_nan_fill_value
        self.clip_scores = clip_scores
        self.clip_counts = clip_counts

        self._feature_cols: Optional[List[str]] = None
        self._source_map: Optional[Dict[str, str]] = None
        self._fill_values: Optional[Dict[str, float]] = None
        self._missing_rate: Optional[Dict[str, float]] = None
        self._all_nan_cols: Optional[Dict[str, bool]] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X).copy()
        X = self._drop_index(X)

        self._feature_cols = self._build_feature_cols(X)
        self._source_map = self._build_source_map(X, self._feature_cols)

        X_num, X_raw = self._prepare_numeric_frame(X, self._feature_cols, self._source_map)

        self._fill_values = {}
        self._missing_rate = {}
        self._all_nan_cols = {}

        for col in self._feature_cols:
            s = X_num[col]
            observed = s[np.isfinite(s)]
            self._missing_rate[col] = float(s.isna().mean())
            self._all_nan_cols[col] = bool(observed.empty)

            if observed.empty:
                fill = float(self.all_nan_fill_value)
            elif self.impute_strategy == "median":
                fill = float(np.median(observed.to_numpy(dtype=float)))
            elif self.impute_strategy == "mean":
                fill = float(np.mean(observed.to_numpy(dtype=float)))
            elif self.impute_strategy == "zero":
                fill = 0.0
            else:
                raise ValueError(f"Unknown impute_strategy: {self.impute_strategy}")

            self._fill_values[col] = fill

        self.feature_names_ = self._build_output_feature_names(self._feature_cols)
        return self

    def transform(self, X: pd.DataFrame):
        if self._feature_cols is None or self._source_map is None or self._fill_values is None:
            raise RuntimeError("KinasePriorPreprocessor must be fitted before transform.")

        X = self._check_input(X).copy()
        X = self._drop_index(X)

        X_num, X_raw = self._prepare_numeric_frame(X, self._feature_cols, self._source_map)
        out = pd.DataFrame(index=X.index)

        for col in self._feature_cols:
            s_raw = X_raw[col]
            has = s_raw.notna().astype(float)
            missing = s_raw.isna().astype(float)

            s = X_num[col].copy()
            fill = self._fill_values.get(col, float(self.all_nan_fill_value))
            s = s.fillna(fill)
            s = self._clip_column(col, s)

            if self.add_has_gates:
                out[f"{col}__has"] = has
            if self.add_missing_indicator:
                out[f"{col}__missing"] = missing

            out[col] = s.astype(float)

        if self.feature_names_ is not None:
            for col in self.feature_names_:
                if col not in out.columns:
                    out[col] = 0.0
            out = out[self.feature_names_]

        if self.return_dataframe:
            return out

        return out.to_numpy(dtype=float)

    def get_feature_names_out(self):
        if self.feature_names_ is None:
            return None
        return list(self.feature_names_)

    def get_impute_summary(self) -> pd.DataFrame:
        if self._feature_cols is None or self._fill_values is None:
            raise RuntimeError("KinasePriorPreprocessor must be fitted before calling get_impute_summary().")

        rows = []
        for col in self._feature_cols:
            rows.append(
                {
                    "feature_name": col,
                    "source_column": self._source_map.get(col, col) if self._source_map else col,
                    "fill_value": self._fill_values.get(col),
                    "missing_rate": self._missing_rate.get(col) if self._missing_rate else None,
                    "all_nan_in_fit": self._all_nan_cols.get(col) if self._all_nan_cols else None,
                }
            )
        return pd.DataFrame(rows)

    def _build_feature_cols(self, X: pd.DataFrame) -> List[str]:
        if self.use_fixed_schema:
            cols = list(KINASE_COLUMNS)
        else:
            cols = [c for c in X.columns if c.startswith(KINASE_PREFIX)]

        if self.keep_extra_kinase_cols:
            extra_cols = [c for c in X.columns if c.startswith(KINASE_PREFIX) and c not in cols]
            cols.extend(sorted(extra_cols))

        return cols

    def _build_source_map(self, X: pd.DataFrame, feature_cols: List[str]) -> Dict[str, str]:
        source_map = {}
        for col in feature_cols:
            candidates = KINASE_COL_CANDIDATES.get(col, [col])
            found = None
            for candidate in candidates:
                if candidate in X.columns:
                    found = candidate
                    break
            source_map[col] = found if found is not None else col
        return source_map

    def _prepare_numeric_frame(self, X: pd.DataFrame, feature_cols: List[str], source_map: Dict[str, str]):
        X_num = pd.DataFrame(index=X.index)
        X_raw = pd.DataFrame(index=X.index)

        for col in feature_cols:
            source_col = source_map.get(col, col)
            if source_col in X.columns:
                raw = X[source_col]
            elif col in X.columns:
                raw = X[col]
            else:
                raw = pd.Series(np.nan, index=X.index)

            X_raw[col] = raw
            numeric = pd.to_numeric(raw, errors="coerce")
            numeric = numeric.replace([np.inf, -np.inf], np.nan)
            numeric = self._clip_column(col, numeric)
            X_num[col] = numeric

        return X_num, X_raw

    def _clip_column(self, col: str, s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")

        if not self.clip_scores and not self.clip_counts:
            return s

        if self.clip_counts and self._is_count_column(col):
            return s.clip(lower=0.0)

        if self.clip_scores and self._is_probability_like_column(col):
            return s.clip(lower=0.0, upper=1.0)

        return s

    def _is_count_column(self, col: str) -> bool:
        name = col.lower()
        return any(k in name for k in ["count", "nkin", "number"])

    def _is_probability_like_column(self, col: str) -> bool:
        name = col.lower()
        if "netphorest" in name:
            return True
        return any(k in name for k in ["prob", "score", "fraction", "frequency", "ratio"])

    def _build_output_feature_names(self, feature_cols: List[str]) -> List[str]:
        names = []
        for col in feature_cols:
            if self.add_has_gates:
                names.append(f"{col}__has")
            if self.add_missing_indicator:
                names.append(f"{col}__missing")
            names.append(col)
        return names

    def _drop_index(self, X: pd.DataFrame) -> pd.DataFrame:
        if "INDEX" in X.columns:
            return X.drop(columns=["INDEX"])
        return X

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("KinasePriorPreprocessor expects a pandas DataFrame.")

        allowed_aliases = set()
        for candidates in KINASE_COL_CANDIDATES.values():
            allowed_aliases.update(candidates)

        bad_cols = []
        for col in X.columns:
            if col == "INDEX":
                continue
            if col.startswith(KINASE_PREFIX):
                continue
            if col in allowed_aliases:
                continue
            bad_cols.append(col)

        if bad_cols:
            raise ValueError(f"Non-kinase feature columns detected: {bad_cols}")

        return X.copy()
