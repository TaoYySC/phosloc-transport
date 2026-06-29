"""
File location:
/mnt/Storage2/home/yangtao/phosloc-PLR/src/preprocess/sequence.py
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


SEQ_PREFIX = "SEQ_"

SEQ_RATIO_COLS = [
    f"{SEQ_PREFIX}ratio_R_over_K",
    f"{SEQ_PREFIX}ratio_E_over_D",
]

SEQ_HYDRO_COL = f"{SEQ_PREFIX}mean_hydropathy_kd"


class SequencePhysicochemPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, return_dataframe: bool = False):
        self.return_dataframe = return_dataframe
        self._imputer = None
        self._all_cols = None
        self._ratio_cols = None
        self._hydro_col = None
        self._other_numeric_cols = None
        self._hydro_median_ = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X)

        self._all_cols = list(X.columns)
        self._ratio_cols = [c for c in SEQ_RATIO_COLS if c in X.columns]
        self._hydro_col = SEQ_HYDRO_COL if SEQ_HYDRO_COL in X.columns else None

        numeric_cols = [c for c in X.columns if c not in self._ratio_cols]
        if self._hydro_col in numeric_cols:
            numeric_cols.remove(self._hydro_col)

        self._other_numeric_cols = numeric_cols

        if self._other_numeric_cols:
            self._imputer = SimpleImputer(strategy="median")
            self._imputer.fit(X[self._other_numeric_cols])

        if self._hydro_col is not None:
            hydro_vals = pd.to_numeric(X[self._hydro_col], errors="coerce").values.astype(float)
            med = np.nanmedian(hydro_vals)
            if np.isnan(med):
                med = 0.0
            self._hydro_median_ = float(med)

        return self

    def transform(self, X: pd.DataFrame):
        X = self._check_input(X).copy()

        if self._ratio_cols:
            for col in self._ratio_cols:
                if col not in X.columns:
                    X[col] = np.nan
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

        if self._hydro_col is not None:
            if self._hydro_col not in X.columns:
                X[self._hydro_col] = np.nan
            X[self._hydro_col] = pd.to_numeric(X[self._hydro_col], errors="coerce").fillna(self._hydro_median_)

        if self._imputer is not None and self._other_numeric_cols:
            missing_cols = [c for c in self._other_numeric_cols if c not in X.columns]
            for c in missing_cols:
                X[c] = np.nan
            X[self._other_numeric_cols] = self._imputer.transform(X[self._other_numeric_cols])

        if self._all_cols is not None:
            missing_cols = [c for c in self._all_cols if c not in X.columns]
            for c in missing_cols:
                X[c] = 0.0
            X = X[self._all_cols]

        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0).astype(float)

        return X if self.return_dataframe else X.to_numpy(dtype=float)

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("SequencePhysicochemPreprocessor expects a DataFrame.")

        X = X.copy()
        bad = [c for c in X.columns if not c.startswith(SEQ_PREFIX)]
        if bad:
            raise ValueError(f"Non-SEQ_ columns found: {bad}")

        return X