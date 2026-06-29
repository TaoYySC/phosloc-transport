"""
File location:
/mnt/Storage2/home/yangtao/phosloc-PLR/src/preprocess/zscale.py
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


ZSCALE_PREFIX = "SEQ_ZScale_"

ZSCALE_MEAN_COLS = [f"{ZSCALE_PREFIX}Window_Z{i}_Mean" for i in range(1, 6)]
ZSCALE_STD_COLS = [f"{ZSCALE_PREFIX}Window_Z{i}_Std" for i in range(1, 6)]
ZSCALE_COLS = ZSCALE_MEAN_COLS + ZSCALE_STD_COLS


class ZScalePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, return_dataframe: bool = False):
        self.return_dataframe = return_dataframe
        self._all_cols = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X)
        self._all_cols = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame):
        X = self._check_input(X).copy()

        for c in self._all_cols:
            if c not in X.columns:
                X[c] = np.nan

        X[self._all_cols] = X[self._all_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X = X[self._all_cols]

        for col in self._all_cols:
            X[col] = X[col].astype(float)

        return X if self.return_dataframe else X.to_numpy(dtype=float)

    def get_feature_names_out(self):
        if self._all_cols is None:
            return None
        return list(self._all_cols)

    def _check_input(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("ZScalePreprocessor expects a pandas DataFrame.")

        X = X.copy()
        bad = [c for c in X.columns if not c.startswith(ZSCALE_PREFIX)]
        if bad:
            raise ValueError(f"Non-{ZSCALE_PREFIX} columns found: {bad}")

        for c in ZSCALE_COLS:
            if c not in X.columns:
                X[c] = np.nan

        return X