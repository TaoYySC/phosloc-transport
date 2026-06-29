"""
File location:
src/preprocess/hotspot.py
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


HOTSPOT_PREFIX = "SEQ_Hotspot_"


class HotspotPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe: bool = False,
        add_missing_indicator: bool = False,
        log_transform_dist: bool = True,
    ):
        self.return_dataframe = return_dataframe
        self.add_missing_indicator = add_missing_indicator
        self.log_transform_dist = log_transform_dist

        self._feature_cols = None
        self._imputer = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X).copy()

        self._feature_cols = list(X.columns)

        if len(self._feature_cols) == 0:
            self._imputer = None
            return self

        self._imputer = SimpleImputer(strategy="median")
        self._imputer.fit(X)

        return self

    def transform(self, X: pd.DataFrame):
        X = self._check_input(X).copy()

        if self._feature_cols is None or len(self._feature_cols) == 0:
            out = pd.DataFrame(index=X.index)
            return out if self.return_dataframe else out.to_numpy(dtype=float)

        missing_cols = [c for c in self._feature_cols if c not in X.columns]
        for c in missing_cols:
            X[c] = np.nan

        X = X[self._feature_cols]

        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        if self.log_transform_dist:
            dist_cols = [
                c for c in X.columns
                if "distance" in c.lower() or "dist" in c.lower()
            ]
            for c in dist_cols:
                X[c] = np.log1p(np.clip(X[c].to_numpy(dtype=float), 0.0, None))

        X_imp = self._imputer.transform(X)
        X_imp = pd.DataFrame(
            X_imp,
            columns=self._feature_cols,
            index=X.index,
        )

        if self.add_missing_indicator:
            for col in self._feature_cols:
                X_imp[f"{col}_missing"] = X[col].isna().astype(int)

        if self.return_dataframe:
            return X_imp

        return X_imp.values.astype(float)

    def get_feature_names_out(self):
        return list(self._feature_cols) if self._feature_cols is not None else None

    def _check_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HotspotPreprocessor expects a DataFrame")

        X = X.copy()

        bad_cols = [c for c in X.columns if not c.startswith(HOTSPOT_PREFIX)]
        if bad_cols:
            raise ValueError(f"Non-{HOTSPOT_PREFIX} columns detected: {bad_cols}")

        return X