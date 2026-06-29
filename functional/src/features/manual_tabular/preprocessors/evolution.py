import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class EvolutionPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, return_dataframe=False, strategy="median"):
        self.return_dataframe = return_dataframe
        self.strategy = strategy
        self._cols = None
        self._imputer = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        X = self._check_input(X)

        cols = [c for c in X.columns if c.startswith("FUNC_Evolution_")]
        self._cols = list(cols)
        self.feature_names_ = list(cols)

        if len(self._cols) == 0:
            self._imputer = None
            return self

        X_num = X[self._cols].copy()
        X_num = X_num.apply(pd.to_numeric, errors="coerce")

        self._imputer = SimpleImputer(strategy=self.strategy)
        self._imputer.fit(X_num)
        return self

    def transform(self, X):
        X = self._check_input(X)

        if self._cols is None:
            out = pd.DataFrame(index=X.index)
            return out if self.return_dataframe else out.to_numpy(dtype=float)

        X_num = X.copy()

        for c in self._cols:
            if c not in X_num.columns:
                X_num[c] = np.nan

        X_num = X_num[self._cols].copy()
        X_num = X_num.apply(pd.to_numeric, errors="coerce")

        if self._imputer is not None and len(self._cols) > 0:
            X_num = pd.DataFrame(
                self._imputer.transform(X_num),
                columns=self._cols,
                index=X.index,
            )

        X_num = X_num.astype(float)

        if self.return_dataframe:
            return X_num

        return X_num.to_numpy(dtype=float)

    def get_feature_names_out(self):
        if self.feature_names_ is None:
            return None
        return list(self.feature_names_)

    def _check_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("EvolutionPreprocessor expects a pandas DataFrame.")
        return X.copy()