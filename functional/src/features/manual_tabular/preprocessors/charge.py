import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


CHARGE_COLUMNS = [
    "SEQ_Charge_Window_PositiveCount",
    "SEQ_Charge_Window_NegativeCount",
    "SEQ_Charge_Window_ChargeSum",
    "SEQ_Charge_Window_PositiveFraction",
    "SEQ_Charge_Window_NegativeFraction",
    "SEQ_Charge_Site_PreWindow_NetCharge",
    "SEQ_Charge_Site_PostWindow_NetCharge",
    "SEQ_Charge_Site_Delta_NetCharge",
    "SEQ_Charge_Window_KRCluster_Score",
    "SEQ_Charge_Window_KRCluster_Density",
]

CHARGE_PREFIX = "SEQ_Charge_"


class ChargeFeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe=False,
        impute_strategy="median",
        standardize=False,
        clip_fraction_range=True,
    ):
        self.return_dataframe = return_dataframe
        self.impute_strategy = impute_strategy
        self.standardize = standardize
        self.clip_fraction_range = clip_fraction_range

        self._cols = None
        self._imputer = None
        self._scaler = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        X = self._check_input(X)

        cols = [c for c in CHARGE_COLUMNS if c in X.columns]
        if len(cols) == 0:
            cols = [c for c in X.columns if c.startswith(CHARGE_PREFIX)]

        self._cols = list(cols)
        self.feature_names_ = list(cols)

        if len(self._cols) == 0:
            self._imputer = None
            self._scaler = None
            return self

        X_num = self._prepare_numeric_frame(X, self._cols)

        self._imputer = SimpleImputer(strategy=self.impute_strategy)
        X_imp = pd.DataFrame(
            self._imputer.fit_transform(X_num),
            columns=self._cols,
            index=X.index,
        )

        if self.standardize:
            self._scaler = StandardScaler()
            self._scaler.fit(X_imp)
        else:
            self._scaler = None

        return self

    def transform(self, X):
        X = self._check_input(X)

        if self._cols is None:
            out = pd.DataFrame(index=X.index)
            return out if self.return_dataframe else out.to_numpy(dtype=float)

        X_num = self._prepare_numeric_frame(X, self._cols)

        if self._imputer is not None and len(self._cols) > 0:
            X_num = pd.DataFrame(
                self._imputer.transform(X_num),
                columns=self._cols,
                index=X.index,
            )

        if self.clip_fraction_range:
            frac_cols = [
                "SEQ_Charge_Window_PositiveFraction",
                "SEQ_Charge_Window_NegativeFraction",
                "SEQ_Charge_Window_KRCluster_Density",
            ]
            for col in frac_cols:
                if col in X_num.columns:
                    X_num[col] = X_num[col].clip(lower=0.0, upper=1.0)

        if self._scaler is not None:
            X_num = pd.DataFrame(
                self._scaler.transform(X_num),
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

    def _prepare_numeric_frame(self, X, cols):
        X_num = X.copy()

        for c in cols:
            if c not in X_num.columns:
                X_num[c] = np.nan

        X_num = X_num[cols].copy()
        X_num = X_num.apply(pd.to_numeric, errors="coerce")
        return X_num

    def _check_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("ChargeFeaturePreprocessor expects a pandas DataFrame.")

        X = X.copy()
        bad = [c for c in X.columns if not c.startswith(CHARGE_PREFIX)]
        if bad:
            raise ValueError(f"Non-{CHARGE_PREFIX} columns found: {bad}")

        return X