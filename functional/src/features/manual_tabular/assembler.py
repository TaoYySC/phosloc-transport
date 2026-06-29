import copy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.features.manual_tabular.family_registry import FAMILY_REGISTRY


class TabularFeatureAssembler(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        feature_families,
        return_dataframe=False,
        fixed_retained_columns_by_family=None,
        missing_rate_threshold=0.5,
        charge_kwargs=None,
        sequence_kwargs=None,
        zscale_kwargs=None,
        hotspot_kwargs=None,
        kinase_kwargs=None,
        evolution_kwargs=None,
        alphamissense_kwargs=None,
        nls_kwargs=None,
        nes_kwargs=None,
        motif1433_kwargs=None,
        idr_kwargs=None,
        domain_kwargs=None,
    ):
        self.feature_families = list(feature_families)
        self.return_dataframe = return_dataframe
        self.fixed_retained_columns_by_family = fixed_retained_columns_by_family
        self.missing_rate_threshold = missing_rate_threshold

        self.charge_kwargs = charge_kwargs or {}
        self.sequence_kwargs = sequence_kwargs or {}
        self.zscale_kwargs = zscale_kwargs or {}
        self.hotspot_kwargs = hotspot_kwargs or {}
        self.kinase_kwargs = kinase_kwargs or {}
        self.evolution_kwargs = evolution_kwargs or {}
        self.alphamissense_kwargs = alphamissense_kwargs or {}
        self.nls_kwargs = nls_kwargs or {}
        self.nes_kwargs = nes_kwargs or {}
        self.motif1433_kwargs = motif1433_kwargs or {}
        self.idr_kwargs = idr_kwargs or {}
        self.domain_kwargs = domain_kwargs or {}

        self.preprocessors = {}
        self.selected_raw_cols = {}
        self.dropped_raw_cols = {}
        self.output_feature_names_by_family = {}
        self.feature_info = {}
        self.output_feature_names_ = []
        self.is_fitted_ = False

    def _filter_by_missing_rate(self, df, cols):
        if len(cols) == 0:
            return [], []

        miss_rate = df[cols].isna().mean()
        keep_cols = miss_rate[miss_rate <= self.missing_rate_threshold].index.tolist()
        drop_cols = miss_rate[miss_rate > self.missing_rate_threshold].index.tolist()

        if len(drop_cols) > 0:
            sample = [(c, round(float(miss_rate[c]), 4)) for c in drop_cols[:10]]
            print(
                f"[INFO] dropped {len(drop_cols)} columns with missing rate > "
                f"{self.missing_rate_threshold}: {sample}"
            )

        return keep_cols, drop_cols

    def fit(self, df_train, y=None):
        self.preprocessors = {}
        self.selected_raw_cols = {}
        self.dropped_raw_cols = {}
        self.output_feature_names_by_family = {}
        self.feature_info = {}
        self.output_feature_names_ = []

        for family in self.feature_families:
            if family not in FAMILY_REGISTRY:
                raise ValueError(f"Unknown family: {family}")

            spec = FAMILY_REGISTRY[family]
            raw_cols = list(spec["selector"](df_train))

            if self.fixed_retained_columns_by_family is not None:
                keep = set(self.fixed_retained_columns_by_family.get(family, []))
                raw_cols = [c for c in raw_cols if c in keep]

            raw_cols, dropped_cols = self._filter_by_missing_rate(df_train, raw_cols)

            if len(raw_cols) == 0:
                self.feature_info[family] = {
                    "n_cols_raw": 0,
                    "n_cols_out": 0,
                    "raw_columns": [],
                    "dropped_raw_columns": dropped_cols,
                    "output_columns": [],
                    "used_fixed_feature_space": self.fixed_retained_columns_by_family is not None,
                    "missing_rate_threshold": float(self.missing_rate_threshold),
                }
                self.dropped_raw_cols[family] = dropped_cols
                continue

            kwargs_key = spec["kwargs_key"]
            kwargs = copy.deepcopy(getattr(self, kwargs_key))
            preprocessor = spec["preprocessor_cls"](return_dataframe=True, **kwargs)

            X_family = df_train[raw_cols].copy()
            Xt = preprocessor.fit_transform(X_family, y=y)

            if not isinstance(Xt, pd.DataFrame):
                cols = preprocessor.get_feature_names_out()
                Xt = pd.DataFrame(Xt, index=df_train.index, columns=cols)

            Xt = self._ensure_numeric(Xt, family_name=family, stage="fit_transform")
            output_cols = [f"{family}::{c}" for c in Xt.columns]

            self.preprocessors[family] = preprocessor
            self.selected_raw_cols[family] = list(raw_cols)
            self.dropped_raw_cols[family] = list(dropped_cols)
            self.output_feature_names_by_family[family] = list(output_cols)

            self.feature_info[family] = {
                "n_cols_raw": int(len(raw_cols)),
                "n_cols_out": int(Xt.shape[1]),
                "raw_columns": list(raw_cols),
                "dropped_raw_columns": list(dropped_cols),
                "output_columns": list(output_cols),
                "used_fixed_feature_space": self.fixed_retained_columns_by_family is not None,
                "missing_rate_threshold": float(self.missing_rate_threshold),
            }

            self.output_feature_names_.extend(output_cols)

        self.is_fitted_ = True
        return self

    def transform(self, df):
        if not self.is_fitted_:
            raise RuntimeError("TabularFeatureAssembler has not been fitted.")

        outputs = []

        for family in self.feature_families:
            if family not in self.preprocessors:
                continue

            raw_cols = self.selected_raw_cols[family]
            preprocessor = self.preprocessors[family]

            X_family = df.copy()
            for c in raw_cols:
                if c not in X_family.columns:
                    X_family[c] = np.nan
            X_family = X_family[raw_cols].copy()

            Xt = preprocessor.transform(X_family)

            if not isinstance(Xt, pd.DataFrame):
                cols = preprocessor.get_feature_names_out()
                Xt = pd.DataFrame(Xt, index=df.index, columns=cols)

            Xt = self._ensure_numeric(Xt, family_name=family, stage="transform")
            Xt.columns = self.output_feature_names_by_family[family]
            Xt = Xt[self.output_feature_names_by_family[family]]
            outputs.append(Xt)

        if len(outputs) == 0:
            out = pd.DataFrame(index=df.index)
            return out if self.return_dataframe else out.to_numpy(dtype=float)

        out = pd.concat(outputs, axis=1)

        if self.return_dataframe:
            return out

        return out.to_numpy(dtype=float)

    def fit_transform(self, df_train, y=None):
        self.fit(df_train, y=y)
        return self.transform(df_train)

    def get_feature_names_out(self):
        return list(self.output_feature_names_)

    def _ensure_numeric(self, X, family_name, stage):
        X = X.copy()

        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        if X.isna().any().any():
            bad_cols = X.columns[X.isna().any()].tolist()
            raise ValueError(
                f"NaN detected after preprocessing. "
                f"family={family_name}, stage={stage}, cols={bad_cols[:20]}"
            )

        arr = X.to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError(
                f"Inf detected after preprocessing. family={family_name}, stage={stage}"
            )

        return X.astype(float)