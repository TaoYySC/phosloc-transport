import copy

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

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
        pvalue_filter=False,
        pvalue_threshold=0.05,
        auroc_top_k_filter=False,
        auroc_top_k=40,
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
        self.pvalue_filter = pvalue_filter
        self.pvalue_threshold = pvalue_threshold
        self.auroc_top_k_filter = bool(auroc_top_k_filter)
        self.auroc_top_k = int(auroc_top_k) if auroc_top_k is not None else None
        self.selected_output_cols_ = []
        self.dropped_auroc_cols_ = []

    def _univariate_auroc_score(self, values, y):
        values = np.asarray(values, dtype=np.float64)
        y = np.asarray(y)
        mask = np.isfinite(values)
        if mask.sum() < 10:
            return np.nan
        y_sub = y[mask]
        x_sub = values[mask]
        if len(np.unique(y_sub)) < 2:
            return np.nan
        try:
            auroc = float(roc_auc_score(y_sub, x_sub))
        except Exception:
            return np.nan
        return max(auroc, 1.0 - auroc)

    def _filter_by_auroc_topk(self, X, y, top_k):
        if top_k is None or int(top_k) <= 0 or X.shape[1] == 0:
            return list(X.columns), []

        scores = {}
        for col in X.columns:
            score = self._univariate_auroc_score(X[col].to_numpy(), y)
            scores[col] = 0.5 if np.isnan(score) else float(score)

        ranked_cols = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        k = min(int(top_k), len(ranked_cols))
        keep_cols = ranked_cols[:k]
        drop_cols = ranked_cols[k:]
        print(
            f"[INFO] AUROC top-{k} filter: kept {len(keep_cols)} / {len(ranked_cols)} "
            f"processed manual features"
        )
        return keep_cols, drop_cols

    def _filter_by_pvalue(self, df, cols, y, pvalue_threshold=0.05):
        keep_cols = []
        drop_cols = []

        y = np.asarray(y)

        for col in cols:
            x = pd.to_numeric(df[col], errors="coerce")
            x0 = x[y == 0].dropna()
            x1 = x[y == 1].dropna()

            if len(x0) < 3 or len(x1) < 3:
                drop_cols.append(col)
                continue

            try:
                _, p = mannwhitneyu(x0, x1, alternative="two-sided")
            except Exception:
                drop_cols.append(col)
                continue

            if p < pvalue_threshold:
                keep_cols.append(col)
            else:
                drop_cols.append(col)

        return keep_cols, drop_cols
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
        self.selected_output_cols_ = []
        self.dropped_auroc_cols_ = []
        family_outputs = []

        for family in self.feature_families:
            if family not in FAMILY_REGISTRY:
                raise ValueError(f"Unknown family: {family}")

            spec = FAMILY_REGISTRY[family]
            raw_cols = list(spec["selector"](df_train))

            if self.fixed_retained_columns_by_family is not None:
                keep = set(self.fixed_retained_columns_by_family.get(family, []))
                raw_cols = [c for c in raw_cols if c in keep]

            raw_cols, dropped_cols = self._filter_by_missing_rate(df_train, raw_cols)
            if self.pvalue_filter:
                raw_cols, dropped_pvalue_cols = self._filter_by_pvalue(
                    df=df_train,
                    cols=raw_cols,
                    y=y,
                    pvalue_threshold=self.pvalue_threshold,
                )
            else:
                dropped_pvalue_cols = []
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
            Xt = Xt.copy()
            Xt.columns = output_cols

            self.preprocessors[family] = preprocessor
            self.selected_raw_cols[family] = list(raw_cols)
            self.dropped_raw_cols[family] = list(dropped_cols)
            self.output_feature_names_by_family[family] = list(output_cols)

            self.feature_info[family] = {
                "n_cols_raw": int(len(raw_cols)),
                "n_cols_out": int(Xt.shape[1]),
                "raw_columns": list(raw_cols),
                "dropped_raw_columns": list(dropped_cols),
                "dropped_pvalue_columns": list(dropped_pvalue_cols),
                "output_columns": list(output_cols),
                "used_fixed_feature_space": self.fixed_retained_columns_by_family is not None,
                "missing_rate_threshold": float(self.missing_rate_threshold),
            }

            family_outputs.append(Xt)

        if family_outputs:
            X_all = pd.concat(family_outputs, axis=1)
        else:
            X_all = pd.DataFrame(index=df_train.index)

        if self.auroc_top_k_filter and y is not None and X_all.shape[1] > 0:
            keep_cols, self.dropped_auroc_cols_ = self._filter_by_auroc_topk(
                X_all,
                y,
                self.auroc_top_k,
            )
            self.selected_output_cols_ = list(keep_cols)
        else:
            self.selected_output_cols_ = list(X_all.columns)
            self.dropped_auroc_cols_ = []

        self.output_feature_names_ = list(self.selected_output_cols_)
        self.feature_info["_global"] = {
            "auroc_top_k_filter": self.auroc_top_k_filter,
            "auroc_top_k": self.auroc_top_k,
            "n_cols_before_auroc_filter": int(X_all.shape[1]),
            "n_cols_after_auroc_filter": int(len(self.selected_output_cols_)),
            "dropped_auroc_columns": list(self.dropped_auroc_cols_),
        }

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
        if self.selected_output_cols_:
            missing = [c for c in self.selected_output_cols_ if c not in out.columns]
            if missing:
                raise RuntimeError(
                    "Selected manual features missing at transform: "
                    f"{missing[:10]}"
                )
            out = out[self.selected_output_cols_]

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