import copy
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from .charge import ChargeFeaturePreprocessor
from .hotspot import HotspotPreprocessor
from .sequence import SequencePhysicochemPreprocessor
from .zscale import ZScalePreprocessor
from .kinase import KinasePriorPreprocessor
from .motif1433 import Motif1433Preprocessor
from .nes import NESFeaturePreprocessor
from .nls import NLSFeaturePreprocessor
from .idr import IDRRegionPreprocessor
from .domain import DomainPreprocessor
from .alphamissense import AlphaMissensePreprocessor


class NumericFallbackPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, return_dataframe=False, strategy="median"):
        self.return_dataframe = return_dataframe
        self.strategy = strategy
        self._cols = None
        self._imputer = None

    def fit(self, X, y=None):
        X = self._check_input(X)
        self._cols = list(X.columns)

        if len(self._cols) == 0:
            self._imputer = None
            return self

        self._imputer = SimpleImputer(strategy=self.strategy)
        self._imputer.fit(X[self._cols])
        return self

    def transform(self, X):
        X = self._check_input(X).copy()

        if self._cols is None:
            out = pd.DataFrame(index=X.index)
            return out if self.return_dataframe else out.to_numpy(dtype=float)

        missing_cols = [c for c in self._cols if c not in X.columns]
        for c in missing_cols:
            X[c] = np.nan

        X = X[self._cols]
        X = X.apply(pd.to_numeric, errors="coerce")

        if self._imputer is not None and len(self._cols) > 0:
            X = pd.DataFrame(
                self._imputer.transform(X),
                columns=self._cols,
                index=X.index,
            )

        X = X.astype(float)

        if self.return_dataframe:
            return X

        return X.to_numpy(dtype=float)

    def get_feature_names_out(self):
        if self._cols is None:
            return None
        return list(self._cols)

    def _check_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("NumericFallbackPreprocessor expects a DataFrame.")
        return X.copy()


class UnifiedFeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        families,
        return_dataframe=False,
        add_namespace=True,
        fixed_columns_by_family=None,
        use_structure_fallback=True,
        use_evolution_fallback=True,
        structure_strategy="median",
        evolution_strategy="median",
        charge_kwargs=None,
        hotspot_kwargs=None,
        sequence_kwargs=None,
        zscale_kwargs=None,
        kinase_kwargs=None,
        motif1433_kwargs=None,
        nes_kwargs=None,
        nls_kwargs=None,
        idr_kwargs=None,
        domain_kwargs=None,
        alphamissense_kwargs=None,
    ):
        self.families = list(families)
        self.return_dataframe = return_dataframe
        self.add_namespace = add_namespace
        self.fixed_columns_by_family = fixed_columns_by_family
        self.use_structure_fallback = use_structure_fallback
        self.use_evolution_fallback = use_evolution_fallback
        self.structure_strategy = structure_strategy
        self.evolution_strategy = evolution_strategy

        self.charge_kwargs = charge_kwargs or {}
        self.hotspot_kwargs = hotspot_kwargs or {}
        self.sequence_kwargs = sequence_kwargs or {}
        self.zscale_kwargs = zscale_kwargs or {}
        self.kinase_kwargs = kinase_kwargs or {}
        self.motif1433_kwargs = motif1433_kwargs or {}
        self.nes_kwargs = nes_kwargs or {}
        self.nls_kwargs = nls_kwargs or {}
        self.idr_kwargs = idr_kwargs or {}
        self.domain_kwargs = domain_kwargs or {}
        self.alphamissense_kwargs = alphamissense_kwargs or {}

        self._transformers = None
        self._raw_columns_by_family = None
        self._output_columns_by_family = None
        self.feature_info = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        X = self._check_input(X)

        self._transformers = OrderedDict()
        self._raw_columns_by_family = OrderedDict()
        self._output_columns_by_family = OrderedDict()
        self.feature_info = OrderedDict()

        all_output_cols = []

        for family_name in self.families:
            transformer = self._build_transformer(family_name)
            raw_cols = self._select_family_columns(X, family_name)

            if len(raw_cols) == 0:
                self._raw_columns_by_family[family_name] = []
                self._output_columns_by_family[family_name] = []
                self.feature_info[family_name] = {
                    "n_cols_raw": 0,
                    "n_cols_out": 0,
                    "raw_columns": [],
                    "output_columns": [],
                    "used_fixed_feature_space": self.fixed_columns_by_family is not None,
                }
                continue

            Xt = transformer.fit_transform(X[raw_cols], y=y)
            Xt_df = self._to_dataframe(
                Xt=Xt,
                transformer=transformer,
                index=X.index,
                fallback_columns=raw_cols,
            )
            Xt_df = self._validate_frame(Xt_df, family_name=family_name, stage="fit_transform")
            Xt_df = self._apply_namespace(Xt_df, family_name)

            self._transformers[family_name] = transformer
            self._raw_columns_by_family[family_name] = list(raw_cols)
            self._output_columns_by_family[family_name] = list(Xt_df.columns)

            self.feature_info[family_name] = {
                "n_cols_raw": int(len(raw_cols)),
                "n_cols_out": int(Xt_df.shape[1]),
                "raw_columns": list(raw_cols),
                "output_columns": list(Xt_df.columns),
                "used_fixed_feature_space": self.fixed_columns_by_family is not None,
            }

            all_output_cols.extend(list(Xt_df.columns))

        self.feature_names_ = list(all_output_cols)
        return self

    def transform(self, X):
        X = self._check_input(X)
        out_frames = []

        for family_name in self.families:
            raw_cols_fit = self._raw_columns_by_family.get(family_name, [])
            out_cols_fit = self._output_columns_by_family.get(family_name, [])

            if len(raw_cols_fit) == 0:
                continue

            transformer = self._transformers[family_name]
            present_cols = [c for c in raw_cols_fit if c in X.columns]
            X_family = X[present_cols].copy()

            Xt = transformer.transform(X_family)
            Xt_df = self._to_dataframe(
                Xt=Xt,
                transformer=transformer,
                index=X.index,
                fallback_columns=raw_cols_fit,
            )
            Xt_df = self._validate_frame(Xt_df, family_name=family_name, stage="transform")
            Xt_df = self._apply_namespace(Xt_df, family_name)

            missing_out = [c for c in out_cols_fit if c not in Xt_df.columns]
            if missing_out:
                raise ValueError(
                    f"Missing transformed columns for family={family_name}: {missing_out[:20]}"
                )

            Xt_df = Xt_df[out_cols_fit]
            out_frames.append(Xt_df)

        if len(out_frames) == 0:
            out = pd.DataFrame(index=X.index)
            return out if self.return_dataframe else out.to_numpy(dtype=float)

        out = pd.concat(out_frames, axis=1)
        out = self._validate_frame(out, family_name="ALL_FAMILIES", stage="concat")

        missing_global = [c for c in self.feature_names_ if c not in out.columns]
        if missing_global:
            raise ValueError(f"Missing global transformed columns: {missing_global[:20]}")

        out = out[self.feature_names_]

        if self.return_dataframe:
            return out

        return out.to_numpy(dtype=float)

    def fit_transform(self, X, y=None):
        return self.fit(X, y=y).transform(X)

    def get_feature_names_out(self):
        if self.feature_names_ is None:
            return None
        return list(self.feature_names_)

    def _build_transformer(self, family_name):
        if family_name == "charge":
            return ChargeFeaturePreprocessor(
                return_dataframe=True,
                **copy.deepcopy(self.charge_kwargs),
            )

        if family_name == "sequence":
            return SequencePhysicochemPreprocessor(
                return_dataframe=True,
                **copy.deepcopy(self.sequence_kwargs),
            )

        if family_name == "zscale":
            return ZScalePreprocessor(
                return_dataframe=True,
                **copy.deepcopy(self.zscale_kwargs),
            )

        if family_name == "hotspot":
            return HotspotPreprocessor(
                return_dataframe=True,
                **copy.deepcopy(self.hotspot_kwargs),
            )

        if family_name == "structure":
            if not self.use_structure_fallback:
                raise ValueError("structure family requested but use_structure_fallback=False")
            return NumericFallbackPreprocessor(
                return_dataframe=True,
                strategy=self.structure_strategy,
            )

        if family_name == "kinase":
            return KinasePriorPreprocessor(
                return_dataframe=True,
                **copy.deepcopy(self.kinase_kwargs),
            )

        if family_name == "evolution":
            if not self.use_evolution_fallback:
                raise ValueError("evolution family requested but use_evolution_fallback=False")
            return NumericFallbackPreprocessor(
                return_dataframe=True,
                strategy=self.evolution_strategy,
            )

        if family_name == "alphamissense":
            return AlphaMissensePreprocessor(
                return_dataframe=True,
                **copy.deepcopy(self.alphamissense_kwargs),
            )

        if family_name == "nls":
            return NLSFeaturePreprocessor(
                return_dataframe=True,
                **copy.deepcopy(self.nls_kwargs),
            )

        if family_name == "nes":
            return NESFeaturePreprocessor(
                return_dataframe=True,
                **copy.deepcopy(self.nes_kwargs),
            )

        if family_name == "motif1433":
            return Motif1433Preprocessor(
                return_dataframe=True,
                **copy.deepcopy(self.motif1433_kwargs),
            )

        if family_name == "idr":
            return IDRRegionPreprocessor(
                return_dataframe=True,
                **copy.deepcopy(self.idr_kwargs),
            )

        if family_name == "domain":
            return DomainPreprocessor(
                return_dataframe=True,
                **copy.deepcopy(self.domain_kwargs),
            )

        raise ValueError(f"Unknown family: {family_name}")

    def _select_family_columns(self, X, family_name):
        cols = []

        if family_name == "charge":
            cols = [c for c in X.columns if c.startswith("SEQ_Charge_")]

        elif family_name == "sequence":
            cols = [
                c for c in X.columns
                if c.startswith("SEQ_")
                and not c.startswith("SEQ_Charge_")
                and not c.startswith("SEQ_Hotspot_")
                and not c.startswith("SEQ_ZScale_")
            ]

        elif family_name == "zscale":
            cols = [c for c in X.columns if c.startswith("SEQ_ZScale_")]

        elif family_name == "hotspot":
            cols = [c for c in X.columns if c.startswith("SEQ_Hotspot_")]

        elif family_name == "structure":
            cols = [c for c in X.columns if c.startswith("STRUCT_")]

        elif family_name == "kinase":
            cols = [c for c in X.columns if c.startswith("FUNC_Kinase_")]

        elif family_name == "evolution":
            cols = [c for c in X.columns if c.startswith("FUNC_Evolution_")]

        elif family_name == "nls":
            cols = [c for c in X.columns if c.startswith("MOTIF_NLS_")]

        elif family_name == "nes":
            cols = [c for c in X.columns if c.startswith("MOTIF_NES_")]

        elif family_name == "motif1433":
            cols = [c for c in X.columns if c.startswith("MOTIF_1433_")]

        elif family_name == "idr":
            cols = [c for c in X.columns if c.startswith("MOTIF_IDR_")]

        elif family_name == "domain":
            cols = [c for c in X.columns if c.startswith("MOTIF_Domain_")]

        if self.fixed_columns_by_family is not None:
            keep = self.fixed_columns_by_family.get(family_name, [])
            keep = set(keep)
            cols = [c for c in cols if c in keep]

        return list(cols)

    def _to_dataframe(self, Xt, transformer, index, fallback_columns):
        if isinstance(Xt, pd.DataFrame):
            df = Xt.copy()
            df.index = index
            return df

        cols = None

        if hasattr(transformer, "get_feature_names_out"):
            cols = transformer.get_feature_names_out()

        if cols is None and hasattr(transformer, "feature_names_"):
            cols = getattr(transformer, "feature_names_")

        if cols is None:
            n_cols = Xt.shape[1] if Xt.ndim == 2 else 1
            cols = list(fallback_columns[:n_cols])

        return pd.DataFrame(Xt, index=index, columns=list(cols))

    def _apply_namespace(self, X, family_name):
        if not self.add_namespace:
            return X
        X = X.copy()
        X.columns = [f"{family_name}::{c}" for c in X.columns]
        return X

    def _validate_frame(self, X, family_name, stage):
        X = X.copy()

        non_numeric_cols = []
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            if not pd.api.types.is_numeric_dtype(X[col]):
                non_numeric_cols.append(col)

        if non_numeric_cols:
            raise ValueError(
                f"Non-numeric columns remain after preprocessing. "
                f"family={family_name}, stage={stage}, cols={non_numeric_cols[:20]}"
            )

        nan_cols = X.columns[X.isna().any()].tolist()
        if nan_cols:
            nan_count_by_col = X[nan_cols].isna().sum().sort_values(ascending=False)
            raise ValueError(
                f"NaN detected after preprocessing. "
                f"family={family_name}, stage={stage}, "
                f"sample_cols={nan_count_by_col.head(20).to_dict()}"
            )

        inf_mask = ~np.isfinite(X.to_numpy(dtype=float))
        if inf_mask.any():
            bad_cols = X.columns[np.any(inf_mask, axis=0)].tolist()
            raise ValueError(
                f"Inf detected after preprocessing. "
                f"family={family_name}, stage={stage}, cols={bad_cols[:20]}"
            )

        return X.astype(float)

    def _check_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("UnifiedFeaturePreprocessor expects a pandas DataFrame.")
        return X.copy()