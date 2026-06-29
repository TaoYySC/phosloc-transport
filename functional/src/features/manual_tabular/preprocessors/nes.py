"""
File location:
/mnt/Storage2/home/yangtao/phosloc-PLR/src/preprocess/nes.py
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional


NES_PREFIX = "MOTIF_NES_"

NES_BOOL_COLS = [
    "MOTIF_NES_HasSegment_Flag",
    "MOTIF_NES_Inside_Flag",
    "MOTIF_NES_Within20AA_Flag",
    "MOTIF_NES_Within50AA_Flag",
]

NES_DIST_COLS = [
    "MOTIF_NES_Nearest_Start",
    "MOTIF_NES_Nearest_End",
    "MOTIF_NES_Nearest_Distance",
    "MOTIF_NES_Nearest_DistanceNorm",
    "MOTIF_NES_Nearest_CenterSignedDistance",
]

NES_INSIDE_ONLY_COLS = [
    "MOTIF_NES_Inside_RelativePosition",
    "MOTIF_NES_Inside_SegmentLength",
    "MOTIF_NES_Inside_SegmentLengthNorm",
]

NES_OTHER_FLOAT_COLS = [
    "MOTIF_NES_Protein_SegmentCount",
    "MOTIF_NES_Nearest_Score",
    "MOTIF_NES_Protein_CoverageFraction",
]

NES_CLASS_COL = "MOTIF_NES_Nearest_Class"
NES_CLASS_PREFIX = "MOTIF_NES_Class_"


class NESFeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe: bool = False,
        drop_weak_cols: bool = True,
        log_transform_dist: bool = True,
        large_value_mode: str = "p99_plus_1",
        one_hot_class: bool = True,
        class_top_k: Optional[int] = None,
        class_other_label: str = "__OTHER__",
    ):
        self.return_dataframe = return_dataframe
        self.drop_weak_cols = drop_weak_cols
        self.log_transform_dist = log_transform_dist
        self.large_value_mode = large_value_mode
        self.one_hot_class = one_hot_class
        self.class_top_k = class_top_k
        self.class_other_label = class_other_label

        self._all_cols = None
        self._keep_cols = None
        self._dist_fill_values = None
        self._class_levels = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X).copy()

        for c in NES_BOOL_COLS:
            if c in X.columns:
                X[c] = self._to_bool01(X[c].values)

        has = (
            X["MOTIF_NES_HasSegment_Flag"].to_numpy(dtype=int)
            if "MOTIF_NES_HasSegment_Flag" in X.columns
            else np.ones(len(X), dtype=int)
        )

        self._dist_fill_values = {}
        for c in NES_DIST_COLS:
            if c not in X.columns:
                continue

            v = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
            v_has = v[has == 1]
            v_has = v_has[np.isfinite(v_has)]

            if v_has.size == 0:
                fill = 1.0
            else:
                if self.large_value_mode == "max_plus_1":
                    fill = float(np.max(v_has) + 1.0)
                elif self.large_value_mode == "p99_plus_1":
                    fill = float(np.quantile(v_has, 0.99) + 1.0)
                elif self.large_value_mode == "p95_times_2":
                    fill = float(np.quantile(v_has, 0.95) * 2.0)
                else:
                    raise ValueError(f"Unknown large_value_mode: {self.large_value_mode}")

            self._dist_fill_values[c] = fill

        self._class_levels = []
        if self.one_hot_class and NES_CLASS_COL in X.columns:
            cls = X[NES_CLASS_COL].astype(str)
            cls = cls.where(cls.notna(), other=self.class_other_label)
            cls = cls.where(cls != "nan", other=self.class_other_label)
            cls = cls.where(cls != "None", other=self.class_other_label)

            if "MOTIF_NES_HasSegment_Flag" in X.columns:
                cls = cls.where(has == 1, other=self.class_other_label)

            vc = cls.value_counts()
            levels = list(vc.index)

            if self.class_top_k is not None and self.class_top_k > 0:
                levels = [c for c in levels if c != self.class_other_label]
                levels = levels[: self.class_top_k]
                levels = levels + [self.class_other_label]

            self._class_levels = levels

        self._all_cols = list(X.columns)
        self._keep_cols = self._build_keep_cols(self._all_cols)
        self.feature_names_ = list(self._keep_cols)
        return self

    def transform(self, X: pd.DataFrame):
        X = self._check_input(X).copy()

        for c in NES_BOOL_COLS:
            if c in X.columns:
                X[c] = self._to_bool01(X[c].values)

        has = (
            X["MOTIF_NES_HasSegment_Flag"].to_numpy(dtype=int)
            if "MOTIF_NES_HasSegment_Flag" in X.columns
            else np.ones(len(X), dtype=int)
        )
        inside = (
            X["MOTIF_NES_Inside_Flag"].to_numpy(dtype=int)
            if "MOTIF_NES_Inside_Flag" in X.columns
            else np.zeros(len(X), dtype=int)
        )

        for c in NES_DIST_COLS:
            if c not in X.columns:
                continue
            X[c] = pd.to_numeric(X[c], errors="coerce")

            fill = self._dist_fill_values.get(c, 1.0)
            X.loc[has == 0, c] = fill
            X[c] = X[c].fillna(fill)

            if self.log_transform_dist:
                X[c] = np.log1p(np.clip(X[c].to_numpy(dtype=float), 0.0, None))

        if "MOTIF_NES_Nearest_Score" in X.columns:
            X["MOTIF_NES_Nearest_Score"] = pd.to_numeric(X["MOTIF_NES_Nearest_Score"], errors="coerce")
            X.loc[has == 0, "MOTIF_NES_Nearest_Score"] = 0.0
            X["MOTIF_NES_Nearest_Score"] = X["MOTIF_NES_Nearest_Score"].fillna(0.0)

        if "MOTIF_NES_Protein_SegmentCount" in X.columns:
            X["MOTIF_NES_Protein_SegmentCount"] = pd.to_numeric(
                X["MOTIF_NES_Protein_SegmentCount"], errors="coerce"
            ).fillna(0.0)

        if "MOTIF_NES_Protein_CoverageFraction" in X.columns:
            X["MOTIF_NES_Protein_CoverageFraction"] = pd.to_numeric(
                X["MOTIF_NES_Protein_CoverageFraction"], errors="coerce"
            ).fillna(0.0)

        if "MOTIF_NES_Inside_RelativePosition" in X.columns:
            X["MOTIF_NES_Inside_RelativePosition"] = pd.to_numeric(
                X["MOTIF_NES_Inside_RelativePosition"], errors="coerce"
            )
            X.loc[inside == 0, "MOTIF_NES_Inside_RelativePosition"] = -1.0
            X["MOTIF_NES_Inside_RelativePosition"] = X["MOTIF_NES_Inside_RelativePosition"].fillna(-1.0)

        for c in ["MOTIF_NES_Inside_SegmentLength", "MOTIF_NES_Inside_SegmentLengthNorm"]:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce")
                X.loc[inside == 0, c] = 0.0
                X[c] = X[c].fillna(0.0)

        if self.one_hot_class and NES_CLASS_COL in X.columns:
            cls = X[NES_CLASS_COL].astype(str)
            cls = cls.where(cls.notna(), other=self.class_other_label)
            cls = cls.where(cls != "nan", other=self.class_other_label)
            cls = cls.where(cls != "None", other=self.class_other_label)
            cls = cls.where(has == 1, other=self.class_other_label)

            allowed = set(self._class_levels or [])
            if allowed:
                cls = cls.where(cls.isin(allowed), other=self.class_other_label)

            for lv in (self._class_levels or []):
                col = f"{NES_CLASS_PREFIX}{lv}"
                X[col] = (cls == lv).astype(int)

            X = X.drop(columns=[NES_CLASS_COL])

        if self._keep_cols is not None:
            missing_cols = [c for c in self._keep_cols if c not in X.columns]
            for c in missing_cols:
                if c.startswith(NES_CLASS_PREFIX):
                    X[c] = 0
                else:
                    X[c] = 0.0
            X = X[self._keep_cols]

        if self.return_dataframe:
            return X

        return X.to_numpy(dtype=float)

    def get_feature_names_out(self):
        if self.feature_names_ is None:
            return None
        return list(self.feature_names_)

    def _build_keep_cols(self, cols):
        cols = list(cols)

        if self.one_hot_class and NES_CLASS_COL in cols:
            cols = [c for c in cols if c != NES_CLASS_COL]
            for lv in (self._class_levels or []):
                cols.append(f"{NES_CLASS_PREFIX}{lv}")

        if not self.drop_weak_cols:
            return cols

        weak = {
            "MOTIF_NES_Nearest_Score",
            "MOTIF_NES_Protein_CoverageFraction",
        }
        return [c for c in cols if c not in weak]

    def _to_bool01(self, v):
        s = pd.Series(v)
        if s.dtype == bool:
            return s.astype(int).to_numpy()

        if s.dtype == object:
            s = s.astype(str).str.strip().str.lower()
            mapped = s.map({
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
            })
            if mapped.notna().any():
                return mapped.fillna(0).astype(int).to_numpy()

        x = pd.to_numeric(s, errors="coerce").fillna(0.0)
        return (x > 0).astype(int).to_numpy()

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("NESFeaturePreprocessor expects a DataFrame.")

        X = X.copy()

        for c in NES_BOOL_COLS:
            if c not in X.columns:
                X[c] = 0.0

        for c in NES_DIST_COLS + NES_INSIDE_ONLY_COLS + NES_OTHER_FLOAT_COLS:
            if c not in X.columns:
                X[c] = np.nan

        if self.one_hot_class and NES_CLASS_COL not in X.columns:
            X[NES_CLASS_COL] = self.class_other_label

        bad = [c for c in X.columns if not (c.startswith(NES_PREFIX) or c.startswith(NES_CLASS_PREFIX) or c == "INDEX")]
        if bad:
            raise ValueError(f"Non-NES feature columns detected: {bad}")

        return X