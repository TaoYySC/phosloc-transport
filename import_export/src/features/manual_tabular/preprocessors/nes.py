"""
Stable preprocessing for NES-related motif features.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


NES_PREFIX = "MOTIF_NES_"
NES_CLASS_COL = "MOTIF_NES_Nearest_Class"
NES_CLASS_PREFIX = "MOTIF_NES_Class_"

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

NES_BOUNDARY_COLS = [
    "MOTIF_NES_Nearest_Start",
    "MOTIF_NES_Nearest_End",
]

NES_SIGNED_DIST_COLS = [
    "MOTIF_NES_Nearest_CenterSignedDistance",
]

NES_UNSIGNED_DIST_COLS = [
    c for c in NES_DIST_COLS if c not in NES_SIGNED_DIST_COLS
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

NES_CANONICAL_NUMERIC_COLS = (
    NES_BOOL_COLS
    + NES_DIST_COLS
    + NES_INSIDE_ONLY_COLS
    + NES_OTHER_FLOAT_COLS
)

NES_COUNT_COLS = [
    "MOTIF_NES_Protein_SegmentCount",
]

NES_FRACTION_COLS = [
    "MOTIF_NES_Protein_CoverageFraction",
    "MOTIF_NES_Nearest_DistanceNorm",
    "MOTIF_NES_Inside_SegmentLengthNorm",
]

NES_SCORE_COLS = [
    "MOTIF_NES_Nearest_Score",
]

NES_RELATIVE_POSITION_COLS = [
    "MOTIF_NES_Inside_RelativePosition",
]


class NESFeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe: bool = False,
        drop_weak_cols: bool = False,
        drop_boundary_cols: bool = True,
        log_transform_dist: bool = True,
        large_value_mode: str = "p99_plus_1",
        one_hot_class: bool = True,
        class_top_k: Optional[int] = None,
        class_other_label: str = "__OTHER__",
        add_missing_indicator: bool = False,
        keep_extra_nes_cols: bool = False,
        clip_physical_ranges: bool = True,
        fixed_schema: bool = True,
    ):
        self.return_dataframe = return_dataframe
        self.drop_weak_cols = drop_weak_cols
        self.drop_boundary_cols = drop_boundary_cols
        self.log_transform_dist = log_transform_dist
        self.large_value_mode = large_value_mode
        self.one_hot_class = one_hot_class
        self.class_top_k = class_top_k
        self.class_other_label = class_other_label
        self.add_missing_indicator = add_missing_indicator
        self.keep_extra_nes_cols = keep_extra_nes_cols
        self.clip_physical_ranges = clip_physical_ranges
        self.fixed_schema = fixed_schema

        self._dist_fill_values: Optional[Dict[str, float]] = None
        self._score_fill_values: Optional[Dict[str, float]] = None
        self._missing_rate_: Optional[Dict[str, float]] = None
        self._class_levels: Optional[List[str]] = None
        self._class_safe_map: Optional[Dict[str, str]] = None
        self._keep_cols: Optional[List[str]] = None
        self._output_feature_names: Optional[List[str]] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X).copy()
        X = self._drop_index(X)
        X = self._ensure_expected_columns(X)

        for c in NES_BOOL_COLS:
            X[c] = self._to_bool01(X[c].values)

        has = X["MOTIF_NES_HasSegment_Flag"].to_numpy(dtype=int)

        self._dist_fill_values = {}
        for c in NES_DIST_COLS:
            s = pd.to_numeric(X[c], errors="coerce")
            v = s.to_numpy(dtype=float)
            v_has = v[has == 1]
            v_has = v_has[np.isfinite(v_has)]

            if c in NES_SIGNED_DIST_COLS:
                fill = float(np.median(v_has)) if v_has.size else 0.0
            else:
                fill = self._compute_large_fill(v_has)

            self._dist_fill_values[c] = fill

        self._score_fill_values = {}
        for c in NES_SCORE_COLS:
            s = pd.to_numeric(X[c], errors="coerce")
            v = s.to_numpy(dtype=float)
            v_has = v[has == 1]
            v_has = v_has[np.isfinite(v_has)]
            self._score_fill_values[c] = float(np.median(v_has)) if v_has.size else 0.0

        self._class_levels = []
        self._class_safe_map = {}
        if self.one_hot_class:
            cls = self._prepare_class_series(X[NES_CLASS_COL], has)
            vc = cls.value_counts(dropna=False)
            levels = list(vc.index.astype(str))

            if self.class_top_k is not None and self.class_top_k > 0:
                levels = [lv for lv in levels if lv != self.class_other_label]
                levels = levels[: self.class_top_k]
                levels = levels + [self.class_other_label]

            if self.class_other_label not in levels:
                levels.append(self.class_other_label)

            self._class_levels = levels
            self._class_safe_map = self._build_class_safe_map(levels)

        self._missing_rate_ = {}
        numeric_source_cols = self._numeric_source_cols(X)
        for c in numeric_source_cols:
            self._missing_rate_[c] = float(pd.to_numeric(X[c], errors="coerce").isna().mean())

        self._keep_cols = self._build_keep_cols(X.columns)
        self._output_feature_names = list(self._keep_cols)

        if self.add_missing_indicator:
            self._output_feature_names.extend([f"{c}__missing" for c in self._keep_cols if not c.startswith(NES_CLASS_PREFIX)])

        self.feature_names_ = list(self._output_feature_names)
        return self

    def transform(self, X: pd.DataFrame):
        if self._keep_cols is None:
            raise RuntimeError("NESFeaturePreprocessor must be fitted before transform.")

        X = self._check_input(X).copy()
        X = self._drop_index(X)
        X = self._ensure_expected_columns(X)

        for c in NES_BOOL_COLS:
            X[c] = self._to_bool01(X[c].values)

        has = X["MOTIF_NES_HasSegment_Flag"].to_numpy(dtype=int)
        inside = X["MOTIF_NES_Inside_Flag"].to_numpy(dtype=int)

        missing_indicators = pd.DataFrame(index=X.index)
        if self.add_missing_indicator:
            for c in self._keep_cols:
                if c in X.columns and not c.startswith(NES_CLASS_PREFIX):
                    missing_indicators[f"{c}__missing"] = pd.to_numeric(X[c], errors="coerce").isna().astype(float)

        X = self._transform_distances(X, has)
        X = self._transform_other_float_features(X, has)
        X = self._transform_inside_only_features(X, inside)
        X = self._transform_class_features(X, has)
        X = self._transform_extra_features(X)

        for c in self._keep_cols:
            if c not in X.columns:
                X[c] = 0.0

        X = X[self._keep_cols].copy()

        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0).astype(float)

        if self.add_missing_indicator:
            for c in self._keep_cols:
                miss_col = f"{c}__missing"
                if miss_col not in missing_indicators.columns:
                    missing_indicators[miss_col] = 0.0
            X = pd.concat([X, missing_indicators[[f"{c}__missing" for c in self._keep_cols]]], axis=1)

        if self.return_dataframe:
            return X

        return X.to_numpy(dtype=float)

    def get_feature_names_out(self):
        if self._output_feature_names is None:
            return None
        return list(self._output_feature_names)

    def get_impute_summary(self) -> pd.DataFrame:
        if self._dist_fill_values is None or self._score_fill_values is None:
            raise RuntimeError("NESFeaturePreprocessor must be fitted before calling get_impute_summary().")

        rows = []
        missing_rate = self._missing_rate_ or {}
        for c in NES_DIST_COLS:
            rows.append(
                {
                    "feature_name": c,
                    "feature_group": "distance",
                    "fill_value": self._dist_fill_values.get(c, 0.0),
                    "missing_rate": missing_rate.get(c, np.nan),
                    "kept": c in (self._keep_cols or []),
                }
            )
        for c in NES_SCORE_COLS:
            rows.append(
                {
                    "feature_name": c,
                    "feature_group": "score",
                    "fill_value": self._score_fill_values.get(c, 0.0),
                    "missing_rate": missing_rate.get(c, np.nan),
                    "kept": c in (self._keep_cols or []),
                }
            )
        return pd.DataFrame(rows)

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("NESFeaturePreprocessor expects a pandas DataFrame.")

        bad = [
            c for c in X.columns
            if c != "INDEX" and not c.startswith(NES_PREFIX) and not c.startswith(NES_CLASS_PREFIX)
        ]
        if bad:
            raise ValueError(f"Non-NES feature columns detected: {bad}")

        return X.copy()

    def _drop_index(self, X: pd.DataFrame) -> pd.DataFrame:
        if "INDEX" in X.columns:
            X = X.drop(columns=["INDEX"])
        return X

    def _ensure_expected_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for c in NES_BOOL_COLS:
            if c not in X.columns:
                X[c] = 0.0

        for c in NES_DIST_COLS + NES_INSIDE_ONLY_COLS + NES_OTHER_FLOAT_COLS:
            if c not in X.columns:
                X[c] = np.nan

        if NES_CLASS_COL not in X.columns:
            X[NES_CLASS_COL] = self.class_other_label

        if self.fixed_schema:
            base_cols = list(NES_CANONICAL_NUMERIC_COLS) + [NES_CLASS_COL]
            extra_cols = self._extra_nes_cols(X.columns) if self.keep_extra_nes_cols else []
            X = X[base_cols + extra_cols]

        return X

    def _numeric_source_cols(self, X: pd.DataFrame) -> List[str]:
        cols = [c for c in X.columns if c != NES_CLASS_COL and not c.startswith(NES_CLASS_PREFIX)]
        return list(cols)

    def _extra_nes_cols(self, cols) -> List[str]:
        canonical = set(NES_CANONICAL_NUMERIC_COLS + [NES_CLASS_COL])
        extra = [
            c for c in cols
            if c.startswith(NES_PREFIX) and c not in canonical and not c.startswith(NES_CLASS_PREFIX)
        ]
        return sorted(extra)

    def _build_keep_cols(self, cols) -> List[str]:
        if self.fixed_schema:
            keep = list(NES_CANONICAL_NUMERIC_COLS)
        else:
            keep = [c for c in cols if c != NES_CLASS_COL and not c.startswith(NES_CLASS_PREFIX)]

        if self.keep_extra_nes_cols:
            for c in self._extra_nes_cols(cols):
                if c not in keep:
                    keep.append(c)

        if self.drop_boundary_cols:
            keep = [c for c in keep if c not in NES_BOUNDARY_COLS]

        if self.drop_weak_cols:
            weak = {
                "MOTIF_NES_Nearest_Score",
                "MOTIF_NES_Protein_CoverageFraction",
            }
            keep = [c for c in keep if c not in weak]

        if self.one_hot_class:
            for lv in self._class_levels or []:
                safe = self._class_safe_map.get(lv, self._sanitize_class_level(lv)) if self._class_safe_map else self._sanitize_class_level(lv)
                keep.append(f"{NES_CLASS_PREFIX}{safe}")

        return keep

    def _transform_distances(self, X: pd.DataFrame, has: np.ndarray) -> pd.DataFrame:
        X = X.copy()
        dist_fills = self._dist_fill_values or {}

        for c in NES_DIST_COLS:
            if c not in X.columns:
                continue

            s = pd.to_numeric(X[c], errors="coerce")
            fill = dist_fills.get(c, 0.0 if c in NES_SIGNED_DIST_COLS else 1.0)

            if c in NES_SIGNED_DIST_COLS:
                s.loc[has == 0] = 0.0
            else:
                s.loc[has == 0] = fill

            s = s.fillna(fill)

            if self.clip_physical_ranges and c in NES_UNSIGNED_DIST_COLS:
                s = s.clip(lower=0.0)

            if self.log_transform_dist:
                values = s.to_numpy(dtype=float)
                if c in NES_SIGNED_DIST_COLS:
                    values = np.sign(values) * np.log1p(np.abs(values))
                else:
                    values = np.log1p(np.clip(values, 0.0, None))
                s = pd.Series(values, index=X.index)

            X[c] = s.astype(float)

        return X

    def _transform_other_float_features(self, X: pd.DataFrame, has: np.ndarray) -> pd.DataFrame:
        X = X.copy()
        score_fills = self._score_fill_values or {}

        if "MOTIF_NES_Protein_SegmentCount" in X.columns:
            s = pd.to_numeric(X["MOTIF_NES_Protein_SegmentCount"], errors="coerce")
            s.loc[has == 0] = 0.0
            s = s.fillna(0.0)
            if self.clip_physical_ranges:
                s = s.clip(lower=0.0)
            X["MOTIF_NES_Protein_SegmentCount"] = s.astype(float)

        if "MOTIF_NES_Protein_CoverageFraction" in X.columns:
            s = pd.to_numeric(X["MOTIF_NES_Protein_CoverageFraction"], errors="coerce")
            s.loc[has == 0] = 0.0
            s = s.fillna(0.0)
            if self.clip_physical_ranges:
                s = s.clip(lower=0.0, upper=1.0)
            X["MOTIF_NES_Protein_CoverageFraction"] = s.astype(float)

        if "MOTIF_NES_Nearest_Score" in X.columns:
            s = pd.to_numeric(X["MOTIF_NES_Nearest_Score"], errors="coerce")
            s.loc[has == 0] = 0.0
            s = s.fillna(score_fills.get("MOTIF_NES_Nearest_Score", 0.0))
            if self.clip_physical_ranges:
                s = s.clip(lower=0.0, upper=1.0)
            X["MOTIF_NES_Nearest_Score"] = s.astype(float)

        return X

    def _transform_inside_only_features(self, X: pd.DataFrame, inside: np.ndarray) -> pd.DataFrame:
        X = X.copy()

        if "MOTIF_NES_Inside_RelativePosition" in X.columns:
            s = pd.to_numeric(X["MOTIF_NES_Inside_RelativePosition"], errors="coerce")
            if self.clip_physical_ranges:
                s.loc[inside == 1] = s.loc[inside == 1].clip(lower=0.0, upper=1.0)
            s.loc[inside == 0] = -1.0
            s = s.fillna(-1.0)
            X["MOTIF_NES_Inside_RelativePosition"] = s.astype(float)

        if "MOTIF_NES_Inside_SegmentLength" in X.columns:
            s = pd.to_numeric(X["MOTIF_NES_Inside_SegmentLength"], errors="coerce")
            s.loc[inside == 0] = 0.0
            s = s.fillna(0.0)
            if self.clip_physical_ranges:
                s = s.clip(lower=0.0)
            X["MOTIF_NES_Inside_SegmentLength"] = s.astype(float)

        if "MOTIF_NES_Inside_SegmentLengthNorm" in X.columns:
            s = pd.to_numeric(X["MOTIF_NES_Inside_SegmentLengthNorm"], errors="coerce")
            s.loc[inside == 0] = 0.0
            s = s.fillna(0.0)
            if self.clip_physical_ranges:
                s = s.clip(lower=0.0, upper=1.0)
            X["MOTIF_NES_Inside_SegmentLengthNorm"] = s.astype(float)

        return X

    def _transform_class_features(self, X: pd.DataFrame, has: np.ndarray) -> pd.DataFrame:
        X = X.copy()

        if not self.one_hot_class:
            if NES_CLASS_COL in X.columns:
                X = X.drop(columns=[NES_CLASS_COL])
            return X

        cls = self._prepare_class_series(X[NES_CLASS_COL], has)
        allowed = set(self._class_levels or [])
        if allowed:
            cls = cls.where(cls.isin(allowed), other=self.class_other_label)
        else:
            cls = pd.Series(self.class_other_label, index=X.index)

        for lv in self._class_levels or []:
            safe = self._class_safe_map.get(lv, self._sanitize_class_level(lv)) if self._class_safe_map else self._sanitize_class_level(lv)
            X[f"{NES_CLASS_PREFIX}{safe}"] = (cls == lv).astype(float)

        if NES_CLASS_COL in X.columns:
            X = X.drop(columns=[NES_CLASS_COL])

        return X

    def _transform_extra_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.keep_extra_nes_cols:
            return X

        X = X.copy()
        for c in self._extra_nes_cols(X.columns):
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0).astype(float)
        return X

    def _prepare_class_series(self, s: pd.Series, has: np.ndarray) -> pd.Series:
        cls = s.copy()
        cls = cls.where(cls.notna(), other=self.class_other_label)
        cls = cls.astype(str).str.strip()
        cls = cls.replace({"": self.class_other_label, "nan": self.class_other_label, "None": self.class_other_label})
        cls = cls.where(has == 1, other=self.class_other_label)
        return cls

    def _build_class_safe_map(self, levels: List[str]) -> Dict[str, str]:
        out = {}
        used = set()
        for lv in levels:
            safe = self._sanitize_class_level(lv)
            base = safe
            i = 2
            while safe in used:
                safe = f"{base}_{i}"
                i += 1
            used.add(safe)
            out[lv] = safe
        return out

    def _sanitize_class_level(self, value: str) -> str:
        s = str(value).strip()
        if s == "":
            s = self.class_other_label
        s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        if s == "":
            s = "OTHER"
        return s

    def _compute_large_fill(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        values = values[values >= 0]

        if values.size == 0:
            return 1.0

        if self.large_value_mode == "max_plus_1":
            return float(np.max(values) + 1.0)
        if self.large_value_mode == "p99_plus_1":
            return float(np.quantile(values, 0.99) + 1.0)
        if self.large_value_mode == "p95_times_2":
            return float(np.quantile(values, 0.95) * 2.0)

        raise ValueError(f"Unknown large_value_mode: {self.large_value_mode}")

    def _to_bool01(self, values) -> np.ndarray:
        s = pd.Series(values)

        if s.dtype == bool:
            return s.astype(int).to_numpy()

        if s.dtype == object:
            s_str = s.astype(str).str.strip().str.lower()
            mapped = s_str.map(
                {
                    "true": 1,
                    "false": 0,
                    "yes": 1,
                    "no": 0,
                    "y": 1,
                    "n": 0,
                    "1": 1,
                    "0": 0,
                }
            )
            if mapped.notna().any():
                return mapped.fillna(0).astype(int).to_numpy()

        x = pd.to_numeric(s, errors="coerce").fillna(0.0)
        return (x > 0).astype(int).to_numpy()
