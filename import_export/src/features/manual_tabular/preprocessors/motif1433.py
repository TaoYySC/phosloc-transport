from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


MOTIF_1433_PREFIX = "MOTIF_1433_"

MOTIF_1433_BOOL_COLS = [
    "MOTIF_1433_HasMotif_Flag",
    "MOTIF_1433_Inside_Flag",
    "MOTIF_1433_Within20AA_Flag",
    "MOTIF_1433_Within50AA_Flag",
    "MOTIF_1433_IsPredictedSite_Flag",
    "MOTIF_1433_Nearest_IsHighConfidence_Flag",
    "MOTIF_1433_Inside_HighConfidence_Flag",
]

MOTIF_1433_COUNT_COLS = [
    "MOTIF_1433_Protein_MotifCount",
    "MOTIF_1433_Inside_OverlapCount",
    "MOTIF_1433_Protein_HighConfidenceSegmentCount",
    "MOTIF_1433_Protein_PredictedSiteCount",
]

MOTIF_1433_FRACTION_COLS = [
    "MOTIF_1433_Protein_CoverageFraction",
    "MOTIF_1433_Protein_HighConfidenceCoverageFraction",
]

MOTIF_1433_SCORE_COLS = [
    "MOTIF_1433_Nearest_ConsensusScore",
]

MOTIF_1433_DIST_COLS = [
    "MOTIF_1433_Nearest_Distance",
    "MOTIF_1433_Nearest_DistanceNorm",
    "MOTIF_1433_Nearest_CenterSignedDistance",
]

MOTIF_1433_UNSIGNED_DIST_COLS = [
    "MOTIF_1433_Nearest_Distance",
    "MOTIF_1433_Nearest_DistanceNorm",
]

MOTIF_1433_SIGNED_DIST_COLS = [
    "MOTIF_1433_Nearest_CenterSignedDistance",
]

MOTIF_1433_INSIDE_ONLY_COLS = [
    "MOTIF_1433_Inside_RelativePosition",
    "MOTIF_1433_Inside_SegmentLength",
    "MOTIF_1433_Inside_SegmentLengthNorm",
]

MOTIF_1433_BOUNDARY_COLS = [
    "MOTIF_1433_Nearest_Start",
    "MOTIF_1433_Nearest_End",
]

MOTIF_1433_CANONICAL_COLS = (
    MOTIF_1433_BOOL_COLS
    + MOTIF_1433_COUNT_COLS
    + MOTIF_1433_FRACTION_COLS
    + MOTIF_1433_SCORE_COLS
    + MOTIF_1433_DIST_COLS
    + MOTIF_1433_INSIDE_ONLY_COLS
    + MOTIF_1433_BOUNDARY_COLS
)


class Motif1433Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe: bool = False,
        strict: bool = True,
        keep_extra_motif_cols: bool = False,
        drop_boundary_cols: bool = True,
        drop_weak_cols: bool = False,
        drop_dead_cols: Optional[bool] = None,
        log_transform_dist: bool = True,
        large_value_mode: str = "p99_plus_1",
        add_missing_indicator: bool = False,
        all_nan_fill_value: float = 0.0,
    ):
        self.return_dataframe = return_dataframe
        self.strict = strict
        self.keep_extra_motif_cols = keep_extra_motif_cols
        self.drop_boundary_cols = drop_boundary_cols
        self.drop_weak_cols = drop_weak_cols
        self.drop_dead_cols = drop_dead_cols
        self.log_transform_dist = log_transform_dist
        self.large_value_mode = large_value_mode
        self.add_missing_indicator = add_missing_indicator
        self.all_nan_fill_value = all_nan_fill_value

        self._input_cols: Optional[List[str]] = None
        self._keep_cols: Optional[List[str]] = None
        self._dist_fill_values: Optional[Dict[str, float]] = None
        self._score_fill_values: Optional[Dict[str, float]] = None
        self._boundary_fill_values: Optional[Dict[str, float]] = None
        self._missing_rate_: Optional[Dict[str, float]] = None
        self._all_nan_cols_: Optional[Set[str]] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X).copy()
        X = self._select_input_columns(X)

        for c in MOTIF_1433_BOOL_COLS:
            if c in X.columns:
                X[c] = self._to_bool01(X[c].values)

        has = self._get_flag_array(X, "MOTIF_1433_HasMotif_Flag", default=0)

        self._dist_fill_values = {}
        self._score_fill_values = {}
        self._boundary_fill_values = {}
        self._missing_rate_ = {}
        self._all_nan_cols_ = set()

        for c in X.columns:
            if c in MOTIF_1433_BOOL_COLS:
                self._missing_rate_[c] = 0.0
                continue
            s = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            self._missing_rate_[c] = float(s.isna().mean())
            if s.notna().sum() == 0:
                self._all_nan_cols_.add(c)

        for c in MOTIF_1433_UNSIGNED_DIST_COLS:
            if c in X.columns:
                self._dist_fill_values[c] = self._large_fill_value(X[c], has)

        for c in MOTIF_1433_SIGNED_DIST_COLS:
            if c in X.columns:
                self._dist_fill_values[c] = self._median_fill_value(X[c], has, default=0.0)

        for c in MOTIF_1433_SCORE_COLS:
            if c in X.columns:
                self._score_fill_values[c] = self._median_fill_value(X[c], has, default=0.0)

        for c in MOTIF_1433_BOUNDARY_COLS:
            if c in X.columns:
                self._boundary_fill_values[c] = self._median_fill_value(X[c], has, default=0.0)

        self._input_cols = list(X.columns)
        self._keep_cols = self._build_keep_cols(self._input_cols)

        self.feature_names_ = list(self._keep_cols)
        if self.add_missing_indicator:
            indicator_cols = [
                f"{c}__missing"
                for c in self._keep_cols
                if c not in MOTIF_1433_BOOL_COLS
            ]
            self.feature_names_.extend(indicator_cols)

        return self

    def transform(self, X: pd.DataFrame):
        if self._input_cols is None or self._keep_cols is None:
            raise RuntimeError("Motif1433Preprocessor must be fitted before transform.")

        X = self._check_input(X).copy()

        for c in self._input_cols:
            if c not in X.columns:
                X[c] = np.nan

        X = X[self._input_cols].copy()
        X = self._transform_internal(X)

        missing_cols = [c for c in self._keep_cols if c not in X.columns]
        for c in missing_cols:
            X[c] = 0.0
        X = X[self._keep_cols].copy()

        if self.add_missing_indicator:
            for c in self._keep_cols:
                if c in MOTIF_1433_BOOL_COLS:
                    continue
                ind_col = f"{c}__missing"
                if ind_col not in X.columns:
                    X[ind_col] = 0.0
            X = X[self.feature_names_]

        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise ValueError(f"Non-numeric columns remain after Motif1433Preprocessor: {non_numeric}")

        X = X.astype(float)
        if self.return_dataframe:
            return X
        return X.to_numpy(dtype=float)

    def get_feature_names_out(self):
        if self.feature_names_ is None:
            return None
        return list(self.feature_names_)

    def get_impute_summary(self) -> pd.DataFrame:
        if self._input_cols is None or self._missing_rate_ is None:
            raise RuntimeError("Motif1433Preprocessor must be fitted before calling get_impute_summary().")

        rows = []
        for c in self._input_cols:
            if c in MOTIF_1433_UNSIGNED_DIST_COLS or c in MOTIF_1433_SIGNED_DIST_COLS:
                fill = (self._dist_fill_values or {}).get(c, np.nan)
                group = "distance"
            elif c in MOTIF_1433_SCORE_COLS:
                fill = (self._score_fill_values or {}).get(c, np.nan)
                group = "score"
            elif c in MOTIF_1433_BOUNDARY_COLS:
                fill = (self._boundary_fill_values or {}).get(c, np.nan)
                group = "boundary"
            elif c in MOTIF_1433_BOOL_COLS:
                fill = 0.0
                group = "boolean"
            elif c in MOTIF_1433_COUNT_COLS:
                fill = 0.0
                group = "count"
            elif c in MOTIF_1433_FRACTION_COLS:
                fill = 0.0
                group = "fraction"
            elif c in MOTIF_1433_INSIDE_ONLY_COLS:
                fill = self._inside_default_value(c)
                group = "inside_only"
            else:
                fill = self.all_nan_fill_value
                group = "extra"

            rows.append(
                {
                    "feature_name": c,
                    "feature_group": group,
                    "fill_value": float(fill) if pd.notna(fill) else np.nan,
                    "missing_rate": float(self._missing_rate_.get(c, np.nan)),
                    "all_nan_in_fit": c in (self._all_nan_cols_ or set()),
                    "kept_in_output": c in (self._keep_cols or []),
                }
            )
        return pd.DataFrame(rows)

    def _select_input_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.strict:
            for c in MOTIF_1433_CANONICAL_COLS:
                if c not in X.columns:
                    X[c] = np.nan
            cols = list(MOTIF_1433_CANONICAL_COLS)
            if self.keep_extra_motif_cols:
                extra_cols = [
                    c for c in X.columns
                    if c.startswith(MOTIF_1433_PREFIX) and c not in cols
                ]
                cols.extend(sorted(extra_cols))
            return X[cols].copy()

        cols = [c for c in X.columns if c.startswith(MOTIF_1433_PREFIX)]
        return X[cols].copy()

    def _transform_internal(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        missing_source = pd.DataFrame(index=X.index)
        for c in X.columns:
            if c in MOTIF_1433_BOOL_COLS:
                missing_source[c] = False
            else:
                missing_source[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan).isna()

        for c in MOTIF_1433_BOOL_COLS:
            if c in X.columns:
                X[c] = self._to_bool01(X[c].values).astype(float)

        has = self._get_flag_array(X, "MOTIF_1433_HasMotif_Flag", default=0)
        inside = self._get_flag_array(X, "MOTIF_1433_Inside_Flag", default=0)

        for c in MOTIF_1433_COUNT_COLS:
            if c not in X.columns:
                continue
            X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if c == "MOTIF_1433_Inside_OverlapCount":
                X.loc[inside == 0, c] = 0.0
            else:
                X.loc[has == 0, c] = 0.0
            X[c] = X[c].fillna(0.0).clip(lower=0.0)

        for c in MOTIF_1433_FRACTION_COLS:
            if c not in X.columns:
                continue
            X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            X.loc[has == 0, c] = 0.0
            X[c] = X[c].fillna(0.0).clip(lower=0.0, upper=1.0)

        for c in MOTIF_1433_SCORE_COLS:
            if c not in X.columns:
                continue
            X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            X.loc[has == 0, c] = 0.0
            fill = (self._score_fill_values or {}).get(c, self.all_nan_fill_value)
            X[c] = X[c].fillna(fill).clip(lower=0.0, upper=1.0)

        for c in MOTIF_1433_UNSIGNED_DIST_COLS:
            if c not in X.columns:
                continue
            X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            fill = (self._dist_fill_values or {}).get(c, 1.0)
            X.loc[has == 0, c] = fill
            X[c] = X[c].fillna(fill).clip(lower=0.0)
            if self.log_transform_dist:
                X[c] = np.log1p(X[c].to_numpy(dtype=float))

        for c in MOTIF_1433_SIGNED_DIST_COLS:
            if c not in X.columns:
                continue
            X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            fill = (self._dist_fill_values or {}).get(c, 0.0)
            X.loc[has == 0, c] = 0.0
            X[c] = X[c].fillna(fill)
            if self.log_transform_dist:
                v = X[c].to_numpy(dtype=float)
                X[c] = np.sign(v) * np.log1p(np.abs(v))

        for c in MOTIF_1433_INSIDE_ONLY_COLS:
            if c not in X.columns:
                continue
            X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            fill = self._inside_default_value(c)
            X.loc[inside == 0, c] = fill
            X[c] = X[c].fillna(fill)
            if c == "MOTIF_1433_Inside_RelativePosition":
                mask_inside = inside == 1
                X.loc[mask_inside, c] = X.loc[mask_inside, c].clip(lower=0.0, upper=1.0)
            else:
                X[c] = X[c].clip(lower=0.0)

        for c in MOTIF_1433_BOUNDARY_COLS:
            if c not in X.columns:
                continue
            X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            fill = (self._boundary_fill_values or {}).get(c, 0.0)
            X.loc[has == 0, c] = 0.0
            X[c] = X[c].fillna(fill).clip(lower=0.0)

        extra_cols = [
            c for c in X.columns
            if c.startswith(MOTIF_1433_PREFIX)
            and c not in MOTIF_1433_CANONICAL_COLS
        ]
        for c in extra_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            X[c] = X[c].fillna(self.all_nan_fill_value)

        if self.add_missing_indicator:
            for c in X.columns:
                if c in MOTIF_1433_BOOL_COLS:
                    continue
                X[f"{c}__missing"] = missing_source.get(c, pd.Series(False, index=X.index)).astype(float)

        return X

    def _build_keep_cols(self, cols: List[str]) -> List[str]:
        cols = list(cols)

        if self.drop_dead_cols is not None:
            drop = {
                "MOTIF_1433_Nearest_Start",
                "MOTIF_1433_Nearest_End",
            }
            if self.drop_dead_cols:
                drop.update(
                    {
                        "MOTIF_1433_Inside_RelativePosition",
                        "MOTIF_1433_Inside_SegmentLength",
                        "MOTIF_1433_Inside_SegmentLengthNorm",
                        "MOTIF_1433_Protein_CoverageFraction",
                    }
                )
            return [c for c in cols if c not in drop]

        drop_cols = set()
        if self.drop_boundary_cols:
            drop_cols.update(MOTIF_1433_BOUNDARY_COLS)

        if self.drop_weak_cols:
            drop_cols.update(
                {
                    "MOTIF_1433_Protein_CoverageFraction",
                    "MOTIF_1433_Protein_HighConfidenceCoverageFraction",
                }
            )

        return [c for c in cols if c not in drop_cols]

    def _large_fill_value(self, s: pd.Series, has: np.ndarray) -> float:
        v = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        v_has = v[has == 1]
        v_has = v_has[np.isfinite(v_has)]
        v_has = v_has[v_has >= 0]

        if v_has.size == 0:
            return 1.0

        if self.large_value_mode == "max_plus_1":
            return float(np.max(v_has) + 1.0)
        if self.large_value_mode == "p99_plus_1":
            return float(np.quantile(v_has, 0.99) + 1.0)
        if self.large_value_mode == "p95_times_2":
            return float(np.quantile(v_has, 0.95) * 2.0)
        raise ValueError(f"Unknown large_value_mode: {self.large_value_mode}")

    def _median_fill_value(self, s: pd.Series, has: np.ndarray, default: float = 0.0) -> float:
        v = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        v_has = v[has == 1]
        v_has = v_has[np.isfinite(v_has)]
        if v_has.size == 0:
            return float(default)
        return float(np.median(v_has))

    def _inside_default_value(self, col: str) -> float:
        if col == "MOTIF_1433_Inside_RelativePosition":
            return -1.0
        return 0.0

    def _get_flag_array(self, X: pd.DataFrame, col: str, default: int = 0) -> np.ndarray:
        if col not in X.columns:
            return np.full(len(X), default, dtype=int)
        return self._to_bool01(X[col].values).astype(int)

    def _to_bool01(self, v) -> np.ndarray:
        s = pd.Series(v)

        if s.dtype == bool:
            return s.astype(int).to_numpy()

        if s.dtype == object:
            s = s.astype(str).str.strip().str.lower()
            mapped = s.map(
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

        x = pd.to_numeric(pd.Series(v), errors="coerce").fillna(0.0)
        return (x > 0).astype(int).to_numpy()

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Motif1433Preprocessor expects a pandas DataFrame.")

        X = X.copy()
        if "INDEX" in X.columns:
            X = X.drop(columns=["INDEX"])

        bad = [c for c in X.columns if not str(c).startswith(MOTIF_1433_PREFIX)]
        if bad:
            raise ValueError(f"Non-MOTIF_1433_ columns detected: {bad}")

        return X
