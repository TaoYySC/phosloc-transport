import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


DOMAIN_FAMILY_PREFIX = "MOTIF_Domain"

DOMAIN_PREFIXES = [
    "DBD",
    "TAD",
    "Linker",
    "Boundary",
    "FunctionalDomain",
]

DOMAIN_BOOL_SUFFIXES = [
    "HasSegment_Flag",
    "Inside_Flag",
    "Within20AA_Flag",
    "Within50AA_Flag",
]

DOMAIN_DIST_SUFFIXES = [
    "Nearest_Start",
    "Nearest_End",
    "Nearest_Distance",
    "Nearest_DistanceNorm",
    "Nearest_CenterSignedDistance",
]

DOMAIN_INSIDE_ONLY_SUFFIXES = [
    "Inside_RelativePosition",
    "Inside_SegmentLength",
    "Inside_SegmentLengthNorm",
]

DOMAIN_OTHER_FLOAT_SUFFIXES = [
    "Protein_SegmentCount",
    "Protein_CoverageFraction",
]


class DomainPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        return_dataframe: bool = False,
        drop_weak_cols: bool = False,
        log_transform_dist: bool = True,
        large_value_mode: str = "p99_plus_1",
    ):
        self.return_dataframe = return_dataframe
        self.drop_weak_cols = drop_weak_cols
        self.log_transform_dist = log_transform_dist
        self.large_value_mode = large_value_mode

        self._all_cols = None
        self._keep_cols = None
        self._dist_fill_values = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check_input(X).copy()

        self._dist_fill_values = {}

        for prefix in DOMAIN_PREFIXES:
            for col in self._bool_cols(prefix):
                if col in X.columns:
                    X[col] = self._to_bool01(X[col].values)

            has_col = self._col(prefix, "HasSegment_Flag")
            has = (
                X[has_col].to_numpy(dtype=int)
                if has_col in X.columns
                else np.ones(len(X), dtype=int)
            )

            for col in self._dist_cols(prefix):
                if col not in X.columns:
                    continue

                v = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
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

                self._dist_fill_values[col] = fill

        self._all_cols = list(X.columns)
        self._keep_cols = self._build_keep_cols(self._all_cols)
        self.feature_names_ = list(self._keep_cols)
        return self

    def transform(self, X: pd.DataFrame):
        X = self._check_input(X).copy()

        for prefix in DOMAIN_PREFIXES:
            for col in self._bool_cols(prefix):
                if col in X.columns:
                    X[col] = self._to_bool01(X[col].values)

            has_col = self._col(prefix, "HasSegment_Flag")
            inside_col = self._col(prefix, "Inside_Flag")

            has = (
                X[has_col].to_numpy(dtype=int)
                if has_col in X.columns
                else np.ones(len(X), dtype=int)
            )
            inside = (
                X[inside_col].to_numpy(dtype=int)
                if inside_col in X.columns
                else np.zeros(len(X), dtype=int)
            )

            for col in self._dist_cols(prefix):
                if col not in X.columns:
                    continue

                X[col] = pd.to_numeric(X[col], errors="coerce")
                fill = self._dist_fill_values.get(col, 1.0)

                X.loc[has == 0, col] = fill
                X[col] = X[col].fillna(fill)

                if self.log_transform_dist:
                    X[col] = np.log1p(np.clip(X[col].to_numpy(dtype=float), 0.0, None))

            n_seg_col = self._col(prefix, "Protein_SegmentCount")
            if n_seg_col in X.columns:
                X[n_seg_col] = pd.to_numeric(X[n_seg_col], errors="coerce").fillna(0.0)

            frac_col = self._col(prefix, "Protein_CoverageFraction")
            if frac_col in X.columns:
                X[frac_col] = pd.to_numeric(X[frac_col], errors="coerce").fillna(0.0)

            rel_col = self._col(prefix, "Inside_RelativePosition")
            if rel_col in X.columns:
                X[rel_col] = pd.to_numeric(X[rel_col], errors="coerce")
                X.loc[inside == 0, rel_col] = -1.0
                X[rel_col] = X[rel_col].fillna(-1.0)

            seg_len_col = self._col(prefix, "Inside_SegmentLength")
            if seg_len_col in X.columns:
                X[seg_len_col] = pd.to_numeric(X[seg_len_col], errors="coerce")
                X.loc[inside == 0, seg_len_col] = 0.0
                X[seg_len_col] = X[seg_len_col].fillna(0.0)

            seg_len_norm_col = self._col(prefix, "Inside_SegmentLengthNorm")
            if seg_len_norm_col in X.columns:
                X[seg_len_norm_col] = pd.to_numeric(X[seg_len_norm_col], errors="coerce")
                X.loc[inside == 0, seg_len_norm_col] = 0.0
                X[seg_len_norm_col] = X[seg_len_norm_col].fillna(0.0)

        if self._keep_cols is not None:
            missing_cols = [c for c in self._keep_cols if c not in X.columns]
            for c in missing_cols:
                X[c] = 0.0
            X = X[self._keep_cols]

        if self.return_dataframe:
            return X

        return X.to_numpy(dtype=float)

    def get_feature_names_out(self):
        if self.feature_names_ is None:
            return None
        return list(self.feature_names_)

    def _col(self, prefix, suffix):
        return f"{DOMAIN_FAMILY_PREFIX}_{prefix}_{suffix}"

    def _build_keep_cols(self, cols):
        cols = list(cols)

        if not self.drop_weak_cols:
            return cols

        weak = set()
        for prefix in DOMAIN_PREFIXES:
            weak.add(self._col(prefix, "Protein_CoverageFraction"))

        return [c for c in cols if c not in weak]

    def _bool_cols(self, prefix):
        return [self._col(prefix, s) for s in DOMAIN_BOOL_SUFFIXES]

    def _dist_cols(self, prefix):
        return [self._col(prefix, s) for s in DOMAIN_DIST_SUFFIXES]

    def _inside_only_cols(self, prefix):
        return [self._col(prefix, s) for s in DOMAIN_INSIDE_ONLY_SUFFIXES]

    def _other_float_cols(self, prefix):
        return [self._col(prefix, s) for s in DOMAIN_OTHER_FLOAT_SUFFIXES]

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

        x = pd.to_numeric(pd.Series(v), errors="coerce").fillna(0.0)
        return (x > 0).astype(int).to_numpy()

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("DomainPreprocessor expects a DataFrame.")

        X = X.copy()

        expected_cols = set()

        for prefix in DOMAIN_PREFIXES:
            for col in self._bool_cols(prefix):
                expected_cols.add(col)
                if col not in X.columns:
                    X[col] = 0.0

            for col in self._dist_cols(prefix):
                expected_cols.add(col)
                if col not in X.columns:
                    X[col] = np.nan

            for col in self._inside_only_cols(prefix):
                expected_cols.add(col)
                if col not in X.columns:
                    X[col] = np.nan

            for col in self._other_float_cols(prefix):
                expected_cols.add(col)
                if col not in X.columns:
                    X[col] = np.nan

        bad = [c for c in X.columns if c not in expected_cols]
        if bad:
            raise ValueError(f"Non-domain feature columns detected: {bad}")

        return X