import copy

import numpy as np

from src.features.esm_window_site_pca.loader import ESMWindowSitePCALoader
from src.features.manual_tabular.block import ManualTabularBlock


class ThreeBranchExtractor:
    """Extract raw window / site / manual feature matrices for stacking."""

    def __init__(self, blocks_cfg, fixed_retained_columns_by_family=None):
        self.blocks_cfg = copy.deepcopy(blocks_cfg)
        self.fixed_retained_columns_by_family = fixed_retained_columns_by_family
        self.esm_loader = None
        self.manual_block = None
        self._build_blocks()

    def _build_blocks(self):
        esm_cfg = copy.deepcopy(self.blocks_cfg["esm"])
        esm_cfg["reducer"] = "none"
        esm_cfg["use_window_embedding"] = True
        esm_cfg["use_site_embedding"] = True
        esm_cfg["separate_reduction_per_embedding"] = True

        self.esm_loader = ESMWindowSitePCALoader(
            embedding_dir=esm_cfg["embedding_dir"],
            acc_col=esm_cfg.get("acc_col", "ACC_ID"),
            pos_col=esm_cfg.get("pos_col", "POSITION"),
            window_size=esm_cfg.get("window_size", 31),
            embedding_dim=esm_cfg.get("embedding_dim", 1280),
            window_pooling=esm_cfg.get("window_pooling", "mean"),
            use_window_embedding=True,
            use_site_embedding=True,
            reducer="none",
            separate_reduction_per_embedding=True,
            standardize_before_reduction=False,
            fill_missing=esm_cfg.get("fill_missing", False),
        )

        tab_cfg = copy.deepcopy(self.blocks_cfg["tabular"])
        self.manual_block = ManualTabularBlock(
            families=tab_cfg.get("families", []),
            index_col=tab_cfg.get("index_col", "INDEX"),
            return_dataframe=False,
            reducer=None,
            standardize_before_reduction=False,
            fixed_retained_columns_by_family=self.fixed_retained_columns_by_family,
            missing_rate_threshold=tab_cfg.get("missing_rate_threshold", 0.5),
            pvalue_filter=tab_cfg.get("pvalue_filter", False),
            pvalue_threshold=tab_cfg.get("pvalue_threshold", 0.05),
            auroc_top_k_filter=tab_cfg.get("auroc_top_k_filter", False),
            auroc_top_k=tab_cfg.get("auroc_top_k", 40),
        )

    def _attach_manual_features(self, df):
        return self.manual_block.attach_features(df.copy())

    def _extract_esm_parts(self, df):
        x_window, x_site = self.esm_loader._build_separate_matrices(df)
        if x_window is None or x_site is None:
            raise ValueError("Both window and site ESM embeddings are required for stacking.")
        return (
            np.asarray(x_window, dtype=np.float32),
            np.asarray(x_site, dtype=np.float32),
        )

    def fit_transform(self, train_df, y=None):
        train_df = self._attach_manual_features(train_df)
        x_window, x_site = self._extract_esm_parts(train_df)
        x_manual = self.manual_block.fit_transform(train_df, y=y)
        x_manual = np.asarray(x_manual, dtype=np.float32)
        features = {
            "window": x_window,
            "site": x_site,
            "manual": x_manual,
        }
        return features, train_df

    def transform(self, df):
        df = self._attach_manual_features(df)
        x_window, x_site = self._extract_esm_parts(df)
        x_manual = self.manual_block.transform(df)
        x_manual = np.asarray(x_manual, dtype=np.float32)
        features = {
            "window": x_window,
            "site": x_site,
            "manual": x_manual,
        }
        return features, df
