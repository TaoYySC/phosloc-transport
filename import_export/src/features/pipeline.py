from src.features.esm_window.loader import ESMWindowLoader
from src.features.alphafold_graph.loader import AlphaFoldGraphLoader
from src.features.classifier_repr.loader import ClassifierReprLoader
from src.features.manual_tabular.block import ManualTabularBlock
from src.features.esm_window_site_pca.loader import ESMWindowSitePCALoader


class FeaturePipeline:
    def __init__(self, feature_cfg, fixed_retained_columns_by_family=None):
        self.feature_cfg = feature_cfg
        self.fixed_retained_columns_by_family = fixed_retained_columns_by_family
        self.blocks = {}
        self.tab_feature_names_ = []
        self.block_feature_names_ = {}
        self.feature_info_ = {}
        self._build_blocks()

    def _build_blocks(self):
        blocks_cfg = self.feature_cfg.get("blocks", {})

        for block_name, cfg in blocks_cfg.items():
            block_type = cfg["type"]

            if block_type == "esm_window":
                self.blocks[block_name] = ESMWindowLoader(
                    embedding_dir=cfg["embedding_dir"],
                    acc_col=cfg.get("acc_col", "ACC_ID"),
                    pos_col=cfg.get("pos_col", "POSITION"),
                    window_size=cfg.get("window_size", 31),
                    embedding_dim=cfg.get("embedding_dim", 1280),
                )

            elif block_type == "esm_window_site_pca":
                self.blocks[block_name] = ESMWindowSitePCALoader(
                    embedding_dir=cfg["embedding_dir"],
                    acc_col=cfg.get("acc_col", "ACC_ID"),
                    pos_col=cfg.get("pos_col", "POSITION"),
                    window_size=cfg.get("window_size", 31),
                    embedding_dim=cfg.get("embedding_dim", 1280),
                    window_pooling=cfg.get("window_pooling", "mean"),
                    use_window_embedding=cfg.get("use_window_embedding", True),
                    use_site_embedding=cfg.get("use_site_embedding", True),
                    reducer=cfg.get("reducer"),
                    pca_components=cfg.get("pca_components", 50),
                    pca_components_window=cfg.get("pca_components_window"),
                    pca_components_site=cfg.get("pca_components_site"),
                    n_components_window=cfg.get("n_components_window"),
                    n_components_site=cfg.get("n_components_site"),
                    pls_components_window=cfg.get("pls_components_window"),
                    pls_components_site=cfg.get("pls_components_site"),
                    separate_pca_per_embedding=cfg.get(
                        "separate_pca_per_embedding",
                        cfg.get("separate_reduction_per_embedding", False),
                    ),
                    separate_reduction_per_embedding=cfg.get(
                        "separate_reduction_per_embedding",
                        cfg.get("separate_pca_per_embedding"),
                    ),
                    standardize_before_pca=cfg.get("standardize_before_pca"),
                    standardize_before_reduction=cfg.get(
                        "standardize_before_reduction",
                        cfg.get("standardize_before_pca", True),
                    ),
                    standardize_after_reduction=cfg.get(
                        "standardize_after_reduction", False
                    ),
                    fill_missing=cfg.get("fill_missing", False),
                    kernel=cfg.get("kernel", "rbf"),
                    kernel_gamma=cfg.get("kernel_gamma"),
                )

            elif block_type == "alphafold_graph":
                self.blocks[block_name] = AlphaFoldGraphLoader(
                    pdb_dir=cfg["pdb_dir"],
                    acc_col=cfg.get("acc_col", "ACC_ID"),
                    pos_col=cfg.get("pos_col", "POSITION"),
                    radius=cfg.get("radius", 10.0),
                    max_nodes=cfg.get("max_nodes", 64),
                    use_plddt=cfg.get("use_plddt", True),
                )

            elif block_type == "classifier_repr":
                self.blocks[block_name] = ClassifierReprLoader(
                    npz_path=cfg["npz_path"],
                    index_path=cfg["index_path"],
                    layer=cfg.get("layer", "l1"),
                    index_col=cfg.get("index_col", "INDEX"),
                    group_filter=cfg.get("group_filter", "positive"),
                    reducer=cfg.get("reducer", "pls"),
                    pls_components=cfg.get("pls_components"),
                    pca_components=cfg.get("pca_components"),
                    n_components=cfg.get("n_components"),
                    standardize_before_reduction=cfg.get(
                        "standardize_before_reduction", True
                    ),
                    standardize_after_reduction=cfg.get(
                        "standardize_after_reduction", False
                    ),
                    kernel=cfg.get("kernel", "rbf"),
                    kernel_gamma=cfg.get("kernel_gamma"),
                )

            elif block_type == "manual_tabular":
                self.blocks[block_name] = ManualTabularBlock(
                    families=cfg.get("families", []),
                    index_col=cfg.get("index_col", "INDEX"),
                    return_dataframe=False,
                    reducer=cfg.get("reducer"),
                    n_components=cfg.get("n_components"),
                    pca_components=cfg.get("pca_components"),
                    pls_components=cfg.get("pls_components"),
                    standardize_before_reduction=cfg.get(
                        "standardize_before_reduction",
                        cfg.get("standardize_before_pca", True),
                    ),
                    standardize_after_reduction=cfg.get(
                        "standardize_after_reduction", False
                    ),
                    kernel=cfg.get("kernel", "rbf"),
                    kernel_gamma=cfg.get("kernel_gamma"),
                    fixed_retained_columns_by_family=self.fixed_retained_columns_by_family,
                    missing_rate_threshold=cfg.get("missing_rate_threshold", 0.5),
                    pvalue_filter=cfg.get("pvalue_filter", False),
                    pvalue_threshold=cfg.get("pvalue_threshold", 0.05),
                    auroc_top_k_filter=cfg.get("auroc_top_k_filter", False),
                    auroc_top_k=cfg.get("auroc_top_k", 40),
                )

            else:
                raise ValueError(f"Unsupported block type: {block_type}")

    def _attach_manual_features(self, df):
        out_df = df.copy()

        for block in self.blocks.values():
            if isinstance(block, ManualTabularBlock):
                out_df = block.attach_features(out_df)

        return out_df

    def _store_feature_names(self, block_name, block):
        if hasattr(block, "get_feature_names_out"):
            names = block.get_feature_names_out()
            self.block_feature_names_[block_name] = list(names)

    def fit_transform(self, train_df):
        train_df = self._attach_manual_features(train_df)

        feature_dict = {}

        for block_name, block in self.blocks.items():
            if isinstance(block, ManualTabularBlock):
                x = block.fit_transform(train_df, y=train_df["LABEL"].values)
                feature_dict[block_name] = x
                self.tab_feature_names_ = block.get_feature_names_out()
                self.feature_info_[block_name] = block.feature_info
                self._store_feature_names(block_name, block)

            elif hasattr(block, "fit_transform"):
                x = block.fit_transform(train_df, y=train_df["LABEL"].values)
                feature_dict[block_name] = x
                self._store_feature_names(block_name, block)

            else:
                x = block.load(train_df)
                feature_dict[block_name] = x
                self._store_feature_names(block_name, block)

        return feature_dict, train_df

    def transform(self, df):
        df = self._attach_manual_features(df)

        feature_dict = {}

        for block_name, block in self.blocks.items():
            if isinstance(block, ManualTabularBlock):
                feature_dict[block_name] = block.transform(df)

            elif hasattr(block, "transform"):
                feature_dict[block_name] = block.transform(df)

            else:
                feature_dict[block_name] = block.load(df)

        return feature_dict, df

    def get_feature_names_out(self):
        names = []

        for block_name, block_names in self.block_feature_names_.items():
            for name in block_names:
                names.append(f"{block_name}::{name}")

        if len(names) == 0:
            return list(self.tab_feature_names_)

        return names

    @property
    def feature_info(self):
        return self.feature_info_