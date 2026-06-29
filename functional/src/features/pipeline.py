from src.features.manual_tabular.block import ManualTabularBlock
from src.features.sequence_cnn.loader import SequenceWindowLoader
from src.features.esm_window.loader import ESMWindowLoader
from src.features.esm_window.site_mean_loader import ESMWindowSiteMeanLoader
from src.features.alphafold_graph.loader import AlphaFoldGraphLoader


class FeaturePipeline:
    def __init__(self, feature_cfg, fixed_retained_columns_by_family=None):
        self.feature_cfg = feature_cfg
        self.fixed_retained_columns_by_family = fixed_retained_columns_by_family
        self.blocks = {}
        self.tab_feature_names_ = []
        self.feature_info_ = {}
        self._build_blocks()

    def _build_blocks(self):
        blocks_cfg = self.feature_cfg.get("blocks", {})

        for block_name, cfg in blocks_cfg.items():
            block_type = cfg["type"]

            if block_type == "manual_tabular":
                self.blocks[block_name] = ManualTabularBlock(
                    families=cfg.get("families", []),
                    index_col="INDEX",
                    return_dataframe=False,
                    fixed_retained_columns_by_family=self.fixed_retained_columns_by_family,
                    missing_rate_threshold=cfg.get("missing_rate_threshold", 0.5),
                )
            elif block_type == "onehot_window":
                self.blocks[block_name] = SequenceWindowLoader(
                    seq_col=cfg.get("seq_col", "FULL_SEQUENCE"),
                    pos_col=cfg.get("pos_col", "POSITION"),
                    window_size=cfg.get("window_size", 31),
                )
            elif block_type == "esm_window":
                self.blocks[block_name] = ESMWindowLoader(
                    embedding_dir=cfg["embedding_dir"],
                    acc_col=cfg.get("acc_col", "ACC_ID"),
                    pos_col=cfg.get("pos_col", "POSITION"),
                    window_size=cfg.get("window_size", 31),
                    embedding_dim=cfg.get("embedding_dim", 1280),
                )
            elif block_type == "esm_window_site_mean":
                self.blocks[block_name] = ESMWindowSiteMeanLoader(
                    embedding_dir=cfg["embedding_dir"],
                    acc_col=cfg.get("acc_col", "ACC_ID"),
                    pos_col=cfg.get("pos_col", "POSITION"),
                    window_size=cfg.get("window_size", 41),
                    embedding_dim=cfg.get("embedding_dim", 1280),
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
            else:
                raise ValueError(f"Unsupported block type: {block_type}")

    def _attach_manual_features(self, df):
        out_df = df
        for _, block in self.blocks.items():
            if isinstance(block, ManualTabularBlock):
                out_df = block.attach_features(out_df)
        return out_df

    def fit_transform(self, train_df):
        train_df = self._attach_manual_features(train_df)
        feature_dict = {}

        for block_name, block in self.blocks.items():
            if isinstance(block, ManualTabularBlock):
                x = block.fit_transform(train_df, y=train_df["LABEL"].values)
                feature_dict[block_name] = x
                self.tab_feature_names_ = block.get_feature_names_out()
                self.feature_info_ = block.feature_info
            else:
                feature_dict[block_name] = block.load(train_df)

        return feature_dict, train_df

    def transform(self, df):
        df = self._attach_manual_features(df)
        feature_dict = {}

        for block_name, block in self.blocks.items():
            if isinstance(block, ManualTabularBlock):
                feature_dict[block_name] = block.transform(df)
            else:
                feature_dict[block_name] = block.load(df)

        return feature_dict, df

    def _collect_feature_names(self):
        names = list(self.tab_feature_names_)
        for block in self.blocks.values():
            if hasattr(block, "get_feature_names_out"):
                names.extend(block.get_feature_names_out())
        return names

    def get_feature_names_out(self):
        if len(self.tab_feature_names_) == 0:
            return self._collect_feature_names()
        extra = []
        for block in self.blocks.values():
            if hasattr(block, "get_feature_names_out"):
                extra.extend(block.get_feature_names_out())
        return list(self.tab_feature_names_) + extra

    @property
    def feature_info(self):
        return self.feature_info_