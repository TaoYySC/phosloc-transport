import torch

from src.features.fusion import FeatureFusion
from src.features.manual_tabular.block import ManualTabularBlock
from src.features.sequence_cnn.block import SequenceCNNBlock
from src.features.sequence_cnn.encoder import SequenceCNNEncoder


class FeaturePipeline:
    def __init__(
        self,
        feature_cfg,
        device="cuda",
        fixed_retained_columns_by_family=None,
    ):
        self.feature_cfg = feature_cfg
        self.device = device
        self.fixed_retained_columns_by_family = fixed_retained_columns_by_family

        self.manual_block = None
        self.sequence_block = None
        self.fusion = None

        self.tab_feature_names_ = []
        self.feature_info_ = {}

        self._build_blocks()

    def _build_blocks(self):
        families = self.feature_cfg.get("families", [])
        if len(families) > 0:
            self.manual_block = ManualTabularBlock(
                families=families,
                index_col="INDEX",
                return_dataframe=False,
                fixed_retained_columns_by_family=self.fixed_retained_columns_by_family,
            )

        if self.feature_cfg.get("use_seq_cnn", False):
            encoder = SequenceCNNEncoder(
                input_dim=self.feature_cfg.get("seq_input_dim", 21),
                conv_channels=tuple(self.feature_cfg.get("seq_conv_channels", [64, 128])),
                kernel_size=self.feature_cfg.get("seq_kernel_size", 3),
                output_dim=self.feature_cfg.get("seq_output_dim", 64),
                dropout=self.feature_cfg.get("seq_dropout", 0.2),
            )

            ckpt_path = self.feature_cfg.get("seq_ckpt_path")
            if ckpt_path is None:
                raise ValueError("seq_ckpt_path is required when use_seq_cnn=True")

            state = torch.load(ckpt_path, map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                encoder.load_state_dict(state["model_state_dict"])
            elif isinstance(state, dict) and "state_dict" in state:
                encoder.load_state_dict(state["state_dict"])
            else:
                encoder.load_state_dict(state)

            self.sequence_block = SequenceCNNBlock(
                encoder=encoder,
                device=self.device,
                seq_col="FULL_SEQUENCE",
                pos_col="POSITION",
                window_size=self.feature_cfg.get("seq_window_size", 31),
            )

        self.fusion = FeatureFusion(
            scale_tabular=self.feature_cfg.get("scale_tabular", True),
            scale_sequence=self.feature_cfg.get("scale_sequence", False),
        )

    def _attach_manual_features(self, df):
        if self.manual_block is None:
            return df
        return self.manual_block.attach_features(df)

    def fit_transform(self, train_df):
        train_df = self._attach_manual_features(train_df)

        x_train_tab = None
        x_train_seq = None

        if self.manual_block is not None:
            x_train_tab = self.manual_block.fit_transform(
                train_df,
                y=train_df["LABEL"].values,
            )
            self.tab_feature_names_ = self.manual_block.get_feature_names_out()
            self.feature_info_ = self.manual_block.feature_info

        if self.sequence_block is not None:
            x_train_seq = self.sequence_block.transform(train_df)

        x_train = self.fusion.fit_transform(
            x_tab=x_train_tab,
            x_seq=x_train_seq,
        )
        return x_train, train_df

    def transform(self, df):
        df = self._attach_manual_features(df)

        x_tab = None
        x_seq = None

        if self.manual_block is not None:
            x_tab = self.manual_block.transform(df)

        if self.sequence_block is not None:
            x_seq = self.sequence_block.transform(df)

        x = self.fusion.transform(
            x_tab=x_tab,
            x_seq=x_seq,
        )
        return x, df

    def get_feature_names_out(self):
        return list(self.tab_feature_names_)

    @property
    def feature_info(self):
        return self.feature_info_