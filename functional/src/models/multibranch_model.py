import torch
import torch.nn as nn

from src.models.tabular_encoder import TabularEncoder
from src.features.sequence_cnn.encoder import SequenceCNNEncoder


class MultiBranchClassifier(nn.Module):
    def __init__(
        self,
        tabular_input_dim=None,
        tabular_hidden_dims=(256, 128),
        tabular_output_dim=64,
        use_sequence=True,
        seq_input_dim=21,
        seq_conv_channels=(64, 128),
        seq_kernel_size=3,
        seq_output_dim=64,
        fusion_hidden_dims=(128, 64),
        dropout=0.2,
        num_classes=1,
    ):
        super().__init__()

        self.use_tabular = tabular_input_dim is not None and tabular_input_dim > 0
        self.use_sequence = use_sequence

        fusion_in_dim = 0

        if self.use_tabular:
            self.tabular_encoder = TabularEncoder(
                input_dim=tabular_input_dim,
                hidden_dims=tabular_hidden_dims,
                output_dim=tabular_output_dim,
                dropout=dropout,
            )
            fusion_in_dim += tabular_output_dim
        else:
            self.tabular_encoder = None

        if self.use_sequence:
            self.sequence_encoder = SequenceCNNEncoder(
                input_dim=seq_input_dim,
                conv_channels=seq_conv_channels,
                kernel_size=seq_kernel_size,
                output_dim=seq_output_dim,
                dropout=dropout,
            )
            fusion_in_dim += seq_output_dim
        else:
            self.sequence_encoder = None

        dims = [fusion_in_dim] + list(fusion_hidden_dims)
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x_tab=None, x_seq=None):
        parts = []

        if self.tabular_encoder is not None:
            if x_tab is None:
                raise ValueError("x_tab is required by the model.")
            z_tab = self.tabular_encoder(x_tab)
            parts.append(z_tab)

        if self.sequence_encoder is not None:
            if x_seq is None:
                raise ValueError("x_seq is required by the model.")
            z_seq = self.sequence_encoder(x_seq)
            parts.append(z_seq)

        if len(parts) == 0:
            raise ValueError("No input branches are enabled.")

        z = torch.cat(parts, dim=1)
        logits = self.classifier(z).squeeze(1)
        return logits