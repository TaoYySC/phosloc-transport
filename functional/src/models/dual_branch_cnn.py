import torch
import torch.nn as nn


class SequenceCNNBranch(nn.Module):
    def __init__(self, input_dim, conv_channels, kernel_size, proj_dim, dropout):
        super().__init__()

        c1, c2 = conv_channels
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, c1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(c1),
            nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(c2),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(c2, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        x = self.proj(x)
        return x


class DualBranchCNNClassifier(nn.Module):
    def __init__(
        self,
        onehot_input_dim=21,
        esm_input_dim=1280,
        onehot_channels=(64, 128),
        esm_channels=(128, 256),
        kernel_size=3,
        onehot_proj_dim=64,
        esm_proj_dim=128,
        fusion_hidden_dims=(128, 64),
        dropout=0.3,
    ):
        super().__init__()

        self.onehot_branch = SequenceCNNBranch(
            input_dim=onehot_input_dim,
            conv_channels=onehot_channels,
            kernel_size=kernel_size,
            proj_dim=onehot_proj_dim,
            dropout=dropout,
        )

        self.esm_branch = SequenceCNNBranch(
            input_dim=esm_input_dim,
            conv_channels=esm_channels,
            kernel_size=kernel_size,
            proj_dim=esm_proj_dim,
            dropout=dropout,
        )

        fusion_in_dim = onehot_proj_dim + esm_proj_dim
        h1, h2 = fusion_hidden_dims

        self.classifier = nn.Sequential(
            nn.Linear(fusion_in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, features):
        z_onehot = self.onehot_branch(features["onehot"])
        z_esm = self.esm_branch(features["esm"])
        z = torch.cat([z_onehot, z_esm], dim=1)
        logits = self.classifier(z).squeeze(1)
        return logits