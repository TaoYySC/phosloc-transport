import torch
import torch.nn as nn

class SequenceCNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim=21,
        conv_channels=(64, 128),
        kernel_size=3,
        output_dim=64,
        dropout=0.2,
    ):
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
            nn.Linear(c2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        x = self.proj(x)
        return x