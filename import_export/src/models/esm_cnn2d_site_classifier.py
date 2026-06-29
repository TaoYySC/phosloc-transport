import torch
import torch.nn as nn


class ESMCNN2DSiteClassifier(nn.Module):
    def __init__(
        self,
        input_dim=1280,
        proj_input_dim=512,
        conv_channels=(256, 128),
        seq_kernel_size=3,
        encoder_output_dim=64,
        site_hidden_dims=(512, 256),
        site_output_dim=64,
        fusion_hidden_dim=32,
        dropout=0.3,
    ):
        super().__init__()

        c1, c2 = conv_channels

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=c1,
                kernel_size=(seq_kernel_size, proj_input_dim),
                padding=(seq_kernel_size // 2, 0),
            ),
            nn.ReLU(),
            nn.BatchNorm2d(c1),
            nn.Conv2d(
                in_channels=c1,
                out_channels=c2,
                kernel_size=(seq_kernel_size, 1),
                padding=(seq_kernel_size // 2, 0),
            ),
            nn.ReLU(),
            nn.BatchNorm2d(c2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.window_proj = nn.Sequential(
            nn.Linear(c2, encoder_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        s1, s2 = site_hidden_dims
        self.site_encoder = nn.Sequential(
            nn.Linear(input_dim, s1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(s1, s2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(s2, site_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        fusion_in_dim = encoder_output_dim + site_output_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, features):
        x = features["esm"]

        center_idx = x.size(1) // 2
        x_site = x[:, center_idx, :]

        x_window = self.input_proj(x)
        x_window = x_window.unsqueeze(1)
        x_window = self.feature_extractor(x_window)
        x_window = self.pool(x_window).squeeze(-1).squeeze(-1)
        z_window = self.window_proj(x_window)

        z_site = self.site_encoder(x_site)

        z = torch.cat([z_window, z_site], dim=1)
        logits = self.classifier(z).squeeze(1)
        return logits
