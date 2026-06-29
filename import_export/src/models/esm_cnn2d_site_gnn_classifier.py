import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class GNNEncoder(nn.Module):
    def __init__(
        self,
        node_input_dim,
        hidden_dim=64,
        output_dim=64,
        heads=4,
        dropout=0.3,
    ):
        super().__init__()

        self.conv1 = GATConv(
            in_channels=node_input_dim,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout,
        )
        self.conv2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=output_dim,
            heads=1,
            dropout=dropout,
        )

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv2(x, data.edge_index)
        x = self.act(x)
        x = self.dropout(x)

        g = global_mean_pool(x, data.batch)
        return g


class ESMCNN2DSiteGNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim=1280,
        proj_input_dim=512,
        conv_channels=(256, 128),
        seq_kernel_size=3,
        encoder_output_dim=64,
        site_hidden_dims=(512, 256),
        site_output_dim=64,
        gnn_node_input_dim=23,
        gnn_hidden_dim=64,
        gnn_output_dim=64,
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

        self.gnn_encoder = GNNEncoder(
            node_input_dim=gnn_node_input_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            dropout=dropout,
        )

        fusion_in_dim = encoder_output_dim + site_output_dim + gnn_output_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, features):
        x = features["esm"]
        graph = features["af_graph"]

        center_idx = x.size(1) // 2
        x_site = x[:, center_idx, :]

        x_window = self.input_proj(x)
        x_window = x_window.unsqueeze(1)
        x_window = self.feature_extractor(x_window)
        x_window = self.pool(x_window).squeeze(-1).squeeze(-1)
        z_window = self.window_proj(x_window)

        z_site = self.site_encoder(x_site)
        z_graph = self.gnn_encoder(graph)

        z = torch.cat([z_window, z_site, z_graph], dim=1)
        logits = self.classifier(z).squeeze(1)
        return logits