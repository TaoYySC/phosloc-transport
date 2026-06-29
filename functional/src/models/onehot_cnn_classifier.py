import torch.nn as nn


class OneHotCNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim=21,
        conv_channels=(64, 128),
        kernel_size=3,
        encoder_output_dim=64,
        hidden_dim=64,
        dropout=0.2,
    ):
        super().__init__()

        c1, c2 = conv_channels

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, c1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(c1),
            nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(c2),
        )

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.proj = nn.Sequential(
            nn.Linear(c2, encoder_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(encoder_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, features):
        x = features["onehot"]
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        x = self.pool(x).squeeze(-1)
        return self.proj(x)

    def forward(self, features):
        z = self.encode(features)
        logits = self.classifier(z).squeeze(1)
        return logits

    def forward_with_repr(self, features, n_repr_layers=2):
        from src.models.classifier_repr import run_classifier_with_repr

        z = self.encode(features)
        return run_classifier_with_repr(self.classifier, z, n_repr_layers=n_repr_layers)