import torch.nn as nn


class OneHotCNN2DClassifier(nn.Module):
    def __init__(
        self,
        input_dim=21,
        conv_channels=(32, 64),
        seq_kernel_size=3,
        encoder_output_dim=64,
        hidden_dim=64,
        dropout=0.2,
    ):
        super().__init__()

        c1, c2 = conv_channels

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=c1,
                kernel_size=(seq_kernel_size, input_dim),
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

        self.pool = nn.AdaptiveMaxPool2d((1, 1))

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
        x = x.unsqueeze(1)
        x = self.feature_extractor(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.proj(x)

    def forward(self, features):
        z = self.encode(features)
        logits = self.classifier(z).squeeze(1)
        return logits

    def forward_with_repr(self, features, n_repr_layers=2):
        from src.models.classifier_repr import run_classifier_with_repr

        z = self.encode(features)
        return run_classifier_with_repr(self.classifier, z, n_repr_layers=n_repr_layers)