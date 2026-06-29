import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionMLP(nn.Module):
    """Projection encoder with a temporary classifier head for supervised training."""

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        projection_dim=16,
        dropout=0.3,
        num_classes=2,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.projection_dim = int(projection_dim)
        self.num_classes = int(num_classes)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.projection_dim),
        )
        self.classifier = nn.Linear(self.projection_dim, self.num_classes)

    def forward_projection(self, x):
        z = self.encoder(x)
        return F.normalize(z, dim=1)

    def forward_logits(self, x):
        z = self.forward_projection(x)
        return self.classifier(z)

    def forward(self, x):
        return self.forward_logits(x)

    def encoder_state_dict(self):
        return self.encoder.state_dict()

    def load_encoder_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict)
