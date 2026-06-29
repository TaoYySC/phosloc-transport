import torch
import torch.nn as nn


class TabularEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128), output_dim=64, dropout=0.2):
        super().__init__()

        dims = [input_dim] + list(hidden_dims)
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], output_dim))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)