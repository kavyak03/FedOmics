import torch
from torch import nn


class GeneExpressionNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim_1: int = 64, hidden_dim_2: int = 32,
                 dropout_1: float = 0.15, dropout_2: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_1),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_2),
            nn.Linear(hidden_dim_2, 2),
        )

    def forward(self, x):
        return self.net(x)
