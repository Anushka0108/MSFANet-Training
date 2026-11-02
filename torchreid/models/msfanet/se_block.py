
import torch
import torch.nn as nn


class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden_channels = max(channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid()
        )

        self.last_attention = None

    def forward(self, x):

        batch, channels, _, _ = x.size()

        weights = self.pool(x).view(batch, channels)

        weights = self.fc(weights).view(
            batch, channels, 1, 1
        )

        self.last_attention = weights.detach()

        return x * weights
