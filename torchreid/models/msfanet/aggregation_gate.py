
import torch
import torch.nn as nn


class AggregationGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden_channels = max(channels // reduction, 1)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, feature):
        weight = self.gate(x)
        return feature * weight
