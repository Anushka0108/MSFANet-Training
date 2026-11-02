
import torch
import torch.nn as nn


class AggregationGate(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, feature):
        weight = self.gate(x)
        return feature * weight
