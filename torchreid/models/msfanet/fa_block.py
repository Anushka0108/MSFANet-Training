import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weights = self.pool(x)
        weights = self.fc1(weights)
        weights = self.relu(weights)
        weights = self.fc2(weights)
        weights = self.sigmoid(weights)
        return x * weights, weights


class FABlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.pointwise = nn.Conv2d(
            channels, channels, kernel_size=1, bias=False
        )
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.se = SEBlock(channels, reduction)
        self.last_attention_map = None

    def forward(self, x):
        x = self.pointwise(x)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.act(x)

        x, weights = self.se(x)
        self.last_attention_map = weights.detach()

        return x
