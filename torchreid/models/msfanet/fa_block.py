
import torch
import torch.nn as nn

from .se_block import SEBlock


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, reduction=16):
        super().__init__()

        hidden_channels = max(out_channels // reduction, 1)

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.activation1 = nn.LeakyReLU(0.1, inplace=True)

        self.expand = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation2 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.expand(x)
        x = self.bn2(x)
        return self.activation2(x)


class FABlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = DepthwiseSeparableConv(
            in_channels, out_channels, kernel_size=1
        )

        self.conv2 = DepthwiseSeparableConv(
            out_channels, out_channels, kernel_size=3, stride=stride
        )

        self.attention = SEBlock(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        return x
