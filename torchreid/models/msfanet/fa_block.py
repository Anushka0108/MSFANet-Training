import torch
import torch.nn as nn

from .se_block import SEBlock


class FABlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pointwise = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1,
            groups=channels, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.se = SEBlock(channels, reduction=reduction)
        self.last_attention_map = None

    def forward(self, x):
        x = self.pointwise(x)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.act(x)
        x, weights = self.se(x)
        self.last_attention_map = x.detach().abs().mean(dim=1, keepdim=True)
        return x
