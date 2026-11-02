
import torch
import torch.nn as nn

from .fa_block import FABlock
from .aggregation_gate import AggregationGate


class MSFABlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.branch1 = nn.Sequential(
            FABlock(in_channels, out_channels)
        )

        self.branch2 = nn.Sequential(
            FABlock(in_channels, out_channels),
            FABlock(out_channels, out_channels),
            FABlock(out_channels, out_channels)
        )

        self.branch3 = nn.Sequential(
            FABlock(in_channels, out_channels),
            FABlock(out_channels, out_channels),
            FABlock(out_channels, out_channels),
            FABlock(out_channels, out_channels),
            FABlock(out_channels, out_channels)
        )

        self.gate1 = AggregationGate(out_channels)
        self.gate2 = AggregationGate(out_channels)

        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        )

        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        identity = self.residual(x)

        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        fused2 = (out1 + self.gate1(out1, out2)) / 2
        fused3 = (fused2 + self.gate2(fused2, out3)) / 2

        return self.activation(identity + fused3)
