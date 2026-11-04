import torch
import torch.nn as nn

from .fa_block import FABlock
from .aggregation_gate import AggregationGate


class MSFABlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        width,
        stride=1,
        reduction=16
    ):
        super().__init__()

        self.project = nn.Sequential(
            nn.Conv2d(
                in_channels,
                width,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm2d(width),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.branch1 = nn.Sequential(
            FABlock(width, reduction)
        )

        self.branch2 = nn.Sequential(
            *[FABlock(width, reduction) for _ in range(3)]
        )

        self.branch3 = nn.Sequential(
            *[FABlock(width, reduction) for _ in range(5)]
        )

        self.gates = nn.ModuleList([
            AggregationGate(width, reduction),
            AggregationGate(width, reduction),
            AggregationGate(width, reduction)
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(
                width * 3,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.last_branch_maps = []
        self.last_gate_maps = []

    def forward(self, x):
        identity = self.shortcut(x)
        branch_input = self.project(x)

        b1 = self.branch1(branch_input)

        b2_outputs = []
        value = branch_input
        for block in self.branch2:
            value = block(value)
            b2_outputs.append(value)
        b2 = b2_outputs[-1]

        b3_outputs = []
        value = branch_input
        for block in self.branch3:
            value = block(value)
            b3_outputs.append(value)
        b3 = b3_outputs[-1]

        # Cross-branch averaging described in Equation (3).
        b1 = 0.5 * (b1 + b2_outputs[0])
        b2 = 0.5 * (b2 + b3_outputs[2])

        gate1 = self.gates[0](branch_input)
        gate2 = self.gates[1](branch_input)
        gate3 = self.gates[2](branch_input)

        g1 = b1 * gate1
        g2 = b2 * gate2
        g3 = b3 * gate3

        fused = g1 + g2 + g3
        fused = self.fuse(fused)

        self.last_branch_maps = [
            g1.detach().abs().mean(dim=1, keepdim=True),
            g2.detach().abs().mean(dim=1, keepdim=True),
            g3.detach().abs().mean(dim=1, keepdim=True)
        ]

        self.last_gate_maps = [
            gate1.detach(),
            gate2.detach(),
            gate3.detach()
        ]

        return self.activation(identity + fused)