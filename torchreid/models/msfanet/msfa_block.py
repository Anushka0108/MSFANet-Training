import torch
import torch.nn as nn

from .fa_block import FABlock
from .aggregation_gate import AggregationGate


class MSFABlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.branch1 = nn.ModuleList([FABlock(out_channels, reduction)])
        self.branch2 = nn.ModuleList([FABlock(out_channels, reduction) for _ in range(3)])
        self.branch3 = nn.ModuleList([FABlock(out_channels, reduction) for _ in range(5)])

        self.gates = nn.ModuleList([
            AggregationGate(out_channels, reduction),
            AggregationGate(out_channels, reduction),
            AggregationGate(out_channels, reduction)
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.last_branch_maps = []

    def _run_branch(self, blocks, x):
        outputs = []
        for block in blocks:
            x = block(x)
            outputs.append(x)
        return outputs

    def forward(self, x):
        identity = self.shortcut(x)

        branch1_outputs = self._run_branch(self.branch1, identity)
        branch2_outputs = self._run_branch(self.branch2, identity)
        branch3_outputs = self._run_branch(self.branch3, identity)

        branch1 = 0.5 * (branch1_outputs[-1] + branch2_outputs[0])
        branch2 = 0.5 * (branch2_outputs[-1] + branch3_outputs[2])
        branch3 = branch3_outputs[-1]

        gated1 = branch1 * self.gates[0](identity)
        gated2 = branch2 * self.gates[1](identity)
        gated3 = branch3 * self.gates[2](identity)

        fused = self.fuse(gated1 + gated2 + gated3)
        self.last_branch_maps = [
            gated1.detach().abs().mean(dim=1, keepdim=True),
            gated2.detach().abs().mean(dim=1, keepdim=True),
            gated3.detach().abs().mean(dim=1, keepdim=True)
        ]
        return identity + fused
