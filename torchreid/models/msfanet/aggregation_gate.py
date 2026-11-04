import torch.nn as nn

from ..osnet import ChannelGate


class AggregationGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.gate = ChannelGate(
            channels,
            return_gates=True,
            reduction=reduction
        )

    def forward(self, x):
        return self.gate(x)
