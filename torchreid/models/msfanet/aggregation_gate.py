import torch.nn as nn


class AggregationGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.expand = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gate = self.pool(x)
        gate = self.reduce(gate)
        gate = self.act(gate)
        gate = self.expand(gate)
        return self.sigmoid(gate)
