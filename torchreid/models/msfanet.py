import torch
import torch.nn as nn
import torch.nn.functional as F
from torchreid.models import register_model

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        mid_channels = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, mid_channels, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc2 = nn.Linear(mid_channels, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean(dim=(2, 3))
        y = self.act(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class FA_Block(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(FA_Block, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv3x3_dw = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, 
            groups=out_channels, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.se = SEBlock(out_channels, reduction=reduction)

    def forward(self, x):
        out = self.conv1x1(x)
        out = self.conv3x3_dw(out)
        out = self.bn(out)
        out = self.act(out)
        return self.se(out)

class AggregationGate(nn.Module):
    def __init__(self, channels):
        super(AggregationGate, self).__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels // 16, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gate(x)

class MSFA_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSFA_Block, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # Daisy-chained 5 FA blocks (receptive fields 3, 7, 11)
        self.fa1 = FA_Block(out_channels, out_channels)
        self.fa2 = FA_Block(out_channels, out_channels)
        self.fa3 = FA_Block(out_channels, out_channels)
        self.fa4 = FA_Block(out_channels, out_channels)
        self.fa5 = FA_Block(out_channels, out_channels)

        self.ag1 = AggregationGate(out_channels)
        self.ag2 = AggregationGate(out_channels)
        self.ag3 = AggregationGate(out_channels)

        self.post_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.post_bn = nn.BatchNorm2d(out_channels)
        self.final_act = nn.LeakyReLU(0.1, inplace=True)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        feat = self.prep(x)

        # Branch 1 (RF = 3)
        b1 = self.fa1(feat)
        # Branch 2 (RF = 7)
        b2 = self.fa3(self.fa2(b1))
        # Branch 3 (RF = 11)
        b3 = self.fa5(self.fa4(b2))

        # Dynamic Scale Fusion
        f1 = 0.5 * (b1 + b2) * self.ag1(feat)
        f2 = 0.5 * (b2 + b3) * self.ag2(feat)
        f3 = b3 * self.ag3(feat)

        out = self.post_conv(f1 + f2 + f3)
        out = self.post_bn(out)
        return self.final_act(out + res)

class MSFANet(nn.Module):
    def __init__(self, num_classes, loss='triplet', **kwargs):
        super(MSFANet, self).__init__()
        self.loss = loss
        self.num_classes = num_classes

        # Stem: 7x7 conv + avg pool
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Stage 1: 1 block, 256 channels
        self.stage1 = nn.Sequential(MSFA_Block(64, 256))
        # Stage 2: 2 blocks, 384 channels
        self.stage2 = nn.Sequential(
            MSFA_Block(256, 384),
            MSFA_Block(384, 384)
        )
        # Stage 3: 3 blocks, 512 channels
        self.stage3 = nn.Sequential(
            MSFA_Block(384, 512),
            MSFA_Block(512, 512),
            MSFA_Block(512, 512)
        )

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes, bias=False)

    def forward(self, x, return_featuremaps=False):
        f = self.stem(x)
        f = self.stage1(f)
        f = self.stage2(f)
        f = self.stage3(f)

        if return_featuremaps:
            return f

        v = self.global_avgpool(f).view(f.size(0), -1)

        if not self.training:
            return v

        y = self.classifier(v)
        if self.loss == 'triplet':
            return y, v
        return y

@register_model('msfanet')
def build_msfanet(num_classes, loss='triplet', **kwargs):
    return MSFANet(num_classes=num_classes, loss=loss, **kwargs)