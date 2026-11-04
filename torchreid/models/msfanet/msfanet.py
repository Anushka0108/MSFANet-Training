import torch
import torch.nn as nn

from .msfa_block import MSFABlock


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class MSFANet(nn.Module):
    def __init__(self, num_classes, loss="softmax", pretrained=False):
        super().__init__()
        self.loss = loss

        self.conv1 = ConvLayer(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._make_stage(64, 256, blocks=1, first_stride=1)
        self.stage2 = self._make_stage(256, 384, blocks=2, first_stride=2)
        self.stage3 = self._make_stage(384, 512, blocks=3, first_stride=2)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

        self.last_activation_map = None

        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, blocks, first_stride):
        layers = []
        for index in range(blocks):
            stride = first_stride if index == 0 else 1
            block_in = in_channels if index == 0 else out_channels
            layers.append(MSFABlock(block_in, out_channels, stride=stride))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.001)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        self.last_activation_map = x.detach().abs().mean(dim=1, keepdim=True)

        features = self.global_avgpool(x).flatten(1)
        if not self.training:
            return features

        logits = self.classifier(features)
        return logits, features

    def get_visual_maps(self):
        maps = {"final_activation": self.last_activation_map}
        index = 0
        for stage_name in ("stage1", "stage2", "stage3"):
            stage = getattr(self, stage_name)
            for block_index, block in enumerate(stage):
                for branch_name in ("branch1", "branch2", "branch3"):
                    branch = getattr(block, branch_name)
                    for fa_index, fa in enumerate(branch):
                        if fa.last_attention_map is not None:
                            maps[
                                f"{stage_name}_block{block_index + 1}_{branch_name}_fa{fa_index + 1}"
                            ] = fa.last_attention_map
                for branch_index, branch_map in enumerate(block.last_branch_maps):
                    maps[
                        f"{stage_name}_block{block_index + 1}_branch{branch_index + 1}"
                    ] = branch_map
                index += 1
        return maps


def msfanet(num_classes, loss="softmax", pretrained=False, **kwargs):
    return MSFANet(num_classes=num_classes, loss=loss, pretrained=pretrained)
