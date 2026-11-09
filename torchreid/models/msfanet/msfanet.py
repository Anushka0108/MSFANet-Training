
import torch
import torch.nn as nn

from ..osnet import ConvLayer
from .msfa_block import MSFABlock


class MSFANet(nn.Module):
    def __init__(
        self,
        num_classes,
        loss="softmax",
        pretrained=False,
        widths=(128, 192, 256),
        reduction=16,
        **kwargs
    ):
        super().__init__()

        self.loss = loss
        self.feature_dim = 512
        self.pretrained = pretrained

        self.conv1 = ConvLayer(
            3, 64, kernel_size=7, stride=2, padding=3
        )

        self.pool1 = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=1
        )

        self.stage1 = self._make_stage(
            64, 256, blocks=1, width=widths[0],
            first_stride=1, reduction=reduction
        )

        self.stage2 = self._make_stage(
            256, 384, blocks=2, width=widths[1],
            first_stride=2, reduction=reduction
        )

        self.stage3 = self._make_stage(
            384, 512, blocks=3, width=widths[2],
            first_stride=2, reduction=reduction
        )

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.feature_bn = nn.BatchNorm1d(self.feature_dim)
        self.feature_bn.bias.requires_grad_(False)

        self.classifier = nn.Linear(
            self.feature_dim, num_classes
        )

        self.last_activation_map = None

        self._initialize_weights()

    def _make_stage(
        self,
        in_channels,
        out_channels,
        blocks,
        width,
        first_stride,
        reduction
    ):
        layers = []

        for index in range(blocks):
            block_in = in_channels if index == 0 else out_channels
            stride = first_stride if index == 0 else 1

            layers.append(
                MSFABlock(
                    block_in,
                    out_channels,
                    width=width,
                    stride=stride,
                    reduction=reduction
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="leaky_relu"
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.001)
                nn.init.zeros_(module.bias)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        self.last_activation_map = (
            x.detach().abs().mean(dim=1, keepdim=True)
        )

        return x

    def forward(self, x):
        x = self.featuremaps(x)
        features = self.global_avgpool(x).flatten(1)

        bn_features = self.feature_bn(features)

        if not self.training:
            return bn_features

        logits = self.classifier(bn_features)

        return logits, features

    def get_visual_maps(self):
        maps = {
            "final_activation": self.last_activation_map
        }

        for stage_name in ("stage1", "stage2", "stage3"):
            stage = getattr(self, stage_name)

            for block_index, block in enumerate(stage):
                prefix = "{}_block{}".format(
                    stage_name,
                    block_index + 1
                )

                for branch_index, activation_map in enumerate(
                    block.last_pre_gate_maps
                ):
                    maps[
                        "{}_branch{}_before_gate".format(
                            prefix, branch_index + 1
                        )
                    ] = activation_map

                for branch_index, activation_map in enumerate(
                    block.last_branch_maps
                ):
                    maps[
                        "{}_branch{}_after_gate".format(
                            prefix, branch_index + 1
                        )
                    ] = activation_map

                for branch_index, gate_map in enumerate(
                    block.last_gate_visual_maps
                ):
                    maps[
                        "{}_gate{}_channel_mean".format(
                            prefix, branch_index + 1
                        )
                    ] = gate_map

                for branch_index, gate_map in enumerate(
                    block.last_gate_maps
                ):
                    maps[
                        "{}_gate{}_channel_weights".format(
                            prefix, branch_index + 1
                        )
                    ] = gate_map

                maps["{}_fused_activation".format(prefix)] = (
                    block.last_fused_map
                )

                for branch_name in (
                    "branch1",
                    "branch2",
                    "branch3"
                ):
                    branch = getattr(block, branch_name)

                    for fa_index, fa in enumerate(branch):
                        if fa.last_attention_map is not None:
                            maps[
                                "{}_{}_fa{}_se_channel_weights".format(
                                    prefix,
                                    branch_name,
                                    fa_index + 1
                                )
                            ] = fa.last_attention_map

        return maps


def msfanet(
    num_classes,
    loss="softmax",
    pretrained=False,
    **kwargs
):
    return MSFANet(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs
    )
