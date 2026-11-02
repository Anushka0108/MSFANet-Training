
import torch
import torch.nn as nn

from .msfa_block import MSFABlock


class MSFANet(nn.Module):
    def __init__(
        self,
        num_classes,
        loss="softmax",
        pretrained=False,
        use_gpu=True,
        feature_dim=512
    ):
        super().__init__()

        if loss not in ["softmax", "triplet"]:
            raise ValueError("Unsupported loss: {}".format(loss))

        self.loss = loss
        self.feature_dim = feature_dim

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.transition1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.stage1 = nn.Sequential(
            MSFABlock(256, 256)
        )

        self.transition2 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.stage2 = nn.Sequential(
            MSFABlock(384, 384),
            MSFABlock(384, 384)
        )

        self.transition3 = nn.Sequential(
            nn.Conv2d(384, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.stage3 = nn.Sequential(
            MSFABlock(512, 512),
            MSFABlock(512, 512),
            MSFABlock(512, 512)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_dim, num_classes)

        self._init_params()

    def _init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="leaky_relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def featuremaps(self, x):
        x = self.stem(x)
        x = self.transition1(x)
        x = self.stage1(x)
        x = self.transition2(x)
        x = self.stage2(x)
        x = self.transition3(x)
        x = self.stage3(x)
        return x

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)

        if return_featuremaps:
            return x

        features = self.global_pool(x)
        features = torch.flatten(features, 1)

        if not self.training:
            return features

        outputs = self.classifier(features)

        if self.loss == "softmax":
            return outputs

        return outputs, features


def msfanet(
    num_classes,
    loss="softmax",
    pretrained=False,
    use_gpu=True
):
    return MSFANet(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu
    )
