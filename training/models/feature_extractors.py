# training/models/feature_extractors.py
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18(nn.Module):
    """
    ResNet-18 feature extractor that returns a SPATIAL feature map:
        out: [B, 512, H', W']
    """

    def __init__(self, in_channels: int, pretrained: bool = True):
        super().__init__()

        try:
            m = resnet18(weights="DEFAULT" if pretrained else None)
        except TypeError:
            m = resnet18(pretrained=pretrained)

        # Adapt first conv if needed
        if in_channels != 3:
            old = m.conv1
            m.conv1 = nn.Conv2d(
                in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )

        # Keep only the convolutional trunk (no avgpool / fc)
        self.stem = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool
        )
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

        self.out_channels = 512

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x