import torch.nn as nn
from torchvision.models import resnet18
from training.models.film import FiLMGenerator

class ResNet18(nn.Module):
    def __init__(
        self,
        in_channels,
        pretrained=True,
        prisma_channels=None,
        film_positions=(),
    ):
        super().__init__()

        self.use_film = prisma_channels is not None
        self.film_positions = set(film_positions)

        try:
            self.model = resnet18(weights="DEFAULT" if pretrained else None)
        except:
            self.model = resnet18(pretrained=pretrained)

        if in_channels != 3:
            old = self.model.conv1
            self.model.conv1 = nn.Conv2d(
                in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )

        self.model.fc = nn.Identity()

        if self.use_film:
            self.film = nn.ModuleDict()
            if "conv1" in self.film_positions:
                self.film["conv1"] = FiLMGenerator(prisma_channels, film_dim=64)
            if "layer1" in self.film_positions:
                self.film["layer1"] = FiLMGenerator(prisma_channels, film_dim=64)
            if "layer2" in self.film_positions:
                self.film["layer2"] = FiLMGenerator(prisma_channels, film_dim=128)
            if "layer3" in self.film_positions:
                self.film["layer3"] = FiLMGenerator(prisma_channels, film_dim=256)

        self.out_channels = 512

    def _film(self, x, g, b):
        return g.unsqueeze(-1).unsqueeze(-1) * x + b.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x, prisma=None):
        m = self.model

        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)

        if self.use_film and "conv1" in self.film:
            g, b = self.film["conv1"](prisma)
            x = self._film(x, g, b)

        x = m.maxpool(x)

        x = m.layer1(x)
        if self.use_film and "layer1" in self.film:
            g, b = self.film["layer1"](prisma)
            x = self._film(x, g, b)

        x = m.layer2(x)
        if self.use_film and "layer2" in self.film:
            g, b = self.film["layer2"](prisma)
            x = self._film(x, g, b)

        x = m.layer3(x)
        if self.use_film and "layer3" in self.film:
            g, b = self.film["layer3"](prisma)
            x = self._film(x, g, b)

        x = m.layer4(x)
        return x