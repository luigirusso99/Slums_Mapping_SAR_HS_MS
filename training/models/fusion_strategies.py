import torch
import torch.nn as nn
from training.models.feature_extractors import ResNet18
from training.models.classification_heads import ClassificationHead
from training.models.film import FiLMGenerator

class SingleSensorModel(nn.Module):
    """
    Single-sensor baseline (SAR or Planet).
    Optional PRISMA FiLM conditioning can be enabled.
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        prisma_channels=None,
        dropout=0.2,
        pool_type="avg",
        pretrained=True,
        film_positions=("layer1", "layer2", "layer3"),
    ):
        super().__init__()

        self.encoder = ResNet18(
            in_channels=in_channels,
            pretrained=pretrained,
            prisma_channels=prisma_channels,
            film_positions=film_positions if prisma_channels is not None else (),
        )

        self.head = ClassificationHead(
            in_channels=self.encoder.out_channels,
            num_classes=num_classes,
            dropout=dropout,
            pool_type=pool_type,
        )

    def forward(self, x, prisma=None):
        """
        Forward pass for single-sensor model.

        If prisma is provided and the encoder was initialized with
        prisma_channels, FiLM conditioning is applied.
        """
        feats = self.encoder(x, prisma) if prisma is not None else self.encoder(x)
        return self.head(feats)

class EarlyFusion(nn.Module):
    def __init__(self, in_channels, num_classes, prisma_channels=None,
                 dropout=0.2, pool_type="avg", pretrained=True):
        super().__init__()

        self.encoder = ResNet18(
            in_channels,
            pretrained,
            prisma_channels,
            film_positions=("conv1",),
        )

        self.head = ClassificationHead(
            self.encoder.out_channels,
            num_classes,
            dropout,
            pool_type,
        )

    def forward(self, x_sar, x_planet, prisma=None):
        """
        Early fusion:
        - concatenate SAR and Planet at input level
        - optional PRISMA FiLM conditioning inside ResNet18
        """
        x = torch.cat([x_sar, x_planet], dim=1)

        if self.encoder.use_film:
            f = self.encoder(x, prisma)
        else:
            f = self.encoder(x)

        return self.head(f)
    
class MidFusion(nn.Module):
    def __init__(self, in_channels, num_classes, prisma_channels=None,
                 dropout=0.2, pool_type="avg", pretrained=True):
        super().__init__()

        self.encoder = ResNet18(
            in_channels,
            pretrained,
            prisma_channels,
            film_positions=("layer1", "layer2", "layer3"),
        )

        self.head = ClassificationHead(
            self.encoder.out_channels,
            num_classes,
            dropout,
            pool_type,
        )

    def forward(self, x_sar, x_planet, prisma=None):
        """
        Mid fusion:
        - concatenate SAR and Planet at input
        - FiLM applied at intermediate ResNet layers only if enabled
        """
        x = torch.cat([x_sar, x_planet], dim=1)

        if self.encoder.use_film:
            f = self.encoder(x, prisma)
        else:
            f = self.encoder(x)

        return self.head(f)
    
class LateFusion(nn.Module):
    def __init__(self, sar_channels, planet_channels, num_classes,
                 prisma_channels=None, dropout=0.2, pool_type="avg", pretrained=True):
        super().__init__()

        self.enc_sar = ResNet18(sar_channels, pretrained)
        self.enc_planet = ResNet18(planet_channels, pretrained)

        self.film = (
            FiLMGenerator(prisma_channels, film_dim=512)
            if prisma_channels is not None
            else None
        )

        self.head = ClassificationHead(
            1024,
            num_classes,
            dropout,
            pool_type,
        )

    def forward(self, x_sar, x_planet, prisma=None):
        if self.film is not None and prisma is None:
            raise RuntimeError(
                "LateFusion was initialized with FiLM, but prisma=None was passed to forward()"
            )
        f_s = self.enc_sar(x_sar)
        f_p = self.enc_planet(x_planet)

        if self.film:
            g, b = self.film(prisma)
            g = g.unsqueeze(-1).unsqueeze(-1)
            b = b.unsqueeze(-1).unsqueeze(-1)
            f_s = g * f_s + b
            f_p = g * f_p + b

        f = torch.cat([f_s, f_p], dim=1)
        return self.head(f)