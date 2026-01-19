import torch
import torch.nn as nn
from training.models.feature_extractors import ResNet18
from training.models.classification_heads import ClassificationHead
from training.models.prisma_embedding import PrismaSpectralEmbedding

class SingleSensorModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        prisma_channels=None,
        prisma_emb_dim=128,
        dropout=0.2,
        pool_type="avg",
        pretrained=True,
    ):
        super().__init__()

        self.encoder = ResNet18(in_channels, pretrained)
        self.use_prisma = prisma_channels is not None

        if self.use_prisma:
            self.prisma_embed = PrismaSpectralEmbedding(prisma_channels, prisma_emb_dim)
            head_in = 512 + prisma_emb_dim
        else:
            self.prisma_embed = None
            head_in = 512

        self.head = ClassificationHead(head_in, num_classes, dropout, pool_type)

    def forward(self, x, prisma=None):
        f = self.encoder(x).mean(dim=(-2, -1))  # [B, 512]

        if self.use_prisma:
            if prisma is None:
                raise RuntimeError("PRISMA enabled but prisma=None")
            z = self.prisma_embed(prisma)
            f = torch.cat([f, z], dim=1)

        return self.head(f)

class EarlyFusion(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        prisma_channels=None,
        prisma_emb_dim=128,
        dropout=0.2,
        pool_type="avg",
        pretrained=True,
    ):
        super().__init__()

        self.encoder = ResNet18(in_channels, pretrained)
        self.use_prisma = prisma_channels is not None

        if self.use_prisma:
            self.prisma_embed = PrismaSpectralEmbedding(prisma_channels, prisma_emb_dim)
            head_in = 512 + prisma_emb_dim
        else:
            self.prisma_embed = None
            head_in = 512

        self.head = ClassificationHead(head_in, num_classes, dropout, pool_type)

    def forward(self, x_sar, x_planet, prisma=None):
        x = torch.cat([x_sar, x_planet], dim=1)
        f = self.encoder(x).mean(dim=(-2, -1))  # [B, 512]

        if self.use_prisma:
            z = self.prisma_embed(prisma)
            f = torch.cat([f, z], dim=1)

        return self.head(f)

class MidFusion(nn.Module):
    def __init__(
        self,
        sar_channels,
        planet_channels,
        num_classes,
        prisma_channels=None,
        prisma_emb_dim=128,
        dropout=0.2,
        pool_type="avg",
        pretrained=True,
    ):
        super().__init__()

        self.enc_sar = ResNet18(sar_channels, pretrained)
        self.enc_planet = ResNet18(planet_channels, pretrained)

        self.fuse = nn.Conv2d(1024, 512, kernel_size=1)
        self.use_prisma = prisma_channels is not None

        if self.use_prisma:
            self.prisma_embed = PrismaSpectralEmbedding(prisma_channels, prisma_emb_dim)
            head_in = 512 + prisma_emb_dim
        else:
            self.prisma_embed = None
            head_in = 512

        self.head = ClassificationHead(head_in, num_classes, dropout, pool_type)

    def forward(self, x_sar, x_planet, prisma=None):
        f_s = self.enc_sar(x_sar)      # [B,512,H,W]
        f_p = self.enc_planet(x_planet)

        f = torch.cat([f_s, f_p], dim=1)  # [B,1024,H,W]
        f = self.fuse(f)                  # [B,512,H,W]
        f = f.mean(dim=(-2, -1))          # [B,512]

        if self.use_prisma:
            z = self.prisma_embed(prisma)
            f = torch.cat([f, z], dim=1)

        return self.head(f)
    
class LateFusion(nn.Module):
    def __init__(
        self,
        sar_channels,
        planet_channels,
        num_classes,
        prisma_channels=None,
        prisma_emb_dim=128,
        dropout=0.2,
        pool_type="avg",
        pretrained=True,
    ):
        super().__init__()

        self.enc_sar = ResNet18(sar_channels, pretrained)
        self.enc_planet = ResNet18(planet_channels, pretrained)
        self.use_prisma = prisma_channels is not None

        if self.use_prisma:
            self.prisma_embed = PrismaSpectralEmbedding(prisma_channels, prisma_emb_dim)
            head_in = 1024 + prisma_emb_dim
        else:
            self.prisma_embed = None
            head_in = 1024

        self.head = ClassificationHead(head_in, num_classes, dropout, pool_type)

    def forward(self, x_sar, x_planet, prisma=None):
        f_s = self.enc_sar(x_sar).mean(dim=(-2, -1))
        f_p = self.enc_planet(x_planet).mean(dim=(-2, -1))

        feats = [f_s, f_p]

        if self.use_prisma:
            z = self.prisma_embed(prisma)
            feats.append(z)

        return self.head(torch.cat(feats, dim=1))