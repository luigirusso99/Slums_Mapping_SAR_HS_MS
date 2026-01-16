# training/models/fusion_strategies.py

from __future__ import annotations
import torch
import torch.nn as nn

from .classification_heads import ClassificationHead

class SingleSensorModel(nn.Module):
    """
    Encoder + classification head for a single sensor
    (e.g., only sar or only Planet).

    encoder must expose `.out_channels` and return [B, C, H, W].
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = 2,
        dropout: float = 0.2,
        pool_type: str = "avg",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        if not hasattr(encoder, "out_channels"):
            raise AttributeError("Encoder must have attribute 'out_channels'")
        self.head = ClassificationHead(
            in_channels=encoder.out_channels,
            num_classes=num_classes,
            dropout=dropout,
            pool_type=pool_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)        # [B, C, H, W]
        logits = self.head(feats)      # [B, num_classes]
        return logits

class EarlyFusionModel(nn.Module):
    """
    Early fusion: concat sar & Planet along channel dimension,
    then pass through a single encoder + head.

      x_sar: [B, C_p, H, W]
      x_planet: [B, C_q, H, W]
      x = cat([x_sar, x_planet], dim=1) -> encoder -> head
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = 2,
        dropout: float = 0.2,
        pool_type: str = "avg",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        if not hasattr(encoder, "out_channels"):
            raise AttributeError("Encoder must have attribute 'out_channels'")
        self.head = ClassificationHead(
            in_channels=encoder.out_channels,
            num_classes=num_classes,
            dropout=dropout,
            pool_type=pool_type,
        )

    def forward(self, x_sar: torch.Tensor, x_planet: torch.Tensor) -> torch.Tensor:
        # x_sar: [B, C_p, H, W], x_planet: [B, C_q, H, W]
        x = torch.cat([x_sar, x_planet], dim=1)  # [B, C_p + C_q, H, W]
        feats = self.encoder(x)
        logits = self.head(feats)
        return logits


class MidFusionModel(nn.Module):
    """
    Mid fusion: two encoders (sar, Planet) -> fuse feature maps -> head.

    Fusion is 1x1 conv over concatenated channels:
      F_p: [B, C_p, H, W]
      F_q: [B, C_q, H, W]
      cat -> [B, C_p + C_q, H, W] -> 1x1 Conv -> [B, fusion_dim, H, W] -> head
    """

    def __init__(
        self,
        encoder_sar: nn.Module,
        encoder_planet: nn.Module,
        fusion_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.2,
        pool_type: str = "avg",
    ) -> None:
        super().__init__()
        self.encoder_sar = encoder_sar
        self.encoder_planet = encoder_planet

        if not hasattr(encoder_sar, "out_channels") or not hasattr(
            encoder_planet, "out_channels"
        ):
            raise AttributeError("Encoders must have attribute 'out_channels'")

        c_p = encoder_sar.out_channels
        c_q = encoder_planet.out_channels
        self.fusion_conv = nn.Conv2d(c_p + c_q, fusion_dim, kernel_size=1)

        self.head = ClassificationHead(
            in_channels=fusion_dim,
            num_classes=num_classes,
            dropout=dropout,
            pool_type=pool_type,
        )

    def forward(self, x_sar: torch.Tensor, x_planet: torch.Tensor) -> torch.Tensor:
        f_p = self.encoder_sar(x_sar)  # [B, C_p, H, W]
        f_q = self.encoder_planet(x_planet)  # [B, C_q, H, W]
        f = torch.cat([f_p, f_q], dim=1)     # [B, C_p + C_q, H, W]
        f = self.fusion_conv(f)              # [B, fusion_dim, H, W]
        logits = self.head(f)                # [B, num_classes]
        return logits

class LateFusionModel(nn.Module):
    """
    Late fusion: two FULL models (SingleSensorModel or simili),
    average (or weighted sum) of logits.

    model_sar(x_sar) -> logits_p
    model_planet(x_planet) -> logits_q
    logits = alpha * logits_p + (1 - alpha) * logits_q
    """

    def __init__(
        self,
        model_sar: nn.Module,
        model_planet: nn.Module,
        alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.model_sar = model_sar
        self.model_planet = model_planet
        self.alpha = alpha

    def forward(self, x_sar: torch.Tensor, x_planet: torch.Tensor) -> torch.Tensor:
        logits_p = self.model_sar(x_sar)  # [B, num_classes]
        logits_q = self.model_planet(x_planet)  # [B, num_classes]
        return self.alpha * logits_p + (1.0 - self.alpha) * logits_q