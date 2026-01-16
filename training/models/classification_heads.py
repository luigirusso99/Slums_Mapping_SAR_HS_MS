# training/models/classification_heads.py

from __future__ import annotations
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Simple classification head:
      - Global pooling (avg / max / avg+max)
      - Dropout
      - Linear layer -> logits [B, 1]

    Assumes encoder output is [B, C, H, W].
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        dropout: float = 0.2,
        pool_type: str = "avg",  # "avg", "max", "avgmax"
    ) -> None:
        super().__init__()

        pool_type = pool_type.lower()
        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif pool_type == "avgmax":
            # avg + max concatenati → 2*C features
            self.pool = None  # gestito in forward
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")

        self.pool_type = pool_type
        self.dropout = nn.Dropout(dropout)

        if pool_type == "avgmax":
            fc_in = in_channels * 2
        else:
            fc_in = in_channels

        self.fc = nn.Linear(fc_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        returns logits: [B, 1]
        """
        if x.dim() != 4:
            raise ValueError(f"ClassificationHead expected [B, C, H, W], got {x.shape}")

        if self.pool_type == "avgmax":
            avg = torch.mean(x, dim=(-2, -1), keepdim=True)  # [B, C, 1, 1]
            mx = torch.amax(x, dim=(-2, -1), keepdim=True)   # [B, C, 1, 1]
            x = torch.cat([avg, mx], dim=1)                  # [B, 2C, 1, 1]
        else:
            x = self.pool(x)  # [B, C, 1, 1]

        x = x.flatten(1)      # [B, C] or [B, 2C]
        x = self.dropout(x)
        logits = self.fc(x)
        return logits