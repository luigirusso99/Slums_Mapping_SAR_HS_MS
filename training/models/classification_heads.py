import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        dropout: float = 0.2,
        pool_type: str = "avg",
    ):
        super().__init__()

        pool_type = pool_type.lower()
        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif pool_type == "avgmax":
            self.pool = None
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")

        self.pool_type = pool_type
        self.dropout = nn.Dropout(dropout)

        fc_in = in_channels * 2 if pool_type == "avgmax" else in_channels
        self.fc = nn.Linear(fc_in, num_classes)

    def forward(self, x):
        if self.pool_type == "avgmax":
            avg = torch.mean(x, dim=(-2, -1), keepdim=True)
            mx = torch.amax(x, dim=(-2, -1), keepdim=True)
            x = torch.cat([avg, mx], dim=1)
        else:
            x = self.pool(x)

        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)