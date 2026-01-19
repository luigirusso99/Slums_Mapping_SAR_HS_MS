#training/models/prisma_embedding
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# OPTION 1 — Spectral 1D-CNN 
# ============================================================
class PrismaSpectralEmbedding(nn.Module):
    """
    PRISMA PCA embedding using spectral-only processing.

    Input:
        x: [B, C_pca, H, W]   (e.g. C_pca=16, H=W=23)

    Strategy:
        - Spatial average → per-pixel spectra
        - 1D CNN along spectral dimension
        - Global pooling → embedding z

    Output:
        z: [B, emb_dim]
    """

    def __init__(self, in_channels, emb_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),  # collapse spectral dimension
        )

        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        # spatial average → [B, C]
        x = x.mean(dim=(2, 3))          # [B, C]

        # add fake spectral length dimension
        x = x.unsqueeze(-1)             # [B, C, 1]

        x = self.net(x)                 # [B, 128, 1]
        x = x.squeeze(-1)               # [B, 128]

        z = self.fc(x)                  # [B, emb_dim]
        return z


# ============================================================
# OPTION 2 — Lightweight 2D-CNN 
# ============================================================
class PrismaSpatialEmbedding(nn.Module):
    """
    PRISMA PCA embedding using lightweight spatial CNN.

    Input:
        x: [B, C_pca, H, W]

    Output:
        z: [B, emb_dim]
    """

    def __init__(self, in_channels, emb_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),  # [B, 64, 1, 1]
        )

        self.fc = nn.Linear(64, emb_dim)

    def forward(self, x):
        x = self.conv(x)                # [B, 64, 1, 1]
        x = x.view(x.size(0), -1)       # [B, 64]
        z = self.fc(x)                  # [B, emb_dim]
        return z