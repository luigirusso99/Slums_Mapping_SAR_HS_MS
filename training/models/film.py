import torch
import torch.nn as nn

class FiLMGenerator(nn.Module):
    def __init__(self, in_channels=230, hidden_dim=128, film_dim=256):
        super().__init__()

        self.bottleneck = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.modulator = nn.Linear(hidden_dim, film_dim * 2)

        nn.init.zeros_(self.modulator.weight)
        nn.init.zeros_(self.modulator.bias)

    def forward(self, x):
        x = x.mean(dim=[2, 3])          # [B, C]
        z = self.bottleneck(x)
        gamma, beta = self.modulator(z).chunk(2, dim=1)
        gamma = 1 + gamma
        return gamma, beta