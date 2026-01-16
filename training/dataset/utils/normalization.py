# training/normalization.py

import pandas as pd
import torch


class ZScoreNormalizer:
    """
    Callable class for Z-score normalization:
        x_norm = (x - mean) / std
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean[:, None, None]   # shape → (C,1,1)
        self.std  = std[:, None, None]
        self.std[self.std == 0] = 1.0     # safety

    def __call__(self, x: torch.Tensor):
        return (x - self.mean) / self.std