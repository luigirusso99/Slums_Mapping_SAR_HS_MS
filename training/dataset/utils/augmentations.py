# training/augmentations.py

import torch
import random

class GeoAugmentations:
    """
    Physically meaningful augmentations for multisensor EO patches.
    Allowed:
        - horizontal flip (urban blocks symmetric)
        - vertical flip
        - 90-degree rotation (valid for scene classification)
    Forbidden:
        - arbitrary rotations
        - color/radiometric jitter
    """

    def __init__(self, hflip=True, vflip=True, rot90=True):
        self.hflip = hflip
        self.vflip = vflip
        self.rot90 = rot90

    def __call__(self, prisma, planet):
        # --- Horizontal flip
        if self.hflip and random.random() < 0.5:
            prisma = prisma.flip(-1) if prisma is not None else None
            planet = planet.flip(-1) if planet is not None else None

        # --- Vertical flip
        if self.vflip and random.random() < 0.5:
            prisma = prisma.flip(-2) if prisma is not None else None
            planet = planet.flip(-2) if planet is not None else None

        # --- 90° rotation
        if self.rot90 and random.random() < 0.5:
            prisma = prisma.transpose(-1, -2) if prisma is not None else None
            planet = planet.transpose(-1, -2) if planet is not None else None

        return prisma, planet