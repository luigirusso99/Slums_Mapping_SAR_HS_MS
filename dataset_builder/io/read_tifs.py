# dataset_builder/io/read_tifs.py
import rasterio
import numpy as np

def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read()  # (C, H, W)
        arr = np.transpose(arr, (1, 2, 0))  # → (H, W, C)
        profile = src.profile
    return arr, profile