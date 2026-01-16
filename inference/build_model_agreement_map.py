#!/usr/bin/env python3
"""
Build Model Agreement Map (Mean Slum Probability)

- Reads full-resolution probability maps from multiple models
- Computes agreement as mean probability across models
- Preserves CRS, resolution, and geotransform
- Saves agreement map as GeoTIFF

Output:
  • model_agreement_mean_probability.tif
"""

import os
import glob
import yaml
import rasterio
import numpy as np
from typing import List


# ============================================================
# CONFIG
# ============================================================
EXPERIMENT_YAML = (
    "/Users/luigi/Desktop/PHD/PROGETTI_E_ATTIVITA_PHD/HEATCORDOBA/"
    "slums/Python_Project_HS_MS/training/configs/experiment.yaml"
)

INFER_BASE_DIR = (
    "/Users/luigi/Desktop/PHD/PROGETTI_E_ATTIVITA_PHD/HEATCORDOBA/"
    "slums/Python_Project_HS_MS/inference/predictions"
)

OUTPUT_NAME = "model_agreement_mean_probability.tif"

MODEL_KEYS = [
    "resnet18",
    "efficientnet_b0",
    "mobilenet_v3_small",
    "convnext_tiny",
    #"swin_tiny_patch4_window7_224",
]


# ============================================================
# CORE LOGIC
# ============================================================
def load_experiment_cfg(cfg_path: str):
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def find_fullres_maps(infer_root: str, model_keys: List[str]):
    maps = []
    for key in model_keys:
        files = glob.glob(os.path.join(infer_root, key, "*fullres.tif"))
        if files:
            maps.append(files[0])
    if not maps:
        raise RuntimeError("No full-resolution probability maps found.")
    return maps


def build_agreement_map(tif_paths: List[str]):
    prob_stack = []
    ref_profile = None

    for i, path in enumerate(tif_paths):
        with rasterio.open(path) as src:
            prob = src.read(1).astype(np.float32)
            prob_stack.append(prob)

            if ref_profile is None:
                ref_profile = src.profile

    agreement = np.nanmean(np.stack(prob_stack, axis=0), axis=0)
    return agreement, ref_profile


def save_geotiff(out_path: str, data: np.ndarray, profile: dict):
    profile = profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        compress="deflate",
        predictor=2,
        nodata=None,
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n▶ Building Model Agreement Map\n")

    cfg = load_experiment_cfg(EXPERIMENT_YAML)

    mode = cfg["data"].get("mode", "planet")
    fusion = cfg["model"].get("fusion_type", "none")

    infer_root = os.path.join(INFER_BASE_DIR, mode, fusion)

    print(f"→ Inference directory: {infer_root}")

    tif_paths = find_fullres_maps(infer_root, MODEL_KEYS)

    print(f"→ Found {len(tif_paths)} model probability maps")

    agreement, profile = build_agreement_map(tif_paths)

    out_path = os.path.join(infer_root, OUTPUT_NAME)

    save_geotiff(out_path, agreement, profile)

    print(f"\n✔ Agreement map saved to:\n  {out_path}\n")


if __name__ == "__main__":
    main()