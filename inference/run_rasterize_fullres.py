# inference/run_rasterize_fullres.py

"""
Builds a full-resolution slum probability map (GeoTIFF) from patch-level OOF predictions.

STRATEGY 2:
-----------
- Each patch prediction contributes only its central "valid" region (approximately stride×stride,
  e.g. 179×179 pixels for patch_size=224, stride=179) at 3 m resolution (Planet grid).
- The valid region is centered within the patch; outer borders are discarded to avoid
  blocky artifacts.
- Overlapping valid regions are averaged across all covering patches (smooth aggregation).
- The final raster has the same pixel size as the original Planet common grid
  and covers the full city extent spanned by all patches.

Inputs:
  • experiment/config YAML (same used for training + inference)
  • metadata.csv        (contains x, y, planet_path, etc.)
  • oof_predictions.csv (contains patch_id, prob_slum from OOF inference)

Output:
  • slum_probability_fullres.tif (float32, in probability [0, 1], with nodata)
"""

import os
import yaml
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from typing import Optional


def run_rasterize_fullres(cfg_path: str, output_dir: Optional[str] = None):
    print("\n▶ Building FULL-RESOLUTION slum probability raster (Strategy 2)...\n")

    # ------------------------
    # Load config
    # ------------------------
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    metadata_csv = data_cfg["metadata_csv"]
    fusion_type = model_cfg["fusion_type"]
    sensor_type = model_cfg.get("sensor_type")
    prisma_flag = "prisma" if model_cfg.get("use_prisma", False) else "no_prisma"

    # Patch geometry from config
    patch_cfg = cfg.get("patch", {})
    patch_size = int(patch_cfg.get("size", 224))
    stride = int(patch_cfg.get("stride", 179))  # not strictly needed, but logged

    print(f"→ Using patch_size={patch_size}, stride={stride}")

    # Compute overlap and central valid window (Stark et al. style):
    # the network effectively uses only the interior (stride x stride) region
    # of each patch, discarding the borders.
    # Removed computation of pad, inner_size.
    # Instead create full patch kernel:
    # Use a circularly weighted, smoother radial kernel.
    Y, X = np.ogrid[:patch_size, :patch_size]
    cy, cx = patch_size // 2, patch_size // 2
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    radius = patch_size / 2
    radial_kernel = np.clip(1 - (dist / radius), 0, 1)

    from scipy.ndimage import gaussian_filter
    kernel = gaussian_filter(radial_kernel.astype(np.float32), sigma=patch_size / 6.0)
    kernel = kernel / kernel.max()

    # Path to OOF predictions (same convention as run_training.py)
    path_parts = [cfg["inference_save_dir"], fusion_type]
    if fusion_type == "single":
        path_parts.append(sensor_type)
    path_parts.append(prisma_flag)
    oof_dir = os.path.join(*path_parts)
    oof_csv = os.path.join(oof_dir, "oof_predictions.csv")

    if not os.path.exists(oof_csv):
        raise FileNotFoundError(f"Missing OOF CSV: {oof_csv}")

    print(f"→ Reading predictions: {oof_csv}")
    print(f"→ Reading metadata:    {metadata_csv}")
    print(
        f"→ Inference mode: fusion={fusion_type} | "
        f"sensor={sensor_type if fusion_type == 'single' else fusion_type} | "
        f"{prisma_flag}"
    )

    df_pred = pd.read_csv(oof_csv)
    df_meta = pd.read_csv(metadata_csv)

    # Merge metadata + predictions on patch_id
    df = df_pred.merge(df_meta, on="patch_id", how="left")

    if df["planet_path"].isnull().any():
        print("⚠ Warning: some predictions are missing 'planet_path' in metadata.")

    # ------------------------
    # Determine full mosaic size in pixels
    # ------------------------
    # x, y are top-left pixel indices of each patch in the global Planet grid.
    max_x = int(df["x"].max()) + patch_size
    max_y = int(df["y"].max()) + patch_size

    W = max_x  # width  in pixels
    H = max_y  # height in pixels

    print(f"→ Full mosaic size: width={W}, height={H} pixels")

    # ------------------------
    # Recover global transform & CRS from Planet patches
    # ------------------------
    # We use georeferencing from Planet (3 m grid).
    # We reconstruct the "global" transform of the original common grid
    # using one reference patch and its (x, y) offset.
    ref_row = df.sort_values(["x", "y"]).iloc[0]
    ref_x = int(ref_row["x"])
    ref_y = int(ref_row["y"])
    ref_planet_path = ref_row["planet_path"]

    if not os.path.exists(ref_planet_path):
        raise FileNotFoundError(f"Reference Planet patch not found: {ref_planet_path}")

    print(f"→ Using reference Planet patch for georeferencing: {ref_planet_path}")
    print(f"  (reference offset x={ref_x}, y={ref_y})")

    with rasterio.open(ref_planet_path) as src:
        t_patch = src.transform
        crs = src.crs

    # Decompose patch transform
    a = t_patch.a
    b = t_patch.b
    c_patch = t_patch.c
    d = t_patch.d
    e = t_patch.e
    f_patch = t_patch.f

    # Reconstruct global transform:
    # t_global(col, row) = t_patch(col + ref_x, row + ref_y)
    # => c_global = c_patch - a*ref_x - b*ref_y
    #    f_global = f_patch - d*ref_x - e*ref_y
    c_global = c_patch - a * ref_x - b * ref_y
    f_global = f_patch - d * ref_x - e * ref_y

    t_global = Affine(a, b, c_global, d, e, f_global)

    print(f"→ Global transform: {t_global}")
    print(f"→ CRS: {crs}")

    # ------------------------
    # Initialize accumulators for averaging overlapping patches
    # ------------------------
    prob_sum = np.zeros((H, W), dtype=np.float32)
    prob_count = np.zeros((H, W), dtype=np.float32)

    # ------------------------
    # Paint each patch probability over its FULL PATCH footprint
    # ------------------------
    for _, row in df.iterrows():
        x0_patch = int(row["x"])
        y0_patch = int(row["y"])
        prob = float(row["prob_slum"])

        # Compute coordinates of the full patch region in global mosaic coordinates
        x0 = x0_patch
        y0 = y0_patch
        x1 = x0 + patch_size
        y1 = y0 + patch_size

        # Safety clamp (in case of edge effects)
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            print(
                f"⚠ Skipping patch {row['patch_id']} "
                f"due to out-of-bounds indices (x0={x0}, y0={y0}, x1={x1}, y1={y1})."
            )
            continue

        prob_sum[y0:y1, x0:x1] += prob * kernel
        prob_count[y0:y1, x0:x1] += kernel

    # ------------------------
    # Compute final probability map with averaging in overlaps
    # ------------------------
    prob_map = np.zeros((H, W), dtype=np.float32)

    mask = prob_count > 0
    prob_map[mask] = prob_sum[mask] / prob_count[mask]

    # Optional: clip to [0, 1] just for safety
    prob_map[mask] = np.clip(prob_map[mask], 0.0, 1.0)

    # ------------------------
    # Save raster
    # ------------------------
    if output_dir is None:
        # Default: same directory as OOF CSV
        out_dir = oof_dir
    else:
        out_dir = output_dir

    os.makedirs(out_dir, exist_ok=True)
    out_prob = os.path.join(out_dir, "slum_probability_fullres.tif")
    print(f"→ Saving probability map to: {out_prob}")

    with rasterio.open(
        out_prob,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=1,
        dtype="float32",
        crs=crs,
        transform=t_global,
    ) as dst:
        dst.write(prob_map, 1)

    # ------------------------
    # Automatic Threshold Selection (maximize F1-score)
    # ------------------------
    print("\n→ Selecting optimal threshold (maximize F1-score)...")

    from sklearn.metrics import f1_score

    # Extract labels and probabilities from OOF predictions DataFrame
    labels = df_pred["label"].values.astype(int)
    probs = df_pred["prob_slum"].values.astype(float)

    thresholds = np.linspace(0, 1, 501)
    best_f1 = -1
    best_thr = 0.5

    for thr in thresholds:
        pred_bin = (probs >= thr).astype(int)
        f1 = f1_score(labels, pred_bin, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    print(f"→ Optimal threshold = {best_thr:.3f}  (F1 = {best_f1:.3f})")

    # ------------------------
    # Build binary map using optimal threshold
    # ------------------------
    binary_map = (prob_map >= best_thr).astype(np.uint8)

    out_bin = os.path.join(out_dir, f"slum_binary_fullres_{best_thr:.3f}.tif")
    print(f"→ Saving binary threshold map to: {out_bin}")

    with rasterio.open(
        out_bin,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=t_global,
    ) as dst:
        dst.write(binary_map, 1)

    print("\n✔ Finished! Full-resolution slum probability map saved.\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="training/configs/experiment.yaml",
        help="Path to config file (same used for training/inference).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where the GeoTIFF will be saved. "
             "If not set, it defaults to the OOF predictions folder.",
    )
    args = parser.parse_args()

    run_rasterize_fullres(args.cfg, output_dir=args.output_dir)