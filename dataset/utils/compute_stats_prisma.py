# dataset/utils/compute_stats_prisma.py

import os
import glob
import numpy as np
import pandas as pd


def compute_prisma_stats(
    prisma_npz_paths,
    nodata=None,
):
    """
    Compute per-band statistics for PRISMA patches stored as .npz.

    Each .npz is expected to contain:
      - data: array of shape (B, H, W)

    Statistics computed per band:
      min, max, p1, p5, p50, p95, p99, mean, std
    """

    bands_data = None

    for i, p in enumerate(prisma_npz_paths):
        data = np.load(p)["data"].astype(np.float32)  # (B, H, W)

        if nodata is not None:
            data[data == nodata] = np.nan

        if bands_data is None:
            nbands = data.shape[0]
            bands_data = [[] for _ in range(nbands)]

        for b in range(nbands):
            bands_data[b].append(data[b].ravel())

    rows = []
    for b, bd in enumerate(bands_data):
        values = np.concatenate(bd)
        values = values[~np.isnan(values)]

        if values.size == 0:
            continue

        rows.append({
            "sensor": "prisma",
            "band": b,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p1": float(np.percentile(values, 1)),
            "p5": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        })

    return pd.DataFrame(rows)


def run(
    prisma_glob,
    out_csv="dataset/prisma_normalization_stats.csv",
    nodata=None,
):
    print("▶ Scanning PRISMA patches...")
    prisma_paths = sorted(glob.glob(prisma_glob))
    print(f"  Found {len(prisma_paths)} PRISMA patches.")

    if len(prisma_paths) == 0:
        raise RuntimeError("No PRISMA patches found.")

    df = compute_prisma_stats(prisma_paths, nodata=nodata)
    df.to_csv(out_csv, index=False)

    print(f"✔ PRISMA statistics saved to: {out_csv}")

    print("\n=== Suggested normalization (PRISMA) ===")
    print(
        "Recommended: robust scaling per band, e.g.\n"
        "  x_norm = (x - p50) / (p95 - p5)\n"
        "or percentile clipping + [0,1] scaling."
    )


if __name__ == "__main__":
    run(
        prisma_glob="dataset/patches_prisma/*.npz",
        out_csv="dataset/prisma_normalization_stats.csv",
        nodata=None
    )