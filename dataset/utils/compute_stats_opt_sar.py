# compute_stats_for_normalization.py

import os
import glob
import rasterio
import numpy as np
import pandas as pd

def compute_stats(raster_paths, nodata=None):
    """
    Given a list of raster file paths (same sensor: e.g. all sar or all Planet), 
    compute per-band statistics across all patches:
      min, max, 1st, 5th, 50th, 95th, 99th percentile, mean, std
    Returns a DataFrame with one row per band/component.
    """
    stats = {}
    first = True
    for p in raster_paths:
        with rasterio.open(p) as src:
            arr = src.read().astype(np.float32)
            if nodata is None:
                # try to read nodata from metadata
                nod = src.nodatavals[0]
            else:
                nod = nodata
            if nod is None:
                mask = np.ones_like(arr, dtype=bool)
            else:
                mask = arr != nod
            arr = np.where(mask, arr, np.nan)
        if first:
            nbands = arr.shape[0]
            # Prepare list of arrays per band
            bands_data = [[] for _ in range(nbands)]
            first = False
        for b in range(nbands):
            bands_data[b].append(arr[b].ravel())
    # Now compute stats per band
    rows = []
    for b, bd in enumerate(bands_data):
        data = np.concatenate(bd)
        data = data[~np.isnan(data)]
        if data.size == 0:
            continue
        rows.append({
            "band": b,
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "p1": float(np.percentile(data, 1)),
            "p5": float(np.percentile(data, 5)),
            "p50": float(np.percentile(data, 50)),
            "p95": float(np.percentile(data, 95)),
            "p99": float(np.percentile(data, 99)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
        })
    df = pd.DataFrame(rows)
    return df

def scan_and_report(sar_glob, planet_glob, out_csv="normalization_stats_opt_sar.csv"):
    """
    sar_glob: glob pattern for sar patch files, e.g. "dataset/patches_sar/*.tif"
    planet_glob: glob pattern for Planet patch files
    """
    print("Scanning sar patches...")
    sar_paths = glob.glob(sar_glob)
    print(f"  Found {len(sar_paths)} files.")
    df_sar = compute_stats(sar_paths, nodata=0)
    df_sar["sensor"] = "sar"

    print("Scanning Planet patches...")
    planet_paths = glob.glob(planet_glob)
    print(f"  Found {len(planet_paths)} files.")
    df_planet = compute_stats(planet_paths, nodata=0)
    df_planet["sensor"] = "planet"

    df = pd.concat([df_sar, df_planet], ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved statistics to {out_csv}")

    # Suggest simple normalizations
    print("\n=== Suggested normalizations ===")
    for sensor, group in df.groupby("sensor"):
        print(f"\nSensor: {sensor}")
        for idx, row in group.iterrows():
            b = int(row["band"])
            minv, maxv = row["min"], row["max"]
            print(f"  Band {b}: min {minv:.3f}, max {maxv:.3f} -> you could scale to [0,1] using (x-min)/(max-min).")
        # Additionally for z-score:
        print(f"  {sensor}: global mean/std per band available. Z-score normalization (x-mean)/std also possible.")

if __name__ == "__main__":
    scan_and_report(
        sar_glob="dataset/patches_sar/*.tif", 
        planet_glob="dataset/patches_planet/*.tif",
        out_csv="dataset/normalization_stats_opt_sar.csv"
    )