# dataset_builder/main_build_dataset.py

from dataset_builder.io.read_tifs import read_raster
from dataset_builder.georef.rasterize_labels import rasterize_shapefile
from dataset_builder.patch_extractor import extract_patches
import os

def main():

    planet_path = "raw_data/PLANET/20250719_8b_clip_aligned.tif"
    sar_path = "raw_data/CSK/CSKS2_GTC_B_HI_04_HH_RA_SF_20250728103626_20250728103634.S01.SBI_aligned.tif"
    prisma_path = "raw_data/PRISMA/prisma_sr_vnir_swir_28_03_2021_40_30_20_georref_230_bands_aligned.tif"
    shp_path    = "raw_data/RENABAP/renabap_2025_CBA.shp"

    out_dir = "dataset"
    os.makedirs(out_dir, exist_ok=True)

    labels_raster = os.path.join(out_dir, "slum_labels_common.tif")

    print("▶ Rasterizing shapefile...")
    rasterize_shapefile(shp_path, planet_path, labels_raster)

    print("▶ Reading aligned rasters...")
    planet, planet_profile = read_raster(planet_path)
    sar, sar_profile       = read_raster(sar_path)
    labels, labels_profile = read_raster(labels_raster)

    print("▶ Extracting patches from Planet, SAR, and PRISMA...")
    n = extract_patches(
        planet,
        sar,
        labels,
        planet_profile,
        sar_profile,
        prisma_path,
        labels_profile,
        out_dir,
        slum_threshold=0.10
    )

    print(f"✔ DONE → {n} patches created.")

if __name__ == "__main__":
    main()