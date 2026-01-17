import os
import numpy as np
import csv
import rasterio
from rasterio.windows import Window, from_bounds

def has_invalid_data(arr, nodata):
    """
    Check if the array is None, empty, or contains a significant fraction of nodata values.
    Nodata values are considered as NaNs or equal to nodata value.
    """
    if arr is None:
        return True
    if arr.size == 0:
        return True
    # Convert nodata to mask
    if nodata is not None:
        if np.isnan(nodata):
            nodata_mask = np.isnan(arr)
        else:
            nodata_mask = (arr == nodata)
        # Consider invalid if more than 5% pixels are nodata
        if np.mean(nodata_mask) > 0.05:
            return True
    # Also consider NaNs as invalid
    if np.isnan(arr).any():
        return True
    return False

def save_patch(arr, base_profile, window, out_path):
    """
    Save a patch array to GeoTIFF with updated transform and profile.
    The transform is updated to reflect the window location.
    """
    transform = base_profile["transform"]
    patch_transform = rasterio.windows.transform(window, transform)

    profile = base_profile.copy()
    profile.update({
        "height": arr.shape[0],
        "width": arr.shape[1],
        "transform": patch_transform,
        "count": arr.shape[2] if arr.ndim == 3 else 1,
        "dtype": arr.dtype.name,
        "driver": "GTiff",
        "compress": "deflate"
    })

    with rasterio.open(out_path, "w", **profile) as dst:
        if arr.ndim == 3:
            dst.write(arr.transpose(2, 0, 1))
        else:
            dst.write(arr, 1)

def extract_patches(
    planet, sar, labels,
    planet_profile, sar_profile, prisma_path, labels_profile,
    out_dir,
    patch_size_planet=224,
    stride_planet=179,
    slum_threshold=0.10
):
    """
    Extract aligned patches from Planet, SAR, and PRISMA datasets.
    Uses Planet grid as reference (pixel-wise patching).
    PRISMA is cropped spatially per Planet patch bounding box.
    Excludes patches if any sensor has invalid data.
    Saves Planet and SAR patches as GeoTIFF, PRISMA patches as compressed npz.
    Writes a single metadata CSV with spatial linkage.
    """
    # Ensure output directories exist
    patches_planet = os.path.join(out_dir, "patches_planet")
    patches_sar = os.path.join(out_dir, "patches_sar")
    patches_prisma = os.path.join(out_dir, "patches_prisma")
    os.makedirs(patches_planet, exist_ok=True)
    os.makedirs(patches_sar, exist_ok=True)
    os.makedirs(patches_prisma, exist_ok=True)

    # Open PRISMA once for reading and CRS check
    with rasterio.open(prisma_path) as prisma_src:
        prisma_crs = prisma_src.crs
        prisma_nodata = prisma_src.nodata

        # Check CRS consistency
        assert prisma_crs == planet_profile["crs"], "CRS mismatch between PRISMA and Planet"
        assert prisma_crs == sar_profile["crs"], "CRS mismatch between PRISMA and SAR"
        assert prisma_crs == labels_profile["crs"], "CRS mismatch between PRISMA and Labels"

        nodata_planet = planet_profile.get("nodata", None)
        nodata_sar = sar_profile.get("nodata", None)
        nodata_labels = labels_profile.get("nodata", None)

        csv_path = os.path.join(out_dir, "metadata.csv")
        with open(csv_path, "w", newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "patch_id", "x", "y", "lat", "lon",
                "planet_path", "sar_path", "prisma_path",
                "prisma_h", "prisma_w",
                "slum_fraction", "label"
            ])

            count = 0
            height_planet = planet.shape[0]
            width_planet = planet.shape[1]

            for row in range(0, height_planet - patch_size_planet + 1, stride_planet):
                for col in range(0, width_planet - patch_size_planet + 1, stride_planet):
                    # Extract Planet patch
                    planet_patch = planet[row:row+patch_size_planet, col:col+patch_size_planet]
                    if has_invalid_data(planet_patch, nodata_planet):
                        continue

                    # Compute bounding box of Planet patch in CRS coords
                    bbox = rasterio.windows.bounds(
                        Window(col, row, patch_size_planet, patch_size_planet),
                        planet_profile["transform"]
                    )

                    # Extract SAR patch using bbox
                    sar_window = from_bounds(*bbox, transform=sar_profile["transform"])
                    sar_window = Window(
                        col_off=int(np.floor(sar_window.col_off)),
                        row_off=int(np.floor(sar_window.row_off)),
                        width=int(np.ceil(sar_window.width)),
                        height=int(np.ceil(sar_window.height))
                    )
                    sar_window = sar_window.intersection(Window(0, 0, sar.shape[1], sar.shape[0]))
                    if sar_window.width < patch_size_planet//2 or sar_window.height < patch_size_planet//2:
                        continue
                    sar_patch = sar[
                        int(sar_window.row_off):int(sar_window.row_off + sar_window.height),
                        int(sar_window.col_off):int(sar_window.col_off + sar_window.width)
                    ]
                    if has_invalid_data(sar_patch, nodata_sar):
                        continue

                    # Extract labels patch using bbox
                    labels_window = from_bounds(*bbox, transform=labels_profile["transform"])
                    labels_window = Window(
                        col_off=int(np.floor(labels_window.col_off)),
                        row_off=int(np.floor(labels_window.row_off)),
                        width=int(np.ceil(labels_window.width)),
                        height=int(np.ceil(labels_window.height))
                    )
                    labels_window = labels_window.intersection(Window(0, 0, labels.shape[1], labels.shape[0]))
                    if labels_window.width < patch_size_planet//2 or labels_window.height < patch_size_planet//2:
                        continue
                    labels_patch = labels[
                        int(labels_window.row_off):int(labels_window.row_off + labels_window.height),
                        int(labels_window.col_off):int(labels_window.col_off + labels_window.width)
                    ]

                    # Determine label based on slum fraction
                    slum_fraction = np.mean(labels_patch > 0)
                    if slum_fraction >= slum_threshold:
                        label = 1
                    elif slum_fraction == 0:
                        label = 0
                    else:
                        continue  # ambiguous patch, skip

                    # Crop PRISMA patch strictly by bbox
                    prisma_patch, prisma_window = None, None
                    try:
                        prisma_window = from_bounds(*bbox, transform=prisma_src.transform)
                        prisma_window = Window(
                            col_off=int(np.floor(prisma_window.col_off)),
                            row_off=int(np.floor(prisma_window.row_off)),
                            width=int(np.ceil(prisma_window.width)),
                            height=int(np.ceil(prisma_window.height))
                        )
                        prisma_window = prisma_window.intersection(Window(0, 0, prisma_src.width, prisma_src.height))
                        if prisma_window.width <= 0 or prisma_window.height <= 0:
                            continue
                        prisma_patch = prisma_src.read(window=prisma_window)
                        if prisma_patch.size == 0:
                            continue
                        if has_invalid_data(prisma_patch, prisma_nodata):
                            continue
                    except Exception:
                        continue

                    patch_id = f"patch_{count:06d}"

                    # Save Planet patch
                    p_planet = os.path.join(patches_planet, patch_id + ".tif")
                    planet_window = Window(col, row, patch_size_planet, patch_size_planet)
                    save_patch(planet_patch, planet_profile, planet_window, p_planet)

                    # Save SAR patch
                    p_sar = os.path.join(patches_sar, patch_id + ".tif")
                    save_patch(sar_patch, sar_profile, sar_window, p_sar)

                    # Save PRISMA patch as compressed npz
                    p_prisma = os.path.join(patches_prisma, patch_id + ".npz")
                    np.savez_compressed(
                        p_prisma,
                        data=prisma_patch,
                        transform=prisma_src.window_transform(prisma_window),
                        crs=str(prisma_src.crs)
                    )

                    # Compute geographic center of patch (Planet CRS assumed geographic or approximate)
                    cx = bbox[0] + (bbox[2] - bbox[0]) / 2
                    cy = bbox[1] + (bbox[3] - bbox[1]) / 2
                    lon, lat = cx, cy

                    writer.writerow([
                        patch_id, col, row, lat, lon,
                        p_planet, p_sar, p_prisma,
                        prisma_patch.shape[1], prisma_patch.shape[2],
                        slum_fraction, label
                    ])

                    count += 1

    return count