import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_origin
from rasterio.coords import BoundingBox


def intersect_bounds(b1: BoundingBox, b2: BoundingBox) -> BoundingBox:
    """Spatial intersection of two bounding boxes"""
    return BoundingBox(
        left=max(b1.left, b2.left),
        bottom=max(b1.bottom, b2.bottom),
        right=min(b1.right, b2.right),
        top=min(b1.top, b2.top),
    )


def read_bounds(path, target_crs):
    with rasterio.open(path) as src:
        if src.crs == target_crs:
            return src.bounds
        return BoundingBox(*transform_bounds(src.crs, target_crs, *src.bounds))


def reproject_to_grid(
    src_path,
    dst_path,
    dst_crs,
    dst_transform,
    dst_width,
    dst_height,
    resampling,
    nodata=0,
):
    with rasterio.open(src_path) as src:
        meta = src.meta.copy()
        meta.update({
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "nodata": nodata,
        })

        with rasterio.open(dst_path, "w", **meta) as dst:
            for b in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, b),
                    destination=rasterio.band(dst, b),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                    dst_nodata=nodata,
                )


def build_grid(bounds: BoundingBox, resolution):
    x_res, y_res = resolution
    width = int((bounds.right - bounds.left) / x_res)
    height = int((bounds.top - bounds.bottom) / y_res)

    transform = from_origin(
        bounds.left,
        bounds.top,
        x_res,
        y_res,
    )
    return transform, width, height


def align_sar_ms_hs(
    sar_path,
    ms_path,
    hs_path,
    out_sar,
    out_ms,
    out_hs,
    sar_resolution,
    ms_resolution,
    hs_resolution,
):
    """
    Align SAR, MS and HS rasters:
    - same CRS
    - same spatial extent (intersection)
    - independent output resolutions
    """

    # ------------------------
    # CRS reference (SAR)
    # ------------------------
    with rasterio.open(sar_path) as src:
        dst_crs = src.crs

    # ------------------------
    # Read bounds in common CRS
    # ------------------------
    sar_bounds = read_bounds(sar_path, dst_crs)
    ms_bounds  = read_bounds(ms_path, dst_crs)
    hs_bounds  = read_bounds(hs_path, dst_crs)

    # ------------------------
    # Intersection
    # ------------------------
    common_bounds = intersect_bounds(sar_bounds, ms_bounds)
    common_bounds = intersect_bounds(common_bounds, hs_bounds)

    if (
        common_bounds.left >= common_bounds.right or
        common_bounds.bottom >= common_bounds.top
    ):
        raise RuntimeError("No spatial intersection between SAR, MS and HS rasters.")

    print("✔ Common spatial extent:")
    print(common_bounds)

    # ------------------------
    # Build grids
    # ------------------------
    sar_tr, sar_w, sar_h = build_grid(common_bounds, sar_resolution)
    ms_tr,  ms_w,  ms_h  = build_grid(common_bounds, ms_resolution)
    hs_tr,  hs_w,  hs_h  = build_grid(common_bounds, hs_resolution)

    print("\n✔ Output grids:")
    print(f"SAR → res={sar_resolution}, size={sar_w}x{sar_h}")
    print(f"MS  → res={ms_resolution},  size={ms_w}x{ms_h}")
    print(f"HS  → res={hs_resolution},  size={hs_w}x{hs_h}")

    # ------------------------
    # Reproject
    # ------------------------
    reproject_to_grid(
        sar_path,
        out_sar,
        dst_crs,
        sar_tr,
        sar_w,
        sar_h,
        resampling=Resampling.nearest,
        nodata=0,
    )

    reproject_to_grid(
        ms_path,
        out_ms,
        dst_crs,
        ms_tr,
        ms_w,
        ms_h,
        resampling=Resampling.bilinear,
        nodata=0,
    )

    reproject_to_grid(
        hs_path,
        out_hs,
        dst_crs,
        hs_tr,
        hs_w,
        hs_h,
        resampling=Resampling.bilinear,
        nodata=0,
    )

    print("\n✔ Alignment completed successfully.")


if __name__ == "__main__":
    align_sar_ms_hs(
        sar_path="raw_data/CSK/CSKS2_GTC_B_HI_04_HH_RA_SF_20250728103626_20250728103634.S01.SBI.tif",
        ms_path="raw_data/PLANET/20250719_8b_clip.tif",
        hs_path="raw_data/PRISMA/prisma_sr_vnir_swir_28_03_2021_40_30_20_georref_230_bands.tif",
        out_sar="raw_data/CSK/CSKS2_GTC_B_HI_04_HH_RA_SF_20250728103626_20250728103634.S01.SBI_aligned.tif",
        out_ms="raw_data/PLANET/20250719_8b_clip_aligned.tif",
        out_hs="raw_data/PRISMA/prisma_sr_vnir_swir_28_03_2021_40_30_20_georref_230_bands_aligned.tif",
        sar_resolution=(3.0, 3.0),     # SAR
        ms_resolution=(3.0, 3.0),      # Planet / Sentinel-2
        hs_resolution=(30.0, 30.0),    # PRISMA / EnMAP
    )