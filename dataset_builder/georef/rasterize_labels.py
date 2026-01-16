# dataset_builder/georef/rasterize_labels.py

import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

def rasterize_shapefile(shp_path, ref_raster_path, out_path):

    with rasterio.open(ref_raster_path) as ref:
        transform = ref.transform
        width = ref.width
        height = ref.height
        crs = ref.crs
        bounds = ref.bounds

    raster_poly = box(*bounds)

    gdf = gpd.read_file(shp_path)

    if gdf.crs.to_epsg() != crs.to_epsg():
        gdf = gdf.to_crs(crs)

    gdf["geometry"] = gdf.buffer(0)              # fix invalid geom
    gdf = gdf[gdf.intersects(raster_poly)]       # select intersection only

    shapes = ((geom, 1) for geom in gdf.geometry)

    out_meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "transform": transform,
        "crs": crs,
        "compress": "deflate"
    }

    with rasterio.open(out_path, "w", **out_meta) as dst:
        burned = rasterize(
            shapes,
            out_shape=(height, width),
            fill=0,
            transform=transform,
            dtype="uint8"
        )
        dst.write(burned, 1)