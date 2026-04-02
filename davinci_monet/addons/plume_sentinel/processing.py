"""Geospatial processing for PlumeSentinel addon.

Transforms raw loaded data into plot-ready forms:
- GOES ABI channels -> true-color RGB arrays
- HMS smoke shapefiles -> clean GeoDataFrames
- MODIS L2 swath AOD -> gridded 2D arrays
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cartopy.crs as ccrs
import numpy as np
import xarray as xr


@dataclass
class GoesRgbResult:
    """Result of GOES true-color RGB assembly."""

    rgb: np.ndarray  # (h, w, 3) float array clipped to [0, 1]
    extent: tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax) in projection coords
    cartopy_crs: ccrs.Projection


@dataclass
class GriddedAodResult:
    """Result of swath-to-grid binning for MODIS AOD."""

    data_2d: np.ndarray  # (nlat, nlon) gridded mean AOD
    lon_centers: np.ndarray
    lat_centers: np.ndarray
    resolution: float
    granule_files: list[str]


def prepare_goes(ds: xr.Dataset, gamma: float = 1.8) -> GoesRgbResult:
    """Assemble GOES ABI channels into a true-color RGB image.

    Parameters
    ----------
    ds
        GOES-16/17 ABI L2 dataset with CMI_C01 (blue), CMI_C02 (red),
        CMI_C03 (veggie), and goes_imager_projection metadata.
    gamma
        Gamma correction exponent. Applied as ``rgb ** (1/gamma)``.

    Returns
    -------
    GoesRgbResult
        RGB array, map extent, and cartopy CRS for plotting.
    """
    red = ds["CMI_C02"].values.astype(np.float64)
    blue = ds["CMI_C01"].values.astype(np.float64)
    veggie = ds["CMI_C03"].values.astype(np.float64)

    # Synthetic green channel (Bah et al. 2018 recipe)
    green = 0.45 * red + 0.10 * blue + 0.45 * veggie

    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)

    if gamma != 1.0:
        rgb = np.power(rgb, 1.0 / gamma)

    # Extract projection metadata
    proj = ds["goes_imager_projection"]
    height = proj.attrs["perspective_point_height"]
    central_lon = proj.attrs["longitude_of_projection_origin"]
    sweep = proj.attrs["sweep_angle_axis"]

    # Scale x/y from radians to projection metres
    x = ds.coords["x"].values
    y = ds.coords["y"].values
    x_proj = x * height
    y_proj = y * height
    extent = (x_proj.min(), x_proj.max(), y_proj.min(), y_proj.max())

    cartopy_crs = ccrs.Geostationary(
        central_longitude=central_lon,
        satellite_height=height,
        sweep_axis=sweep,
    )

    return GoesRgbResult(rgb=rgb, extent=extent, cartopy_crs=cartopy_crs)


def prepare_hms(gdf: Any) -> Any:
    """Clean and reproject an HMS smoke GeoDataFrame.

    Parameters
    ----------
    gdf
        GeoDataFrame loaded from an HMS smoke shapefile.

    Returns
    -------
    GeoDataFrame
        Reprojected to EPSG:4326 with invalid geometries fixed.
    """
    import geopandas as gpd  # noqa: F811

    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"Expected GeoDataFrame, got {type(gdf).__name__}")

    # Fix invalid geometries
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.buffer(0)

    # Reproject to WGS84 if needed
    if gdf.crs is not None and not gdf.crs.equals("EPSG:4326"):
        gdf = gdf.to_crs(epsg=4326)

    return gdf


def prepare_modis_aod(
    loaded: dict[str, Any],
    grid_spec: dict[str, Any],
) -> GriddedAodResult:
    """Bin MODIS L2 swath AOD onto a regular lat/lon grid.

    Parameters
    ----------
    loaded
        Dictionary from ``_load_modis_aod`` with keys: latitude, longitude,
        data, granule_files.
    grid_spec
        Grid specification with keys: resolution, lon_range, lat_range,
        min_obs_count.

    Returns
    -------
    GriddedAodResult
        Gridded 2D AOD array with coordinate vectors.
    """
    from davinci_monet.pairing.grid_binning import (
        bin_swath_to_grid,
        edges_from_centers,
        normalize_grid,
    )

    resolution = grid_spec["resolution"]
    lon_min, lon_max = grid_spec["lon_range"]
    lat_min, lat_max = grid_spec["lat_range"]
    min_obs_count = grid_spec.get("min_obs_count", 1)

    # Build center coordinate arrays
    lon_centers = np.arange(lon_min + resolution / 2, lon_max, resolution)
    lat_centers = np.arange(lat_min + resolution / 2, lat_max, resolution)

    lon_edges = edges_from_centers(lon_centers)
    lat_edges = edges_from_centers(lat_centers)

    # Single time bin
    time_edges = np.array([0.0, 1.0])
    nlon = len(lon_centers)
    nlat = len(lat_centers)

    count_grid = np.zeros((1, nlon, nlat), dtype=np.int64)
    data_grid = np.zeros((1, nlon, nlat), dtype=np.float64)

    lat_obs = loaded["latitude"].astype(np.float64)
    lon_obs = loaded["longitude"].astype(np.float64)
    data_obs = loaded["data"].astype(np.float64)
    time_obs = np.zeros(len(data_obs), dtype=np.float64)

    if len(data_obs) > 0:
        bin_swath_to_grid(
            time_edges,
            lon_edges,
            lat_edges,
            time_obs,
            lon_obs,
            lat_obs,
            data_obs,
            count_grid,
            data_grid,
        )
        normalize_grid(count_grid, data_grid)

        # Apply minimum observation count filter
        if min_obs_count > 1:
            data_grid[count_grid < min_obs_count] = np.nan
    else:
        data_grid[:] = np.nan

    # Transpose from (lon, lat) to (lat, lon) for imshow with origin="lower"
    data_2d = data_grid[0].T

    return GriddedAodResult(
        data_2d=data_2d,
        lon_centers=lon_centers,
        lat_centers=lat_centers,
        resolution=resolution,
        granule_files=loaded.get("granule_files", []),
    )
