"""Deterministic metric extraction for plume_sentinel stage outputs.

Given an AOD field (xarray.DataArray, dims y/x or lat/lon), returns
percentile metrics, peak, retrieval failure %, and area-thresholded
coverage in km². Given an HMS GeoDataFrame (with a 'density' column),
returns area per density class in km².
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr


def _grid_cell_area_km2(da: xr.DataArray) -> xr.DataArray:
    """Compute per-cell area on a lat/lon grid (cosine-weighted), km².

    Uses a spherical-Earth approximation: each cell's area is
    ``R^2 * dlat_rad * dlon_rad * cos(lat)`` with ``R = 6371 km``.
    """
    lat_dim = "y" if "y" in da.dims else "lat"
    lon_dim = "x" if "x" in da.dims else "lon"
    lat = da[lat_dim].values
    lon = da[lon_dim].values
    dlat = float(abs(lat[1] - lat[0])) if len(lat) > 1 else 1.0
    dlon = float(abs(lon[1] - lon[0])) if len(lon) > 1 else 1.0
    R = 6371.0
    lat_rad = np.deg2rad(lat)
    cell_area = R * R * np.deg2rad(dlat) * np.deg2rad(dlon) * np.cos(lat_rad)
    return xr.DataArray(
        np.broadcast_to(cell_area[:, None], da.shape),
        dims=(lat_dim, lon_dim),
    )


def aod_metrics(aod: xr.DataArray, threshold: float = 2.0) -> dict[str, Any]:
    """Extract AOD summary metrics from a 2D field.

    Returns a dict with: ``aod_peak``, ``aod_p50``, ``aod_p95``,
    ``area_aod_gt_2_km2``, ``retrieval_failure_pct``.
    """
    flat = aod.values.flatten()
    n_total = flat.size
    n_nan = int(np.isnan(flat).sum())
    valid = flat[~np.isnan(flat)]

    if valid.size == 0:
        return {
            "aod_peak": None,
            "aod_p50": None,
            "aod_p95": None,
            "area_aod_gt_2_km2": 0.0,
            "retrieval_failure_pct": 100.0,
        }

    cell_area = _grid_cell_area_km2(aod)
    over = (aod > threshold) & (~aod.isnull())
    area_over = float((over * cell_area).sum().item())

    return {
        "aod_peak": float(np.max(valid)),
        "aod_p50": float(np.percentile(valid, 50)),
        "aod_p95": float(np.percentile(valid, 95)),
        "area_aod_gt_2_km2": area_over,
        "retrieval_failure_pct": 100.0 * n_nan / n_total,
    }


def hms_metrics(hms_gdf: Any) -> dict[str, float]:
    """Areas (km²) per HMS density class.

    Projects the GeoDataFrame to NA Albers Equal-Area (EPSG:5070) before
    computing geometry areas, so results are real km² rather than
    degrees-squared. Accepts ``density`` or ``Density`` column names
    (the operational HMS shapefiles use ``Density`` with a capital D).
    """
    import geopandas as gpd  # heavy import — keep local

    if not isinstance(hms_gdf, gpd.GeoDataFrame):
        raise TypeError("expected a GeoDataFrame")

    # Project to an equal-area CRS for area computation (NA Albers).
    projected = hms_gdf.to_crs("EPSG:5070") if hms_gdf.crs else hms_gdf

    out = {"hms_heavy_km2": 0.0, "hms_medium_km2": 0.0, "hms_light_km2": 0.0}

    # Locate the density column (case-insensitive).
    density_col: str | None = None
    for col in projected.columns:
        if str(col).lower() == "density":
            density_col = col
            break
    if density_col is None:
        return out

    by_class = {
        "Heavy": "hms_heavy_km2",
        "Medium": "hms_medium_km2",
        "Light": "hms_light_km2",
    }
    # Compare case-insensitively against the canonical class names.
    classes_lower = projected[density_col].astype(str).str.strip().str.lower()
    for cls, key in by_class.items():
        subset = projected[classes_lower == cls.lower()]
        if not subset.empty:
            # Dissolve overlapping polygons before measuring area; raw
            # subset.geometry.area.sum() double-counts where polygons overlap.
            # Operational HMS shapefiles routinely contain overlapping plumes
            # (e.g., 2020-09-09 had 14 Heavy polygons with ~47% overlap).
            out[key] = float(subset.geometry.union_all().area / 1e6)  # m² → km²
    return out
