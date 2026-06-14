"""Numba-accelerated binning of sparse datasets onto a uniform grid.

Ported from MELODIES-MONET grid_util.py. Bins satellite swath pixels
(or any sparse 2D datasets) into (time, lon, lat) grid cells using
simple floor-division arithmetic, accumulated in-place.

The numba JIT inner loop operates on flat numpy arrays with zero
Python/xarray overhead, making it suitable for 100M+ pixel datasets.
"""

from __future__ import annotations

import math

import numba
import numpy as np


@numba.jit(nopython=True)
def bin_swath_to_grid(
    time_edges: np.ndarray,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    time_values: np.ndarray,
    lon_values: np.ndarray,
    lat_values: np.ndarray,
    data_values: np.ndarray,
    count_grid: np.ndarray,
    data_grid: np.ndarray,
) -> None:
    """Accumulate swath pixels into (time, lon, lat) grid cells.

    Each valid (non-NaN) pixel is assigned to a grid cell via floor
    division on the bin edges. Running sums and counts are accumulated
    in-place in *data_grid* and *count_grid*.

    Parameters
    ----------
    time_edges
        1D array of time bin edges (epoch seconds), length ntime+1.
    lon_edges
        1D array of longitude bin edges, length nlon+1.
    lat_edges
        1D array of latitude bin edges, length nlat+1.
    time_values
        Flat array of dataset timestamps (epoch seconds).
    lon_values
        Flat array of dataset longitudes.
    lat_values
        Flat array of dataset latitudes.
    data_values
        Flat array of dataset data values.
    count_grid
        Pre-allocated (ntime, nlon, nlat) int array — modified in-place.
    data_grid
        Pre-allocated (ntime, nlon, nlat) float array — modified in-place.
    """
    dt = time_edges[1] - time_edges[0]
    dx = lon_edges[1] - lon_edges[0]
    dy = lat_edges[1] - lat_edges[0]
    nt, nx, ny = data_grid.shape
    for i in range(len(data_values)):
        if (
            not math.isnan(data_values[i])
            and not math.isnan(time_values[i])
            and not math.isnan(lon_values[i])
            and not math.isnan(lat_values[i])
            and time_values[i] >= time_edges[0]
            and time_values[i] <= time_edges[-1]
            and lon_values[i] >= lon_edges[0]
            and lon_values[i] <= lon_edges[-1]
            and lat_values[i] >= lat_edges[0]
            and lat_values[i] <= lat_edges[-1]
        ):
            it = int((time_values[i] - time_edges[0]) / dt)
            ix = int((lon_values[i] - lon_edges[0]) / dx)
            iy = int((lat_values[i] - lat_edges[0]) / dy)
            # Clamp exact upper-edge coordinates into the final bin.
            if it < 0:
                it = 0
            elif it >= nt:
                it = nt - 1
            if ix < 0:
                ix = 0
            elif ix >= nx:
                ix = nx - 1
            if iy < 0:
                iy = 0
            elif iy >= ny:
                iy = ny - 1
            count_grid[it, ix, iy] += 1
            data_grid[it, ix, iy] += data_values[i]


def normalize_grid(
    count_grid: np.ndarray,
    data_grid: np.ndarray,
) -> None:
    """Divide accumulated sums by counts; set empty cells to NaN.

    Modifies *data_grid* in-place.

    Parameters
    ----------
    count_grid
        Array of dataset counts per cell.
    data_grid
        Array of accumulated sums — converted to means in-place.
    """
    mask = count_grid > 0
    data_grid[~mask] = np.nan
    data_grid[mask] /= count_grid[mask]


def edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Derive bin edges from uniformly-spaced center coordinates.

    Edges are placed at midpoints between consecutive centers, with
    half-spacing extensions at the boundaries.

    Parameters
    ----------
    centers
        1D array of grid center values (must be uniformly spaced).

    Returns
    -------
    np.ndarray
        Array of length ``len(centers) + 1`` containing bin edges.
    """
    if len(centers) == 1:
        # Single center — use a default half-day spacing for time,
        # or 1.0 for spatial coords
        half_spacing = max(abs(centers[0]) * 0.01, 0.5)
        return np.array([centers[0] - half_spacing, centers[0] + half_spacing], dtype=np.float64)

    half_spacing = (centers[1] - centers[0]) / 2.0
    edges = np.empty(len(centers) + 1, dtype=np.float64)
    edges[0] = centers[0] - half_spacing
    edges[-1] = centers[-1] + half_spacing
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    return edges
