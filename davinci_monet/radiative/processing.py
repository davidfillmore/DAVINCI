"""Processing utilities for radiative analysis.

Provides anomaly computation, smoke AOD derivation, nearest-neighbor
regridding, and semi-empirical surface dimming estimation.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def compute_background(ds: xr.Dataset, window: int) -> xr.Dataset:
    """Compute background as the mean of the first *window* time steps.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with a ``time`` dimension.
    window : int
        Number of leading time steps to average.  Clamped to the actual
        number of time steps when *window* exceeds the dataset length.

    Returns
    -------
    xr.Dataset
        Time-averaged background (no ``time`` dimension).
    """
    n = min(window, ds.sizes["time"])
    return ds.isel(time=slice(0, n)).mean(dim="time")


def compute_anomalies(ds: xr.Dataset, background: xr.Dataset) -> xr.Dataset:
    """Subtract a background from every time step.

    Parameters
    ----------
    ds : xr.Dataset
        Full time-varying dataset.
    background : xr.Dataset
        Background fields (no ``time`` dimension).

    Returns
    -------
    xr.Dataset
        Anomaly dataset with the same shape as *ds*.
    """
    return ds - background


def derive_smoke_aod(ds: xr.Dataset, species: list[str]) -> xr.DataArray:
    """Sum aerosol species to obtain total smoke AOD.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing one variable per aerosol species.
    species : list[str]
        Variable names to sum.

    Returns
    -------
    xr.DataArray
        Total smoke AOD.
    """
    total = ds[species[0]]
    for name in species[1:]:
        total = total + ds[name]
    return total


def regrid_nearest(
    source: xr.DataArray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> xr.DataArray:
    """Nearest-neighbor regridding onto a new lat/lon grid.

    Parameters
    ----------
    source : xr.DataArray
        2-D field with ``lat`` and ``lon`` dimensions.
    target_lat, target_lon : np.ndarray
        1-D arrays of target coordinates.

    Returns
    -------
    xr.DataArray
        Regridded field on the target grid.
    """
    src_lat = source.lat.values
    src_lon = source.lon.values

    out = np.empty((len(target_lat), len(target_lon)))
    for i, tlat in enumerate(target_lat):
        ilat = int(np.argmin(np.abs(src_lat - tlat)))
        for j, tlon in enumerate(target_lon):
            jlon = int(np.argmin(np.abs(src_lon - tlon)))
            out[i, j] = source.values[ilat, jlon]

    return xr.DataArray(
        out,
        dims=["lat", "lon"],
        coords={"lat": target_lat, "lon": target_lon},
    )


def semi_empirical_surface_dimming(
    smoke_aod: np.ndarray,
    toa_insol: np.ndarray,
    ssa: float = 0.92,
    asymmetry: float = 0.65,
) -> np.ndarray:
    """Estimate surface shortwave dimming from smoke aerosol.

    Uses a Beer-Lambert extinction with a forward-scattering correction.

    Parameters
    ----------
    smoke_aod : array-like
        Smoke aerosol optical depth.
    toa_insol : array-like
        Top-of-atmosphere insolation (W m-2).
    ssa : float
        Single-scattering albedo (default 0.92).
    asymmetry : float
        Asymmetry parameter (default 0.65).

    Returns
    -------
    np.ndarray
        Surface dimming (negative values, W m-2).
    """
    mu = 0.5  # effective daily-mean cosine solar zenith angle
    extinct = 1.0 - np.exp(-smoke_aod / mu)
    fwd_scatter_frac = ssa * (1.0 + asymmetry) / 2.0
    return -toa_insol * extinct * (1.0 - fwd_scatter_frac)
