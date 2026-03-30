"""MERRA-2 radiation data loader (tavg1_2d_rad_Nx).

Loads surface radiation fields and computes the aerosol radiative effect
as the difference between all-sky and clean-sky net surface SW flux.
"""

from __future__ import annotations

import glob

import numpy as np
import xarray as xr

from davinci_monet.logging import get_logger

logger = get_logger(__name__)


def load_merra2_rad(
    files: str,
    domain: tuple[float, float, float, float],
) -> xr.Dataset:
    """Load MERRA-2 radiation data, compute daily means.

    Parameters
    ----------
    files : str
        Glob pattern matching MERRA-2 tavg1_2d_rad_Nx NetCDF files.
    domain : tuple
        ``(west, east, south, north)`` bounding box.

    Returns
    -------
    xr.Dataset
        Combined dataset with daily-mean values and derived variables:
        ``ALBEDO`` (surface albedo) and ``m2_sfc_effect`` (aerosol surface
        SW effect = SWGNT - SWGNTCLN).
        Dims: ``(time, lat, lon)``.
    """
    paths = sorted(glob.glob(files))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {files}")
    logger.info("Loading %d MERRA-2 radiation file(s)", len(paths))

    west, east, south, north = domain
    daily: list[xr.Dataset] = []

    for p in paths:
        ds = xr.open_dataset(p)
        # MERRA-2 has ascending latitudes
        ds = ds.sel(lat=slice(south, north), lon=slice(west, east))
        ds = ds.mean(dim="time", keepdims=True)

        # Derive surface albedo if not present
        if "ALBEDO" not in ds:
            swgdn = ds["SWGDN"].values
            swgdn_safe = np.clip(swgdn, 1.0, None)
            ds["ALBEDO"] = xr.DataArray(
                ds["SWGNT"].values / swgdn_safe,
                dims=ds["SWGNT"].dims,
                coords=ds["SWGNT"].coords,
            )

        # Aerosol surface SW effect: all-sky net minus clean-sky net
        ds["m2_sfc_effect"] = ds["SWGNT"] - ds["SWGNTCLN"]

        daily.append(ds)

    combined = xr.concat(daily, dim="time")
    return combined
