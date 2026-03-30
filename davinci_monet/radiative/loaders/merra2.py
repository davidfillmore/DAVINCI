"""MERRA-2 aerosol data loader."""

from __future__ import annotations

import glob

import xarray as xr

from davinci_monet.logging import get_logger
from davinci_monet.radiative.processing import derive_smoke_aod

logger = get_logger(__name__)


def load_merra2(
    files: str,
    domain: tuple[float, float, float, float],
    smoke_species: list[str],
) -> xr.Dataset:
    """Load MERRA-2 aerosol files, compute daily means, and derive smoke AOD.

    Parameters
    ----------
    files : str
        Glob pattern matching MERRA-2 NetCDF files.
    domain : tuple
        ``(west, east, south, north)`` bounding box.
    smoke_species : list[str]
        Variable names to sum for smoke AOD (e.g. ``["OCEXTTAU", "BCEXTTAU"]``).

    Returns
    -------
    xr.Dataset
        Combined dataset with daily-mean values and a ``SMOKEAOD`` variable.
    """
    paths = sorted(glob.glob(files))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {files}")
    logger.info("Loading %d MERRA-2 file(s)", len(paths))

    west, east, south, north = domain
    daily: list[xr.Dataset] = []

    for p in paths:
        ds = xr.open_dataset(p)
        # MERRA-2 has ascending latitudes — slice(south, north)
        ds = ds.sel(lat=slice(south, north), lon=slice(west, east))
        ds = ds.mean(dim="time", keepdims=True)
        ds["SMOKEAOD"] = derive_smoke_aod(ds, smoke_species)
        daily.append(ds)

    combined = xr.concat(daily, dim="time")
    return combined
