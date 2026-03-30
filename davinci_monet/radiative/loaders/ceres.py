"""CERES data loader — local files and OPeNDAP fetch utilities.

The local loader (``load_ceres_local``) reads SYN1deg-Day NetCDF files and
returns a domain-subsetted xarray Dataset.

The OPeNDAP helpers (``fetch_ebaf``, ``fetch_syn1deg_day``,
``fetch_syn1deg_range``) download data from the NASA LaRC OPeNDAP server
using a DAP2 binary parser to avoid HDF4 compatibility issues.
"""

from __future__ import annotations

import glob
import struct
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr

from davinci_monet.logging import get_logger

logger = get_logger(__name__)

# OPeNDAP base URL for CERES SYN1deg-Day
_SYN1DEG_BASE_URL = (
    "https://opendap.larc.nasa.gov/opendap/CERES/" "SYN1deg-Day/Terra-Aqua-NOAA20_Edition4B"
)

_NLAT, _NLON = 180, 360


# ---------------------------------------------------------------------------
# Local loader
# ---------------------------------------------------------------------------


def load_ceres_local(
    files: str,
    domain: tuple[float, float, float, float],
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Load CERES NetCDF files from a glob pattern and subset to *domain*.

    Parameters
    ----------
    files : str
        Glob pattern matching one or more NetCDF files.
    domain : tuple
        ``(west, east, south, north)`` bounding box.
    variables : list[str] | None
        If given, keep only these data variables.

    Returns
    -------
    xr.Dataset
        Combined dataset with a ``time`` dimension.
    """
    paths = sorted(glob.glob(files))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {files}")
    logger.info("Loading %d CERES file(s)", len(paths))

    datasets: list[xr.Dataset] = []
    for p in paths:
        ds = xr.open_dataset(p)
        # Replace fill values
        ds = ds.where(ds > -900.0)
        if variables is not None:
            available = [v for v in variables if v in ds]
            ds = ds[available]
        datasets.append(ds)

    if len(datasets) == 1:
        combined = datasets[0].expand_dims("time")
    else:
        # Add a dummy time dim to each then concat
        for i, ds in enumerate(datasets):
            datasets[i] = ds.expand_dims("time")
        combined = xr.concat(datasets, dim="time")

    # Subset to domain
    west, east, south, north = domain
    lats = combined.lat.values
    if lats[0] > lats[-1]:
        # Descending latitudes — slice is (north, south)
        combined = combined.sel(
            lat=slice(north, south),
            lon=slice(west, east),
        )
    else:
        combined = combined.sel(
            lat=slice(south, north),
            lon=slice(west, east),
        )

    return combined


# ---------------------------------------------------------------------------
# OPeNDAP helpers (not tested — require network access)
# ---------------------------------------------------------------------------


def _build_syn1deg_url(d: date) -> str:
    """Build the OPeNDAP HDF URL for a given date."""
    return (
        f"{_SYN1DEG_BASE_URL}/{d.year}/{d.month:02d}/"
        f"CER_SYN1deg-Day_Terra-Aqua-NOAA20_Edition4B_408412."
        f"{d.strftime('%Y%m%d')}.hdf"
    )


def _parse_dods_grid(data: bytes, offset: int, shape: tuple[int, ...]) -> tuple[np.ndarray, int]:
    """Parse a DAP2 Grid from binary .dods response.

    A Grid consists of the array data followed by each map vector.
    Each array/vector is preceded by two int32 (type-length, data-length).
    """
    n = int(np.prod(shape))

    # Array header
    _arr_n1, _arr_n2 = struct.unpack_from(">II", data, offset)
    offset += 8
    arr = np.frombuffer(data, dtype=">f4", count=n, offset=offset).copy()
    offset += n * 4
    arr = arr.reshape(shape)

    # Skip map vectors (one per dimension)
    for dim_size in shape:
        offset += 8  # two int32 header
        offset += dim_size * 4  # float32 vector

    return arr, offset


def _fetch_var_dods(hdf_url: str, varname: str, shape: tuple[int, ...]) -> np.ndarray:
    """Fetch a single variable via the DAP2 binary interface."""
    import requests

    url = f"{hdf_url}.dods?{varname}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    sep = b"\nData:\n"
    idx = r.content.index(sep) + len(sep)
    arr, _ = _parse_dods_grid(r.content, idx, shape)
    return arr


def _fetch_coords(hdf_url: str) -> tuple[np.ndarray, np.ndarray]:
    """Fetch lat/lon coordinate arrays from an OPeNDAP endpoint."""
    lat = _fetch_var_dods(hdf_url, "latitude", (_NLAT,)).ravel()
    lon = _fetch_var_dods(hdf_url, "longitude", (_NLON,)).ravel()
    return lat, lon


def fetch_syn1deg_day(d: date, output_dir: str | Path) -> Path:
    """Fetch one day of CERES SYN1deg-Day data via OPeNDAP.

    Parameters
    ----------
    d : date
        Date to fetch.
    output_dir : str or Path
        Directory where the NetCDF file will be written.

    Returns
    -------
    Path
        Path to the written NetCDF file.
    """
    import netCDF4

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"CERES_SYN1deg-Day_{d.strftime('%Y%m%d')}.nc"

    if out_path.exists():
        logger.info("Already exists: %s", out_path.name)
        return out_path

    hdf_url = _build_syn1deg_url(d)
    logger.info("Fetching CERES SYN1deg-Day %s", d)
    lat, lon = _fetch_coords(hdf_url)

    variables_2d = [
        "obs_all_toa_sw",
        "obs_all_toa_lw",
        "obs_all_toa_net",
        "obs_clr_toa_sw",
        "obs_clr_toa_lw",
        "obs_clr_toa_net",
        "toa_sw_insol",
        "init_all_sfc_sw_dn",
        "init_all_sfc_sw_up",
        "init_all_sfc_lw_dn",
        "init_all_sfc_lw_up",
        "init_clr_sfc_sw_dn",
        "init_clr_sfc_lw_dn",
        "init_match_aod55",
        "init_skin_temp",
        "init_pw",
    ]
    variables_3d = ["obs_cld_amount", "obs_cld_od"]

    dst = netCDF4.Dataset(str(out_path), "w", format="NETCDF4")
    dst.title = f"CERES SYN1deg-Day {d.isoformat()}"
    dst.source = hdf_url
    dst.date = d.isoformat()

    dst.createDimension("lat", _NLAT)
    dst.createDimension("lon", _NLON)

    lat_var = dst.createVariable("lat", "f4", ("lat",))
    lat_var[:] = lat
    lat_var.units = "degrees_north"

    lon_var = dst.createVariable("lon", "f4", ("lon",))
    lon_var[:] = lon
    lon_var.units = "degrees_east"

    for vname in variables_2d:
        try:
            data = _fetch_var_dods(hdf_url, vname, (_NLAT, _NLON))
            dv = dst.createVariable(vname, "f4", ("lat", "lon"), fill_value=-999.0, zlib=True)
            dv[:] = np.where(data > -900, data, -999.0)
            logger.debug("  %s ok", vname)
        except Exception as exc:
            logger.warning("  %s FAILED: %s", vname, exc)

    for vname in variables_3d:
        try:
            data = _fetch_var_dods(hdf_url, vname, (5, _NLAT, _NLON))
            dv = dst.createVariable(vname, "f4", ("lat", "lon"), fill_value=-999.0, zlib=True)
            dv[:] = np.where(data[0] > -900, data[0], -999.0)
            logger.debug("  %s (total column) ok", vname)
        except Exception as exc:
            logger.warning("  %s FAILED: %s", vname, exc)

    dst.close()
    logger.info("Wrote %s", out_path.name)
    return out_path


def fetch_syn1deg_range(start: date, end: date, output_dir: str | Path) -> list[Path]:
    """Fetch a range of CERES SYN1deg-Day files.

    Parameters
    ----------
    start, end : date
        Inclusive date range.
    output_dir : str or Path
        Directory for output files.

    Returns
    -------
    list[Path]
        Paths to all fetched/existing files.
    """
    paths: list[Path] = []
    d = start
    while d <= end:
        paths.append(fetch_syn1deg_day(d, output_dir))
        d += timedelta(days=1)
    return paths


def fetch_ebaf(
    start_time: str,
    end_time: str,
    output_dir: str | Path,
    variables: list[str] | None = None,
) -> Path:
    """Fetch CERES EBAF monthly data via OPeNDAP.

    Parameters
    ----------
    start_time, end_time : str
        ISO date strings defining the time range.
    output_dir : str or Path
        Directory for the output file.
    variables : list[str] | None
        Subset of variables to fetch; all if None.

    Returns
    -------
    Path
        Path to the written NetCDF file.
    """
    import requests

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ebaf_url = (
        "https://opendap.larc.nasa.gov/opendap/CERES/"
        "EBAF/Edition4.2/Subset/CERES_EBAF_Edition4.2_200003-202310.nc"
    )

    out_path = output_dir / "CERES_EBAF.nc"
    logger.info("Fetching CERES EBAF %s to %s", start_time, end_time)

    # Use xarray's OPeNDAP support for EBAF (standard netCDF)
    ds = xr.open_dataset(ebaf_url)
    ds = ds.sel(time=slice(start_time, end_time))
    if variables is not None:
        ds = ds[variables]
    ds.to_netcdf(out_path)
    logger.info("Wrote %s", out_path.name)
    return out_path
