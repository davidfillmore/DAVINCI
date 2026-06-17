"""EOF (Empirical Orthogonal Function) decomposition of a gridded field."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from davinci_monet.config.schema import EOFSpec

logger = logging.getLogger(__name__)

_LAT_NAMES = ("latitude", "lat", "LAT", "Latitude")
_LON_NAMES = ("longitude", "lon", "LON", "Longitude")


def _named_coord(da: xr.DataArray, names: tuple[str, ...], kind: str) -> xr.DataArray:
    for name in names:
        if name in da.coords:
            return da.coords[name]
    raise ValueError(f"EOF requires a {kind} coordinate (one of {names})")


def _lat_coord(da: xr.DataArray) -> xr.DataArray:
    return _named_coord(da, _LAT_NAMES, "latitude")


def _lon_coord(da: xr.DataArray) -> xr.DataArray:
    return _named_coord(da, _LON_NAMES, "longitude")


def _vertical_dim(da: xr.DataArray, lat: xr.DataArray, lon: xr.DataArray) -> str | None:
    horiz = set(lat.dims) | set(lon.dims)
    verts = [d for d in da.dims if d != "time" and d not in horiz]
    if len(verts) > 1:
        raise ValueError(f"EOF: ambiguous vertical dims {verts}; expected one")
    return verts[0] if verts else None


def _area_weight(da: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
    """sqrt(cos(lat)) broadcast over the latitude dimension."""
    coslat = np.cos(np.deg2rad(lat)).clip(min=0.0)
    return np.sqrt(coslat)


def _layer_mass_weight(data: xr.Dataset, vdim: str) -> xr.DataArray | None:
    """sqrt(normalized layer pressure thickness) over the vertical dim, or None.

    Uses ``ilev`` pressure edges if present, else CESM hybrid coefficients
    (hyai/hybi + PS or P0). Returns None when no vertical thickness info exists;
    the caller then falls back to equal layer weight (logged, not warned).
    """
    nlev = int(data.sizes[vdim])
    dp: np.ndarray | None = None
    if "ilev" in data.coords and int(data.sizes.get("ilev", 0)) == nlev + 1:
        dp = np.abs(np.diff(np.asarray(data["ilev"].values, dtype=float)))
    elif {"hyai", "hybi"} <= set(data.variables):
        p0 = float(data["P0"]) if "P0" in data.variables else 1.0e5
        ps = float(np.asarray(data["PS"].values).mean()) if "PS" in data.variables else p0
        edges = np.asarray(data["hyai"].values, float) * p0 + np.asarray(data["hybi"].values, float) * ps
        if edges.size == nlev + 1:
            dp = np.abs(np.diff(edges))
    if dp is None:
        return None
    dpn = dp / dp.sum()
    return xr.DataArray(np.sqrt(dpn), dims=[vdim])


def _fix_sign(mode: xr.DataArray, pc: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Flip each mode so its largest-|loading| spatial point is positive.

    Deterministic and robust for dipole modes (a domain-mean rule is not).
    """
    spatial = [d for d in mode.dims if d != "mode"]
    flat = mode.stack(_pt=spatial)
    idx = np.abs(flat).argmax("_pt")
    peak = flat.isel(_pt=idx)
    signs = xr.where(peak >= 0, 1.0, -1.0)
    return mode * signs, pc * signs


def _effective_n(anom: xr.DataArray, lat: xr.DataArray) -> float:
    """Effective independent sample count from the area-mean series lag-1 autocorr."""
    coslat = np.cos(np.deg2rad(lat)).clip(min=0.0)
    spatial = [d for d in anom.dims if d != "time"]
    am = anom.weighted(coslat).mean(dim=spatial)
    x = np.asarray(am.values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(len(x))
    if n < 3:
        return float(max(n, 1))
    r1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    r1 = float(np.clip(r1, -0.99, 0.99))
    return n * (1.0 - r1) / (1.0 + r1)
