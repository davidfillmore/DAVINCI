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


from davinci_monet.analysis.base import DerivedAnalysis  # noqa: E402
from davinci_monet.core.protocols import DataGeometry  # noqa: E402
from davinci_monet.core.registry import analysis_registry  # noqa: E402


@analysis_registry.register("eof")
class EOFAnalysis(DerivedAnalysis):
    """Empirical Orthogonal Function decomposition of a gridded field."""

    name = "eof"
    long_name = "Empirical Orthogonal Function Decomposition"
    output_geometry = DataGeometry.GRID

    def analyze(self, data: xr.Dataset, spec: "EOFSpec") -> xr.Dataset:
        from xeofs.models import EOF, EOFRotator

        da = data[spec.variable]
        lat = _lat_coord(da)
        lon = _lon_coord(da)
        vdim = _vertical_dim(da, lat, lon)
        if spec.level is not None and vdim is not None:
            da = da.isel({vdim: spec.level})
            vdim = None

        anom = da - da.mean("time")
        if spec.remove_seasonal_cycle:
            clim = anom.groupby("time.month").mean("time")
            anom = anom.groupby("time.month") - clim
        if spec.standardize:
            std = anom.std("time")
            anom = anom / std.where(std > 0)

        weight = _area_weight(anom, lat)
        if vdim is not None and not spec.standardize:
            mw = _layer_mass_weight(data, vdim)
            if mw is None:
                logger.warning(
                    "EOF 3-D mass weighting unavailable for '%s'; using equal layer weight",
                    spec.variable,
                )
            else:
                weight = weight * mw
        elif vdim is not None and spec.standardize:
            logger.warning(
                "EOF standardize=True with a 3-D field: vertical mass weighting disabled "
                "(per-cell standardization already equalizes variance)"
            )
        weight = weight.fillna(0.0)

        weighted = (anom * weight).fillna(0.0)
        model: object = EOF(n_modes=spec.n_modes, use_coslat=False, standardize=False)
        model.fit(weighted, dim="time")
        if spec.rotation == "varimax":
            model = EOFRotator(n_modes=spec.n_modes).fit(model)

        scores = model.scores()                       # (mode, time)
        ev_ratio = model.explained_variance_ratio()   # (mode,)

        pc = scores / scores.std("time")              # (mode, time), unit variance
        # Regression of anomaly onto unit-variance PCs → physical spatial modes.
        # Result has dims (lat, ..., mode); transpose to (mode, <spatial>).
        mode_raw = (anom * pc).mean("time")
        spatial_dims = [d for d in mode_raw.dims if d != "mode"]
        mode_raw = mode_raw.transpose("mode", *spatial_dims)
        mode_raw, pc = _fix_sign(mode_raw, pc)

        n_modes = int(ev_ratio.sizes["mode"])
        mode_idx = np.arange(1, n_modes + 1)

        # Build Dataset from numpy arrays to avoid DataArray dim-name collisions.
        # xarray does not allow a data variable to share its name with one of its
        # dimensions, so spatial patterns are stored as "eofs" (not "mode") while
        # the "mode" dimension carries an integer index 1..n_modes.
        pc_transposed = pc.transpose("time", "mode")
        ds = xr.Dataset(
            {
                "eofs": xr.Variable(
                    ("mode", *spatial_dims),
                    mode_raw.values,
                    attrs=dict(
                        units=str(da.attrs.get("units", "")),
                        long_name=f"EOF spatial pattern of {spec.variable}",
                        kind="eofs",
                    ),
                ),
                "pc": xr.Variable(
                    ("time", "mode"),
                    pc_transposed.values,
                    attrs=dict(units="1", long_name=f"Principal component of {spec.variable}", kind="pc"),
                ),
                "explained_variance": xr.Variable(
                    ("mode",),
                    ev_ratio.values,
                    attrs=dict(kind="scalar", percent=True),
                ),
            }
        )
        if spec.rotation == "none":
            n_eff = _effective_n(anom, lat)
            err_vals = ev_ratio.values * np.sqrt(2.0 / n_eff)
            ds["explained_variance_error"] = xr.Variable(
                ("mode",), err_vals, attrs=dict(kind="scalar")
            )

        # Assign coordinates: integer mode index, spatial dim values, time.
        coord_kwargs: dict[str, object] = {"mode": mode_idx}
        for dim in spatial_dims:
            if dim in anom.coords:
                coord_kwargs[dim] = anom.coords[dim].values
        if "time" in anom.coords:
            coord_kwargs["time"] = anom.coords["time"].values
        ds = ds.assign_coords(**coord_kwargs)
        ds.attrs["eof_quantity"] = spec.variable
        return ds
