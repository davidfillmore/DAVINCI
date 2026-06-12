"""CERES L3 gridded readers (EBAF and SYN1deg).

EBAF (Energy Balanced and Filled) ships as a single whole-record monthly
netCDF on a 1-degree grid with CF-standard ``(time, lat, lon)`` dims plus
``ctime``/``sc``-dimensioned climatology variables. This reader:

* selects requested variables (native EBAF names, e.g. ``toa_lw_all_mon``);
* drops climatology dims when no selected variable uses them;
* normalizes longitude from EBAF's 0-360 to the repo convention of
  sorted [-180, 180);
* tags GRID geometry for the pairing engine.

SYN1deg ships as HDF4 — one file per month, day, or day-of-hours, with 2-D
``(latitude, longitude)`` SDS (3-D ``(gmt_hr_index, latitude, longitude)``
for hourly) and explicit 1-D coordinate SDS (lat descending). Timestamps
come from the filename tail (``.YYYYMM`` or ``.YYYYMMDD``); hourly files
expand to 24 steps. Fill/valid_range/scale handling uses the shared
``apply_hdf4_scale``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.io.reader_utils import (
    apply_hdf4_scale,
    retry_transient_open,
    select_variables,
    set_geometry_attr,
    validate_file_list,
)


def _drop_unused_dims(ds: xr.Dataset) -> xr.Dataset:
    """Drop dims (e.g. EBAF's ``ctime``/``sc``) used by no data variable.

    ``select_variables`` already drops climatology dims on the selection
    path; this also covers orphan dims in a ``variables=None`` open. Note
    ``drop_dims`` removes *any* variable touching a dropped dim (e.g. a
    hypothetical bounds coord) — acceptable for CERES L3 files, which
    carry none.
    """
    used: set[Any] = set()
    for var in ds.data_vars.values():
        used.update(var.dims)
    unused = [d for d in ds.dims if d not in used]
    return ds.drop_dims(unused) if unused else ds


def _normalize_longitude(ds: xr.Dataset) -> xr.Dataset:
    """Wrap a 0-360 ``lon`` coord to [-180, 180) and sort ascending."""
    if "lon" not in ds.coords:
        return ds
    lon = ds["lon"].values
    if lon.size and float(lon.max()) > 180.0:
        ds = ds.assign_coords(lon=(((lon + 180.0) % 360.0) - 180.0))
        ds = ds.sortby("lon")
    return ds


@source_registry.register("ceres_ebaf")
class CERESEBAFReader:
    """Reader for CERES EBAF monthly gridded netCDF (TOA + surface fluxes)."""

    @property
    def name(self) -> str:
        """Return reader name."""
        return "ceres_ebaf"

    @property
    def geometry(self) -> DataGeometry:
        """EBAF is gridded."""
        return DataGeometry.GRID

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open EBAF netCDF file(s) and standardize to (time, lat, lon).

        Parameters
        ----------
        file_paths
            Paths to EBAF ``.nc`` files (resource-fork ``._*`` sidecars are
            ignored). EBAF normally ships as one whole-record file.
        variables
            Native EBAF variable names to load (e.g. ``toa_lw_all_mon``).
            If None, loads all, including climatology variables.
        **kwargs
            Passed through to xarray's open functions.

        Returns
        -------
        xr.Dataset
            Standardized dataset with GRID geometry tagged.
        """
        real = [Path(f) for f in file_paths if not Path(f).name.startswith("._")]
        file_list = validate_file_list(real, source_label="CERES EBAF")

        def _open() -> xr.Dataset:
            if len(file_list) > 1:
                ds = xr.open_mfdataset(
                    [str(f) for f in file_list],
                    combine="by_coords",
                    parallel=True,
                    **kwargs,
                )
            else:
                ds = xr.open_dataset(str(file_list[0]), **kwargs)
            return select_variables(ds, variables)

        ds = retry_transient_open(_open, context="Opening CERES EBAF files")
        return self._standardize_dataset(ds)

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Drop unused climatology dims, fix longitudes, tag GRID geometry."""
        ds = _drop_unused_dims(ds)
        ds = _normalize_longitude(ds)
        return set_geometry_attr(ds, DataGeometry.GRID)


_SYN_DATE_RE = re.compile(r"\.(\d{6}|\d{8})$")


def _syn_time_from_filename(path: Path) -> np.datetime64:
    """Parse the date stamp from a SYN1deg filename tail.

    ``...Edition4B_415412.202512`` -> 2025-12-01;
    ``...Edition4B_415412.20251229`` -> 2025-12-29.
    """
    m = _SYN_DATE_RE.search(path.name)
    if m is None:
        raise ValueError(
            f"Cannot parse a SYN1deg date from filename {path.name!r}: expected a "
            "'.YYYYMM' or '.YYYYMMDD' tail"
        )
    stamp = m.group(1)
    try:
        if len(stamp) == 6:
            return np.datetime64(f"{stamp[:4]}-{stamp[4:6]}-01")
        return np.datetime64(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}")
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse a SYN1deg date from filename {path.name!r}: "
            f"{stamp!r} is not a valid date"
        ) from exc


@source_registry.register("ceres_syn1deg")
class CERESSYN1degReader:
    """Reader for CERES SYN1deg HDF4 files (monthly, daily, or hourly)."""

    @property
    def name(self) -> str:
        """Return reader name."""
        return "ceres_syn1deg"

    @property
    def geometry(self) -> DataGeometry:
        """SYN1deg is gridded."""
        return DataGeometry.GRID

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open SYN1deg HDF4 file(s) and assemble (time, lat, lon).

        Parameters
        ----------
        file_paths
            Paths to SYN1deg HDF4 files (resource-fork ``._*`` sidecars are
            ignored). Monthly/daily/hourly cadences may not be mixed in one
            open.
        variables
            Native SYN1deg SDS names (e.g. ``obs_all_toa_lw_reg``). If None,
            loads all regional/hourly SDS on the (latitude, longitude) grid;
            zonal-mean and layered SDS are skipped.
        **kwargs
            Unused; accepted for protocol compatibility.

        Returns
        -------
        xr.Dataset
            Standardized dataset with GRID geometry tagged.
        """
        real = [Path(f) for f in file_paths if not Path(f).name.startswith("._")]
        file_list = validate_file_list(real, source_label="CERES SYN1deg")

        per_file = [self._open_one(path, variables) for path in file_list]
        ds = per_file[0] if len(per_file) == 1 else xr.concat(per_file, dim="time")
        ds = ds.sortby("time")
        if "lat" in ds.coords and ds["lat"].values[0] > ds["lat"].values[-1]:
            ds = ds.sortby("lat")
        return set_geometry_attr(ds, DataGeometry.GRID)

    def _open_one(self, path: Path, variables: Sequence[str] | None) -> xr.Dataset:
        """Read one SYN1deg HDF4 file into an in-memory (time, lat, lon) Dataset."""
        try:
            from pyhdf.SD import SD, SDC
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "pyhdf is required to read CERES SYN1deg HDF4 files " "(conda install pyhdf)"
            ) from exc

        base_time = _syn_time_from_filename(path)
        explicit = variables is not None
        hdf = SD(str(path), SDC.READ)
        try:
            available = hdf.datasets()
            if variables is not None:
                missing = [v for v in variables if v not in available]
                if missing:
                    raise ValueError(f"SYN1deg variable(s) not found in {path.name}: {missing!r}")
                names = list(variables)
            else:
                names = list(available)
            names = [n for n in names if n not in ("latitude", "longitude")]
            lat = np.asarray(hdf.select("latitude").get(), dtype=np.float64)
            lon = np.asarray(hdf.select("longitude").get(), dtype=np.float64)

            data_vars: dict[str, Any] = {}
            hourly = False
            for name in names:
                sds = hdf.select(name)
                dims_tuple = tuple(sds.dimensions().keys())
                if dims_tuple not in (
                    ("latitude", "longitude"),
                    ("gmt_hr_index", "latitude", "longitude"),
                ):
                    sds.endaccess()
                    if explicit:
                        raise ValueError(
                            f"SYN1deg variable {name!r} has unsupported dims "
                            f"{dims_tuple}; this reader handles regional "
                            "(latitude, longitude) and hourly "
                            "(gmt_hr_index, latitude, longitude) fields"
                        )
                    continue  # variables=None scan: skip zonal/layered SDS
                attrs = sds.attributes()
                values = apply_hdf4_scale(np.asarray(sds.get()), attrs)
                sds.endaccess()
                keep_attrs = {k: attrs[k] for k in ("units", "long_name") if k in attrs}
                if dims_tuple == ("gmt_hr_index", "latitude", "longitude"):
                    hourly = True
                    data_vars[name] = (("time", "lat", "lon"), values, keep_attrs)
                else:
                    data_vars[name] = (("time", "lat", "lon"), values[None, ...], keep_attrs)
        finally:
            hdf.end()

        if hourly:
            times: np.ndarray = base_time + np.arange(24) * np.timedelta64(1, "h")
        else:
            times = np.array([base_time])
        return xr.Dataset(data_vars, coords={"time": times, "lat": lat, "lon": lon})
