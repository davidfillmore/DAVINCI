"""CERES L3 gridded readers (EBAF; SYN1deg arrives in Phase 2).

EBAF (Energy Balanced and Filled) ships as a single whole-record monthly
netCDF on a 1-degree grid with CF-standard ``(time, lat, lon)`` dims plus
``ctime``/``sc``-dimensioned climatology variables. This reader:

* selects requested variables (native EBAF names, e.g. ``toa_lw_all_mon``);
* drops climatology dims when no selected variable uses them;
* normalizes longitude from EBAF's 0-360 to the repo convention of
  sorted [-180, 180);
* tags GRID geometry for the pairing engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.io.reader_utils import (
    retry_transient_open,
    select_variables,
    set_geometry_attr,
    validate_file_list,
)


def _drop_unused_dims(ds: xr.Dataset) -> xr.Dataset:
    """Drop dims (e.g. EBAF's ``ctime``/``sc``) used by no data variable."""
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
