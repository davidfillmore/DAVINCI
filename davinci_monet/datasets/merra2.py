"""MERRA-2 reader.

Reads MERRA-2 (NASA GMAO) gridded NetCDF4 output as a GRID source. Handles
2D collections (e.g. ``tavg1_2d_slv_Nx``, ``tavgM_2d_aer_Nx``) and 3D
collections on dataset levels (``*_Nv``) or pressure levels (``*_Np``).

Vertical convention
-------------------
The reader renames the vertical dim ``lev`` to the canonical ``z`` and
preserves its ordering. It does NOT slice a surface level — surface
extraction is single-sourced downstream (``_extract_surface`` /
``surface_level_index``), which auto-detects the surface by whether the
vertical coordinate increases with index. For MERRA-2 that means:

* ``*_Nv`` (dataset levels): values increase with index, surface = last.
* ``*_Np`` (pressure levels): values decrease with index (1000 hPa first),
  surface = first.

MERRA-2 files on external drives carry macOS resource-fork sidecars
(``._*.nc4``); these are filtered before opening.
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
    standardize_dims,
    validate_file_list,
)


@source_registry.register("merra2")
class MERRA2Reader:
    """Reader for MERRA-2 gridded NetCDF4 output."""

    @property
    def name(self) -> str:
        """Return reader name."""
        return "merra2"

    @property
    def geometry(self) -> DataGeometry:
        """MERRA-2 output is gridded."""
        return DataGeometry.GRID

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open MERRA-2 NetCDF4 files and standardize to (time, z, lat, lon).

        Parameters
        ----------
        file_paths
            Paths to MERRA-2 ``.nc4`` files (resource-fork ``._*`` sidecars
            are ignored).
        variables
            Variables to load. If None, loads all.
        **kwargs
            Passed through to xarray's open functions.

        Returns
        -------
        xr.Dataset
            Standardized dataset with GRID geometry tagged.
        """
        # Filter macOS resource-fork sidecars before validation so the count
        # reflects real data files (external drives carry ``._*.nc4``).
        real = [Path(f) for f in file_paths if not Path(f).name.startswith("._")]
        file_list = validate_file_list(real, dataset_label="MERRA-2")

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

        ds = retry_transient_open(_open, context="Opening MERRA-2 files")
        return self._standardize_dataset(ds)

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename ``lev``->``z`` (when present) and tag GRID geometry."""
        ds = standardize_dims(ds, {"lev": "z"})
        return set_geometry_attr(ds, DataGeometry.GRID)
