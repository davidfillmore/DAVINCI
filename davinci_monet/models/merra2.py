"""MERRA-2 reader.

Reads MERRA-2 (NASA GMAO) gridded NetCDF4 output as a GRID source. Handles
2D collections (e.g. ``tavg1_2d_slv_Nx``, ``tavgM_2d_aer_Nx``) and 3D
collections on model levels (``*_Nv``) or pressure levels (``*_Np``).

Vertical convention
-------------------
The reader renames the vertical dim ``lev`` to the canonical ``z`` and
preserves its ordering. It does NOT slice a surface level — surface
extraction is single-sourced downstream (``_extract_surface`` /
``surface_level_index``), which auto-detects the surface by whether the
vertical coordinate increases with index. For MERRA-2 that means:

* ``*_Nv`` (model levels): values increase with index, surface = last.
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
