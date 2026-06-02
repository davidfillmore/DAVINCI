"""Catalog-driven MODIS/VIIRS reader (L3 grid slice).

Initial vertical slice of the modis_viirs catalog reader design: Level-3
regular-grid atmosphere products (MOD08_M3 / MYD08_M3). See
docs/superpowers/specs/2026-06-01-modis-viirs-catalog-readers-design.md.
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.core.exceptions import DataNotFoundError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.observations.satellite.catalog import ProductEntry, get_catalog

_DATE_TOKEN = re.compile(r"\.A(\d{7})\.")  # ".A2024032." -> 2024032


@source_registry.register("modis_viirs")
class MODISVIIRSReader:
    """Catalog-driven reader for MODIS/VIIRS products (L3 grid path)."""

    def __init__(self) -> None:
        self.geometry: DataGeometry = DataGeometry.GRID

    @property
    def name(self) -> str:
        return "modis_viirs"

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        *,
        product: str | None = None,
        level: str | None = None,
        time_range: tuple[Any, Any] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open MODIS/VIIRS L3 grid files.

        Parameters
        ----------
        file_paths
            Paths to MODIS/VIIRS HDF4 or NetCDF files.
        variables
            Catalog display names of variables to load (e.g. ``"aod_550nm"``).
            Passing ``None`` or ``["*"]`` loads all cataloged variables.
        product
            Catalog product ID (e.g. ``"MOD08_M3"``). Required.
        level
            Optional level string for validation against catalog entry.
        time_range
            Accepted for interface compatibility but not applied here: time
            filtering is performed by the pipeline after ``open()``, so the
            reader does not apply it.
        **kwargs
            Ignored; present for interface compatibility.

        Returns
        -------
        xr.Dataset
            Dataset with display-name variables, ``lat``/``lon`` coords,
            and a ``time`` dimension parsed from filenames.
        """
        if not product:
            raise ValueError("modis_viirs source requires a 'product' (e.g. MOD08_M3).")
        entry = get_catalog().resolve(product)  # raises UnknownProductError if absent
        if level and level.upper() != entry.level.upper():
            raise ValueError(
                f"Configured level '{level}' != catalog level '{entry.level}' for {product}."
            )
        if entry.geometry.upper() != "GRID":
            raise NotImplementedError(
                f"modis_viirs slice supports GRID products only; " f"{product} is {entry.geometry}."
            )

        files = [Path(f) for f in file_paths]
        if not files:
            raise DataNotFoundError("No MODIS/VIIRS files provided")
        missing = [f for f in files if not f.exists()]
        if missing:
            raise DataNotFoundError(f"MODIS/VIIRS files not found: {missing}")

        per_file = [self._open_one(f, variables, entry) for f in files]
        valid = [d for d in per_file if d is not None]
        if not valid:
            raise DataNotFoundError(
                f"No valid MODIS/VIIRS data found (0 of {len(files)} files opened successfully)"
            )

        ds = xr.concat(valid, dim="time").sortby("time")
        self.geometry = DataGeometry.GRID
        ds.attrs["geometry"] = DataGeometry.GRID.value
        ds.attrs.update(
            product_id=entry.product_id,
            instrument=entry.instrument,
            platform=entry.platform,
            level=entry.level,
            daac=entry.daac,
            collection=entry.collection,
        )
        return ds

    def _open_one(
        self, fpath: Path, variables: Sequence[str] | None, entry: ProductEntry
    ) -> xr.Dataset | None:
        # MOD08_M3/MYD08_M3 HDF4 files contain 4-D histogram variables with
        # duplicate dimension names.  xarray emits a UserWarning about these
        # on open(), set_coords(), and variable-subset operations (which all
        # copy the full raw dataset).  The warning is harmless for our use-case
        # — we immediately drop those variables — but the message is misleading
        # and breaks test suites with filterwarnings=error.  We therefore
        # suppress it for the entire raw-dataset phase (open → subset); all
        # operations on the clean subsetted dataset proceed normally.
        _dup_dim_msg = "Duplicate dimension names present"
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", _dup_dim_msg, UserWarning)
                raw = xr.open_dataset(str(fpath), engine="netcdf4", mask_and_scale=True)
        except Exception as e:  # pragma: no cover - exercised via smoke test
            warnings.warn(f"Failed to open {fpath}: {e}", UserWarning)
            return None

        # Resolve which catalog variables to keep (display names -> SDS names).
        wanted: list[str] | None = list(variables) if variables else None
        if wanted is None or wanted == ["*"]:
            selected = entry.variables
        else:
            selected = []
            for name in wanted:
                v = entry.variable_by_display(name) or entry.variable_by_sds(name)
                if v is None:
                    warnings.warn(
                        f"{entry.product_id}: variable '{name}' not in catalog", UserWarning
                    )
                    continue
                selected.append(v)

        keep = {v.sds_name: v for v in selected if v.sds_name in raw.data_vars}
        if not keep:
            warnings.warn(f"{fpath.name}: none of the requested SDS present", UserWarning)
            return None

        # Promote grid-axis variables (e.g. XDim, YDim) to coordinates before
        # selecting, so they survive the variable-subset and can be renamed to
        # lon/lat by _standardize_grid.  In real HDF4 files these are plain
        # data variables (not coords), so raw[list(keep)] would otherwise drop
        # them; the synthetic unit-test fixture creates them as coords, so this
        # is a no-op there.
        # set_coords() and variable selection both copy the full raw dataset,
        # triggering the duplicate-dim warning again — keep the filter active.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", _dup_dim_msg, UserWarning)
            axis_vars = [k for k in entry.dim_aliases if k in raw.data_vars and k not in raw.coords]
            if axis_vars:
                raw = raw.set_coords(axis_vars)
            ds = raw[list(keep)]

        # Rename SDS -> display name and attach variable metadata.
        ds = ds.rename({sds: v.display_name for sds, v in keep.items()})
        for v in keep.values():
            attrs = ds[v.display_name].attrs
            attrs["units"] = v.units
            if v.wavelength_nm is not None:
                attrs["wavelength_nm"] = v.wavelength_nm
            if v.long_name:
                attrs["long_name"] = v.long_name

        # Standardize grid coords: dim_aliases maps file dim/coord names -> lon/lat.
        ds = self._standardize_grid(ds, entry)

        # Parse the month from the filename and assign a time coordinate.
        ds = ds.expand_dims(time=[self._parse_time(fpath.name, entry)])
        return ds

    @staticmethod
    def _standardize_grid(ds: xr.Dataset, entry: ProductEntry) -> xr.Dataset:
        """Apply dim_aliases to rename dims and coords to lon/lat.

        xarray does not allow renaming a dim and a coord to the same target
        name in one call when a coord shares a different name than the dim.
        We therefore rename dims first, then coords separately.
        """
        aliases = entry.dim_aliases

        # --- 1. rename dims ---------------------------------------------------
        dim_renames = {k: v for k, v in aliases.items() if k in ds.dims}
        if dim_renames:
            ds = ds.rename_dims(dim_renames)

        # --- 2. rename coords -------------------------------------------------
        coord_renames = {k: v for k, v in aliases.items() if k in ds.coords}
        # Only rename if target name not already a coord (avoids collision).
        coord_renames = {k: v for k, v in coord_renames.items() if v not in ds.coords}
        if coord_renames:
            ds = ds.rename_vars(coord_renames)

        # Promote lon/lat to coordinates if they are bare 1-D variables.
        for axis in ("lon", "lat"):
            if axis in ds.variables and axis not in ds.coords:
                ds = ds.set_coords(axis)
        return ds

    @staticmethod
    def _parse_time(filename: str, entry: ProductEntry) -> np.datetime64:
        m = _DATE_TOKEN.search(filename)
        if not m:
            raise ValueError(f"Cannot parse date token from filename: {filename}")
        dt = datetime.strptime("A" + m.group(1), entry.time_parse)
        # Monthly products: snap to first-of-month for clean monthly alignment.
        return np.datetime64(pd.Timestamp(dt).to_period("M").to_timestamp(), "ns")
