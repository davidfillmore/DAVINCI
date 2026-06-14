"""TEMPO L2 NO2 dataset reader.

This module provides the TEMPOL2NO2Reader class for reading TEMPO
(Tropospheric Emissions: Monitoring of Pollution) L2 NO2 products.

TEMPO is a geostationary satellite instrument over North America providing
hourly datasets of air quality.

Note
----
This reader requires monetio.sat._tempo_l2_no2_mm for full functionality.
Without monetio, it falls back to basic xarray reading which may not handle
all TEMPO-specific features.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Sequence

import xarray as xr

from davinci_monet.core.exceptions import DataNotFoundError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.io.reader_utils import (
    select_variables,
    set_geometry_attr,
    validate_file_list,
)

# Standard variable name mappings for TEMPO NO2
TEMPO_NO2_VARIABLE_MAPPING: dict[str, str] = {
    "no2": "nitrogendioxide_tropospheric_column",
    "no2_total": "nitrogendioxide_total_column",
    "qa_value": "qa_value",
    "sza": "solar_zenith_angle",
    "vza": "viewing_zenith_angle",
}


@source_registry.register("tempo_l2_no2")
class TEMPOL2NO2Reader:
    """Reader for TEMPO L2 NO2 satellite datasets.

    Reads TEMPO L2 NO2 data from NetCDF files. TEMPO provides hourly
    datasets over North America from geostationary orbit.

    Data is returned as swath geometry with appropriate dimensions.

    Note
    ----
    Full functionality requires monetio. Without monetio, the reader falls
    back to basic xarray which may not handle TEMPO-specific features
    correctly.

    Examples
    --------
    >>> reader = TEMPOL2NO2Reader()
    >>> ds = reader.open(["TEMPO_NO2_L2_*.nc"])
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "tempo_l2_no2"

    @property
    def geometry(self) -> DataGeometry:
        """Return produced geometry."""
        return DataGeometry.SWATH

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        *,
        qa_threshold: float | None = 0.75,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open TEMPO L2 NO2 dataset files.

        Parameters
        ----------
        file_paths
            Paths to TEMPO L2 NO2 files.
        variables
            Variables to load. If None, loads main product variables.
        qa_threshold
            Quality assurance threshold (0-1). Pixels below threshold
            are masked. Set to None to disable filtering.
        **kwargs
            Additional options.

        Returns
        -------
        xr.Dataset
            TEMPO datasets with swath dimensions.
        """
        file_list = validate_file_list(file_paths, source_label="TEMPO")

        # Try monetio first
        try:
            ds = self._open_with_monetio(file_list, variables, **kwargs)
        except ImportError:
            warnings.warn(
                "monetio not available, using basic xarray reader. "
                "TEMPO-specific handling may be incomplete.",
                UserWarning,
            )
            ds = self._open_with_xarray(file_list, variables, **kwargs)

        # Apply QA filtering
        if qa_threshold is not None:
            ds = self._apply_qa_filter(ds, qa_threshold)

        return self._standardize_dataset(ds)

    def _open_with_monetio(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open TEMPO files using monetio."""
        import monetio.sat._tempo_l2_no2_mm as tempo_module

        files = [str(f) for f in file_paths]

        if len(files) == 1:
            ds: xr.Dataset = tempo_module.open_dataset(files[0], **kwargs)
        else:
            ds_list = []
            for f in files:
                try:
                    ds_i: xr.Dataset = tempo_module.open_dataset(f, **kwargs)
                    ds_list.append(ds_i)
                except Exception as e:
                    warnings.warn(f"Failed to open {f}: {e}", UserWarning)
                    continue

            if not ds_list:
                raise DataNotFoundError("No valid TEMPO data found")

            ds = xr.concat(ds_list, dim="time")

        return select_variables(ds, variables)

    def _open_with_xarray(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open TEMPO files using xarray."""
        ds_list = []
        for fpath in file_paths:
            try:
                # TEMPO files may have nested groups
                ds = xr.open_dataset(str(fpath), group="product", **kwargs)
                ds_list.append(ds)
            except Exception:
                try:
                    ds = xr.open_dataset(str(fpath), **kwargs)
                    ds_list.append(ds)
                except Exception as e:
                    warnings.warn(f"Failed to open {fpath}: {e}", UserWarning)
                    continue

        if not ds_list:
            raise DataNotFoundError("No valid TEMPO data found")

        if len(ds_list) > 1:
            ds = xr.concat(ds_list, dim="time")
        else:
            ds = ds_list[0]

        return select_variables(ds, variables)

    def _apply_qa_filter(self, ds: xr.Dataset, qa_threshold: float) -> xr.Dataset:
        """Apply QA value filtering to dataset."""
        qa_var = None
        for name in ["qa_value", "quality_flag", "main_data_quality_flag"]:
            if name in ds.data_vars:
                qa_var = name
                break

        if qa_var is not None:
            mask = ds[qa_var] >= qa_threshold
            for var in ds.data_vars:
                if var != qa_var:
                    ds[var] = ds[var].where(mask)

        return ds

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize TEMPO dataset dimensions and coordinates."""
        coord_renames: dict[str, str] = {}

        if "latitude" in ds.coords and "lat" not in ds.coords:
            coord_renames["latitude"] = "lat"
        if "longitude" in ds.coords and "lon" not in ds.coords:
            coord_renames["longitude"] = "lon"

        if coord_renames:
            ds = ds.rename(coord_renames)

        return set_geometry_attr(ds, DataGeometry.SWATH)
