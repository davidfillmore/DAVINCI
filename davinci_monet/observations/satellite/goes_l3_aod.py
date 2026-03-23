"""GOES L3 AOD (Aerosol Optical Depth) observation reader.

This module provides the GOESL3AODReader class for reading GOES-ABI L3 AOD
(Aerosol Optical Depth) gridded products.

Note
----
This reader is optimized for GOES-ABI AOD products and relies on monetio.sat.goes
for full functionality. Without monetio, it falls back to basic xarray reading
which may not handle all GOES-specific features (projection, etc.).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import xarray as xr

from davinci_monet.core.exceptions import DataNotFoundError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import observation_registry
from davinci_monet.observations.base import ObservationData, create_observation_data

# Standard variable name mappings for GOES AOD
GOES_AOD_VARIABLE_MAPPING: dict[str, str] = {
    "aod": "AOD",
    "aod_550": "AOD",
    "dqf": "DQF",
    "quality_flag": "DQF",
}

# Backward compatibility alias
GOES_VARIABLE_MAPPING = GOES_AOD_VARIABLE_MAPPING


@observation_registry.register("goes_l3_aod")
class GOESL3AODReader:
    """Reader for GOES-ABI L3 AOD satellite observations.

    Reads GOES L3 AOD data from NetCDF files. This reader is specifically
    designed for GOES-ABI Aerosol Optical Depth products.

    Data is returned as grid geometry with (time, y, x) dimensions.

    Note
    ----
    Full functionality requires monetio. Without monetio, the reader falls
    back to basic xarray which may not handle GOES-specific features like
    fixed grid projections correctly.

    Examples
    --------
    >>> reader = GOESL3AODReader()
    >>> ds = reader.open(["OR_ABI-L2-AODC*.nc"])
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "goes_l3_aod"

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        *,
        product: str = "AOD",
        dqf_filter: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open GOES L3 AOD observation files.

        Parameters
        ----------
        file_paths
            Paths to GOES-ABI L3 AOD files.
        variables
            Variables to load.
        product
            Product type (default 'AOD').
        dqf_filter
            Data Quality Flag values to keep. If None, keeps all.
            Common values: 0=high quality, 1=medium, 2=low.
        **kwargs
            Additional options.

        Returns
        -------
        xr.Dataset
            GOES observations with grid dimensions.
        """
        file_list = [Path(f) for f in file_paths]

        if not file_list:
            raise DataNotFoundError("No GOES files provided")

        missing = [f for f in file_list if not f.exists()]
        if missing:
            raise DataNotFoundError(f"GOES files not found: {missing}")

        # Try monetio first
        try:
            ds = self._open_with_monetio(file_list, variables, **kwargs)
        except ImportError:
            warnings.warn(
                "monetio not available, using basic xarray reader. "
                "GOES projection handling may be incomplete.",
                UserWarning,
            )
            ds = self._open_with_xarray(file_list, variables, **kwargs)

        # Apply DQF filtering
        if dqf_filter is not None:
            ds = self._apply_dqf_filter(ds, dqf_filter)

        return self._standardize_dataset(ds)

    def _open_with_monetio(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open GOES files using monetio."""
        import monetio.sat.goes as goes_mod

        files = [str(f) for f in file_paths]

        # monetio.sat.goes expects a single file or list
        if len(files) == 1:
            ds: xr.Dataset = goes_mod.open_dataset(files[0], **kwargs)
        else:
            ds_list = []
            for f in files:
                try:
                    ds_i: xr.Dataset = goes_mod.open_dataset(f, **kwargs)
                    ds_list.append(ds_i)
                except Exception as e:
                    warnings.warn(f"Failed to open {f}: {e}", UserWarning)
                    continue

            if not ds_list:
                raise DataNotFoundError("No valid GOES data found")

            ds = xr.concat(ds_list, dim="time")

        if variables is not None:
            available = [v for v in variables if v in ds.data_vars]
            if available:
                ds = ds[available]

        return ds

    def _open_with_xarray(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open GOES files using xarray."""
        ds_list = []
        for fpath in file_paths:
            try:
                ds = xr.open_dataset(str(fpath), **kwargs)
                ds_list.append(ds)
            except Exception as e:
                warnings.warn(f"Failed to open {fpath}: {e}", UserWarning)
                continue

        if not ds_list:
            raise DataNotFoundError("No valid GOES data found")

        if len(ds_list) > 1:
            ds = xr.concat(ds_list, dim="time")
        else:
            ds = ds_list[0]

        if variables is not None:
            available = [v for v in variables if v in ds.data_vars]
            if available:
                ds = ds[available]

        return ds

    def _apply_dqf_filter(self, ds: xr.Dataset, dqf_filter: Sequence[int]) -> xr.Dataset:
        """Apply Data Quality Flag filtering to dataset."""
        dqf_var = None
        for name in ["DQF", "dqf", "data_quality_flag"]:
            if name in ds.data_vars:
                dqf_var = name
                break

        if dqf_var is not None:
            # Create mask for valid DQF values
            mask = ds[dqf_var].isin(list(dqf_filter))
            # Apply mask to all data variables
            for var in ds.data_vars:
                if var != dqf_var:
                    ds[var] = ds[var].where(mask)

        return ds

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize GOES dataset dimensions and coordinates."""
        coord_renames: dict[str, str] = {}

        # GOES data often has lat/lon as 2D arrays
        if "latitude" in ds.coords and "lat" not in ds.coords:
            coord_renames["latitude"] = "lat"
        if "longitude" in ds.coords and "lon" not in ds.coords:
            coord_renames["longitude"] = "lon"

        if coord_renames:
            ds = ds.rename(coord_renames)

        ds.attrs["geometry"] = DataGeometry.GRID.value

        return ds

    def get_variable_mapping(self) -> Mapping[str, str]:
        """Return GOES AOD variable name mapping."""
        return GOES_AOD_VARIABLE_MAPPING


# Backward compatibility alias
GOESReader = GOESL3AODReader


def open_goes_l3_aod(
    files: str | Path | Sequence[str | Path],
    variables: Sequence[str] | None = None,
    label: str = "goes_aod",
    product: str = "AOD",
    dqf_filter: Sequence[int] | None = None,
    **kwargs: Any,
) -> ObservationData:
    """Open GOES L3 AOD observation data.

    Parameters
    ----------
    files
        File path(s) or glob pattern.
    variables
        Variables to load.
    label
        Observation label.
    product
        Product type (default 'AOD').
    dqf_filter
        Data Quality Flag values to keep.
    **kwargs
        Additional reader options.

    Returns
    -------
    ObservationData
        GOES observation data container with GRID geometry.

    Note
    ----
    Full functionality requires monetio. Without monetio, GOES projection
    handling may be incomplete.
    """
    from glob import glob

    reader = GOESL3AODReader()

    if isinstance(files, (str, Path)):
        file_str = str(files)
        if "*" in file_str or "?" in file_str:
            file_list = sorted(glob(file_str))
            if not file_list:
                raise DataNotFoundError(f"No files match pattern: {files}")
            file_paths: Sequence[str | Path] = file_list
        else:
            file_paths = [files]
    else:
        file_paths = list(files)

    ds = reader.open(file_paths, variables, product=product, dqf_filter=dqf_filter, **kwargs)

    obs = create_observation_data(
        label=label,
        obs_type="gridded",
        data=ds,
        variables=dict.fromkeys(variables) if variables else {},
    )
    obs.geometry = DataGeometry.GRID

    return obs


# Backward compatibility alias
def open_goes(
    files: str | Path | Sequence[str | Path],
    variables: Sequence[str] | None = None,
    label: str = "goes",
    product: str = "AOD",
    dqf_filter: Sequence[int] | None = None,
    **kwargs: Any,
) -> ObservationData:
    """Open GOES observation data (deprecated, use open_goes_l3_aod).

    .. deprecated::
        Use :func:`open_goes_l3_aod` instead.
    """
    warnings.warn(
        "open_goes is deprecated, use open_goes_l3_aod instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return open_goes_l3_aod(
        files, variables, label=label, product=product, dqf_filter=dqf_filter, **kwargs
    )
