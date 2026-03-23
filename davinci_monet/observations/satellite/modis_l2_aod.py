"""MODIS L2 AOD observation reader.

This module provides the MODISL2AODReader class for reading MODIS
(Moderate Resolution Imaging Spectroradiometer) L2 AOD products.

MODIS is on NASA's Terra and Aqua satellites, providing global
aerosol optical depth measurements since 2000.

Note
----
This reader requires monetio.sat._modis_l2_mm for full functionality.
Without monetio, it falls back to basic xarray reading which may not handle
all MODIS-specific features (HDF-EOS, swath geometry).
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

# Standard variable name mappings for MODIS AOD
MODIS_AOD_VARIABLE_MAPPING: dict[str, str] = {
    "aod": "Optical_Depth_Land_And_Ocean",
    "aod_550": "Optical_Depth_Land_And_Ocean",
    "aod_land": "Corrected_Optical_Depth_Land",
    "aod_ocean": "Effective_Optical_Depth_Best_Ocean",
    "angstrom": "Deep_Blue_Angstrom_Exponent_Land",
    "qa": "Land_Ocean_Quality_Flag",
}


@observation_registry.register("modis_l2_aod")
class MODISL2AODReader:
    """Reader for MODIS L2 AOD satellite observations.

    Reads MODIS L2 AOD (Aerosol Optical Depth) data from HDF-EOS files.
    Supports both Terra (MOD04) and Aqua (MYD04) products.

    Data is returned as swath geometry with appropriate dimensions.

    Note
    ----
    Full functionality requires monetio. Without monetio, the reader falls
    back to basic xarray which may not handle MODIS HDF-EOS swath geometry
    correctly.

    Examples
    --------
    >>> reader = MODISL2AODReader()
    >>> ds = reader.open(["MOD04_L2.A*.hdf"])
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "modis_l2_aod"

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        *,
        qa_threshold: int | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open MODIS L2 AOD observation files.

        Parameters
        ----------
        file_paths
            Paths to MODIS L2 AOD files (MOD04 or MYD04).
        variables
            Variables to load. If None, loads AOD at 550nm.
        qa_threshold
            Quality assurance threshold. Pixels with QA below threshold
            are masked. Common values: 3=best, 2=good, 1=marginal.
            Set to None to disable filtering.
        **kwargs
            Additional options.

        Returns
        -------
        xr.Dataset
            MODIS observations with swath dimensions.
        """
        file_list = [Path(f) for f in file_paths]

        if not file_list:
            raise DataNotFoundError("No MODIS files provided")

        missing = [f for f in file_list if not f.exists()]
        if missing:
            raise DataNotFoundError(f"MODIS files not found: {missing}")

        # Try monetio first
        try:
            ds = self._open_with_monetio(file_list, variables, **kwargs)
        except ImportError:
            warnings.warn(
                "monetio not available, using basic xarray reader. "
                "MODIS HDF-EOS swath handling may be incomplete.",
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
        """Open MODIS files using monetio."""
        import monetio.sat._modis_l2_mm as modis_mod

        files = [str(f) for f in file_paths]

        if len(files) == 1:
            ds: xr.Dataset = modis_mod.read_dataset(files[0], **kwargs)
        else:
            ds = modis_mod.read_mfdataset(files, **kwargs)

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
        """Open MODIS files using xarray."""
        ds_list = []
        for fpath in file_paths:
            try:
                # MODIS HDF-EOS files need special handling
                ds = xr.open_dataset(str(fpath), engine="netcdf4", **kwargs)
                ds_list.append(ds)
            except Exception as e:
                warnings.warn(f"Failed to open {fpath}: {e}", UserWarning)
                continue

        if not ds_list:
            raise DataNotFoundError("No valid MODIS data found")

        if len(ds_list) > 1:
            # Try to concatenate - may fail for swath data
            try:
                ds = xr.concat(ds_list, dim="time")
            except Exception:
                ds = ds_list[0]
                warnings.warn(
                    "Could not concatenate MODIS files, using first file only",
                    UserWarning,
                )
        else:
            ds = ds_list[0]

        if variables is not None:
            available = [v for v in variables if v in ds.data_vars]
            if available:
                ds = ds[available]

        return ds

    def _apply_qa_filter(self, ds: xr.Dataset, qa_threshold: int) -> xr.Dataset:
        """Apply quality assurance filtering to dataset."""
        qa_var = None
        for name in ["Land_Ocean_Quality_Flag", "qa", "Quality_Assurance_Land"]:
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
        """Standardize MODIS dataset dimensions and coordinates."""
        coord_renames: dict[str, str] = {}

        if "Latitude" in ds.coords and "lat" not in ds.coords:
            coord_renames["Latitude"] = "lat"
        elif "latitude" in ds.coords and "lat" not in ds.coords:
            coord_renames["latitude"] = "lat"

        if "Longitude" in ds.coords and "lon" not in ds.coords:
            coord_renames["Longitude"] = "lon"
        elif "longitude" in ds.coords and "lon" not in ds.coords:
            coord_renames["longitude"] = "lon"

        if coord_renames:
            ds = ds.rename(coord_renames)

        ds.attrs["geometry"] = DataGeometry.SWATH.value

        return ds

    def get_variable_mapping(self) -> Mapping[str, str]:
        """Return MODIS AOD variable name mapping."""
        return MODIS_AOD_VARIABLE_MAPPING


def open_modis_l2_aod(
    files: str | Path | Sequence[str | Path],
    variables: Sequence[str] | None = None,
    label: str = "modis_aod",
    qa_threshold: int | None = None,
    **kwargs: Any,
) -> ObservationData:
    """Open MODIS L2 AOD observation data.

    Parameters
    ----------
    files
        File path(s) or glob pattern.
    variables
        Variables to load.
    label
        Observation label.
    qa_threshold
        Quality assurance threshold (3=best, 2=good, 1=marginal).
    **kwargs
        Additional reader options.

    Returns
    -------
    ObservationData
        MODIS observation data container with SWATH geometry.

    Note
    ----
    Full functionality requires monetio. Without monetio, MODIS HDF-EOS
    swath handling may be incomplete.
    """
    from glob import glob

    reader = MODISL2AODReader()

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

    ds = reader.open(file_paths, variables, qa_threshold=qa_threshold, **kwargs)

    obs = create_observation_data(
        label=label,
        obs_type="satellite",
        data=ds,
        variables=dict.fromkeys(variables) if variables else {},
    )
    obs.geometry = DataGeometry.SWATH

    return obs
