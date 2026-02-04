"""Base model class for atmospheric model output.

This module provides the BaseModel class that wraps model output data
with common operations for loading, processing, and transforming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Mapping, Sequence

import xarray as xr

from davinci_monet.core.base import DataContainer
from davinci_monet.core.exceptions import (
    DataFormatError,
    DataNotFoundError,
    DataValidationError,
    write_error_log,
)
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import PathLike, VariableMapping


@dataclass
class ModelData(DataContainer):
    """Container for atmospheric model output data.

    Wraps model output (CMAQ, WRF-Chem, UFS, etc.) with metadata
    and common processing operations.

    Attributes
    ----------
    data : xr.Dataset | None
        The model output dataset.
    label : str
        Model run identifier.
    mod_type : str
        Model type (e.g., 'cmaq', 'wrfchem', 'ufs').
    files : list[Path]
        List of model output files.
    file_pattern : str | None
        Glob pattern used to find files.
    radius_of_influence : float
        Default spatial search radius for pairing (meters).
    obs_mapping : dict[str, VariableMapping]
        Mapping of observation labels to variable mappings.
    is_global : bool
        Whether this is a global model.
    mod_kwargs : dict[str, Any]
        Additional keyword arguments for the model reader.
    """

    mod_type: str = ""
    files: list[Path] = field(default_factory=list)
    file_pattern: str | None = None
    radius_of_influence: float = 12000.0
    obs_mapping: dict[str, VariableMapping] = field(default_factory=dict)
    is_global: bool = False
    mod_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def geometry(self) -> DataGeometry:
        """Model data is always gridded."""
        return DataGeometry.GRID

    def _copy_with_data(self, data: xr.Dataset) -> ModelData:
        """Create a copy with new data."""
        return ModelData(
            data=data,
            label=self.label,
            variables=self.variables.copy(),
            variable_mapping=dict(self.variable_mapping),
            mod_type=self.mod_type,
            files=self.files.copy(),
            file_pattern=self.file_pattern,
            radius_of_influence=self.radius_of_influence,
            obs_mapping={k: dict(v) for k, v in self.obs_mapping.items()},
            is_global=self.is_global,
            mod_kwargs=self.mod_kwargs.copy(),
        )

    def resolve_files(self, pattern: str | None = None) -> list[Path]:
        """Resolve file pattern to list of files.

        Parameters
        ----------
        pattern
            Glob pattern for model files. If None, uses self.file_pattern.

        Returns
        -------
        list[Path]
            Sorted list of matching file paths.

        Raises
        ------
        DataNotFoundError
            If no files match the pattern.
        """
        import numpy as np

        if pattern is None:
            pattern = self.file_pattern

        if pattern is None:
            return self.files

        # Check if pattern is a text file containing file list
        if pattern.endswith(".txt") and Path(pattern).exists():
            try:
                with open(pattern) as f:
                    file_list = [Path(line.strip()) for line in f if line.strip()]
            except OSError as e:
                error_file = write_error_log(e, f"Reading file list from '{pattern}'")
                msg = f"Failed to read file list from '{pattern}': {e}"
                if error_file:
                    msg += f" (details: {error_file})"
                raise DataFormatError(msg) from e
            self.files = file_list
            return file_list

        # Glob for files
        matched = glob(pattern)
        if not matched:
            raise DataNotFoundError(f"No files match pattern: {pattern}")

        # Sort files
        sorted_files = sorted(matched)
        self.files = [Path(f) for f in sorted_files]
        return self.files

    def get_variable_list_for_obs(
        self,
        obs_label: str,
    ) -> list[str]:
        """Get list of model variables needed for an observation source.

        Parameters
        ----------
        obs_label
            Observation source label.

        Returns
        -------
        list[str]
            List of model variable names.
        """
        if obs_label not in self.obs_mapping:
            return []
        return list(self.obs_mapping[obs_label].values())

    def get_mapping_for_obs(self, obs_label: str) -> VariableMapping:
        """Get variable mapping for an observation source.

        Parameters
        ----------
        obs_label
            Observation source label.

        Returns
        -------
        VariableMapping
            Mapping from observation variable names to model variable names.
        """
        return self.obs_mapping.get(obs_label, {})

    def apply_variable_config(self) -> None:
        """Apply variable configuration (scaling, masking, renaming).

        Processes all variables in self.variables dictionary.
        """
        if self.data is None:
            return

        # Use list() to avoid modifying dict during iteration
        for var_name, config in list(self.variables.items()):
            if var_name not in self.data:
                continue

            # Apply unit scaling
            if "unit_scale" in config:
                scale = config["unit_scale"]
                method = config.get("unit_scale_method", "*")
                self.apply_unit_scale(var_name, scale, method)

            # Apply masking
            min_val = config.get("obs_min")
            max_val = config.get("obs_max")
            nan_val = config.get("nan_value")
            if any(v is not None for v in [min_val, max_val, nan_val]):
                self.apply_mask(var_name, min_val, max_val, nan_val)

            # Apply renaming (do this last)
            new_name = config.get("rename")
            if new_name:
                self.rename_variable(var_name, new_name)
                # Update the config dict key
                self.variables[new_name] = self.variables.pop(var_name)

    def sum_variables(
        self,
        new_var: str,
        source_vars: Sequence[str],
        config: dict[str, Any] | None = None,
    ) -> None:
        """Create a new variable by summing existing variables.

        Parameters
        ----------
        new_var
            Name for the new summed variable.
        source_vars
            List of variable names to sum.
        config
            Optional configuration for the new variable.
        """
        if self.data is None:
            return

        # Check source variables exist
        missing = [v for v in source_vars if v not in self.data]
        if missing:
            raise DataValidationError(
                f"Cannot sum variables, missing: {missing}"
            )

        # Check new variable doesn't already exist
        if new_var in self.data:
            raise DataValidationError(
                f"Variable '{new_var}' already exists, cannot create with summing"
            )

        # Sum the variables
        result = self.data[source_vars[0]].copy()
        for var in source_vars[1:]:
            result = result + self.data[var]

        self.data[new_var] = result

        # Store config if provided
        if config is not None:
            self.variables[new_var] = config

    def extract_surface(self, level_dim: str = "z") -> ModelData:
        """Extract surface level from 3D model data.

        Parameters
        ----------
        level_dim
            Name of the vertical dimension.

        Returns
        -------
        ModelData
            New ModelData with only surface level.
        """
        if self.data is None:
            return self

        if level_dim not in self.data.dims:
            return self

        # Select lowest level (index 0)
        surface = self.data.isel({level_dim: 0})
        return self._copy_with_data(surface)

    def extract_level(
        self,
        level: int | float,
        level_dim: str = "z",
        method: str = "nearest",
    ) -> ModelData:
        """Extract a specific vertical level.

        Parameters
        ----------
        level
            Level value or index.
        level_dim
            Name of the vertical dimension.
        method
            Selection method ('nearest', 'exact', 'index').

        Returns
        -------
        ModelData
            New ModelData with single level.
        """
        if self.data is None:
            return self

        if level_dim not in self.data.dims:
            return self

        if method == "index":
            selected = self.data.isel({level_dim: int(level)})
        elif method == "nearest":
            selected = self.data.sel({level_dim: level}, method="nearest")
        else:
            selected = self.data.sel({level_dim: level})

        return self._copy_with_data(selected)

    def interpolate_vertical(
        self,
        target_levels: Sequence[float],
        level_coord: str = "z",
        method: str = "linear",
    ) -> ModelData:
        """Interpolate to new vertical levels.

        Parameters
        ----------
        target_levels
            Target level values.
        level_coord
            Name of the vertical coordinate.
        method
            Interpolation method ('linear', 'nearest').

        Returns
        -------
        ModelData
            New ModelData with interpolated levels.
        """
        if self.data is None:
            return self

        if level_coord not in self.data.dims:
            return self

        interpolated = self.data.interp(
            {level_coord: list(target_levels)},
            method=method,  # type: ignore[arg-type]
        )
        return self._copy_with_data(interpolated)

    def regrid_horizontal(
        self,
        target_lats: Sequence[float],
        target_lons: Sequence[float],
        method: str = "nearest",
    ) -> ModelData:
        """Regrid to new horizontal grid.

        Parameters
        ----------
        target_lats
            Target latitude values.
        target_lons
            Target longitude values.
        method
            Interpolation method ('nearest', 'linear').

        Returns
        -------
        ModelData
            New ModelData with regridded data.
        """
        if self.data is None:
            return self

        # Determine lat/lon dimension names
        lat_dim = None
        lon_dim = None
        for name in ["lat", "latitude", "y"]:
            if name in self.data.dims:
                lat_dim = name
                break
        for name in ["lon", "longitude", "x"]:
            if name in self.data.dims:
                lon_dim = name
                break

        if lat_dim is None or lon_dim is None:
            return self

        regridded = self.data.interp(
            {lat_dim: list(target_lats), lon_dim: list(target_lons)},
            method=method,  # type: ignore[arg-type]
        )
        return self._copy_with_data(regridded)

    def compute_derived_variable(
        self,
        name: str,
        expression: str,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """Compute a derived variable using an expression.

        Parameters
        ----------
        name
            Name for the new variable.
        expression
            Python expression using existing variable names.
            Variables are accessed as ds['varname'].
        attrs
            Optional attributes for the new variable.
        """
        if self.data is None:
            return

        # Create local namespace with dataset variables
        local_vars: dict[str, Any] = {"ds": self.data}
        for var in self.data.data_vars:
            local_vars[str(var)] = self.data[str(var)]

        # Evaluate expression
        result = eval(expression, {"__builtins__": {}}, local_vars)

        if attrs is not None:
            result.attrs.update(attrs)

        self.data[name] = result


def create_model_data(
    label: str,
    mod_type: str = "",
    data: xr.Dataset | None = None,
    files: str | Path | Sequence[str | Path] | None = None,
    radius_of_influence: float = 12000.0,
    mapping: dict[str, VariableMapping] | None = None,
    variables: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ModelData:
    """Factory function to create ModelData instance.

    Parameters
    ----------
    label
        Model run identifier.
    mod_type
        Model type (e.g., 'cmaq', 'wrfchem').
    data
        Pre-loaded xarray Dataset.
    files
        File path(s) or glob pattern.
    radius_of_influence
        Spatial search radius in meters.
    mapping
        Observation-to-model variable mappings.
    variables
        Variable configuration.
    **kwargs
        Additional model-specific options.

    Returns
    -------
    ModelData
        Configured ModelData instance.
    """
    model = ModelData(
        data=data,
        label=label,
        mod_type=mod_type,
        radius_of_influence=radius_of_influence,
        obs_mapping=mapping or {},
        variables=variables or {},
        mod_kwargs=kwargs,
    )

    if files is not None:
        if isinstance(files, (str, Path)):
            file_str = str(files)
            if "*" in file_str or "?" in file_str:
                model.file_pattern = file_str
                model.resolve_files()
            else:
                model.files = [Path(files)]
        else:
            model.files = [Path(f) for f in files]

    return model
