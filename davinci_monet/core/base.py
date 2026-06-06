"""Base data classes for DAVINCI.

This module provides the paired-data container and the paired-variable / plot
series helpers that wrap xarray Datasets with role metadata.

Key Classes:
    - PairedData: Container for paired source data
    - PlotSeries: One plottable series with role metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import xarray as xr

from davinci_monet.core.exceptions import DataValidationError, VariableNotFoundError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import TimeRange


def paired_variable_role(dataset: xr.Dataset, var_name: str) -> str | None:
    """Source role of a paired variable (renderer rewire R-5).

    Returns the variable's ``role`` attr (``"obs"``/``"model"``) when present,
    otherwise infers it from the legacy ``obs_``/``model_`` prefix. Returns
    ``None`` for variables that are neither (coordinates, unrelated vars).
    """
    role = dataset[var_name].attrs.get("role") if var_name in dataset.data_vars else None
    if role is None:
        lname = str(var_name).lower()
        if lname.startswith("obs_"):
            role = "obs"
        elif lname.startswith("model_"):
            role = "model"
    return role


def paired_variable_pair_role(dataset: xr.Dataset, var_name: str) -> str | None:
    """Reference/comparand role of a paired variable.

    ``pair_role`` is the role-neutral pairing position introduced for unified
    source pairs. Legacy paired data falls back to source role/prefix:
    ``obs`` -> reference and ``model`` -> comparand.
    """
    if var_name in dataset.data_vars:
        pair_role = dataset[var_name].attrs.get("pair_role")
        if pair_role in ("reference", "comparand"):
            return str(pair_role)
    role = paired_variable_role(dataset, var_name)
    if role == "obs":
        return "reference"
    if role == "model":
        return "comparand"
    return None


def paired_canonical_name(dataset: xr.Dataset, var_name: str) -> str:
    """Canonical (unprefixed) name of a paired variable (renderer rewire R-5).

    Strips the source-label prefix (from the variable's ``source_label`` attr,
    e.g. ``cam_o3`` -> ``o3``) or the legacy ``obs_``/``model_`` prefix. Names
    with no recognised prefix are returned unchanged.
    """
    if var_name in dataset.data_vars:
        source_label = dataset[var_name].attrs.get("source_label")
        if source_label and var_name.startswith(f"{source_label}_"):
            return var_name[len(source_label) + 1 :]
    # Case-insensitive prefix match, consistent with ``paired_variable_role``.
    lname = str(var_name).lower()
    for prefix in ("obs_", "model_"):
        if lname.startswith(prefix):
            return var_name[len(prefix) :]
    return var_name


def iter_paired_variable_pairs(dataset: xr.Dataset) -> list[tuple[str, str, str]]:
    """Pair reference variables with their comparand counterparts.
    counterparts by canonical name (renderer rewire R-5).

    Returns ``(reference_var, comparand_var, canonical)`` triples. Pair roles
    come from ``pair_role`` with a fallback to the legacy ``obs``/``model``
    source role and prefixes, so this works for source-label-named paired output
    as well as legacy or untagged paired data. One variable per position is used, so
    dual-named data never double-counts.
    """
    refs: dict[str, str] = {}
    comps: dict[str, str] = {}
    for v in dataset.data_vars:
        name = str(v)
        role = paired_variable_pair_role(dataset, name)
        if role not in ("reference", "comparand"):
            continue
        canonical = paired_canonical_name(dataset, name)
        (refs if role == "reference" else comps).setdefault(canonical, name)
    return [(refs[c], comps[c], c) for c in comps if c in refs]


@dataclass
class PlotSeries:
    """One plottable series — a variable from one data source, with role metadata.

    The value object the unified renderer contract (``render(series)``) consumes.
    ``index`` is the series' 0-based position within its canonical group and is
    the styling hook for palette cycling (``get_color_for_role(role, index)``).

    Attributes
    ----------
    dataset
        The dataset the variable lives in.
    var_name
        The actual variable name in ``dataset`` (e.g. ``cam_o3``/``airnow_o3``/``o3``).
    canonical
        The unprefixed canonical name (e.g. ``o3``).
    role
        Source role (``"obs"``/``"model"``/custom) or ``None`` when untagged.
    pair_role
        Pairing position (``"reference"``/``"comparand"``) or ``None`` when unpaired.
    source_label
        The source's identity in a unified pair (e.g. ``airnow``/``cam``) or ``None``.
    index
        Position within the canonical group (0-based).
    """

    dataset: xr.Dataset
    var_name: str
    canonical: str
    role: str | None
    pair_role: str | None
    source_label: str | None
    index: int


def iter_canonical_variable_series(dataset: xr.Dataset) -> dict[str, list[PlotSeries]]:
    """Group a dataset's source variables by canonical name into :class:`PlotSeries`.

    N-capable sibling of :func:`iter_paired_variable_pairs`: where that returns a
    single ``(reference, comparand)`` pair per canonical, this returns *every*
    source variable for each canonical as an ordered list (1 → single series,
    2 → reference + comparand, N → multi-source overlay). Only variables carrying
    a recognised ``role`` or ``pair_role`` (attr, or the legacy ``obs_``/``model_``
    prefix) are included; bookkeeping vars without a role (e.g. obs counts) are
    skipped. Series preserve ``data_vars`` order; ``index`` is the 0-based position
    within the canonical group.
    """
    groups: dict[str, list[PlotSeries]] = {}
    for v in dataset.data_vars:
        name = str(v)
        role = paired_variable_role(dataset, name)
        pair_role = paired_variable_pair_role(dataset, name)
        if role is None and pair_role is None:
            continue
        canonical = paired_canonical_name(dataset, name)
        source_label = dataset[name].attrs.get("source_label")
        group = groups.setdefault(canonical, [])
        group.append(
            PlotSeries(
                dataset=dataset,
                var_name=name,
                canonical=canonical,
                role=role,
                pair_role=pair_role,
                source_label=str(source_label) if source_label else None,
                index=len(group),
            )
        )
    return groups


@dataclass
class PairedData:
    """Container for paired source data.

    Canonically, paired data has a reference source and a comparand source.
    The historical ``obs_label`` and ``model_label`` fields are retained as
    compatibility aliases for reference and comparand, respectively.

    Attributes
    ----------
    data : xr.Dataset
        The paired dataset with reference and comparand variables.
    model_label : str
        Compatibility alias for the comparand source label.
    obs_label : str
        Compatibility alias for the reference source label.
    geometry : DataGeometry
        The geometry type of the reference data.
    pairing_info : dict[str, Any]
        Metadata about the pairing process.
    """

    data: xr.Dataset
    model_label: str
    obs_label: str
    geometry: DataGeometry
    pairing_info: dict[str, Any] = field(default_factory=dict)

    @property
    def pair_label(self) -> str:
        """Get combined pair label (reference_comparand format)."""
        return f"{self.reference_label}_{self.comparand_label}"

    @property
    def reference_label(self) -> str:
        """Label of the reference source."""
        return str(self.pairing_info.get("reference_label") or self.obs_label)

    @property
    def comparand_label(self) -> str:
        """Label of the comparand source."""
        return str(self.pairing_info.get("comparand_label") or self.model_label)

    @property
    def reference_variables(self) -> list[str]:
        """List of reference-role variables in the paired data."""
        return [
            str(v)
            for v in self.data.data_vars
            if paired_variable_pair_role(self.data, str(v)) == "reference"
        ]

    @property
    def comparand_variables(self) -> list[str]:
        """List of comparand-role variables in the paired data."""
        return [
            str(v)
            for v in self.data.data_vars
            if paired_variable_pair_role(self.data, str(v)) == "comparand"
        ]

    @property
    def model_variables(self) -> list[str]:
        """Compatibility alias for comparand-role variables."""
        return self.comparand_variables

    @property
    def obs_variables(self) -> list[str]:
        """Compatibility alias for reference-role variables."""
        return self.reference_variables

    @property
    def paired_variable_names(self) -> list[tuple[str, str]]:
        """List of (obs_var, model_var) pairs, matched by canonical name."""
        return [
            (obs_var, model_var) for obs_var, model_var, _ in iter_paired_variable_pairs(self.data)
        ]

    def _resolve_role_var(self, variable: str, role: str) -> str | None:
        """Resolve a paired variable for ``role``.

        ``role`` accepts canonical pair roles (``reference``/``comparand``) and
        compatibility names (``obs``/``model``). Variable lookup accepts exact,
        legacy-prefixed, bare canonical, or source-label names.
        """
        wanted = "reference" if role in {"obs", "reference"} else "comparand"
        if (
            variable in self.data.data_vars
            and paired_variable_pair_role(self.data, variable) == wanted
        ):
            return variable
        prefix = "obs_" if wanted == "reference" else "model_"
        legacy = variable if variable.startswith(prefix) else f"{prefix}{variable}"
        if legacy in self.data.data_vars and paired_variable_pair_role(self.data, legacy) == wanted:
            return legacy
        target = paired_canonical_name(self.data, variable)
        for v in self.data.data_vars:
            name = str(v)
            if (
                paired_variable_pair_role(self.data, name) == wanted
                and paired_canonical_name(self.data, name) == target
            ):
                return name
        return None

    def get_reference(self, variable: str) -> xr.DataArray:
        """Get a reference-role variable."""
        name = self._resolve_role_var(variable, "reference")
        if name is None:
            raise VariableNotFoundError(
                f"Reference variable '{variable}' not found. "
                f"Available: {self.reference_variables}"
            )
        result: xr.DataArray = self.data[name]
        return result

    def get_comparand(self, variable: str) -> xr.DataArray:
        """Get a comparand-role variable."""
        name = self._resolve_role_var(variable, "comparand")
        if name is None:
            raise VariableNotFoundError(
                f"Comparand variable '{variable}' not found. "
                f"Available: {self.comparand_variables}"
            )
        result: xr.DataArray = self.data[name]
        return result

    def get_obs(self, variable: str) -> xr.DataArray:
        """Get reference variable through the legacy observation accessor.

        Parameters
        ----------
        variable
            Variable name: exact, legacy ``obs_``-prefixed, bare canonical, or
            source-label form.

        Returns
        -------
        xr.DataArray
            Reference data.
        """
        return self.get_reference(variable)

    def get_model(self, variable: str) -> xr.DataArray:
        """Get comparand variable through the legacy model accessor.

        Parameters
        ----------
        variable
            Variable name: exact, legacy ``model_``-prefixed, bare canonical, or
            source-label form.

        Returns
        -------
        xr.DataArray
            Comparand data.
        """
        return self.get_comparand(variable)

    def get_pair(self, variable: str) -> tuple[xr.DataArray, xr.DataArray]:
        """Get both reference and comparand arrays for a variable.

        Parameters
        ----------
        variable
            Base variable name (without prefix).

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            (reference_array, comparand_array) tuple.
        """
        reference = self.get_reference(variable)
        comparand = self.get_comparand(variable)
        return (reference, comparand)

    @property
    def time_range(self) -> TimeRange | None:
        """Get the time range of paired data."""
        if "time" not in self.data.dims:
            return None
        times = self.data["time"].values
        if len(times) == 0:
            return None
        return (times[0], times[-1])

    @property
    def n_points(self) -> int:
        """Get total number of paired data points."""
        # Count non-NaN pairs for first paired variable
        if not self.paired_variable_names:
            return 0
        obs_var, model_var = self.paired_variable_names[0]
        obs_data = self.data[obs_var]
        model_data = self.data[model_var]
        valid = ~obs_data.isnull() & ~model_data.isnull()
        return int(valid.sum().values)

    def to_dataframe(self) -> Any:
        """Convert paired data to pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with time/location index and paired variables.
        """
        import pandas as pd

        df: pd.DataFrame = self.data.to_dataframe().reset_index()
        return df

    def subset_time(
        self,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> PairedData:
        """Subset paired data to a time range.

        Parameters
        ----------
        start
            Start time (inclusive).
        end
            End time (inclusive).

        Returns
        -------
        PairedData
            New PairedData with subsetted data.
        """
        if "time" not in self.data.dims:
            return self

        subset = self.data
        if start is not None:
            subset = subset.sel(time=slice(start, None))
        if end is not None:
            subset = subset.sel(time=slice(None, end))

        return PairedData(
            data=subset,
            model_label=self.model_label,
            obs_label=self.obs_label,
            geometry=self.geometry,
            pairing_info=self.pairing_info,
        )


def validate_dataset_geometry(
    data: xr.Dataset,
    expected_geometry: DataGeometry,
) -> None:
    """Validate that a dataset has the expected geometry.

    Parameters
    ----------
    data
        Dataset to validate.
    expected_geometry
        Expected geometry type.

    Raises
    ------
    DataValidationError
        If dataset doesn't match expected geometry.
    """
    # Check geometry attribute if present
    if "geometry" in data.attrs:
        actual = data.attrs["geometry"]
        if isinstance(actual, DataGeometry):
            if actual != expected_geometry:
                raise DataValidationError(
                    f"Expected geometry {expected_geometry.name}, got {actual.name}"
                )
            return
        if isinstance(actual, str):
            try:
                actual_geom = DataGeometry[actual.upper()]
                if actual_geom != expected_geometry:
                    raise DataValidationError(
                        f"Expected geometry {expected_geometry.name}, got {actual}"
                    )
                return
            except KeyError:
                pass

    # Validate based on expected dimensions
    dims = set(data.dims)

    if expected_geometry == DataGeometry.POINT:
        # Point: (time, site) or (time, x)
        if not (("time" in dims and ("site" in dims or "x" in dims)) or "time" in dims):
            raise DataValidationError(f"POINT geometry expects dims (time, site), got {dims}")

    elif expected_geometry == DataGeometry.TRACK:
        # Track: (time,) with lat/lon/alt coords
        if "time" not in dims:
            raise DataValidationError(f"TRACK geometry expects 'time' dimension, got {dims}")

    elif expected_geometry == DataGeometry.PROFILE:
        # Profile: (time, level) with lat/lon coords
        if not ("time" in dims and ("level" in dims or "z" in dims)):
            raise DataValidationError(f"PROFILE geometry expects dims (time, level), got {dims}")

    elif expected_geometry == DataGeometry.SWATH:
        # Swath: (time, scanline, pixel) or similar
        if "time" not in dims:
            raise DataValidationError(f"SWATH geometry expects 'time' dimension, got {dims}")

    elif expected_geometry == DataGeometry.GRID:
        # Grid: (time, lat, lon) or (time, y, x)
        has_spatial = ("lat" in dims or "latitude" in dims or "y" in dims) and (
            "lon" in dims or "longitude" in dims or "x" in dims
        )
        if not has_spatial:
            raise DataValidationError(f"GRID geometry expects spatial dims (lat, lon), got {dims}")
