"""Unified pairing engine for source matching.

This module provides the main pairing orchestrator that dispatches to
geometry-specific strategies based on reference and comparand data geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Mapping, Sequence

import xarray as xr

from davinci_monet.core.base import PairedData
from davinci_monet.core.exceptions import GeometryMismatchError, NoOverlapError, PairingError
from davinci_monet.core.protocols import DataGeometry, PairingStrategy
from davinci_monet.core.types import TimeDelta


@dataclass
class PairingConfig:
    """Configuration for pairing operations.

    Attributes
    ----------
    radius_of_influence : float
        Spatial search radius in meters.
    time_tolerance : TimeDelta | None
        Maximum time difference for matching.
    vertical_method : str
        Vertical interpolation method ('nearest', 'linear', 'log').
    horizontal_method : str
        Horizontal interpolation method ('nearest', 'bilinear').
    time_method : str
        Time interpolation method ('nearest', 'linear'). Use 'linear' for
        models with sparse time output (e.g. 6-hourly WRF-Chem) to avoid
        step-function artifacts in the paired time series.
    apply_averaging_kernel : bool
        Whether to apply satellite averaging kernels.
    require_overlap : bool
        Whether to require temporal overlap.
    """

    radius_of_influence: float = 12000.0
    time_tolerance: TimeDelta | None = None
    vertical_method: str = "nearest"
    horizontal_method: str = "nearest"
    time_method: str = "nearest"
    apply_averaging_kernel: bool = False
    require_overlap: bool = True


class PairingEngine:
    """Unified pairing engine that dispatches to geometry-specific strategies.

    The engine automatically selects the appropriate pairing strategy based
    on the reference data's geometry attribute.

    Examples
    --------
    >>> engine = PairingEngine()
    >>> paired = engine.pair_sources(
    ...     reference=reference_data,
    ...     comparand=comparand_data,
    ...     reference_vars=["O3"],
    ...     comparand_vars=["OZONE"],
    ... )
    """

    def __init__(self, register_defaults: bool = True) -> None:
        """Initialize pairing engine.

        Parameters
        ----------
        register_defaults
            If True, register default strategies for all geometries.
        """
        self._strategies: dict[DataGeometry, PairingStrategy] = {}

        if register_defaults:
            self._register_default_strategies()

    def _register_default_strategies(self) -> None:
        """Register the default strategies for each geometry type."""
        # Import here to avoid circular imports
        from davinci_monet.pairing.strategies import (
            GridStrategy,
            PointStrategy,
            ProfileStrategy,
            SwathGridStrategy,
            TrackStrategy,
        )

        self.register_strategy(PointStrategy())
        self.register_strategy(TrackStrategy())
        self.register_strategy(ProfileStrategy())
        # SwathGridStrategy (numba binning onto a target grid) is the production
        # SWATH handler. SwathStrategy (per-pixel nearest-neighbor) is preserved
        # for direct use but is intentionally NOT the engine default.
        self.register_strategy(SwathGridStrategy())
        self.register_strategy(GridStrategy())

    def register_strategy(self, strategy: PairingStrategy) -> None:
        """Register a pairing strategy for a geometry type.

        Parameters
        ----------
        strategy
            Strategy instance implementing PairingStrategy protocol.
        """
        self._strategies[strategy.geometry] = strategy

    def get_strategy(self, geometry: DataGeometry) -> PairingStrategy:
        """Get the strategy for a given geometry.

        Parameters
        ----------
        geometry
            The data geometry type.

        Returns
        -------
        PairingStrategy
            The registered strategy.

        Raises
        ------
        PairingError
            If no strategy registered for geometry.
        """
        if geometry not in self._strategies:
            raise PairingError(
                f"No pairing strategy registered for geometry {geometry.name}. "
                f"Available: {[g.name for g in self._strategies.keys()]}"
            )
        return self._strategies[geometry]

    def supported_pairing_combinations(self) -> set[tuple[DataGeometry, DataGeometry]]:
        """Return supported ``(reference, comparand)`` geometry combinations."""
        return {(reference_geometry, DataGeometry.GRID) for reference_geometry in self._strategies}

    def supports_pairing_combination(
        self,
        reference_geometry: DataGeometry,
        comparand_geometry: DataGeometry,
    ) -> bool:
        """Return whether the engine can pair this geometry combination."""
        return (
            reference_geometry,
            comparand_geometry,
        ) in self.supported_pairing_combinations()

    def get_strategy_for(
        self,
        reference_geometry: DataGeometry,
        comparand_geometry: DataGeometry,
    ) -> PairingStrategy:
        """Get the strategy for a ``(reference, comparand)`` geometry pair.

        The comparand is resampled onto the reference's geometry. Supported
        combinations are a GRID comparand sampled onto any registered reference
        geometry (POINT/TRACK/PROFILE/SWATH/GRID). The reference geometry selects
        the strategy.

        Parameters
        ----------
        reference_geometry
            Geometry of the reference source (sampled *onto*).
        comparand_geometry
            Geometry of the comparand source (sampled *from*).

        Returns
        -------
        PairingStrategy
            The strategy handling this combination.

        Raises
        ------
        PairingError
            If the combination is not supported.
        """
        if not self.supports_pairing_combination(reference_geometry, comparand_geometry):
            raise PairingError(
                f"Unsupported pairing combination "
                f"(reference={reference_geometry.name}, comparand={comparand_geometry.name}). "
                f"Supported combinations sample a GRID comparand onto a "
                f"{[g.name for g in self._strategies.keys()]} reference."
            )
        return self.get_strategy(reference_geometry)

    def pair_sources(
        self,
        reference: xr.Dataset,
        comparand: xr.Dataset,
        reference_vars: Sequence[str],
        comparand_vars: Sequence[str],
        reference_geometry: DataGeometry | None = None,
        comparand_geometry: DataGeometry | None = None,
        config: PairingConfig | None = None,
        reference_label: str = "reference",
        comparand_label: str = "comparand",
        **kwargs: Any,
    ) -> PairedData:
        """Pair two role-neutral sources.

        ``comparand`` is sampled onto ``reference``. The paired output uses
        ``<source_label>_<source_variable>`` variable names with ``pair_role``,
        ``source_label``, and ``source_variable`` attrs.
        """
        if config is None:
            config = PairingConfig()
        if reference_geometry is None:
            reference_geometry = self._detect_geometry(reference)
        if comparand_geometry is None:
            comparand_geometry = self._detect_geometry(comparand)

        if config.require_overlap:
            self._check_temporal_overlap(comparand, reference)

        strategy = self.get_strategy_for(reference_geometry, comparand_geometry)
        paired_ds = strategy.pair_sources(
            reference=reference,
            comparand=comparand,
            radius_of_influence=config.radius_of_influence,
            time_tolerance=config.time_tolerance,
            vertical_method=config.vertical_method,
            horizontal_method=config.horizontal_method,
            time_method=config.time_method,
            reference_vars=list(reference_vars),
            comparand_vars=list(comparand_vars),
            reference_var=str(reference_vars[0]) if reference_vars else None,
            comparand_var=str(comparand_vars[0]) if comparand_vars else None,
            **kwargs,
        )
        result_ds = self._assemble_paired_dataset(
            paired_ds,
            reference_vars=reference_vars,
            comparand_vars=comparand_vars,
            reference_label=reference_label,
            comparand_label=comparand_label,
        )
        return PairedData.from_sources(
            data=result_ds,
            reference_label=reference_label,
            comparand_label=comparand_label,
            geometry=reference_geometry,
            pairing_info={
                "reference_label": reference_label,
                "comparand_label": comparand_label,
                "reference_geometry": reference_geometry.name,
                "comparand_geometry": comparand_geometry.name,
                "radius_of_influence": config.radius_of_influence,
                "time_tolerance": config.time_tolerance,
                "vertical_method": config.vertical_method,
                "horizontal_method": config.horizontal_method,
                "strategy": strategy.__class__.__name__,
            },
        )

    @staticmethod
    def _select_var(ds: xr.Dataset, candidates: Sequence[str]) -> str | None:
        """Return the first matching data variable name from candidates."""
        for name in candidates:
            if name in ds.data_vars:
                return name
        return None

    def _assemble_paired_dataset(
        self,
        paired_ds: xr.Dataset,
        reference_vars: Sequence[str],
        comparand_vars: Sequence[str],
        reference_label: str,
        comparand_label: str,
    ) -> xr.Dataset:
        """Build a source-label paired dataset from strategy output."""
        data_vars: dict[str, xr.DataArray] = {}
        coords = dict(paired_ds.coords)

        for reference_var, comparand_var in zip(reference_vars, comparand_vars):
            ref_name = str(reference_var)
            comp_name = str(comparand_var)
            reference_key = self._select_var(
                paired_ds,
                [f"reference_{ref_name}", ref_name, f"obs_{ref_name}"],
            )
            comparand_key = self._select_var(
                paired_ds,
                [
                    f"comparand_{comp_name}",
                    comp_name,
                    f"model_{comp_name}",
                    f"model_{ref_name}",
                ],
            )

            if reference_key is None or comparand_key is None:
                continue

            reference_output = f"{reference_label}_{ref_name}"
            comparand_output = f"{comparand_label}_{comp_name}"
            reference_da = paired_ds[reference_key].copy()
            comparand_da = paired_ds[comparand_key].copy()
            reference_da.attrs.update(
                {
                    "pair_role": "reference",
                    "source_label": reference_label,
                    "source_variable": ref_name,
                    "canonical_name": ref_name,
                }
            )
            comparand_da.attrs.update(
                {
                    "pair_role": "comparand",
                    "source_label": comparand_label,
                    "source_variable": comp_name,
                    "canonical_name": ref_name,
                }
            )
            data_vars[reference_output] = reference_da
            data_vars[comparand_output] = comparand_da

        result = xr.Dataset(data_vars, coords=coords)
        result.attrs = dict(paired_ds.attrs)
        result.attrs.update({"created_by": "davinci_monet", "paired": True})
        return result

    def _detect_geometry(self, data: xr.Dataset) -> DataGeometry:
        """Detect source geometry from dataset attributes or structure.

        Parameters
        ----------
        data
            Source dataset.

        Returns
        -------
        DataGeometry
            Detected geometry type.

        Raises
        ------
        GeometryMismatchError
            If geometry cannot be determined.
        """
        # Check explicit geometry attribute
        if "geometry" in data.attrs:
            geom = data.attrs["geometry"]
            if isinstance(geom, DataGeometry):
                return geom
            if isinstance(geom, str):
                try:
                    return DataGeometry[geom.upper()]
                except KeyError:
                    pass

        # Infer from dimensions
        dims = set(data.dims)

        # Grid: has lat/lon as dimensions
        if ("lat" in dims or "latitude" in dims) and ("lon" in dims or "longitude" in dims):
            return DataGeometry.GRID

        # Swath: has scanline/pixel or similar
        if "scanline" in dims or "pixel" in dims or "cross_track" in dims:
            return DataGeometry.SWATH

        # Profile: has level/z dimension with time
        if "time" in dims and ("level" in dims or "z" in dims or "altitude" in dims):
            return DataGeometry.PROFILE

        # Point: has site/station dimension
        if "site" in dims or "station" in dims or "x" in dims:
            return DataGeometry.POINT

        # Track: has time dimension with lat/lon as coordinates
        if "time" in dims:
            coords = set(data.coords)
            if ("lat" in coords or "latitude" in coords) and (
                "lon" in coords or "longitude" in coords
            ):
                return DataGeometry.TRACK

        raise GeometryMismatchError(
            f"Cannot determine source geometry from dims {dims}. "
            "Please set the 'geometry' attribute on the dataset."
        )

    def _check_temporal_overlap(self, comparand: xr.Dataset, reference: xr.Dataset) -> None:
        """Check if the two sources have temporal overlap.

        Parameters
        ----------
        comparand
            Source sampled from.
        reference
            Source sampled onto.

        Raises
        ------
        NoOverlapError
            If no temporal overlap exists.
        """
        if "time" not in comparand.dims or "time" not in reference.dims:
            return

        comparand_times = comparand["time"].values
        reference_times = reference["time"].values

        if len(comparand_times) == 0 or len(reference_times) == 0:
            return

        comparand_start = comparand_times.min()
        comparand_end = comparand_times.max()
        reference_start = reference_times.min()
        reference_end = reference_times.max()

        if comparand_end < reference_start or reference_end < comparand_start:
            raise NoOverlapError(
                f"No temporal overlap between comparand ({comparand_start} to {comparand_end}) "
                f"and reference ({reference_start} to {reference_end})"
            )


def create_default_engine() -> PairingEngine:
    """Create a pairing engine with all default strategies registered.

    Returns
    -------
    PairingEngine
        Engine with point, track, profile, swath, and grid strategies.
    """
    from davinci_monet.pairing.strategies.grid import GridStrategy
    from davinci_monet.pairing.strategies.point import PointStrategy
    from davinci_monet.pairing.strategies.profile import ProfileStrategy
    from davinci_monet.pairing.strategies.swath_grid import SwathGridStrategy
    from davinci_monet.pairing.strategies.track import TrackStrategy

    engine = PairingEngine()
    engine.register_strategy(PointStrategy())
    engine.register_strategy(TrackStrategy())
    engine.register_strategy(ProfileStrategy())
    # Production SWATH handler: bin onto a grid (see _register_default_strategies).
    engine.register_strategy(SwathGridStrategy())
    engine.register_strategy(GridStrategy())

    return engine
