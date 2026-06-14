"""Unified pairing engine for source matching.

This module provides the main pairing orchestrator that dispatches to
geometry-specific strategies based on geometry and dataset data geometry.
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
        datasets with sparse time output (e.g. 6-hourly WRF-Chem) to avoid
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
    on the geometry data's geometry attribute.

    Examples
    --------
    >>> engine = PairingEngine()
    >>> paired = engine.pair_sources(
    ...     geometry=geometry_data,
    ...     dataset=dataset_data,
    ...     geometry_vars=["O3"],
    ...     dataset_vars=["OZONE"],
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
        """Return supported ``(geometry, dataset)`` geometry combinations."""
        return {(geometry, DataGeometry.GRID) for geometry in self._strategies}

    def supports_pairing_combination(
        self,
        geometry: DataGeometry,
        dataset_geometry: DataGeometry,
    ) -> bool:
        """Return whether the engine can pair this geometry combination."""
        return (
            geometry,
            dataset_geometry,
        ) in self.supported_pairing_combinations()

    def get_strategy_for(
        self,
        geometry: DataGeometry,
        dataset_geometry: DataGeometry,
    ) -> PairingStrategy:
        """Get the strategy for a ``(geometry, dataset)`` geometry pair.

        The dataset is resampled onto the geometry's geometry. Supported
        combinations are a GRID dataset sampled onto any registered geometry
        geometry (POINT/TRACK/PROFILE/SWATH/GRID). The geometry geometry selects
        the strategy.

        Parameters
        ----------
        geometry
            Geometry of the geometry source (sampled *onto*).
        dataset_geometry
            Geometry of the dataset source (sampled *from*).

        Returns
        -------
        PairingStrategy
            The strategy handling this combination.

        Raises
        ------
        PairingError
            If the combination is not supported.
        """
        if not self.supports_pairing_combination(geometry, dataset_geometry):
            raise PairingError(
                f"Unsupported pairing combination "
                f"(geometry={geometry.name}, dataset={dataset_geometry.name}). "
                f"Supported combinations sample a GRID dataset onto a "
                f"{[g.name for g in self._strategies.keys()]} geometry."
            )
        return self.get_strategy(geometry)

    def pair_sources(
        self,
        geometry_data: xr.Dataset,
        dataset_data: xr.Dataset,
        geometry_vars: Sequence[str],
        dataset_vars: Sequence[str],
        output_geometry: DataGeometry | None = None,
        dataset_geometry: DataGeometry | None = None,
        config: PairingConfig | None = None,
        geometry_label: str = "geometry",
        dataset_label: str = "dataset",
        **kwargs: Any,
    ) -> PairedData:
        """Pair two sources.

        ``dataset`` is sampled onto ``geometry``. The paired output uses
        ``<dataset_label>_<dataset_variable>`` variable names with ``pair_axis``,
        ``dataset_label``, and ``dataset_variable`` attrs.
        """
        if config is None:
            config = PairingConfig()
        if output_geometry is None:
            output_geometry = self._detect_geometry(geometry_data)
        if dataset_geometry is None:
            dataset_geometry = self._detect_geometry(dataset_data)

        if config.require_overlap:
            self._check_temporal_overlap(dataset_data, geometry_data)

        strategy = self.get_strategy_for(output_geometry, dataset_geometry)
        paired_ds = strategy.pair_sources(
            geometry_data=geometry_data,
            dataset_data=dataset_data,
            radius_of_influence=config.radius_of_influence,
            time_tolerance=config.time_tolerance,
            vertical_method=config.vertical_method,
            horizontal_method=config.horizontal_method,
            time_method=config.time_method,
            geometry_vars=list(geometry_vars),
            dataset_vars=list(dataset_vars),
            geometry_var=str(geometry_vars[0]) if geometry_vars else None,
            dataset_var=str(dataset_vars[0]) if dataset_vars else None,
            **kwargs,
        )
        result_ds = self._assemble_paired_dataset(
            paired_ds,
            geometry_vars=geometry_vars,
            dataset_vars=dataset_vars,
            geometry_label=geometry_label,
            dataset_label=dataset_label,
        )
        return PairedData.from_sources(
            data=result_ds,
            geometry_label=geometry_label,
            dataset_label=dataset_label,
            geometry=output_geometry,
            pairing_info={
                "geometry_label": geometry_label,
                "dataset_label": dataset_label,
                "geometry": output_geometry.name,
                "dataset_geometry": dataset_geometry.name,
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
        geometry_vars: Sequence[str],
        dataset_vars: Sequence[str],
        geometry_label: str,
        dataset_label: str,
    ) -> xr.Dataset:
        """Build a source-label paired dataset from strategy output."""
        data_vars: dict[str, xr.DataArray] = {}
        coords = dict(paired_ds.coords)

        for geometry_var, dataset_var in zip(geometry_vars, dataset_vars):
            geometry_name = str(geometry_var)
            dataset_name = str(dataset_var)
            geometry_key = self._select_var(
                paired_ds,
                [geometry_name, f"geometry_{geometry_name}"],
            )
            dataset_key = self._select_var(
                paired_ds,
                [
                    dataset_name,
                    f"dataset_{dataset_name}",
                    f"dataset_{geometry_name}",
                ],
            )

            if geometry_key is None or dataset_key is None:
                continue

            geometry_output = f"{geometry_label}_{geometry_name}"
            dataset_output = f"{dataset_label}_{dataset_name}"
            geometry_da = paired_ds[geometry_key].copy()
            dataset_da = paired_ds[dataset_key].copy()
            geometry_da.attrs.update(
                {
                    "pair_axis": "geometry",
                    "dataset_label": geometry_label,
                    "dataset_variable": geometry_name,
                    "canonical_name": geometry_name,
                }
            )
            dataset_da.attrs.update(
                {
                    "pair_axis": "dataset",
                    "dataset_label": dataset_label,
                    "dataset_variable": dataset_name,
                    "canonical_name": geometry_name,
                }
            )
            data_vars[geometry_output] = geometry_da
            data_vars[dataset_output] = dataset_da

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

    def _check_temporal_overlap(self, dataset: xr.Dataset, geometry: xr.Dataset) -> None:
        """Check if the two sources have temporal overlap.

        Parameters
        ----------
        dataset
            Source sampled from.
        geometry
            Source sampled onto.

        Raises
        ------
        NoOverlapError
            If no temporal overlap exists.
        """
        if "time" not in dataset.dims or "time" not in geometry.dims:
            return

        dataset_times = dataset["time"].values
        geometry_times = geometry["time"].values

        if len(dataset_times) == 0 or len(geometry_times) == 0:
            return

        dataset_start = dataset_times.min()
        dataset_end = dataset_times.max()
        geometry_start = geometry_times.min()
        geometry_end = geometry_times.max()

        if dataset_end < geometry_start or geometry_end < dataset_start:
            raise NoOverlapError(
                f"No temporal overlap between dataset ({dataset_start} to {dataset_end}) "
                f"and geometry ({geometry_start} to {geometry_end})"
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
