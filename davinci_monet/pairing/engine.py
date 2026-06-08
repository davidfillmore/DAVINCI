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
    >>> paired = engine.pair_sources(reference=obs_data, comparand=model_data,
    ...                              reference_vars=['O3'], comparand_vars=['OZONE'])
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

    def get_strategy_for(
        self,
        reference_geometry: DataGeometry,
        comparand_geometry: DataGeometry,
    ) -> PairingStrategy:
        """Get the strategy for a ``(reference, comparand)`` geometry pair.

        Role-neutral dispatch (Phase 4). The comparand is resampled onto the
        reference's geometry. The seeded combinations mirror today's behavior:
        a GRID comparand sampled onto any irregular reference (POINT/TRACK/
        PROFILE/SWATH), and GRID onto GRID. The reference geometry selects the
        strategy, exactly as the legacy obs-geometry dispatch did.

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
        if comparand_geometry is not DataGeometry.GRID:
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

        ``comparand`` is sampled onto ``reference``. The paired output still
        uses the internal ``obs_``/``model_`` assembly prefixes before the
        pipeline tags them with source labels.
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
            **kwargs,
        )
        result_ds = self._assemble_paired_dataset(
            paired_ds,
            obs_vars=reference_vars,
            model_vars=comparand_vars,
        )
        return PairedData(
            data=result_ds,
            model_label=comparand_label,
            obs_label=reference_label,
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
        obs_vars: Sequence[str],
        model_vars: Sequence[str],
    ) -> xr.Dataset:
        """Build a paired dataset with obs_/model_ prefixes from strategy output."""
        data_vars: dict[str, xr.DataArray] = {}
        coords = dict(paired_ds.coords)

        for obs_var, model_var in zip(obs_vars, model_vars):
            obs_key = self._select_var(paired_ds, [f"obs_{obs_var}", obs_var])
            model_key = self._select_var(
                paired_ds,
                [f"model_{model_var}", f"model_{obs_var}", model_var],
            )

            if obs_key is not None:
                data_vars[f"obs_{obs_var}"] = paired_ds[obs_key]
            if model_key is not None:
                data_vars[f"model_{obs_var}"] = paired_ds[model_key]

        result = xr.Dataset(data_vars, coords=coords)
        result.attrs = dict(paired_ds.attrs)
        result.attrs.update({"created_by": "davinci_monet", "paired": True})
        return result

    def _detect_geometry(self, obs: xr.Dataset) -> DataGeometry:
        """Detect observation geometry from dataset attributes or structure.

        Parameters
        ----------
        obs
            Observation dataset.

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
        if "geometry" in obs.attrs:
            geom = obs.attrs["geometry"]
            if isinstance(geom, DataGeometry):
                return geom
            if isinstance(geom, str):
                try:
                    return DataGeometry[geom.upper()]
                except KeyError:
                    pass

        # Infer from dimensions
        dims = set(obs.dims)

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
            coords = set(obs.coords)
            if ("lat" in coords or "latitude" in coords) and (
                "lon" in coords or "longitude" in coords
            ):
                return DataGeometry.TRACK

        raise GeometryMismatchError(
            f"Cannot determine observation geometry from dims {dims}. "
            "Please set the 'geometry' attribute on the dataset."
        )

    def _check_temporal_overlap(self, model: xr.Dataset, obs: xr.Dataset) -> None:
        """Check if model and observation have temporal overlap.

        Parameters
        ----------
        model
            Model dataset.
        obs
            Observation dataset.

        Raises
        ------
        NoOverlapError
            If no temporal overlap exists.
        """
        if "time" not in model.dims or "time" not in obs.dims:
            return

        model_times = model["time"].values
        obs_times = obs["time"].values

        if len(model_times) == 0 or len(obs_times) == 0:
            return

        model_start = model_times.min()
        model_end = model_times.max()
        obs_start = obs_times.min()
        obs_end = obs_times.max()

        if model_end < obs_start or obs_end < model_start:
            raise NoOverlapError(
                f"No temporal overlap between model ({model_start} to {model_end}) "
                f"and observations ({obs_start} to {obs_end})"
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
