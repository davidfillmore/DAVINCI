"""Pre-built test scenarios for common evaluation cases.

This module provides ready-to-use test scenarios that combine dataset
and dataset data with known relationships for testing pairing,
statistics, and plotting components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.tests.synthetic.datasets import (
    create_dataset_dataset,
    create_gridded_geometries,
    create_point_geometries,
    create_profile_geometries,
    create_swath_geometries,
    create_track_geometries,
)
from davinci_monet.tests.synthetic.generators import (
    Domain,
    TimeConfig,
    add_random_noise,
    get_variable_spec,
)


@dataclass
class Scenario(ABC):
    """Base class for test scenarios.

    A scenario provides matched dataset and dataset data with known
    statistical properties for testing.

    Parameters
    ----------
    variables
        Variables to include in the scenario.
    domain
        Geographic domain.
    time_config
        Time configuration.
    seed
        Random seed for reproducibility.
    geometry
        Dataset geometry type.
    """

    variables: list[str] = field(default_factory=lambda: ["O3", "PM25"])
    domain: Domain = field(default_factory=Domain)
    time_config: TimeConfig = field(default_factory=TimeConfig)
    seed: int = 42
    geometry: DataGeometry = DataGeometry.POINT

    @abstractmethod
    def generate(self) -> tuple[xr.Dataset, xr.Dataset]:
        """Generate dataset and dataset datasets.

        Returns
        -------
        tuple[xr.Dataset, xr.Dataset]
            Tuple of (dataset, datasets) datasets.
        """
        ...

    @property
    @abstractmethod
    def expected_statistics(self) -> dict[str, dict[str, float]]:
        """Return expected statistics for each variable.

        Returns
        -------
        dict[str, dict[str, float]]
            Nested dict of {variable: {metric: value}}.
        """
        ...


@dataclass
class PerfectMatchScenario(Scenario):
    """Scenario where dataset perfectly matches datasets.

    In this scenario, datasets are sampled directly from the dataset
    at dataset locations, resulting in:
    - Mean Bias (MB) ≈ 0
    - RMSE ≈ 0
    - R² ≈ 1

    Useful for testing that pairing and interpolation work correctly.

    Parameters
    ----------
    geometry
        Dataset geometry type.
    n_geometry
        Number of datasets (sites for point, points for track, etc.).
    noise_level
        Small noise to add (0 = perfect match).
    """

    geometry: DataGeometry = DataGeometry.POINT
    n_geometry: int = 20
    noise_level: float = 0.0

    def generate(self) -> tuple[xr.Dataset, xr.Dataset]:
        """Generate perfectly matched dataset and dataset data."""
        # Generate dataset data
        dataset = create_dataset_dataset(
            variables=self.variables,
            domain=self.domain,
            time_config=self.time_config,
            seed=self.seed,
        )

        # Generate datasets by sampling from dataset
        if self.geometry == DataGeometry.POINT:
            geometry = self._generate_point_geometry(dataset)
        elif self.geometry == DataGeometry.TRACK:
            geometry = self._generate_track_geometry(dataset)
        elif self.geometry == DataGeometry.PROFILE:
            geometry = self._generate_profile_geometry(dataset)
        elif self.geometry == DataGeometry.SWATH:
            geometry = self._generate_swath_geometry(dataset)
        elif self.geometry == DataGeometry.GRID:
            geometry = self._generate_grid_geometry(dataset)
        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}")

        return dataset, geometry

    def _generate_point_geometry(self, dataset: xr.Dataset) -> xr.Dataset:
        """Generate point datasets sampled from dataset."""
        rng = np.random.default_rng(self.seed)

        # Random site locations
        n_sites = self.n_geometry
        site_lons = rng.uniform(self.domain.lon_min, self.domain.lon_max, n_sites)
        site_lats = rng.uniform(self.domain.lat_min, self.domain.lat_max, n_sites)

        # Sample dataset at dataset locations
        geometry_data: dict[str, Any] = {}
        for var in self.variables:
            # Interpolate dataset to dataset locations
            sampled = dataset[var].interp(
                lon=xr.DataArray(site_lons, dims=["site"]),
                lat=xr.DataArray(site_lats, dims=["site"]),
                method="linear",
            )
            if self.noise_level > 0:
                sampled = add_random_noise(sampled, self.noise_level, seed=self.seed)
            geometry_data[var] = sampled

        geometry = xr.Dataset(geometry_data)
        geometry.coords["longitude"] = ("site", site_lons)
        geometry.coords["latitude"] = ("site", site_lats)
        geometry.coords["site_id"] = ("site", [f"SITE{i:03d}" for i in range(n_sites)])
        geometry.attrs = {"geometry": "point", "title": "Perfect Match Point Datasets"}

        return geometry

    def _generate_track_geometry(self, dataset: xr.Dataset) -> xr.Dataset:
        """Generate track datasets sampled from dataset."""
        rng = np.random.default_rng(self.seed)
        n_points = self.n_geometry

        # Generate track path through domain
        t = np.linspace(0, 2 * np.pi, n_points)
        center_lon, center_lat = self.domain.center
        track_lons = center_lon + self.domain.lon_range * 0.3 * np.sin(t)
        track_lats = center_lat + self.domain.lat_range * 0.3 * np.cos(t)

        # Sample times
        time_indices = np.linspace(0, len(dataset.time) - 1, n_points).astype(int)
        track_times = dataset.time.values[time_indices]

        # Sample dataset at track locations
        geometry_data: dict[str, Any] = {}
        for var in self.variables:
            sampled_values = []
            for i, (lon, lat, tidx) in enumerate(zip(track_lons, track_lats, time_indices)):
                val = float(dataset[var].isel(time=tidx).interp(lon=lon, lat=lat, method="linear"))
                sampled_values.append(val)
            sampled = xr.DataArray(sampled_values, dims=["time"])
            if self.noise_level > 0:
                sampled = add_random_noise(sampled, self.noise_level, seed=self.seed)
            geometry_data[var] = sampled

        geometry = xr.Dataset(geometry_data)
        geometry.coords["time"] = ("time", track_times)
        geometry.coords["longitude"] = ("time", track_lons)
        geometry.coords["latitude"] = ("time", track_lats)
        geometry.coords["altitude"] = ("time", rng.uniform(1000, 5000, n_points))
        geometry.attrs = {"geometry": "track", "title": "Perfect Match Track Datasets"}

        return geometry

    def _generate_profile_geometry(self, dataset: xr.Dataset) -> xr.Dataset:
        """Generate profile datasets (requires 3D dataset)."""
        # For simplicity, create synthetic profiles
        geometry = create_profile_geometries(
            n_profiles=self.n_geometry,
            variables=self.variables,
            domain=self.domain,
            time_config=self.time_config,
            seed=self.seed,
        )
        geometry.attrs["title"] = "Perfect Match Profile Datasets"
        return geometry

    def _generate_swath_geometry(self, dataset: xr.Dataset) -> xr.Dataset:
        """Generate swath datasets."""
        geometry = create_swath_geometries(
            n_scans=self.n_geometry,
            variables=self.variables,
            domain=self.domain,
            time_config=self.time_config,
            seed=self.seed,
        )
        geometry.attrs["title"] = "Perfect Match Swath Datasets"
        return geometry

    def _generate_grid_geometry(self, dataset: xr.Dataset) -> xr.Dataset:
        """Generate gridded datasets from dataset."""
        # Regrid dataset to coarser dataset grid
        geometry_domain = Domain(
            lon_min=self.domain.lon_min,
            lon_max=self.domain.lon_max,
            lat_min=self.domain.lat_min,
            lat_max=self.domain.lat_max,
            n_lon=self.domain.n_lon // 2,
            n_lat=self.domain.n_lat // 2,
        )
        geometry = create_gridded_geometries(
            variables=self.variables,
            domain=geometry_domain,
            time_config=self.time_config,
            seed=self.seed,
        )
        geometry.attrs["title"] = "Perfect Match Gridded Datasets"
        return geometry

    @property
    def expected_statistics(self) -> dict[str, dict[str, float]]:
        """Expected statistics for perfect match scenario."""
        stats: dict[str, dict[str, float]] = {}
        for var in self.variables:
            if self.noise_level == 0:
                stats[var] = {"MB": 0.0, "RMSE": 0.0, "R2": 1.0}
            else:
                # With noise, expect small but non-zero errors
                spec = get_variable_spec(var)
                expected_rmse = spec.std * self.noise_level
                stats[var] = {"MB": 0.0, "RMSE": expected_rmse, "R2": 0.99}
        return stats


@dataclass
class BiasScenario(Scenario):
    """Scenario with known dataset bias.

    Dataset values have a systematic offset from datasets.
    Useful for testing bias statistics (MB, NMB, etc.).

    Parameters
    ----------
    bias
        Additive bias (dataset = geometry + bias).
    relative_bias
        If True, bias is relative (dataset = geometry * (1 + bias)).
    geometry
        Dataset geometry type.
    n_geometry
        Number of datasets.
    """

    bias: float = 5.0
    relative_bias: bool = False
    geometry: DataGeometry = DataGeometry.POINT
    n_geometry: int = 20

    def generate(self) -> tuple[xr.Dataset, xr.Dataset]:
        """Generate dataset and dataset data with known bias."""
        # First generate datasets
        if self.geometry == DataGeometry.POINT:
            geometry = create_point_geometries(
                n_sites=self.n_geometry,
                variables=self.variables,
                domain=self.domain,
                time_config=self.time_config,
                seed=self.seed,
            )
        elif self.geometry == DataGeometry.TRACK:
            geometry = create_track_geometries(
                n_points=self.n_geometry,
                variables=self.variables,
                domain=self.domain,
                time_config=self.time_config,
                seed=self.seed,
            )
        else:
            geometry = create_point_geometries(
                n_sites=self.n_geometry,
                variables=self.variables,
                domain=self.domain,
                time_config=self.time_config,
                seed=self.seed,
            )

        # Generate dataset as biased version of datasets
        dataset = create_dataset_dataset(
            variables=self.variables,
            domain=self.domain,
            time_config=self.time_config,
            seed=self.seed + 1,  # Different seed for independent data
        )

        # Apply bias
        for var in self.variables:
            if self.relative_bias:
                # Multiplicative bias
                dataset[var] = dataset[var] * (1 + self.bias)
            else:
                # Additive bias
                dataset[var] = dataset[var] + self.bias

        return dataset, geometry

    @property
    def expected_statistics(self) -> dict[str, dict[str, float]]:
        """Expected statistics for bias scenario."""
        stats: dict[str, dict[str, float]] = {}
        for var in self.variables:
            spec = get_variable_spec(var)
            if self.relative_bias:
                expected_mb = spec.mean * self.bias
            else:
                expected_mb = self.bias
            stats[var] = {"MB": expected_mb}
        return stats


@dataclass
class MismatchScenario(Scenario):
    """Scenario with spatial or temporal mismatch.

    Dataset and datasets cover different domains or time periods,
    useful for testing edge cases and error handling.

    Parameters
    ----------
    mismatch_type
        Type of mismatch: 'spatial', 'temporal', or 'both'.
    overlap_fraction
        Fraction of domain/time that overlaps (0 to 1).
    geometry
        Dataset geometry type.
    n_geometry
        Number of datasets.
    """

    mismatch_type: str = "spatial"
    overlap_fraction: float = 0.5
    geometry: DataGeometry = DataGeometry.POINT
    n_geometry: int = 20

    def generate(self) -> tuple[xr.Dataset, xr.Dataset]:
        """Generate mismatched dataset and dataset data."""
        # Dataset uses standard domain
        dataset = create_dataset_dataset(
            variables=self.variables,
            domain=self.domain,
            time_config=self.time_config,
            seed=self.seed,
        )

        # Create offset domain/time for datasets
        if self.mismatch_type in ["spatial", "both"]:
            geometry_domain = Domain(
                lon_min=self.domain.lon_min + self.domain.lon_range * (1 - self.overlap_fraction),
                lon_max=self.domain.lon_max + self.domain.lon_range * (1 - self.overlap_fraction),
                lat_min=self.domain.lat_min,
                lat_max=self.domain.lat_max,
                n_lon=self.domain.n_lon,
                n_lat=self.domain.n_lat,
            )
        else:
            geometry_domain = self.domain

        if self.mismatch_type in ["temporal", "both"]:
            import pandas as pd

            original_range = self.time_config.time_range
            offset = pd.Timedelta(hours=int(len(original_range) * (1 - self.overlap_fraction)))
            geometry_time_config = TimeConfig(
                start=pd.Timestamp(self.time_config.start) + offset,
                end=pd.Timestamp(self.time_config.end) + offset,
                freq=self.time_config.freq,
            )
        else:
            geometry_time_config = self.time_config

        # Generate datasets with offset
        if self.geometry == DataGeometry.POINT:
            geometry = create_point_geometries(
                n_sites=self.n_geometry,
                variables=self.variables,
                domain=geometry_domain,
                time_config=geometry_time_config,
                seed=self.seed,
            )
        else:
            geometry = create_point_geometries(
                n_sites=self.n_geometry,
                variables=self.variables,
                domain=geometry_domain,
                time_config=geometry_time_config,
                seed=self.seed,
            )

        return dataset, geometry

    @property
    def expected_statistics(self) -> dict[str, dict[str, float]]:
        """Expected statistics depend on overlap."""
        # Mismatched data do not have fixed statistics.
        return {var: {} for var in self.variables}


def sample_geometry_from(
    dataset_ds: xr.Dataset,
    geometry: str = "point",
    *,
    scenario: "PerfectMatchScenario | None" = None,
) -> xr.Dataset:
    """Generate datasets sampled from *dataset_ds* for a given geometry.

    This is a public convenience wrapper around
    :meth:`PerfectMatchScenario._generate_point_geometry` (and its siblings) so
    that test helpers can call it without reaching into private methods.

    Parameters
    ----------
    dataset_ds:
        Dataset dataset to sample from.
    geometry:
        Geometry key: ``"point"`` (default), ``"track"``, ``"profile"``,
        ``"swath"``, or ``"grid"``.
    scenario:
        An existing :class:`PerfectMatchScenario` whose configuration
        (domain, seed, variables, …) should be used.  If *None*, a default
        scenario is constructed from the dataset's domain extents.

    Returns
    -------
    xr.Dataset
        Dataset dataset sampled from the dataset at the requested geometry.
    """
    if scenario is None:
        scenario = PerfectMatchScenario()

    geom_key = geometry.lower()
    _dispatch: dict[str, Any] = {
        "point": scenario._generate_point_geometry,
        "track": scenario._generate_track_geometry,
        "profile": scenario._generate_profile_geometry,
        "swath": scenario._generate_swath_geometry,
        "grid": scenario._generate_grid_geometry,
    }
    if geom_key not in _dispatch:
        valid = ", ".join(_dispatch)
        raise ValueError(f"Unknown geometry {geometry!r}. Valid choices: {valid}")
    return _dispatch[geom_key](dataset_ds)


def create_scenario(
    scenario_type: str,
    geometry: DataGeometry = DataGeometry.POINT,
    **kwargs: Any,
) -> Scenario:
    """Factory function to create test scenarios.

    Parameters
    ----------
    scenario_type
        Type of scenario: 'perfect_match', 'bias', 'mismatch'.
    geometry
        Dataset geometry.
    **kwargs
        Additional scenario parameters.

    Returns
    -------
    Scenario
        Configured scenario instance.

    Examples
    --------
    >>> scenario = create_scenario("perfect_match", geometry=DataGeometry.POINT)
    >>> dataset, geometry = scenario.generate()
    """
    scenarios: dict[str, type[PerfectMatchScenario | BiasScenario | MismatchScenario]] = {
        "perfect_match": PerfectMatchScenario,
        "bias": BiasScenario,
        "mismatch": MismatchScenario,
    }

    if scenario_type not in scenarios:
        valid = ", ".join(scenarios.keys())
        raise ValueError(f"Unknown scenario type: {scenario_type}. Valid types: {valid}")

    scenario_cls = scenarios[scenario_type]
    return scenario_cls(geometry=geometry, **kwargs)
