"""Pre-built test scenarios for common evaluation cases.

This module provides ready-to-use test scenarios that combine model
and observation data with known relationships for testing pairing,
statistics, and plotting components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.tests.synthetic.generators import (
    Domain,
    TimeConfig,
    add_random_noise,
    get_variable_spec,
)
from davinci_monet.tests.synthetic.models import create_model_dataset
from davinci_monet.tests.synthetic.observations import (
    create_gridded_observations,
    create_point_observations,
    create_profile_observations,
    create_swath_observations,
    create_track_observations,
)


@dataclass
class Scenario(ABC):
    """Base class for test scenarios.

    A scenario provides matched model and observation data with known
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
        Observation geometry type.
    """

    variables: list[str] = field(default_factory=lambda: ["O3", "PM25"])
    domain: Domain = field(default_factory=Domain)
    time_config: TimeConfig = field(default_factory=TimeConfig)
    seed: int = 42
    geometry: DataGeometry = DataGeometry.POINT

    @abstractmethod
    def generate(self) -> tuple[xr.Dataset, xr.Dataset]:
        """Generate model and observation datasets.

        Returns
        -------
        tuple[xr.Dataset, xr.Dataset]
            Tuple of (model, observations) datasets.
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
    """Scenario where model perfectly matches observations.

    In this scenario, observations are sampled directly from the model
    at observation locations, resulting in:
    - Mean Bias (MB) ≈ 0
    - RMSE ≈ 0
    - R² ≈ 1

    Useful for testing that pairing and interpolation work correctly.

    Parameters
    ----------
    geometry
        Observation geometry type.
    n_obs
        Number of observations (sites for point, points for track, etc.).
    noise_level
        Small noise to add (0 = perfect match).
    """

    geometry: DataGeometry = DataGeometry.POINT
    n_obs: int = 20
    noise_level: float = 0.0

    def generate(self) -> tuple[xr.Dataset, xr.Dataset]:
        """Generate perfectly matched model and observation data."""
        # Generate model data
        model = create_model_dataset(
            variables=self.variables,
            domain=self.domain,
            time_config=self.time_config,
            seed=self.seed,
        )

        # Generate observations by sampling from model
        if self.geometry == DataGeometry.POINT:
            obs = self._generate_point_obs(model)
        elif self.geometry == DataGeometry.TRACK:
            obs = self._generate_track_obs(model)
        elif self.geometry == DataGeometry.PROFILE:
            obs = self._generate_profile_obs(model)
        elif self.geometry == DataGeometry.SWATH:
            obs = self._generate_swath_obs(model)
        elif self.geometry == DataGeometry.GRID:
            obs = self._generate_grid_obs(model)
        else:
            raise ValueError(f"Unsupported geometry: {self.geometry}")

        return model, obs

    def _generate_point_obs(self, model: xr.Dataset) -> xr.Dataset:
        """Generate point observations sampled from model."""
        rng = np.random.default_rng(self.seed)

        # Random site locations
        n_sites = self.n_obs
        site_lons = rng.uniform(self.domain.lon_min, self.domain.lon_max, n_sites)
        site_lats = rng.uniform(self.domain.lat_min, self.domain.lat_max, n_sites)

        # Sample model at observation locations
        obs_data: dict[str, Any] = {}
        for var in self.variables:
            # Interpolate model to observation locations
            sampled = model[var].interp(
                lon=xr.DataArray(site_lons, dims=["site"]),
                lat=xr.DataArray(site_lats, dims=["site"]),
                method="linear",
            )
            if self.noise_level > 0:
                sampled = add_random_noise(sampled, self.noise_level, seed=self.seed)
            obs_data[var] = sampled

        obs = xr.Dataset(obs_data)
        obs.coords["longitude"] = ("site", site_lons)
        obs.coords["latitude"] = ("site", site_lats)
        obs.coords["site_id"] = ("site", [f"SITE{i:03d}" for i in range(n_sites)])
        obs.attrs = {"geometry": "point", "title": "Perfect Match Point Observations"}

        return obs

    def _generate_track_obs(self, model: xr.Dataset) -> xr.Dataset:
        """Generate track observations sampled from model."""
        rng = np.random.default_rng(self.seed)
        n_points = self.n_obs

        # Generate track path through domain
        t = np.linspace(0, 2 * np.pi, n_points)
        center_lon, center_lat = self.domain.center
        track_lons = center_lon + self.domain.lon_range * 0.3 * np.sin(t)
        track_lats = center_lat + self.domain.lat_range * 0.3 * np.cos(t)

        # Sample times
        time_indices = np.linspace(0, len(model.time) - 1, n_points).astype(int)
        track_times = model.time.values[time_indices]

        # Sample model at track locations
        obs_data: dict[str, Any] = {}
        for var in self.variables:
            sampled_values = []
            for i, (lon, lat, tidx) in enumerate(zip(track_lons, track_lats, time_indices)):
                val = float(model[var].isel(time=tidx).interp(lon=lon, lat=lat, method="linear"))
                sampled_values.append(val)
            sampled = xr.DataArray(sampled_values, dims=["time"])
            if self.noise_level > 0:
                sampled = add_random_noise(sampled, self.noise_level, seed=self.seed)
            obs_data[var] = sampled

        obs = xr.Dataset(obs_data)
        obs.coords["time"] = ("time", track_times)
        obs.coords["longitude"] = ("time", track_lons)
        obs.coords["latitude"] = ("time", track_lats)
        obs.coords["altitude"] = ("time", rng.uniform(1000, 5000, n_points))
        obs.attrs = {"geometry": "track", "title": "Perfect Match Track Observations"}

        return obs

    def _generate_profile_obs(self, model: xr.Dataset) -> xr.Dataset:
        """Generate profile observations (requires 3D model)."""
        # For simplicity, create synthetic profiles
        obs = create_profile_observations(
            n_profiles=self.n_obs,
            variables=self.variables,
            domain=self.domain,
            time_config=self.time_config,
            seed=self.seed,
        )
        obs.attrs["title"] = "Perfect Match Profile Observations"
        return obs

    def _generate_swath_obs(self, model: xr.Dataset) -> xr.Dataset:
        """Generate swath observations."""
        obs = create_swath_observations(
            n_scans=self.n_obs,
            variables=self.variables,
            domain=self.domain,
            time_config=self.time_config,
            seed=self.seed,
        )
        obs.attrs["title"] = "Perfect Match Swath Observations"
        return obs

    def _generate_grid_obs(self, model: xr.Dataset) -> xr.Dataset:
        """Generate gridded observations from model."""
        # Regrid model to coarser observation grid
        obs_domain = Domain(
            lon_min=self.domain.lon_min,
            lon_max=self.domain.lon_max,
            lat_min=self.domain.lat_min,
            lat_max=self.domain.lat_max,
            n_lon=self.domain.n_lon // 2,
            n_lat=self.domain.n_lat // 2,
        )
        obs = create_gridded_observations(
            variables=self.variables,
            domain=obs_domain,
            time_config=self.time_config,
            seed=self.seed,
        )
        obs.attrs["title"] = "Perfect Match Gridded Observations"
        return obs

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
    """Scenario with known model bias.

    Model values have a systematic offset from observations.
    Useful for testing bias statistics (MB, NMB, etc.).

    Parameters
    ----------
    bias
        Additive bias (model = obs + bias).
    relative_bias
        If True, bias is relative (model = obs * (1 + bias)).
    geometry
        Observation geometry type.
    n_obs
        Number of observations.
    """

    bias: float = 5.0
    relative_bias: bool = False
    geometry: DataGeometry = DataGeometry.POINT
    n_obs: int = 20

    def generate(self) -> tuple[xr.Dataset, xr.Dataset]:
        """Generate model and observation data with known bias."""
        # First generate observations
        if self.geometry == DataGeometry.POINT:
            obs = create_point_observations(
                n_sites=self.n_obs,
                variables=self.variables,
                domain=self.domain,
                time_config=self.time_config,
                seed=self.seed,
            )
        elif self.geometry == DataGeometry.TRACK:
            obs = create_track_observations(
                n_points=self.n_obs,
                variables=self.variables,
                domain=self.domain,
                time_config=self.time_config,
                seed=self.seed,
            )
        else:
            obs = create_point_observations(
                n_sites=self.n_obs,
                variables=self.variables,
                domain=self.domain,
                time_config=self.time_config,
                seed=self.seed,
            )

        # Generate model as biased version of observations
        model = create_model_dataset(
            variables=self.variables,
            domain=self.domain,
            time_config=self.time_config,
            seed=self.seed + 1,  # Different seed for independent data
        )

        # Apply bias
        for var in self.variables:
            if self.relative_bias:
                # Multiplicative bias
                model[var] = model[var] * (1 + self.bias)
            else:
                # Additive bias
                model[var] = model[var] + self.bias

        return model, obs

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

    Model and observations cover different domains or time periods,
    useful for testing edge cases and error handling.

    Parameters
    ----------
    mismatch_type
        Type of mismatch: 'spatial', 'temporal', or 'both'.
    overlap_fraction
        Fraction of domain/time that overlaps (0 to 1).
    geometry
        Observation geometry type.
    n_obs
        Number of observations.
    """

    mismatch_type: str = "spatial"
    overlap_fraction: float = 0.5
    geometry: DataGeometry = DataGeometry.POINT
    n_obs: int = 20

    def generate(self) -> tuple[xr.Dataset, xr.Dataset]:
        """Generate mismatched model and observation data."""
        # Model uses standard domain
        model = create_model_dataset(
            variables=self.variables,
            domain=self.domain,
            time_config=self.time_config,
            seed=self.seed,
        )

        # Create offset domain/time for observations
        if self.mismatch_type in ["spatial", "both"]:
            obs_domain = Domain(
                lon_min=self.domain.lon_min + self.domain.lon_range * (1 - self.overlap_fraction),
                lon_max=self.domain.lon_max + self.domain.lon_range * (1 - self.overlap_fraction),
                lat_min=self.domain.lat_min,
                lat_max=self.domain.lat_max,
                n_lon=self.domain.n_lon,
                n_lat=self.domain.n_lat,
            )
        else:
            obs_domain = self.domain

        if self.mismatch_type in ["temporal", "both"]:
            import pandas as pd

            original_range = self.time_config.time_range
            offset = pd.Timedelta(hours=int(len(original_range) * (1 - self.overlap_fraction)))
            obs_time_config = TimeConfig(
                start=pd.Timestamp(self.time_config.start) + offset,
                end=pd.Timestamp(self.time_config.end) + offset,
                freq=self.time_config.freq,
            )
        else:
            obs_time_config = self.time_config

        # Generate observations with offset
        if self.geometry == DataGeometry.POINT:
            obs = create_point_observations(
                n_sites=self.n_obs,
                variables=self.variables,
                domain=obs_domain,
                time_config=obs_time_config,
                seed=self.seed,
            )
        else:
            obs = create_point_observations(
                n_sites=self.n_obs,
                variables=self.variables,
                domain=obs_domain,
                time_config=obs_time_config,
                seed=self.seed,
            )

        return model, obs

    @property
    def expected_statistics(self) -> dict[str, dict[str, float]]:
        """Expected statistics depend on overlap."""
        # Cannot easily predict statistics for mismatched data
        return {var: {} for var in self.variables}


def sample_obs_from(
    model_ds: xr.Dataset,
    geometry: str = "point",
    *,
    scenario: "PerfectMatchScenario | None" = None,
) -> xr.Dataset:
    """Generate observations sampled from *model_ds* for a given geometry.

    This is a public convenience wrapper around
    :meth:`PerfectMatchScenario._generate_point_obs` (and its siblings) so
    that test helpers can call it without reaching into private methods.

    Parameters
    ----------
    model_ds:
        Model dataset to sample from.
    geometry:
        Geometry key: ``"point"`` (default), ``"track"``, ``"profile"``,
        ``"swath"``, or ``"grid"``.
    scenario:
        An existing :class:`PerfectMatchScenario` whose configuration
        (domain, seed, variables, …) should be used.  If *None*, a default
        scenario is constructed from the model's domain extents.

    Returns
    -------
    xr.Dataset
        Observation dataset sampled from the model at the requested geometry.
    """
    if scenario is None:
        scenario = PerfectMatchScenario()

    geom_key = geometry.lower()
    _dispatch: dict[str, Any] = {
        "point": scenario._generate_point_obs,
        "track": scenario._generate_track_obs,
        "profile": scenario._generate_profile_obs,
        "swath": scenario._generate_swath_obs,
        "grid": scenario._generate_grid_obs,
    }
    if geom_key not in _dispatch:
        valid = ", ".join(_dispatch)
        raise ValueError(f"Unknown geometry {geometry!r}. Valid choices: {valid}")
    return _dispatch[geom_key](model_ds)


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
        Observation geometry.
    **kwargs
        Additional scenario parameters.

    Returns
    -------
    Scenario
        Configured scenario instance.

    Examples
    --------
    >>> scenario = create_scenario("perfect_match", geometry=DataGeometry.POINT)
    >>> model, obs = scenario.generate()
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
