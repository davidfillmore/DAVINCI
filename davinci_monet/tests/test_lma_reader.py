"""Tests for the Lightning Mapping Array (LMA) observation reader.

Tests the LMAReader class using synthetic datasets that mimic CF-compliant
NetCDF grids from LMA networks (OKLMA, COLMA, NALMA).
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import observation_registry

# Ensure LMA reader is registered
import davinci_monet.observations.lightning.lma  # noqa: F401


# =============================================================================
# Synthetic LMA Data
# =============================================================================


def create_synthetic_lma_grid(
    n_lat: int = 40,
    n_lon: int = 40,
    n_times: int = 6,
    network: str = "oklma",
) -> xr.Dataset:
    """Create a synthetic LMA flash density grid.

    Mimics CF-compliant NetCDF produced by LMA grid processing pipelines.
    """
    center = {
        "oklma": (35.2, -97.4),
        "colma": (40.4, -104.6),
        "nalma": (34.7, -86.6),
    }
    lat_c, lon_c = center.get(network, (36.0, -97.0))

    times = pd.date_range("2012-05-30", periods=n_times, freq="10min")
    lats = np.linspace(lat_c - 1.5, lat_c + 1.5, n_lat)
    lons = np.linspace(lon_c - 1.5, lon_c + 1.5, n_lon)

    rng = np.random.default_rng(42)

    # Flash extent density (flashes per grid cell per time step)
    flash_extent = rng.poisson(lam=2.0, size=(n_times, n_lat, n_lon)).astype(np.float32)

    # Source density (VHF sources per grid cell)
    source = rng.poisson(lam=10.0, size=(n_times, n_lat, n_lon)).astype(np.float32)

    # Flash initiation density
    flash_init = rng.poisson(lam=1.0, size=(n_times, n_lat, n_lon)).astype(np.float32)

    ds = xr.Dataset(
        {
            "flash_extent_density": (["time", "latitude", "longitude"], flash_extent),
            "source_density": (["time", "latitude", "longitude"], source),
            "flash_init_density": (["time", "latitude", "longitude"], flash_init),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
        attrs={
            "title": f"LMA grid product - {network.upper()}",
            "Conventions": "CF-1.6",
        },
    )

    return ds


def create_synthetic_lma_grid_alt_coords(
    n_lat: int = 20,
    n_lon: int = 20,
    n_times: int = 3,
) -> xr.Dataset:
    """Create LMA grid with non-standard coordinate names (lat/lon instead of latitude/longitude)."""
    times = pd.date_range("2012-05-30", periods=n_times, freq="10min")
    lats = np.linspace(34.0, 36.5, n_lat)
    lons = np.linspace(-98.5, -96.0, n_lon)

    rng = np.random.default_rng(99)
    flash_extent = rng.poisson(lam=3.0, size=(n_times, n_lat, n_lon)).astype(np.float32)

    ds = xr.Dataset(
        {
            "flash_extent_density": (["time", "lat", "lon"], flash_extent),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )

    return ds


def create_synthetic_lma_grid_no_time(
    n_lat: int = 20,
    n_lon: int = 20,
) -> xr.Dataset:
    """Create a single-time LMA grid without explicit time dimension."""
    lats = np.linspace(34.0, 36.5, n_lat)
    lons = np.linspace(-98.5, -96.0, n_lon)

    rng = np.random.default_rng(77)
    flash_extent = rng.poisson(lam=2.0, size=(n_lat, n_lon)).astype(np.float32)

    ds = xr.Dataset(
        {
            "flash_extent_density": (["latitude", "longitude"], flash_extent),
        },
        coords={
            "latitude": lats,
            "longitude": lons,
        },
    )

    return ds


# =============================================================================
# Registry Tests
# =============================================================================


class TestLMARegistry:
    """Test LMA reader registration."""

    def test_lma_registered(self):
        """Test that LMA reader is registered."""
        assert "lma" in observation_registry

    def test_get_reader_class(self):
        """Test getting LMA reader class from registry."""
        lma_cls = observation_registry.get("lma")
        assert lma_cls is not None
        reader = lma_cls()
        assert reader.name == "lma"


# =============================================================================
# Reader Tests
# =============================================================================


class TestLMAReader:
    """Test LMA reader."""

    def test_reader_name(self):
        """Test reader name."""
        from davinci_monet.observations.lightning.lma import LMAReader

        reader = LMAReader()
        assert reader.name == "lma"

    def test_variable_mapping(self):
        """Test variable mapping."""
        from davinci_monet.observations.lightning.lma import LMAReader

        reader = LMAReader()
        mapping = reader.get_variable_mapping()
        assert "flash_density" in mapping
        assert "source_density" in mapping
        assert mapping["flash_density"] == "flash_extent_density"

    def test_open_netcdf_file(self):
        """Test opening a standard LMA NetCDF grid."""
        from davinci_monet.observations.lightning.lma import LMAReader

        ds = create_synthetic_lma_grid()

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            ds.to_netcdf(f.name)
            reader = LMAReader()
            result = reader.open([f.name])

            assert "flash_extent_density" in result.data_vars
            assert "source_density" in result.data_vars
            assert "flash_init_density" in result.data_vars
            assert "time" in result.dims
            assert "latitude" in result.dims
            assert "longitude" in result.dims
            assert result.attrs["geometry"] == DataGeometry.GRID.value

            Path(f.name).unlink()

    def test_coordinate_standardization(self):
        """Test that lat/lon coords are renamed to latitude/longitude."""
        from davinci_monet.observations.lightning.lma import LMAReader

        ds = create_synthetic_lma_grid_alt_coords()

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            ds.to_netcdf(f.name)
            reader = LMAReader()
            result = reader.open([f.name])

            assert "latitude" in result.dims
            assert "longitude" in result.dims
            assert "lat" not in result.dims
            assert "lon" not in result.dims

            Path(f.name).unlink()

    def test_time_dimension_added(self):
        """Test that time dimension is added if missing."""
        from davinci_monet.observations.lightning.lma import LMAReader

        ds = create_synthetic_lma_grid_no_time()

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            ds.to_netcdf(f.name)
            reader = LMAReader()
            result = reader.open([f.name])

            assert "time" in result.dims
            assert result.sizes["time"] == 1

            Path(f.name).unlink()

    def test_network_detection_oklma(self):
        """Test auto-detection of OKLMA network from filename."""
        from davinci_monet.observations.lightning.lma import LMAReader

        ds = create_synthetic_lma_grid(network="oklma")

        with tempfile.NamedTemporaryFile(
            suffix=".nc", prefix="oklma_20120530_", delete=False
        ) as f:
            ds.to_netcdf(f.name)
            reader = LMAReader()
            result = reader.open([f.name])

            assert result.attrs.get("lma_network_id") == "oklma"
            assert "Oklahoma" in result.attrs.get("lma_network", "")

            Path(f.name).unlink()

    def test_explicit_network(self):
        """Test passing network explicitly."""
        from davinci_monet.observations.lightning.lma import LMAReader

        ds = create_synthetic_lma_grid(network="colma")

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            ds.to_netcdf(f.name)
            reader = LMAReader()
            result = reader.open([f.name], network="colma")

            assert result.attrs.get("lma_network_id") == "colma"
            assert "Colorado" in result.attrs.get("lma_network", "")

            Path(f.name).unlink()

    def test_variable_selection(self):
        """Test loading specific variables."""
        from davinci_monet.observations.lightning.lma import LMAReader

        ds = create_synthetic_lma_grid()

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            ds.to_netcdf(f.name)
            reader = LMAReader()
            result = reader.open([f.name], variables=["flash_extent_density"])

            assert "flash_extent_density" in result.data_vars
            assert "source_density" not in result.data_vars

            Path(f.name).unlink()

    def test_multiple_files(self):
        """Test concatenating multiple LMA files."""
        from davinci_monet.observations.lightning.lma import LMAReader

        ds1 = create_synthetic_lma_grid(n_times=3)
        ds2 = create_synthetic_lma_grid(n_times=3)
        # Shift time for second dataset
        ds2["time"] = pd.date_range("2012-05-30 01:00", periods=3, freq="10min")

        paths = []
        for ds in [ds1, ds2]:
            f = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
            ds.to_netcdf(f.name)
            paths.append(f.name)
            f.close()

        reader = LMAReader()
        result = reader.open(paths)
        assert result.sizes["time"] == 6

        for p in paths:
            Path(p).unlink()

    def test_missing_files_raises(self):
        """Test that missing files raise DataNotFoundError."""
        from davinci_monet.observations.lightning.lma import LMAReader
        from davinci_monet.core.exceptions import DataNotFoundError

        reader = LMAReader()
        with pytest.raises(DataNotFoundError):
            reader.open(["/nonexistent/path/oklma_grid.nc"])

    def test_empty_file_list_raises(self):
        """Test that empty file list raises DataNotFoundError."""
        from davinci_monet.observations.lightning.lma import LMAReader
        from davinci_monet.core.exceptions import DataNotFoundError

        reader = LMAReader()
        with pytest.raises(DataNotFoundError):
            reader.open([])


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestOpenLMA:
    """Test open_lma convenience function."""

    def test_open_single_file(self):
        """Test opening a single file with open_lma."""
        from davinci_monet.observations.lightning.lma import open_lma

        ds = create_synthetic_lma_grid()

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            ds.to_netcdf(f.name)
            obs = open_lma(f.name, label="oklma_test")

            assert obs.label == "oklma_test"
            assert obs.geometry == DataGeometry.GRID
            assert obs.data is not None
            assert "flash_extent_density" in obs.data.data_vars

            Path(f.name).unlink()

    def test_open_with_glob_pattern(self, tmp_path: Path):
        """Test opening files with glob pattern."""
        from davinci_monet.observations.lightning.lma import open_lma

        for i, day in enumerate([28, 29, 30]):
            ds = create_synthetic_lma_grid(n_times=2)
            ds["time"] = pd.date_range(
                f"2012-05-{day}", periods=2, freq="10min"
            )
            ds.to_netcdf(tmp_path / f"oklma_201205{day:02d}_grid.nc")

        obs = open_lma(str(tmp_path / "oklma_*.nc"), label="oklma_multi")
        assert obs.data is not None
        assert obs.data.sizes["time"] == 6


# =============================================================================
# Geometry Mapping Tests
# =============================================================================


class TestLMAGeometryMapping:
    """Test that LMA obs_type maps to GRID geometry."""

    def test_geometry_from_obs_type(self):
        """Test geometry_from_obs_type for lma."""
        from davinci_monet.observations.base import ObservationData

        geom = ObservationData.geometry_from_obs_type("lma")
        assert geom == DataGeometry.GRID

    def test_create_observation_data(self):
        """Test factory function with lma obs_type."""
        from davinci_monet.observations.base import create_observation_data

        ds = create_synthetic_lma_grid()
        obs = create_observation_data(
            label="lma_test",
            obs_type="lma",
            data=ds,
        )
        assert obs.geometry == DataGeometry.GRID
        assert obs.obs_type == "lma"
