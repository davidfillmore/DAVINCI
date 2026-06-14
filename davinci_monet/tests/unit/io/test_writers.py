"""Tests for io module (writers)."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from davinci_monet.io import (
    write_dataset,
    write_pickle,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_dataset() -> xr.Dataset:
    """Create a sample xarray Dataset."""
    n_times = 10
    n_lat = 5
    n_lon = 6

    times = np.datetime64("2020-01-01") + np.arange(n_times) * np.timedelta64(1, "h")
    lats = np.linspace(30, 40, n_lat)
    lons = np.linspace(-110, -100, n_lon)

    np.random.seed(42)
    data = np.random.randn(n_times, n_lat, n_lon)

    return xr.Dataset(
        {"temperature": (["time", "lat", "lon"], data)},
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
        attrs={"title": "Test Dataset"},
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Writer Tests
# =============================================================================


class TestWriteDataset:
    """Tests for write_dataset function."""

    def test_write_netcdf(self, sample_dataset: xr.Dataset, temp_dir: Path):
        """Test writing NetCDF file."""
        filepath = temp_dir / "output.nc"
        write_dataset(sample_dataset, filepath)

        assert filepath.exists()

        # Verify by reading back
        result = xr.open_dataset(filepath)
        assert "temperature" in result.data_vars

    def test_write_pickle(self, sample_dataset: xr.Dataset, temp_dir: Path):
        """Test writing pickle file."""
        filepath = temp_dir / "output.pkl"
        write_dataset(sample_dataset, filepath)

        assert filepath.exists()

        # Verify by reading back
        with open(filepath, "rb") as f:
            result = pickle.load(f)
        assert isinstance(result, xr.Dataset)

    def test_creates_parent_directory(self, sample_dataset: xr.Dataset, temp_dir: Path):
        """Test parent directory is created."""
        filepath = temp_dir / "subdir" / "nested" / "output.nc"
        write_dataset(sample_dataset, filepath)

        assert filepath.exists()

    def test_write_zarr(self, sample_dataset: xr.Dataset, temp_dir: Path):
        """Test writing Zarr file."""
        pytest.importorskip("zarr")

        filepath = temp_dir / "output.zarr"
        write_dataset(sample_dataset, filepath)

        assert filepath.exists()


class TestWritePickle:
    """Tests for write_pickle function."""

    def test_write_dataset(self, sample_dataset: xr.Dataset, temp_dir: Path):
        """Test writing Dataset to pickle."""
        filepath = temp_dir / "data.pkl"
        write_pickle(sample_dataset, filepath)

        assert filepath.exists()

        with open(filepath, "rb") as f:
            result = pickle.load(f)
        assert isinstance(result, xr.Dataset)

    def test_write_any_object(self, temp_dir: Path):
        """Test writing any object to pickle."""
        data = {"key": "value", "list": [1, 2, 3]}
        filepath = temp_dir / "data.pkl"
        write_pickle(data, filepath)

        assert filepath.exists()

        with open(filepath, "rb") as f:
            result = pickle.load(f)
        assert result == data

    def test_creates_parent_directory(self, temp_dir: Path):
        """Test parent directory is created."""
        filepath = temp_dir / "subdir" / "data.pkl"
        write_pickle({"test": True}, filepath)

        assert filepath.exists()


# =============================================================================
# Round-trip Tests
# =============================================================================


class TestRoundTrip:
    """Tests for reading back written data."""

    def test_netcdf_roundtrip(self, sample_dataset: xr.Dataset, temp_dir: Path):
        """Test NetCDF write and read."""
        filepath = temp_dir / "roundtrip.nc"

        write_dataset(sample_dataset, filepath)
        result = xr.open_dataset(filepath)

        xr.testing.assert_equal(result, sample_dataset)

    def test_pickle_roundtrip(self, sample_dataset: xr.Dataset, temp_dir: Path):
        """Test pickle write and read."""
        filepath = temp_dir / "roundtrip.pkl"

        write_pickle(sample_dataset, filepath)
        with open(filepath, "rb") as f:
            result = pickle.load(f)

        xr.testing.assert_equal(result, sample_dataset)
