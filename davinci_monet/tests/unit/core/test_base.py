"""Tests for core base data classes."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.base import (
    PairedData,
    validate_dataset_geometry,
)
from davinci_monet.core.exceptions import DataValidationError
from davinci_monet.core.protocols import DataGeometry


class TestPairedData:
    """Tests for PairedData class."""

    @pytest.fixture
    def paired_dataset(self) -> xr.Dataset:
        """Create a paired dataset for testing."""
        times = np.arange(10)
        sites = np.arange(5)
        return xr.Dataset(
            {
                "obs_ozone": (["time", "site"], np.random.randn(10, 5) + 40),
                "model_ozone": (["time", "site"], np.random.randn(10, 5) + 42),
                "obs_pm25": (["time", "site"], np.random.randn(10, 5) + 10),
                "model_pm25": (["time", "site"], np.random.randn(10, 5) + 12),
            },
            coords={
                "time": times,
                "site": sites,
                "lat": ("site", np.linspace(30, 40, 5)),
                "lon": ("site", np.linspace(-100, -90, 5)),
            },
        )

    def test_pair_label(self, paired_dataset: xr.Dataset) -> None:
        """Test pair label generation."""
        paired = PairedData(
            data=paired_dataset,
            model_label="cmaq",
            obs_label="airnow",
            geometry=DataGeometry.POINT,
        )
        assert paired.pair_label == "airnow_cmaq"

    def test_model_variables(self, paired_dataset: xr.Dataset) -> None:
        """Test listing model variables."""
        paired = PairedData(
            data=paired_dataset,
            model_label="cmaq",
            obs_label="airnow",
            geometry=DataGeometry.POINT,
        )
        assert "model_ozone" in paired.model_variables
        assert "model_pm25" in paired.model_variables

    def test_obs_variables(self, paired_dataset: xr.Dataset) -> None:
        """Test listing observation variables."""
        paired = PairedData(
            data=paired_dataset,
            model_label="cmaq",
            obs_label="airnow",
            geometry=DataGeometry.POINT,
        )
        assert "obs_ozone" in paired.obs_variables
        assert "obs_pm25" in paired.obs_variables

    def test_paired_variable_names(self, paired_dataset: xr.Dataset) -> None:
        """Test getting paired variable names."""
        paired = PairedData(
            data=paired_dataset,
            model_label="cmaq",
            obs_label="airnow",
            geometry=DataGeometry.POINT,
        )
        pairs = paired.paired_variable_names
        assert ("obs_ozone", "model_ozone") in pairs
        assert ("obs_pm25", "model_pm25") in pairs

    def test_get_obs(self, paired_dataset: xr.Dataset) -> None:
        """Test getting observation variable."""
        paired = PairedData(
            data=paired_dataset,
            model_label="cmaq",
            obs_label="airnow",
            geometry=DataGeometry.POINT,
        )
        # With prefix
        obs = paired.get_obs("obs_ozone")
        assert obs is not None
        # Without prefix
        obs = paired.get_obs("ozone")
        assert obs is not None

    def test_get_model(self, paired_dataset: xr.Dataset) -> None:
        """Test getting model variable."""
        paired = PairedData(
            data=paired_dataset,
            model_label="cmaq",
            obs_label="airnow",
            geometry=DataGeometry.POINT,
        )
        # With prefix
        model = paired.get_model("model_ozone")
        assert model is not None
        # Without prefix
        model = paired.get_model("ozone")
        assert model is not None

    def test_get_pair(self, paired_dataset: xr.Dataset) -> None:
        """Test getting paired arrays."""
        paired = PairedData(
            data=paired_dataset,
            model_label="cmaq",
            obs_label="airnow",
            geometry=DataGeometry.POINT,
        )
        obs, model = paired.get_pair("ozone")
        assert obs is not None
        assert model is not None

    def test_n_points(self, paired_dataset: xr.Dataset) -> None:
        """Test counting data points."""
        paired = PairedData(
            data=paired_dataset,
            model_label="cmaq",
            obs_label="airnow",
            geometry=DataGeometry.POINT,
        )
        assert paired.n_points == 50  # 10 times * 5 sites

    def test_to_dataframe(self, paired_dataset: xr.Dataset) -> None:
        """Test conversion to DataFrame."""
        paired = PairedData(
            data=paired_dataset,
            model_label="cmaq",
            obs_label="airnow",
            geometry=DataGeometry.POINT,
        )
        df = paired.to_dataframe()
        assert len(df) == 50
        assert "obs_ozone" in df.columns
        assert "model_ozone" in df.columns

    def test_subset_time(self, paired_dataset: xr.Dataset) -> None:
        """Test subsetting paired data by time."""
        paired = PairedData(
            data=paired_dataset,
            model_label="cmaq",
            obs_label="airnow",
            geometry=DataGeometry.POINT,
        )
        subset = paired.subset_time(start=2, end=5)  # type: ignore[arg-type]
        assert len(subset.data["time"]) == 4


class TestValidateDatasetGeometry:
    """Tests for validate_dataset_geometry function."""

    def test_point_geometry_valid(self) -> None:
        """Test valid POINT geometry."""
        ds = xr.Dataset(
            {"temp": (["time", "site"], np.random.randn(10, 5))},
            coords={"time": np.arange(10), "site": np.arange(5)},
            attrs={"geometry": "POINT"},
        )
        # Should not raise
        validate_dataset_geometry(ds, DataGeometry.POINT)

    def test_track_geometry_valid(self) -> None:
        """Test valid TRACK geometry."""
        ds = xr.Dataset(
            {"temp": (["time"], np.random.randn(100))},
            coords={
                "time": np.arange(100),
                "lat": ("time", np.random.randn(100)),
                "lon": ("time", np.random.randn(100)),
            },
            attrs={"geometry": "TRACK"},
        )
        validate_dataset_geometry(ds, DataGeometry.TRACK)

    def test_grid_geometry_valid(self) -> None:
        """Test valid GRID geometry."""
        ds = xr.Dataset(
            {"temp": (["time", "lat", "lon"], np.random.randn(10, 20, 30))},
            coords={
                "time": np.arange(10),
                "lat": np.linspace(-90, 90, 20),
                "lon": np.linspace(-180, 180, 30),
            },
        )
        validate_dataset_geometry(ds, DataGeometry.GRID)

    def test_geometry_mismatch_raises(self) -> None:
        """Test geometry mismatch raises error."""
        ds = xr.Dataset(
            {"temp": (["time"], np.random.randn(10))},
            attrs={"geometry": "TRACK"},
        )
        with pytest.raises(DataValidationError):
            validate_dataset_geometry(ds, DataGeometry.POINT)
