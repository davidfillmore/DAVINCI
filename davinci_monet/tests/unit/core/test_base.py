"""Tests for core base data classes."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.base import (
    DataContainer,
    PairedData,
    create_paired_dataset,
    validate_dataset_geometry,
)
from davinci_monet.core.exceptions import DataValidationError, VariableNotFoundError
from davinci_monet.core.protocols import DataGeometry


# Concrete implementation for testing abstract DataContainer
class ConcreteContainer(DataContainer):
    """Concrete implementation of DataContainer for testing."""

    _geometry: DataGeometry = DataGeometry.POINT

    @property
    def geometry(self) -> DataGeometry:
        return self._geometry

    def _copy_with_data(self, data: xr.Dataset) -> ConcreteContainer:
        return ConcreteContainer(
            data=data,
            label=self.label,
            variables=self.variables.copy(),
            variable_mapping=dict(self.variable_mapping),
        )


class TestDataContainer:
    """Tests for DataContainer base class."""

    @pytest.fixture
    def sample_dataset(self) -> xr.Dataset:
        """Create a sample dataset for testing."""
        times = np.arange(10)
        sites = np.arange(5)
        return xr.Dataset(
            {
                "temperature": (["time", "site"], np.random.randn(10, 5)),
                "pressure": (["time", "site"], np.random.randn(10, 5) + 1000),
            },
            coords={
                "time": times,
                "site": sites,
                "lat": ("site", np.linspace(30, 40, 5)),
                "lon": ("site", np.linspace(-100, -90, 5)),
            },
        )

    def test_is_loaded_false_when_no_data(self) -> None:
        """Test is_loaded returns False when no data."""
        container = ConcreteContainer()
        assert container.is_loaded is False

    def test_is_loaded_true_when_data_present(self, sample_dataset: xr.Dataset) -> None:
        """Test is_loaded returns True when data present."""
        container = ConcreteContainer(data=sample_dataset)
        assert container.is_loaded is True

    def test_get_variable_direct(self, sample_dataset: xr.Dataset) -> None:
        """Test getting variable by direct name."""
        container = ConcreteContainer(data=sample_dataset, label="test")
        temp = container.get_variable("temperature")
        assert temp is not None
        assert temp.name == "temperature"

    def test_get_variable_mapped(self, sample_dataset: xr.Dataset) -> None:
        """Test getting variable through mapping."""
        container = ConcreteContainer(
            data=sample_dataset,
            label="test",
            variable_mapping={"temp": "temperature"},
        )
        temp = container.get_variable("temp")
        assert temp is not None

    def test_get_variable_not_found(self, sample_dataset: xr.Dataset) -> None:
        """Test VariableNotFoundError when variable missing."""
        container = ConcreteContainer(data=sample_dataset, label="test")
        with pytest.raises(VariableNotFoundError):
            container.get_variable("nonexistent")

    def test_has_variable_true(self, sample_dataset: xr.Dataset) -> None:
        """Test has_variable returns True for existing variable."""
        container = ConcreteContainer(data=sample_dataset)
        assert container.has_variable("temperature") is True

    def test_has_variable_false(self, sample_dataset: xr.Dataset) -> None:
        """Test has_variable returns False for missing variable."""
        container = ConcreteContainer(data=sample_dataset)
        assert container.has_variable("nonexistent") is False

    def test_available_variables(self, sample_dataset: xr.Dataset) -> None:
        """Test listing available variables."""
        container = ConcreteContainer(data=sample_dataset)
        vars_list = container.available_variables
        assert "temperature" in vars_list
        assert "pressure" in vars_list

    def test_time_range(self, sample_dataset: xr.Dataset) -> None:
        """Test getting time range."""
        container = ConcreteContainer(data=sample_dataset)
        time_range = container.time_range
        assert time_range is not None
        assert time_range[0] == 0
        assert time_range[1] == 9

    def test_bounds(self, sample_dataset: xr.Dataset) -> None:
        """Test getting geographic bounds."""
        container = ConcreteContainer(data=sample_dataset)
        bounds = container.bounds
        assert bounds is not None
        lon_min, lon_max, lat_min, lat_max = bounds
        assert lat_min == pytest.approx(30.0)
        assert lat_max == pytest.approx(40.0)
        assert lon_min == pytest.approx(-100.0)
        assert lon_max == pytest.approx(-90.0)

    def test_subset_time(self, sample_dataset: xr.Dataset) -> None:
        """Test subsetting by time."""
        container = ConcreteContainer(data=sample_dataset, label="test")
        subset = container.subset_time(start=2, end=5)  # type: ignore[arg-type]
        assert subset.data is not None
        assert len(subset.data["time"]) == 4

    def test_apply_unit_scale_multiply(self, sample_dataset: xr.Dataset) -> None:
        """Test unit scaling with multiplication."""
        container = ConcreteContainer(data=sample_dataset.copy(deep=True))
        original = container.data["temperature"].values.copy()  # type: ignore
        container.apply_unit_scale("temperature", 2.0, "*")
        np.testing.assert_array_almost_equal(
            container.data["temperature"].values,  # type: ignore
            original * 2.0,
        )

    def test_apply_unit_scale_add(self, sample_dataset: xr.Dataset) -> None:
        """Test unit scaling with addition."""
        container = ConcreteContainer(data=sample_dataset.copy(deep=True))
        original = container.data["temperature"].values.copy()  # type: ignore
        container.apply_unit_scale("temperature", 273.15, "+")
        np.testing.assert_array_almost_equal(
            container.data["temperature"].values,  # type: ignore
            original + 273.15,
        )

    def test_rename_variable(self, sample_dataset: xr.Dataset) -> None:
        """Test renaming a variable."""
        container = ConcreteContainer(data=sample_dataset.copy(deep=True))
        container.rename_variable("temperature", "temp_k")
        assert "temp_k" in container.data  # type: ignore
        assert "temperature" not in container.data  # type: ignore

    def test_apply_mask(self, sample_dataset: xr.Dataset) -> None:
        """Test applying mask to variable."""
        container = ConcreteContainer(data=sample_dataset.copy(deep=True))
        # Add some values outside the mask range
        container.data["temperature"].values[0, 0] = 100.0  # type: ignore
        container.apply_mask("temperature", min_val=-10.0, max_val=10.0)
        assert np.isnan(container.data["temperature"].values[0, 0])  # type: ignore


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


class TestCreatePairedDataset:
    """Tests for create_paired_dataset function."""

    def test_basic_pairing(self) -> None:
        """Test basic paired dataset creation."""
        obs = xr.Dataset(
            {"ozone": (["time", "site"], np.random.randn(10, 5))},
            coords={"time": np.arange(10), "site": np.arange(5)},
        )
        model = xr.Dataset(
            {"o3": (["time", "site"], np.random.randn(10, 5))},
            coords={"time": np.arange(10), "site": np.arange(5)},
        )
        paired = create_paired_dataset(obs, model, obs_vars=["ozone"], model_vars=["o3"])
        assert "obs_ozone" in paired
        assert "model_ozone" in paired

    def test_mismatched_vars_raises(self) -> None:
        """Test mismatched variable lists raises error."""
        obs = xr.Dataset({"ozone": (["time"], np.random.randn(10))})
        model = xr.Dataset({"o3": (["time"], np.random.randn(10))})
        with pytest.raises(DataValidationError):
            create_paired_dataset(obs, model, obs_vars=["ozone", "pm25"], model_vars=["o3"])

    def test_custom_prefixes(self) -> None:
        """Test custom prefixes."""
        obs = xr.Dataset({"ozone": (["time"], np.random.randn(10))})
        model = xr.Dataset({"o3": (["time"], np.random.randn(10))})
        paired = create_paired_dataset(
            obs,
            model,
            obs_vars=["ozone"],
            model_vars=["o3"],
            prefix_obs="observation_",
            prefix_model="simulation_",
        )
        assert "observation_ozone" in paired
        assert "simulation_ozone" in paired
