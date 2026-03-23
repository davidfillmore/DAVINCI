"""Tests for models base class."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.exceptions import (
    DataNotFoundError,
    DataValidationError,
)
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.models.base import (
    ModelData,
    create_model_data,
)


class TestModelData:
    """Tests for ModelData class."""

    @pytest.fixture
    def sample_model_data(self) -> xr.Dataset:
        """Create a sample 4D model dataset."""
        return xr.Dataset(
            {
                "O3": (
                    ["time", "z", "lat", "lon"],
                    np.random.randn(10, 5, 20, 30) + 40,
                ),
                "PM25": (
                    ["time", "z", "lat", "lon"],
                    np.random.randn(10, 5, 20, 30) + 10,
                ),
                "NO2": (
                    ["time", "z", "lat", "lon"],
                    np.random.randn(10, 5, 20, 30) + 5,
                ),
            },
            coords={
                "time": np.arange(10),
                "z": np.arange(5),
                "lat": np.linspace(30, 50, 20),
                "lon": np.linspace(-100, -70, 30),
            },
        )

    def test_geometry_is_grid(self) -> None:
        """Test that model data geometry is always GRID."""
        model = ModelData()
        assert model.geometry == DataGeometry.GRID

    def test_is_loaded(self, sample_model_data: xr.Dataset) -> None:
        """Test is_loaded property."""
        model = ModelData()
        assert model.is_loaded is False

        model.data = sample_model_data
        assert model.is_loaded is True

    def test_default_radius_of_influence(self) -> None:
        """Test default radius of influence."""
        model = ModelData()
        assert model.radius_of_influence == 12000.0

    def test_custom_radius_of_influence(self) -> None:
        """Test custom radius of influence."""
        model = ModelData(radius_of_influence=50000.0)
        assert model.radius_of_influence == 50000.0

    def test_mod_type(self) -> None:
        """Test mod_type attribute."""
        model = ModelData(mod_type="cmaq", label="test_cmaq")
        assert model.mod_type == "cmaq"
        assert model.label == "test_cmaq"

    def test_obs_mapping(self) -> None:
        """Test observation mapping."""
        mapping = {"airnow": {"O3": "OZONE", "PM25": "PM2.5"}}
        model = ModelData(obs_mapping=mapping)  # type: ignore[arg-type]
        assert model.obs_mapping["airnow"]["O3"] == "OZONE"

    def test_get_mapping_for_obs(self) -> None:
        """Test getting mapping for specific observation."""
        mapping = {"airnow": {"O3": "OZONE"}, "aeronet": {"AOD": "AOD550"}}
        model = ModelData(obs_mapping=mapping)  # type: ignore[arg-type]
        airnow_mapping = model.get_mapping_for_obs("airnow")
        assert airnow_mapping["O3"] == "OZONE"

        # Missing obs returns empty dict
        empty = model.get_mapping_for_obs("nonexistent")
        assert empty == {}

    def test_get_variable_list_for_obs(self) -> None:
        """Test getting variable list for observation."""
        mapping = {"airnow": {"O3": "OZONE", "PM25": "PM2.5"}}
        model = ModelData(obs_mapping=mapping)  # type: ignore[arg-type]
        vars_list = model.get_variable_list_for_obs("airnow")
        assert "OZONE" in vars_list
        assert "PM2.5" in vars_list

    def test_apply_variable_config(self, sample_model_data: xr.Dataset) -> None:
        """Test applying variable configuration."""
        model = ModelData(
            data=sample_model_data.copy(deep=True),
            variables={
                "O3": {"unit_scale": 1000.0, "unit_scale_method": "*"},
                "PM25": {"rename": "pm25_ug"},
            },
        )
        original_o3 = sample_model_data["O3"].values.copy()

        model.apply_variable_config()

        # Check scaling was applied
        np.testing.assert_array_almost_equal(
            model.data["O3"].values,  # type: ignore
            original_o3 * 1000.0,
        )
        # Check renaming was applied
        assert "pm25_ug" in model.data  # type: ignore
        assert "PM25" not in model.data  # type: ignore

    def test_sum_variables(self, sample_model_data: xr.Dataset) -> None:
        """Test summing variables."""
        model = ModelData(data=sample_model_data.copy(deep=True))
        model.sum_variables("NOx", ["O3", "NO2"])
        assert "NOx" in model.data  # type: ignore
        expected = sample_model_data["O3"] + sample_model_data["NO2"]
        np.testing.assert_array_almost_equal(
            model.data["NOx"].values,  # type: ignore
            expected.values,
        )

    def test_sum_variables_existing_raises(self, sample_model_data: xr.Dataset) -> None:
        """Test summing into existing variable raises error."""
        model = ModelData(data=sample_model_data.copy(deep=True))
        with pytest.raises(DataValidationError):
            model.sum_variables("O3", ["PM25", "NO2"])

    def test_sum_variables_missing_raises(self, sample_model_data: xr.Dataset) -> None:
        """Test summing missing variables raises error."""
        model = ModelData(data=sample_model_data.copy(deep=True))
        with pytest.raises(DataValidationError):
            model.sum_variables("new_var", ["O3", "nonexistent"])

    def test_extract_surface(self, sample_model_data: xr.Dataset) -> None:
        """Test extracting surface level."""
        model = ModelData(data=sample_model_data)
        surface = model.extract_surface()
        assert "z" not in surface.data.dims  # type: ignore
        assert surface.data["O3"].shape == (10, 20, 30)  # type: ignore

    def test_extract_surface_cesm_convention(self) -> None:
        """Test surface extraction with CESM-style increasing pressure levels.

        Regression test for review finding #3: CESM hybrid sigma-pressure
        coordinates have pressure increasing with index, so surface is at
        the last index (highest pressure), not the first (TOA).
        """
        # CESM-like levels: low values (TOA) to high values (surface)
        lev_vals = np.array([3.0, 10.0, 50.0, 200.0, 500.0, 1000.0])
        ds = xr.Dataset(
            {"O3": (["time", "lev", "lat", "lon"], np.random.randn(2, 6, 3, 3))},
            coords={
                "time": np.arange(2),
                "lev": lev_vals,
                "lat": np.linspace(30, 50, 3),
                "lon": np.linspace(-100, -70, 3),
            },
        )
        # Put distinct values at TOA (index 0) and surface (index -1)
        ds["O3"][:, 0, :, :] = 9999.0  # stratospheric (TOA)
        ds["O3"][:, -1, :, :] = 50.0  # surface

        model = ModelData(data=ds)
        surface = model.extract_surface(level_dim="lev")

        assert "lev" not in surface.data.dims  # type: ignore
        # Should get surface values (~50), not stratospheric (~9999)
        assert float(surface.data["O3"].mean()) == pytest.approx(50.0)  # type: ignore

    def test_extract_surface_standard_convention(self) -> None:
        """Test surface extraction with standard decreasing levels."""
        # Standard convention: values decrease with index (surface first)
        lev_vals = np.array([1000.0, 500.0, 200.0, 50.0, 10.0, 3.0])
        ds = xr.Dataset(
            {"O3": (["time", "lev", "lat", "lon"], np.random.randn(2, 6, 3, 3))},
            coords={
                "time": np.arange(2),
                "lev": lev_vals,
                "lat": np.linspace(30, 50, 3),
                "lon": np.linspace(-100, -70, 3),
            },
        )
        ds["O3"][:, 0, :, :] = 50.0  # surface (first index)
        ds["O3"][:, -1, :, :] = 9999.0  # TOA

        model = ModelData(data=ds)
        surface = model.extract_surface(level_dim="lev")

        assert "lev" not in surface.data.dims  # type: ignore
        assert float(surface.data["O3"].mean()) == pytest.approx(50.0)  # type: ignore

    def test_extract_level(self, sample_model_data: xr.Dataset) -> None:
        """Test extracting specific level."""
        model = ModelData(data=sample_model_data)
        level2 = model.extract_level(2, method="index")
        assert "z" not in level2.data.dims  # type: ignore

    def test_copy_with_data(self, sample_model_data: xr.Dataset) -> None:
        """Test _copy_with_data creates new instance."""
        model = ModelData(
            data=sample_model_data,
            label="test",
            mod_type="cmaq",
            radius_of_influence=20000.0,
        )
        subset = sample_model_data.isel(time=slice(0, 5))
        copy = model._copy_with_data(subset)

        assert copy is not model
        assert copy.label == "test"
        assert copy.mod_type == "cmaq"
        assert copy.radius_of_influence == 20000.0
        assert len(copy.data["time"]) == 5  # type: ignore

    def test_bounds(self, sample_model_data: xr.Dataset) -> None:
        """Test geographic bounds."""
        model = ModelData(data=sample_model_data)
        bounds = model.bounds
        assert bounds is not None
        lon_min, lon_max, lat_min, lat_max = bounds
        assert lat_min == pytest.approx(30.0)
        assert lat_max == pytest.approx(50.0)
        assert lon_min == pytest.approx(-100.0)
        assert lon_max == pytest.approx(-70.0)

    def test_time_range(self, sample_model_data: xr.Dataset) -> None:
        """Test time range."""
        model = ModelData(data=sample_model_data)
        time_range = model.time_range
        assert time_range is not None
        assert time_range[0] == 0
        assert time_range[1] == 9


class TestModelDataFileHandling:
    """Tests for ModelData file handling."""

    def test_resolve_files_glob(self) -> None:
        """Test resolving files from glob pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(3):
                path = Path(tmpdir) / f"model_{i:02d}.nc"
                path.touch()

            model = ModelData()
            files = model.resolve_files(f"{tmpdir}/model_*.nc")
            assert len(files) == 3

    def test_resolve_files_no_match_raises(self) -> None:
        """Test no matching files raises error."""
        model = ModelData()
        with pytest.raises(DataNotFoundError):
            model.resolve_files("/nonexistent/path/*.nc")

    def test_resolve_files_from_txt(self) -> None:
        """Test resolving files from text file list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create list file
            list_file = Path(tmpdir) / "files.txt"
            file_paths = [f"/data/model_{i}.nc" for i in range(3)]
            list_file.write_text("\n".join(file_paths))

            model = ModelData()
            files = model.resolve_files(str(list_file))
            assert len(files) == 3
            assert files[0] == Path("/data/model_0.nc")


class TestCreateModelData:
    """Tests for create_model_data factory function."""

    def test_basic_creation(self) -> None:
        """Test basic model data creation."""
        model = create_model_data(
            label="cmaq_test",
            mod_type="cmaq",
            radius_of_influence=15000.0,
        )
        assert model.label == "cmaq_test"
        assert model.mod_type == "cmaq"
        assert model.radius_of_influence == 15000.0

    def test_with_mapping(self) -> None:
        """Test creation with variable mapping."""
        mapping = {"airnow": {"O3": "OZONE"}}
        model = create_model_data(
            label="test",
            mapping=mapping,  # type: ignore[arg-type]
        )
        assert model.obs_mapping["airnow"]["O3"] == "OZONE"

    def test_with_variables(self) -> None:
        """Test creation with variable config."""
        variables = {"O3": {"unit_scale": 1000.0}}
        model = create_model_data(
            label="test",
            variables=variables,
        )
        assert model.variables["O3"]["unit_scale"] == 1000.0

    def test_with_glob_pattern(self) -> None:
        """Test creation with glob pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(2):
                path = Path(tmpdir) / f"model_{i:02d}.nc"
                path.touch()

            model = create_model_data(
                label="test",
                files=f"{tmpdir}/model_*.nc",
            )
            assert len(model.files) == 2

    def test_with_single_file(self) -> None:
        """Test creation with single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.nc"
            path.touch()

            model = create_model_data(
                label="test",
                files=str(path),
            )
            assert len(model.files) == 1
            assert model.files[0] == path


class TestModelDataVerticalInterpolation:
    """Tests for vertical interpolation methods."""

    @pytest.fixture
    def model_with_levels(self) -> ModelData:
        """Create model with vertical levels."""
        ds = xr.Dataset(
            {
                "O3": (
                    ["time", "z", "lat", "lon"],
                    np.random.randn(5, 10, 10, 10) + 40,
                ),
            },
            coords={
                "time": np.arange(5),
                "z": np.linspace(0, 5000, 10),  # 0-5000m
                "lat": np.linspace(30, 40, 10),
                "lon": np.linspace(-100, -90, 10),
            },
        )
        return ModelData(data=ds)

    def test_interpolate_vertical(self, model_with_levels: ModelData) -> None:
        """Test vertical interpolation."""
        target_levels = [0, 500, 1000, 2000]
        interp = model_with_levels.interpolate_vertical(target_levels)
        assert interp.data is not None
        assert len(interp.data["z"]) == 4


class TestModelDataHorizontalRegridding:
    """Tests for horizontal regridding."""

    @pytest.fixture
    def model_with_grid(self) -> ModelData:
        """Create model with lat/lon grid."""
        ds = xr.Dataset(
            {
                "O3": (
                    ["time", "lat", "lon"],
                    np.random.randn(5, 20, 30) + 40,
                ),
            },
            coords={
                "time": np.arange(5),
                "lat": np.linspace(30, 50, 20),
                "lon": np.linspace(-100, -70, 30),
            },
        )
        return ModelData(data=ds)

    def test_regrid_horizontal(self, model_with_grid: ModelData) -> None:
        """Test horizontal regridding."""
        target_lats = np.linspace(35, 45, 10)
        target_lons = np.linspace(-90, -80, 15)
        regridded = model_with_grid.regrid_horizontal(target_lats, target_lons, method="nearest")  # type: ignore[arg-type]
        assert regridded.data is not None
        assert len(regridded.data["lat"]) == 10
        assert len(regridded.data["lon"]) == 15
