"""Tests for observations base class."""

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
from davinci_monet.observations.base import (
    GriddedObservation,
    ObservationData,
    PointObservation,
    ProfileObservation,
    SwathObservation,
    TrackObservation,
    create_observation_data,
)


class TestObservationData:
    """Tests for ObservationData class."""

    @pytest.fixture
    def point_obs_data(self) -> xr.Dataset:
        """Create sample point observation data."""
        return xr.Dataset(
            {
                "ozone": (["time", "site"], np.random.randn(24, 10) + 40),
                "pm25": (["time", "site"], np.random.randn(24, 10) + 10),
            },
            coords={
                "time": np.arange(24),
                "site": np.arange(10),
                "lat": ("site", np.linspace(30, 40, 10)),
                "lon": ("site", np.linspace(-100, -90, 10)),
            },
        )

    @pytest.fixture
    def track_obs_data(self) -> xr.Dataset:
        """Create sample track observation data."""
        n_points = 100
        return xr.Dataset(
            {
                "ozone": (["time"], np.random.randn(n_points) + 50),
                "altitude": (["time"], np.linspace(0, 10000, n_points)),
            },
            coords={
                "time": np.arange(n_points),
                "lat": ("time", np.linspace(30, 40, n_points)),
                "lon": ("time", np.linspace(-100, -90, n_points)),
            },
        )

    def test_default_geometry_is_point(self) -> None:
        """Test default geometry is POINT."""
        obs = ObservationData()
        assert obs.geometry == DataGeometry.POINT

    def test_geometry_setter(self) -> None:
        """Test geometry can be set."""
        obs = ObservationData()
        obs.geometry = DataGeometry.TRACK
        assert obs.geometry == DataGeometry.TRACK

    def test_is_loaded(self, point_obs_data: xr.Dataset) -> None:
        """Test is_loaded property."""
        obs = ObservationData()
        assert obs.is_loaded is False

        obs.data = point_obs_data
        assert obs.is_loaded is True

    def test_obs_type(self) -> None:
        """Test obs_type attribute."""
        obs = ObservationData(obs_type="aircraft", label="firex")
        assert obs.obs_type == "aircraft"
        assert obs.label == "firex"


class TestObservationDataGeometryDetection:
    """Tests for geometry detection from obs_type."""

    @pytest.mark.parametrize(
        "obs_type,expected",
        [
            ("pt_sfc", DataGeometry.POINT),
            ("surface", DataGeometry.POINT),
            ("ground", DataGeometry.POINT),
            ("airnow", DataGeometry.POINT),
            ("aircraft", DataGeometry.TRACK),
            ("mobile", DataGeometry.TRACK),
            ("ship", DataGeometry.TRACK),
            ("sonde", DataGeometry.PROFILE),
            ("ozonesonde", DataGeometry.PROFILE),
            ("satellite", DataGeometry.SWATH),
            ("l2", DataGeometry.SWATH),
            ("gridded", DataGeometry.GRID),
            ("l3", DataGeometry.GRID),
            ("unknown", DataGeometry.POINT),  # default
        ],
    )
    def test_geometry_from_obs_type(self, obs_type: str, expected: DataGeometry) -> None:
        """Test geometry detection from obs_type string."""
        assert ObservationData.geometry_from_obs_type(obs_type) == expected


class TestObservationDataProcessing:
    """Tests for observation data processing methods."""

    @pytest.fixture
    def obs_with_data(self) -> ObservationData:
        """Create observation with sample data."""
        ds = xr.Dataset(
            {
                "ozone": (["time", "site"], np.random.randn(24, 10) + 40),
                "pm25": (["time", "site"], np.random.randn(24, 10) + 10),
                "flag": (["time", "site"], np.random.randint(0, 3, (24, 10))),
            },
            coords={
                "time": np.arange(24),
                "site": np.arange(10),
                "lat": ("site", np.linspace(30, 40, 10)),
                "lon": ("site", np.linspace(-100, -90, 10)),
            },
        )
        return ObservationData(data=ds, label="test")

    def test_apply_variable_config(self, obs_with_data: ObservationData) -> None:
        """Test applying variable configuration."""
        obs_with_data.variables = {
            "ozone": {"unit_scale": 1000.0, "unit_scale_method": "*"},
            "pm25": {"rename": "pm25_ug"},
        }
        original_ozone = obs_with_data.data["ozone"].values.copy()  # type: ignore

        obs_with_data.apply_variable_config()

        # Check scaling
        np.testing.assert_array_almost_equal(
            obs_with_data.data["ozone"].values,  # type: ignore
            original_ozone * 1000.0,
        )
        # Check renaming
        assert "pm25_ug" in obs_with_data.data  # type: ignore
        assert "pm25" not in obs_with_data.data  # type: ignore

    def test_remove_nans(self) -> None:
        """Test removing NaN values."""
        ds = xr.Dataset(
            {
                "ozone": (["time"], [1.0, np.nan, 3.0, 4.0, np.nan]),
            },
            coords={"time": np.arange(5)},
        )
        obs = ObservationData(data=ds)
        obs.remove_nans(["ozone"])
        assert len(obs.data["time"]) == 3  # type: ignore

    def test_filter_by_flag_valid(self, obs_with_data: ObservationData) -> None:
        """Test filtering by valid flags."""
        obs_with_data.filter_by_flag("flag", valid_flags=[0, 1])
        # Data should be filtered
        assert obs_with_data.data is not None

    def test_filter_by_flag_invalid(self, obs_with_data: ObservationData) -> None:
        """Test filtering by invalid flags."""
        obs_with_data.filter_by_flag("flag", invalid_flags=[2])
        assert obs_with_data.data is not None

    def test_filter_by_time(self, obs_with_data: ObservationData) -> None:
        """Test filtering by time range."""
        obs_with_data.filter_by_time(start=5, end=15)
        assert len(obs_with_data.data["time"]) == 11  # type: ignore

    def test_filter_by_bbox(self, obs_with_data: ObservationData) -> None:
        """Test filtering by bounding box."""
        obs_with_data.filter_by_bbox(lon_min=-98, lon_max=-92, lat_min=32, lat_max=38)
        # Should filter to subset of sites
        assert obs_with_data.data is not None

    def test_sum_variables(self, obs_with_data: ObservationData) -> None:
        """Test summing variables."""
        obs_with_data.sum_variables("total", ["ozone", "pm25"])
        assert "total" in obs_with_data.data  # type: ignore

    def test_sum_variables_existing_raises(self, obs_with_data: ObservationData) -> None:
        """Test summing into existing variable raises error."""
        with pytest.raises(DataValidationError):
            obs_with_data.sum_variables("ozone", ["pm25", "flag"])

    def test_n_sites(self, obs_with_data: ObservationData) -> None:
        """Test n_sites property."""
        assert obs_with_data.n_sites == 10

    def test_n_times(self, obs_with_data: ObservationData) -> None:
        """Test n_times property."""
        assert obs_with_data.n_times == 24

    def test_to_point_dataframe(self, obs_with_data: ObservationData) -> None:
        """Test conversion to DataFrame."""
        df = obs_with_data.to_point_dataframe()
        assert len(df) == 240  # 24 times * 10 sites
        assert "ozone" in df.columns


class TestObservationDataFileHandling:
    """Tests for file handling."""

    def test_resolve_files_single(self) -> None:
        """Test resolving single file."""
        with tempfile.NamedTemporaryFile(suffix=".nc") as f:
            obs = ObservationData()
            files = obs.resolve_files(f.name)
            assert len(files) == 1
            assert files[0] == Path(f.name)

    def test_resolve_files_glob(self) -> None:
        """Test resolving glob pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                path = Path(tmpdir) / f"obs_{i:02d}.nc"
                path.touch()

            obs = ObservationData()
            files = obs.resolve_files(f"{tmpdir}/obs_*.nc")
            assert len(files) == 3

    def test_resolve_files_no_match_raises(self) -> None:
        """Test no matching files raises error."""
        obs = ObservationData()
        with pytest.raises(DataNotFoundError):
            obs.resolve_files("/nonexistent/path/*.nc")


class TestCreateObservationData:
    """Tests for create_observation_data factory function."""

    def test_basic_creation(self) -> None:
        """Test basic observation data creation."""
        obs = create_observation_data(
            label="airnow",
            obs_type="pt_sfc",
        )
        assert obs.label == "airnow"
        assert obs.obs_type == "pt_sfc"
        assert obs.geometry == DataGeometry.POINT

    def test_aircraft_geometry(self) -> None:
        """Test aircraft observation gets TRACK geometry."""
        obs = create_observation_data(
            label="firex",
            obs_type="aircraft",
        )
        assert obs.geometry == DataGeometry.TRACK

    def test_with_variables(self) -> None:
        """Test creation with variable config."""
        variables = {"ozone": {"unit_scale": 1000.0}}
        obs = create_observation_data(
            label="test",
            variables=variables,
        )
        assert obs.variables["ozone"]["unit_scale"] == 1000.0

    def test_with_data_proc(self) -> None:
        """Test creation with data processing config."""
        data_proc = {"rem_obs_nan": True}
        obs = create_observation_data(
            label="test",
            data_proc=data_proc,
        )
        assert obs.data_proc["rem_obs_nan"] is True

    def test_with_resample(self) -> None:
        """Test creation with resample setting."""
        obs = create_observation_data(
            label="test",
            resample="h",
        )
        assert obs.resample == "h"


class TestSpecializedObservationClasses:
    """Tests for geometry-specific observation classes."""

    def test_point_observation(self) -> None:
        """Test PointObservation class."""
        obs = PointObservation(label="airnow")
        assert obs.geometry == DataGeometry.POINT
        assert obs.obs_type == "pt_sfc"

    def test_track_observation(self) -> None:
        """Test TrackObservation class."""
        obs = TrackObservation(label="firex")
        assert obs.geometry == DataGeometry.TRACK
        assert obs.obs_type == "aircraft"

    def test_profile_observation(self) -> None:
        """Test ProfileObservation class."""
        obs = ProfileObservation(label="ozonesonde")
        assert obs.geometry == DataGeometry.PROFILE
        assert obs.obs_type == "sonde"

    def test_swath_observation(self) -> None:
        """Test SwathObservation class."""
        obs = SwathObservation(label="tropomi")
        assert obs.geometry == DataGeometry.SWATH
        assert obs.obs_type == "satellite"

    def test_gridded_observation(self) -> None:
        """Test GriddedObservation class."""
        obs = GriddedObservation(label="merra2")
        assert obs.geometry == DataGeometry.GRID
        assert obs.obs_type == "gridded"


class TestObservationDataCopyWithData:
    """Tests for _copy_with_data method."""

    def test_copy_preserves_attributes(self) -> None:
        """Test that copy preserves all attributes."""
        ds = xr.Dataset(
            {"ozone": (["time"], np.random.randn(10))},
            coords={"time": np.arange(10)},
        )
        obs = ObservationData(
            data=ds,
            label="test",
            obs_type="pt_sfc",
            resample="h",
            time_var="datetime",
            data_proc={"rem_obs_nan": True},
        )

        subset = ds.isel(time=slice(0, 5))
        copy = obs._copy_with_data(subset)

        assert copy is not obs
        assert copy.label == "test"
        assert copy.obs_type == "pt_sfc"
        assert copy.resample == "h"
        assert copy.time_var == "datetime"
        assert copy.data_proc["rem_obs_nan"] is True
        assert len(copy.data["time"]) == 5  # type: ignore


class TestObservationDataCoordinates:
    """Tests for coordinate handling."""

    def test_add_site_coordinates(self) -> None:
        """Test adding site coordinates."""
        ds = xr.Dataset(
            {"ozone": (["time"], np.random.randn(24))},
            coords={"time": np.arange(24)},
        )
        obs = ObservationData(data=ds, label="ground_site")
        obs.add_site_coordinates(lat=35.5, lon=-97.5, alt=350.0)

        assert obs.data is not None
        assert float(obs.data.coords["latitude"]) == 35.5
        assert float(obs.data.coords["longitude"]) == -97.5
        assert float(obs.data.coords["altitude"]) == 350.0

    def test_bounds(self) -> None:
        """Test geographic bounds."""
        ds = xr.Dataset(
            {"ozone": (["time", "site"], np.random.randn(10, 5))},
            coords={
                "time": np.arange(10),
                "site": np.arange(5),
                "lat": ("site", np.array([30, 32, 35, 38, 40])),
                "lon": ("site", np.array([-100, -95, -90, -85, -80])),
            },
        )
        obs = ObservationData(data=ds)
        bounds = obs.bounds
        assert bounds is not None
        lon_min, lon_max, lat_min, lat_max = bounds
        assert lat_min == pytest.approx(30.0)
        assert lat_max == pytest.approx(40.0)
        assert lon_min == pytest.approx(-100.0)
        assert lon_max == pytest.approx(-80.0)

    def test_time_range(self) -> None:
        """Test time range."""
        ds = xr.Dataset(
            {"ozone": (["time"], np.random.randn(24))},
            coords={"time": np.arange(24)},
        )
        obs = ObservationData(data=ds)
        time_range = obs.time_range
        assert time_range is not None
        assert time_range[0] == 0
        assert time_range[1] == 23


class TestObservationDataResampling:
    """Tests for data resampling."""

    def test_resample_data(self) -> None:
        """Test resampling observation data."""
        # Create hourly data
        import pandas as pd

        times = pd.date_range("2024-01-01", periods=24, freq="h")
        ds = xr.Dataset(
            {"ozone": (["time"], np.random.randn(24) + 40)},
            coords={"time": times},
        )
        obs = ObservationData(data=ds, resample="6h")
        obs.resample_data()

        # Should have 4 6-hourly values
        assert len(obs.data["time"]) == 4  # type: ignore
