"""Tests for observation reader implementations.

Tests the observation reader classes using synthetic datasets.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import davinci_monet.observations.aircraft.icartt  # noqa: F401
import davinci_monet.observations.satellite.goes_l3_aod  # noqa: F401
import davinci_monet.observations.satellite.modis_l2_aod  # noqa: F401
import davinci_monet.observations.satellite.mopitt_l3_co  # noqa: F401
import davinci_monet.observations.satellite.omps_l3_o3  # noqa: F401
import davinci_monet.observations.satellite.tempo_l2_no2  # noqa: F401
import davinci_monet.observations.satellite.tropomi  # noqa: F401
import davinci_monet.observations.sonde.ozonesonde  # noqa: F401
import davinci_monet.observations.surface.aeronet  # noqa: F401
import davinci_monet.observations.surface.airnow  # noqa: F401

# Import readers to ensure they are registered
import davinci_monet.observations.surface.aqs  # noqa: F401
import davinci_monet.observations.surface.openaq  # noqa: F401
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry as observation_registry

# =============================================================================
# Helper functions for creating synthetic observation data
# =============================================================================


def create_synthetic_surface_obs(
    n_sites: int = 10,
    n_times: int = 24,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Create synthetic surface observation data."""
    if variables is None:
        variables = ["O3", "PM25", "NO2"]

    times = pd.date_range("2024-01-01", periods=n_times, freq="h")
    sites = [f"site_{i:03d}" for i in range(n_sites)]

    data_vars = {}
    for var in variables:
        data_vars[var] = (
            ["time", "x"],
            np.random.rand(n_times, n_sites) * 50,
        )

    coords = {
        "time": times,
        "x": sites,
        "lat": ("x", np.random.uniform(30, 45, n_sites)),
        "lon": ("x", np.random.uniform(-120, -70, n_sites)),
    }

    return xr.Dataset(data_vars, coords=coords)


def create_synthetic_aircraft_track(
    n_times: int = 1000,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Create synthetic aircraft track data."""
    if variables is None:
        variables = ["O3", "CO", "NO"]

    times = pd.date_range("2024-01-01 10:00", periods=n_times, freq="s")

    # Create flight path
    lat = np.linspace(35, 40, n_times) + np.random.randn(n_times) * 0.01
    lon = np.linspace(-100, -95, n_times) + np.random.randn(n_times) * 0.01
    alt = 5000 + np.sin(np.linspace(0, 4 * np.pi, n_times)) * 2000

    data_vars = {}
    for var in variables:
        data_vars[var] = ("time", np.random.rand(n_times) * 100)

    coords = {
        "time": times,
        "lat": ("time", lat),
        "lon": ("time", lon),
        "alt": ("time", alt),
    }

    return xr.Dataset(data_vars, coords=coords)


def create_synthetic_satellite_swath(
    n_scanlines: int = 100,
    n_pixels: int = 450,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Create synthetic satellite swath data."""
    if variables is None:
        variables = ["NO2", "qa_value"]

    data_vars = {}
    for var in variables:
        if var == "qa_value":
            data_vars[var] = (
                ["scanline", "pixel"],
                np.random.uniform(0, 1, (n_scanlines, n_pixels)),
            )
        else:
            data_vars[var] = (
                ["scanline", "pixel"],
                np.random.rand(n_scanlines, n_pixels) * 1e15,
            )

    # Generate lat/lon for swath
    lats = np.zeros((n_scanlines, n_pixels))
    lons = np.zeros((n_scanlines, n_pixels))
    for i in range(n_scanlines):
        lats[i, :] = np.linspace(30, 50, n_pixels) + i * 0.1
        lons[i, :] = np.linspace(-120, -70, n_pixels)

    coords = {
        "scanline": range(n_scanlines),
        "pixel": range(n_pixels),
        "lat": (["scanline", "pixel"], lats),
        "lon": (["scanline", "pixel"], lons),
    }

    return xr.Dataset(data_vars, coords=coords)


def create_synthetic_gridded_obs(
    n_lat: int = 50,
    n_lon: int = 100,
    n_times: int = 24,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Create synthetic gridded observation data."""
    if variables is None:
        variables = ["AOD"]

    times = pd.date_range("2024-01-01", periods=n_times, freq="h")
    lats = np.linspace(25, 50, n_lat)
    lons = np.linspace(-130, -65, n_lon)

    data_vars = {}
    for var in variables:
        data_vars[var] = (
            ["time", "lat", "lon"],
            np.random.rand(n_times, n_lat, n_lon),
        )

    coords = {
        "time": times,
        "lat": lats,
        "lon": lons,
    }

    return xr.Dataset(data_vars, coords=coords)


def create_synthetic_profile_obs(
    n_levels: int = 100,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Create synthetic vertical profile data."""
    if variables is None:
        variables = ["O3", "Temp", "Press"]

    data_vars = {}
    for var in variables:
        if var == "Press":
            data_vars[var] = ("level", np.linspace(1000, 10, n_levels))
        elif var == "Temp":
            data_vars[var] = ("level", np.linspace(290, 200, n_levels))
        else:
            data_vars[var] = ("level", np.random.rand(n_levels) * 10)

    coords = {
        "level": range(n_levels),
        "lat": 40.0,
        "lon": -105.0,
    }

    ds = xr.Dataset(data_vars, coords=coords)
    ds = ds.expand_dims("time")
    ds = ds.assign_coords(time=[pd.Timestamp("2024-01-01 12:00")])

    return ds


# =============================================================================
# Registry Tests
# =============================================================================


class TestObservationRegistry:
    """Test observation registry."""

    def test_surface_readers_registered(self):
        """Test that surface readers are registered."""
        assert "aqs" in observation_registry
        assert "airnow" in observation_registry
        assert "aeronet" in observation_registry
        assert "openaq" in observation_registry

    def test_aircraft_readers_registered(self):
        """Test that aircraft readers are registered."""
        assert "icartt" in observation_registry

    def test_satellite_readers_registered(self):
        """Test that satellite readers are registered."""
        assert "tropomi" in observation_registry
        assert "goes_l3_aod" in observation_registry
        assert "tempo_l2_no2" in observation_registry
        assert "modis_l2_aod" in observation_registry
        assert "mopitt_l3_co" in observation_registry
        assert "omps_l3_o3" in observation_registry

    def test_sonde_readers_registered(self):
        """Test that sonde readers are registered."""
        assert "ozonesonde" in observation_registry

    def test_get_reader_class(self):
        """Test getting reader classes from registry."""
        aqs_cls = observation_registry.get("aqs")
        assert aqs_cls is not None
        reader = aqs_cls()
        assert reader.name == "aqs"


# =============================================================================
# Surface Reader Tests
# =============================================================================


class TestAQSReader:
    """Test AQS reader."""

    def test_reader_name(self):
        """Test reader name."""
        from davinci_monet.observations.surface.aqs import AQSReader

        reader = AQSReader()
        assert reader.name == "aqs"

    def test_variable_mapping(self):
        """Test variable mapping."""
        from davinci_monet.observations.surface.aqs import AQSReader

        reader = AQSReader()
        mapping = reader.get_variable_mapping()
        assert "ozone" in mapping
        assert "pm25" in mapping

    def test_open_netcdf_file(self):
        """Test opening NetCDF file."""
        from davinci_monet.observations.surface.aqs import AQSReader

        ds = create_synthetic_surface_obs()

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            ds.to_netcdf(f.name)
            reader = AQSReader()
            result = reader.open([f.name])
            assert "O3" in result.data_vars
            Path(f.name).unlink()

    def test_standardization(self):
        """Test coordinate standardization."""
        from davinci_monet.observations.surface.aqs import AQSReader

        # Create data with latitude/longitude names
        ds = create_synthetic_surface_obs()
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            ds.to_netcdf(f.name)
            reader = AQSReader()
            result = reader.open([f.name])
            # Should be renamed to lat/lon
            assert "lat" in result.coords
            assert "lon" in result.coords
            Path(f.name).unlink()


class TestAirNowReader:
    """Test AirNow reader."""

    def test_reader_name(self):
        """Test reader name."""
        from davinci_monet.observations.surface.airnow import AirNowReader

        reader = AirNowReader()
        assert reader.name == "airnow"

    def test_open_netcdf_file(self):
        """Test opening NetCDF file."""
        from davinci_monet.observations.surface.airnow import AirNowReader

        ds = create_synthetic_surface_obs()

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            ds.to_netcdf(f.name)
            reader = AirNowReader()
            result = reader.open([f.name])
            assert "O3" in result.data_vars
            Path(f.name).unlink()


class TestAERONETReader:
    """Test AERONET reader."""

    def test_reader_name(self):
        """Test reader name."""
        from davinci_monet.observations.surface.aeronet import AERONETReader

        reader = AERONETReader()
        assert reader.name == "aeronet"

    def test_variable_mapping(self):
        """Test variable mapping."""
        from davinci_monet.observations.surface.aeronet import AERONETReader

        reader = AERONETReader()
        mapping = reader.get_variable_mapping()
        assert "aod_500" in mapping
        assert "angstrom" in mapping


class TestOpenAQReader:
    """Test OpenAQ reader."""

    def test_reader_name(self):
        """Test reader name."""
        from davinci_monet.observations.surface.openaq import OpenAQReader

        reader = OpenAQReader()
        assert reader.name == "openaq"


# =============================================================================
# Aircraft Reader Tests
# =============================================================================


class TestICARTTReader:
    """Test ICARTT reader."""

    def test_reader_name(self):
        """Test reader name."""
        from davinci_monet.observations.aircraft.icartt import ICARTTReader

        reader = ICARTTReader()
        assert reader.name == "icartt"

    def test_variable_mapping(self):
        """Test variable mapping."""
        from davinci_monet.observations.aircraft.icartt import ICARTTReader

        reader = ICARTTReader()
        mapping = reader.get_variable_mapping()
        assert "ozone" in mapping
        assert "co" in mapping

    def test_dataframe_to_dataset(self):
        """Test converting DataFrame to Dataset."""
        from davinci_monet.observations.aircraft.icartt import ICARTTReader

        # Create sample DataFrame like ICARTT output
        df = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01 10:00", periods=100, freq="s"),
                "O3": np.random.rand(100) * 100,
                "CO": np.random.rand(100) * 200,
                "Latitude": np.linspace(35, 40, 100),
                "Longitude": np.linspace(-100, -95, 100),
            }
        )

        reader = ICARTTReader()
        ds = reader._dataframe_to_dataset(df)
        assert "time" in ds.dims
        assert "O3" in ds.data_vars


# =============================================================================
# Satellite Reader Tests
# =============================================================================


class TestTROPOMIReader:
    """Test TROPOMI reader."""

    def test_reader_name(self):
        """Test reader name."""
        from davinci_monet.observations.satellite.tropomi import TROPOMIReader

        reader = TROPOMIReader()
        assert reader.name == "tropomi"

    def test_variable_mapping(self):
        """Test variable mapping."""
        from davinci_monet.observations.satellite.tropomi import TROPOMIReader

        reader = TROPOMIReader()
        mapping = reader.get_variable_mapping()
        assert "no2" in mapping
        assert "o3" in mapping

    def test_qa_filtering(self):
        """Test QA value filtering."""
        from davinci_monet.observations.satellite.tropomi import TROPOMIReader

        ds = create_synthetic_satellite_swath()

        reader = TROPOMIReader()
        result = reader._apply_qa_filter(ds, qa_threshold=0.75)

        # Values below threshold should be masked
        assert result["NO2"].isnull().any()

    def test_open_with_xarray(self):
        """Test opening file with xarray fallback."""
        from davinci_monet.observations.satellite.tropomi import TROPOMIReader

        ds = create_synthetic_satellite_swath()

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            ds.to_netcdf(f.name)
            reader = TROPOMIReader()
            # Use xarray method directly to bypass monetio
            result = reader._open_with_xarray([Path(f.name)], None)
            assert "NO2" in result.data_vars
            Path(f.name).unlink()

    def test_standardization(self):
        """Test dataset standardization."""
        from davinci_monet.observations.satellite.tropomi import TROPOMIReader

        ds = create_synthetic_satellite_swath()

        reader = TROPOMIReader()
        result = reader._standardize_dataset(ds)
        assert result.attrs.get("geometry") == DataGeometry.SWATH.value


class TestGOESL3AODReader:
    """Test GOES L3 AOD reader."""

    def test_reader_name(self):
        """Test reader name."""
        from davinci_monet.observations.satellite.goes_l3_aod import GOESL3AODReader

        reader = GOESL3AODReader()
        assert reader.name == "goes_l3_aod"

    def test_variable_mapping(self):
        """Test variable mapping."""
        from davinci_monet.observations.satellite.goes_l3_aod import GOESL3AODReader

        reader = GOESL3AODReader()
        mapping = reader.get_variable_mapping()
        assert "aod" in mapping

    def test_open_with_xarray(self):
        """Test opening file with xarray fallback."""
        from davinci_monet.observations.satellite.goes_l3_aod import GOESL3AODReader

        ds = create_synthetic_gridded_obs()

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            ds.to_netcdf(f.name)
            reader = GOESL3AODReader()
            # Use xarray method directly to bypass monetio
            result = reader._open_with_xarray([Path(f.name)], None)
            assert "AOD" in result.data_vars
            Path(f.name).unlink()

    def test_standardization(self):
        """Test dataset standardization."""
        from davinci_monet.observations.satellite.goes_l3_aod import GOESL3AODReader

        ds = create_synthetic_gridded_obs()

        reader = GOESL3AODReader()
        result = reader._standardize_dataset(ds)
        assert result.attrs.get("geometry") == DataGeometry.GRID.value

    def test_backward_compatibility_alias(self):
        """Test GOESReader alias still works."""
        from davinci_monet.observations.satellite.goes_l3_aod import GOESReader

        reader = GOESReader()
        assert reader.name == "goes_l3_aod"


class TestTEMPOL2NO2Reader:
    """Test TEMPO L2 NO2 reader."""

    def test_reader_name(self):
        """Test reader name."""
        from davinci_monet.observations.satellite.tempo_l2_no2 import TEMPOL2NO2Reader

        reader = TEMPOL2NO2Reader()
        assert reader.name == "tempo_l2_no2"

    def test_variable_mapping(self):
        """Test variable mapping."""
        from davinci_monet.observations.satellite.tempo_l2_no2 import TEMPOL2NO2Reader

        reader = TEMPOL2NO2Reader()
        mapping = reader.get_variable_mapping()
        assert "no2" in mapping

    def test_standardization(self):
        """Test dataset standardization."""
        from davinci_monet.observations.satellite.tempo_l2_no2 import TEMPOL2NO2Reader

        ds = create_synthetic_satellite_swath()
        reader = TEMPOL2NO2Reader()
        result = reader._standardize_dataset(ds)
        assert result.attrs.get("geometry") == DataGeometry.SWATH.value


class TestMODISL2AODReader:
    """Test MODIS L2 AOD reader."""

    def test_reader_name(self):
        """Test reader name."""
        from davinci_monet.observations.satellite.modis_l2_aod import MODISL2AODReader

        reader = MODISL2AODReader()
        assert reader.name == "modis_l2_aod"

    def test_variable_mapping(self):
        """Test variable mapping."""
        from davinci_monet.observations.satellite.modis_l2_aod import MODISL2AODReader

        reader = MODISL2AODReader()
        mapping = reader.get_variable_mapping()
        assert "aod" in mapping

    def test_standardization(self):
        """Test dataset standardization."""
        from davinci_monet.observations.satellite.modis_l2_aod import MODISL2AODReader

        ds = create_synthetic_satellite_swath()
        reader = MODISL2AODReader()
        result = reader._standardize_dataset(ds)
        assert result.attrs.get("geometry") == DataGeometry.SWATH.value


class TestMOPITTL3COReader:
    """Test MOPITT L3 CO reader."""

    def test_reader_name(self):
        """Test reader name."""
        from davinci_monet.observations.satellite.mopitt_l3_co import MOPITTL3COReader

        reader = MOPITTL3COReader()
        assert reader.name == "mopitt_l3_co"

    def test_variable_mapping(self):
        """Test variable mapping."""
        from davinci_monet.observations.satellite.mopitt_l3_co import MOPITTL3COReader

        reader = MOPITTL3COReader()
        mapping = reader.get_variable_mapping()
        assert "co" in mapping

    def test_standardization(self):
        """Test dataset standardization."""
        from davinci_monet.observations.satellite.mopitt_l3_co import MOPITTL3COReader

        ds = create_synthetic_gridded_obs()
        reader = MOPITTL3COReader()
        result = reader._standardize_dataset(ds)
        assert result.attrs.get("geometry") == DataGeometry.GRID.value


class TestOMPSL3O3Reader:
    """Test OMPS L3 O3 reader."""

    def test_reader_name(self):
        """Test reader name."""
        from davinci_monet.observations.satellite.omps_l3_o3 import OMPSL3O3Reader

        reader = OMPSL3O3Reader()
        assert reader.name == "omps_l3_o3"

    def test_variable_mapping(self):
        """Test variable mapping."""
        from davinci_monet.observations.satellite.omps_l3_o3 import OMPSL3O3Reader

        reader = OMPSL3O3Reader()
        mapping = reader.get_variable_mapping()
        assert "o3" in mapping

    def test_standardization(self):
        """Test dataset standardization."""
        from davinci_monet.observations.satellite.omps_l3_o3 import OMPSL3O3Reader

        ds = create_synthetic_gridded_obs()
        reader = OMPSL3O3Reader()
        result = reader._standardize_dataset(ds)
        assert result.attrs.get("geometry") == DataGeometry.GRID.value


# =============================================================================
# Sonde Reader Tests
# =============================================================================


class TestOzonesondeReader:
    """Test ozonesonde reader."""

    def test_reader_name(self):
        """Test reader name."""
        from davinci_monet.observations.sonde.ozonesonde import OzonesondeReader

        reader = OzonesondeReader()
        assert reader.name == "ozonesonde"

    def test_variable_mapping(self):
        """Test variable mapping."""
        from davinci_monet.observations.sonde.ozonesonde import OzonesondeReader

        reader = OzonesondeReader()
        mapping = reader.get_variable_mapping()
        assert "ozone" in mapping
        assert "pressure" in mapping

    def test_open_netcdf_file(self):
        """Test opening NetCDF file."""
        from davinci_monet.observations.sonde.ozonesonde import OzonesondeReader

        ds = create_synthetic_profile_obs()

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            ds.to_netcdf(f.name)
            reader = OzonesondeReader()
            result = reader.open([f.name])
            assert "O3" in result.data_vars
            assert "level" in result.dims
            Path(f.name).unlink()


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_open_aqs(self):
        """Test open_aqs function."""
        from davinci_monet.observations import open_aqs

        ds = create_synthetic_surface_obs()

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            ds.to_netcdf(f.name)
            obs = open_aqs(f.name, label="test_aqs")
            assert obs.label == "test_aqs"
            assert obs.geometry == DataGeometry.POINT
            Path(f.name).unlink()

    def test_open_ozonesonde(self):
        """Test open_ozonesonde function."""
        from davinci_monet.observations import open_ozonesonde

        ds = create_synthetic_profile_obs()

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            ds.to_netcdf(f.name)
            obs = open_ozonesonde(f.name, label="test_sonde")
            assert obs.label == "test_sonde"
            assert obs.geometry == DataGeometry.PROFILE
            Path(f.name).unlink()

    def test_create_observation_data_with_data(self):
        """Test create_observation_data with pre-loaded data."""
        from davinci_monet.observations import create_observation_data

        ds = create_synthetic_surface_obs()
        obs = create_observation_data(
            label="test_obs",
            obs_type="pt_sfc",
            data=ds,
        )
        assert obs.label == "test_obs"
        assert obs.data is not None
        assert obs.geometry == DataGeometry.POINT


# =============================================================================
# Geometry Attribute Tests
# =============================================================================


class TestGeometryAttributes:
    """Test that readers set correct geometry attributes."""

    def test_surface_geometry(self):
        """Test surface readers set POINT geometry."""
        from davinci_monet.observations.surface.aqs import AQSReader

        ds = create_synthetic_surface_obs()

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            ds.to_netcdf(f.name)
            reader = AQSReader()
            result = reader.open([f.name])
            assert result.attrs.get("geometry") == DataGeometry.POINT.value
            Path(f.name).unlink()

    def test_satellite_swath_geometry(self):
        """Test satellite readers set SWATH geometry via standardization."""
        from davinci_monet.observations.satellite.tropomi import TROPOMIReader

        ds = create_synthetic_satellite_swath()
        reader = TROPOMIReader()
        result = reader._standardize_dataset(ds)
        assert result.attrs.get("geometry") == DataGeometry.SWATH.value

    def test_gridded_geometry(self):
        """Test gridded readers set GRID geometry via standardization."""
        from davinci_monet.observations.satellite.goes_l3_aod import GOESL3AODReader

        ds = create_synthetic_gridded_obs()
        reader = GOESL3AODReader()
        result = reader._standardize_dataset(ds)
        assert result.attrs.get("geometry") == DataGeometry.GRID.value

    def test_profile_geometry(self):
        """Test profile readers set PROFILE geometry."""
        from davinci_monet.observations.sonde.ozonesonde import OzonesondeReader

        ds = create_synthetic_profile_obs()

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            ds.to_netcdf(f.name)
            reader = OzonesondeReader()
            result = reader.open([f.name])
            assert result.attrs.get("geometry") == DataGeometry.PROFILE.value
            Path(f.name).unlink()
