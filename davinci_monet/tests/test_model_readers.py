"""Tests for model reader implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.core.exceptions import DataNotFoundError
from davinci_monet.core.registry import source_registry as model_registry
from davinci_monet.models import (
    CESMFVReader,
    CESMSEReader,
    CMAQReader,
    GenericReader,
    RRFSReader,
    UFSReader,
    WRFChemReader,
)

# =============================================================================
# Fixtures for synthetic model data
# =============================================================================


@pytest.fixture
def cmaq_dataset() -> xr.Dataset:
    """Create a synthetic CMAQ-style dataset."""
    times = pd.date_range("2024-01-01", periods=24, freq="h")
    layers = np.arange(35)
    rows = np.arange(100)
    cols = np.arange(120)

    # Create 2D lat/lon
    lat = 30 + 0.1 * rows[:, np.newaxis] + np.zeros((100, 120))
    lon = -120 + 0.1 * cols + np.zeros((100, 1))

    return xr.Dataset(
        {
            "O3": (["TSTEP", "LAY", "ROW", "COL"], np.random.rand(24, 35, 100, 120)),
            "PM25_TOT": (["TSTEP", "LAY", "ROW", "COL"], np.random.rand(24, 35, 100, 120)),
            "NO2": (["TSTEP", "LAY", "ROW", "COL"], np.random.rand(24, 35, 100, 120)),
        },
        coords={
            "TSTEP": times,
            "LAY": layers,
            "ROW": rows,
            "COL": cols,
            "latitude": (["ROW", "COL"], lat),
            "longitude": (["ROW", "COL"], lon),
        },
    )


@pytest.fixture
def wrfchem_dataset() -> xr.Dataset:
    """Create a synthetic WRF-Chem-style dataset."""
    times = pd.date_range("2024-01-01", periods=24, freq="h")
    levels = np.arange(35)
    y = np.arange(100)
    x = np.arange(120)

    lat = 30 + 0.1 * y[:, np.newaxis] + np.zeros((100, 120))
    lon = -120 + 0.1 * x + np.zeros((100, 1))

    return xr.Dataset(
        {
            "o3": (
                ["Time", "bottom_top", "south_north", "west_east"],
                np.random.rand(24, 35, 100, 120),
            ),
            "PM2_5_DRY": (
                ["Time", "bottom_top", "south_north", "west_east"],
                np.random.rand(24, 35, 100, 120),
            ),
            "T2": (["Time", "south_north", "west_east"], 280 + 20 * np.random.rand(24, 100, 120)),
        },
        coords={
            "Time": times,
            "bottom_top": levels,
            "south_north": y,
            "west_east": x,
            "XLAT": (["south_north", "west_east"], lat),
            "XLONG": (["south_north", "west_east"], lon),
        },
    )


@pytest.fixture
def ufs_dataset() -> xr.Dataset:
    """Create a synthetic UFS-style dataset."""
    times = pd.date_range("2024-01-01", periods=6, freq="h")
    levels = np.arange(65)
    lat = np.linspace(20, 55, 100)
    lon = np.linspace(-130, -60, 200)

    return xr.Dataset(
        {
            "o3": (["time", "pfull", "grid_yt", "grid_xt"], np.random.rand(6, 65, 100, 200)),
            "pm25": (["time", "grid_yt", "grid_xt"], np.random.rand(6, 100, 200)),
            "tmp2m": (["time", "grid_yt", "grid_xt"], 280 + 20 * np.random.rand(6, 100, 200)),
        },
        coords={
            "time": times,
            "pfull": levels,
            "grid_yt": lat,
            "grid_xt": lon,
        },
    )


@pytest.fixture
def cesm_fv_dataset() -> xr.Dataset:
    """Create a synthetic CESM-FV-style dataset."""
    times = pd.date_range("2024-01-01", periods=31, freq="D")
    levels = np.arange(32)
    lat = np.linspace(-90, 90, 192)
    lon = np.linspace(0, 360, 288)

    return xr.Dataset(
        {
            "O3": (["time", "lev", "lat", "lon"], np.random.rand(31, 32, 192, 288)),
            "T": (["time", "lev", "lat", "lon"], 200 + 100 * np.random.rand(31, 32, 192, 288)),
            "PS": (["time", "lat", "lon"], 100000 + 5000 * np.random.rand(31, 192, 288)),
        },
        coords={
            "time": times,
            "lev": levels,
            "lat": lat,
            "lon": lon,
        },
    )


@pytest.fixture
def generic_dataset() -> xr.Dataset:
    """Create a generic NetCDF-style dataset."""
    times = pd.date_range("2024-01-01", periods=10, freq="D")
    lat = np.linspace(-90, 90, 50)
    lon = np.linspace(-180, 180, 100)

    return xr.Dataset(
        {
            "temperature": (["time", "lat", "lon"], 280 + 20 * np.random.rand(10, 50, 100)),
            "precipitation": (["time", "lat", "lon"], np.random.rand(10, 50, 100)),
        },
        coords={
            "time": times,
            "lat": lat,
            "lon": lon,
        },
    )


@pytest.fixture
def temp_netcdf_file(tmp_path: Path, generic_dataset: xr.Dataset) -> Path:
    """Create a temporary NetCDF file for testing."""
    file_path = tmp_path / "test_model.nc"
    generic_dataset.to_netcdf(file_path)
    return file_path


# =============================================================================
# Tests for Model Registry
# =============================================================================


class TestModelRegistry:
    """Tests for model registry functionality."""

    def test_cmaq_registered(self) -> None:
        """Test that CMAQ reader is registered."""
        reader_cls = model_registry.get("cmaq")
        reader = reader_cls()
        assert isinstance(reader, CMAQReader)

    def test_wrfchem_registered(self) -> None:
        """Test that WRF-Chem reader is registered."""
        reader_cls = model_registry.get("wrfchem")
        reader = reader_cls()
        assert isinstance(reader, WRFChemReader)

    def test_ufs_registered(self) -> None:
        """Test that UFS reader is registered."""
        reader_cls = model_registry.get("ufs")
        reader = reader_cls()
        assert isinstance(reader, UFSReader)

    def test_rrfs_registered(self) -> None:
        """Test that RRFS reader is registered (alias for UFS)."""
        reader_cls = model_registry.get("rrfs")
        reader = reader_cls()
        assert isinstance(reader, RRFSReader)

    def test_cesm_fv_registered(self) -> None:
        """Test that CESM-FV reader is registered."""
        reader_cls = model_registry.get("cesm_fv")
        reader = reader_cls()
        assert isinstance(reader, CESMFVReader)

    def test_cesm_se_registered(self) -> None:
        """Test that CESM-SE reader is registered."""
        reader_cls = model_registry.get("cesm_se")
        reader = reader_cls()
        assert isinstance(reader, CESMSEReader)

    def test_generic_registered(self) -> None:
        """Test that generic reader is registered."""
        reader_cls = model_registry.get("generic")
        reader = reader_cls()
        assert isinstance(reader, GenericReader)


# =============================================================================
# Tests for CMAQReader
# =============================================================================


class TestCMAQReader:
    """Tests for CMAQ model reader."""

    def test_name_property(self) -> None:
        """Test reader name property."""
        reader = CMAQReader()
        assert reader.name == "cmaq"

    def test_variable_mapping(self) -> None:
        """Test variable mapping contains expected entries."""
        reader = CMAQReader()
        mapping = reader.get_variable_mapping()
        assert "ozone" in mapping
        assert mapping["ozone"] == "O3"
        assert "pm25" in mapping
        assert mapping["pm25"] == "PM25_TOT"

    def test_standardize_dataset(self, cmaq_dataset: xr.Dataset) -> None:
        """Test that CMAQ dimensions are standardized."""
        reader = CMAQReader()
        standardized = reader._standardize_dataset(cmaq_dataset)

        # Check dimension renames
        assert "time" in standardized.dims
        assert "z" in standardized.dims
        assert "y" in standardized.dims
        assert "x" in standardized.dims

    def test_open_missing_files(self) -> None:
        """Test that opening missing files raises error."""
        reader = CMAQReader()
        with pytest.raises(DataNotFoundError):
            reader.open(["/nonexistent/file.nc"])


# =============================================================================
# Tests for WRFChemReader
# =============================================================================


class TestWRFChemReader:
    """Tests for WRF-Chem model reader."""

    def test_name_property(self) -> None:
        """Test reader name property."""
        reader = WRFChemReader()
        assert reader.name == "wrfchem"

    def test_variable_mapping(self) -> None:
        """Test variable mapping contains expected entries."""
        reader = WRFChemReader()
        mapping = reader.get_variable_mapping()
        assert "ozone" in mapping
        assert mapping["ozone"] == "o3"
        assert "temperature" in mapping
        assert mapping["temperature"] == "T2"

    def test_standardize_dataset(self, wrfchem_dataset: xr.Dataset) -> None:
        """Test that WRF-Chem dimensions are standardized."""
        reader = WRFChemReader()
        standardized = reader._standardize_dataset(wrfchem_dataset)

        # Check dimension renames
        assert "time" in standardized.dims
        assert "z" in standardized.dims
        assert "y" in standardized.dims
        assert "x" in standardized.dims

    def test_open_falls_back_to_xarray_on_compat_error(
        self, tmp_path: Path, wrfchem_dataset: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When monetio's wrf-python path fails with NotImplementedError
        (the wrf-python ↔ newer netCDF4 incompatibility), open() falls back
        to the xarray reader, strips monetio-only kwargs, and warns."""
        f = tmp_path / "wrfout_d01_2025-08-01_00:00:00.nc"
        f.touch()

        reader = WRFChemReader()

        def fake_monetio(
            self_: object, file_paths: object, variables: object, **kw: Any
        ) -> xr.Dataset:
            raise NotImplementedError("Dataset is not picklable")

        captured: dict[str, Any] = {}

        def fake_xarray(
            self_: object, file_paths: object, variables: object, **kw: Any
        ) -> xr.Dataset:
            captured["kwargs"] = dict(kw)
            return wrfchem_dataset

        monkeypatch.setattr(WRFChemReader, "_open_with_monetio", fake_monetio)
        monkeypatch.setattr(WRFChemReader, "_open_with_xarray", fake_xarray)

        with pytest.warns(UserWarning, match="monetio WRF-Chem reader unavailable"):
            ds = reader.open([f], mech="racm_esrl_vcp", convert_to_ppb=True, foo="bar")

        # monetio-only kwargs stripped
        assert "mech" not in captured["kwargs"]
        assert "convert_to_ppb" not in captured["kwargs"]
        # other kwargs pass through
        assert captured["kwargs"].get("foo") == "bar"
        # result is the xarray fallback dataset (post-standardize)
        assert "time" in ds.dims

    def test_open_keeps_monetio_kwargs_when_monetio_succeeds(
        self, tmp_path: Path, wrfchem_dataset: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Happy-path regression: when monetio succeeds, mech and other
        monetio kwargs pass through unchanged and no warning fires."""
        f = tmp_path / "wrfout_d01_2025-08-01_00:00:00.nc"
        f.touch()

        reader = WRFChemReader()
        captured: dict[str, Any] = {}

        def fake_monetio(
            self_: object, file_paths: object, variables: object, **kw: Any
        ) -> xr.Dataset:
            captured["kwargs"] = dict(kw)
            return wrfchem_dataset

        monkeypatch.setattr(WRFChemReader, "_open_with_monetio", fake_monetio)

        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            ds = reader.open([f], mech="racm_esrl_vcp")

        assert captured["kwargs"].get("mech") == "racm_esrl_vcp"
        assert "time" in ds.dims

    def test_standardize_dataset_decodes_wrf_times_chararray(self) -> None:
        """The xarray fallback path leaves WRF time as a `Times` char-array
        variable (no coord on the time dim). _standardize_dataset must decode
        it into a datetime coord so downstream pairing can align with obs."""
        times_bytes = np.array(
            [b"2025-08-01_00:00:00", b"2025-08-01_01:00:00", b"2025-08-01_02:00:00"]
        )
        ds = xr.Dataset(
            {
                "Times": (["Time"], times_bytes),
                "o3": (
                    ["Time", "bottom_top", "south_north", "west_east"],
                    np.zeros((3, 2, 4, 5)),
                ),
            }
        )

        out = WRFChemReader()._standardize_dataset(ds)

        assert "time" in out.coords
        assert out.time.dtype == np.dtype("datetime64[ns]")
        assert str(out.time.values[0]) == "2025-08-01T00:00:00.000000000"
        assert "Times" not in out.variables  # consumed by decode

    def test_standardize_dataset_squeezes_time_dim_from_xlat_xlong(self) -> None:
        """WRF replicates XLAT/XLONG across the Time dim. lat/lon coords must
        be reduced to 2D (y, x) so the point-pairing strategy can use them
        (it only handles 1D or 2D coords, not 3D)."""
        ntime, ny, nx = 3, 4, 5
        lat = 30 + 0.1 * np.arange(ny)[:, None] + np.zeros((ny, nx))
        lon = -120 + 0.1 * np.arange(nx)[None, :] + np.zeros((ny, nx))
        # WRF stores XLAT/XLONG with Time dim (constant across time)
        lat_3d = np.broadcast_to(lat, (ntime, ny, nx)).copy()
        lon_3d = np.broadcast_to(lon, (ntime, ny, nx)).copy()

        ds = xr.Dataset(
            {
                "o3": (
                    ["Time", "bottom_top", "south_north", "west_east"],
                    np.zeros((ntime, 2, ny, nx)),
                ),
                "XLAT": (["Time", "south_north", "west_east"], lat_3d),
                "XLONG": (["Time", "south_north", "west_east"], lon_3d),
            }
        )

        out = WRFChemReader()._standardize_dataset(ds)

        assert "lat" in out.coords and "lon" in out.coords
        # lat/lon must be 2D, not 3D (time dim squeezed)
        assert out.lat.ndim == 2
        assert out.lon.ndim == 2
        assert out.lat.shape == (ny, nx)
        assert out.lon.shape == (ny, nx)

    def test_drop_uninitialized_chem_steps_drops_zero_pm_step(self) -> None:
        """The WRF-Chem reader must drop timesteps where the PM2_5_DRY
        chemistry diagnostic is identically zero across the grid.

        Background: in the operational AQ_WATCH WRF-Chem cycle, the hour-0
        wrfout file is the IC dump — written before any chemistry tendency
        step has run — so the PM2_5_DRY diagnostic is exactly zero across
        the entire CONUS grid. Pairing such a timestep against AirNow
        observations silently biases stats negative by ~1.5 µg/m³ and
        produces a visible 0-to-realistic discontinuity in timeseries plots.

        The reader detects this and drops the affected timesteps with a
        loud warning so downstream pairing/stats see only valid model fields.
        """
        # Two timesteps: t=0 has PM2_5_DRY all zero (the IC dump), t=1 has
        # realistic values.
        ny, nx, nz = 5, 8, 3
        times = np.array(["2026-05-16T00:00:00", "2026-05-16T06:00:00"], dtype="datetime64[ns]")
        pm = np.zeros((2, nz, ny, nx))
        pm[1] = 5.0  # realistic at t=1
        # Other variables stay populated at both timesteps
        o3 = np.full((2, nz, ny, nx), 40.0)

        ds = xr.Dataset(
            {
                "PM2_5_DRY": (["time", "z", "y", "x"], pm),
                "o3": (["time", "z", "y", "x"], o3),
            },
            coords={"time": times},
        )

        reader = WRFChemReader()
        with pytest.warns(UserWarning, match="identically zero"):
            out = reader._drop_uninitialized_chem_steps(ds)

        assert (
            out.sizes["time"] == 1
        ), f"Expected 1 timestep after dropping zero-PM step, got {out.sizes['time']}"
        # Remaining timestep is the realistic one
        assert float(out["PM2_5_DRY"].max()) == 5.0
        # Other variables retained for the surviving timestep
        assert float(out["o3"].max()) == 40.0

    def test_drop_uninitialized_chem_steps_passthrough_when_clean(self) -> None:
        """No drop, no warning when all timesteps have populated diagnostics."""
        ny, nx, nz = 5, 8, 3
        times = np.array(["2026-05-16T06:00:00", "2026-05-16T12:00:00"], dtype="datetime64[ns]")
        pm = np.full((2, nz, ny, nx), 5.0)
        ds = xr.Dataset(
            {"PM2_5_DRY": (["time", "z", "y", "x"], pm)},
            coords={"time": times},
        )

        reader = WRFChemReader()
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            out = reader._drop_uninitialized_chem_steps(ds)

        assert out.sizes["time"] == 2

    def test_drop_uninitialized_chem_steps_no_diagnostic_present(self) -> None:
        """If no chemistry diagnostic is present in the dataset, no-op."""
        ny, nx = 5, 8
        ds = xr.Dataset(
            {"T2": (["time", "y", "x"], np.full((2, ny, nx), 280.0))},
            coords={
                "time": np.array(
                    ["2026-05-16T00:00:00", "2026-05-16T06:00:00"], dtype="datetime64[ns]"
                )
            },
        )
        out = WRFChemReader()._drop_uninitialized_chem_steps(ds)
        assert out.sizes["time"] == 2


# =============================================================================
# Tests for UFSReader
# =============================================================================


class TestUFSReader:
    """Tests for UFS model reader."""

    def test_name_property(self) -> None:
        """Test reader name property."""
        reader = UFSReader()
        assert reader.name == "ufs"

    def test_rrfs_alias(self) -> None:
        """Test RRFS reader is an alias for UFS."""
        reader = RRFSReader()
        assert reader.name == "rrfs"
        assert isinstance(reader, UFSReader)

    def test_variable_mapping(self) -> None:
        """Test variable mapping contains expected entries."""
        reader = UFSReader()
        mapping = reader.get_variable_mapping()
        assert "ozone" in mapping
        assert mapping["ozone"] == "o3"
        assert "pm25" in mapping

    def test_standardize_dataset(self, ufs_dataset: xr.Dataset) -> None:
        """Test that UFS dimensions are standardized."""
        reader = UFSReader()
        standardized = reader._standardize_dataset(ufs_dataset)

        # Check dimension renames
        assert "z" in standardized.dims
        assert "y" in standardized.dims
        assert "x" in standardized.dims


# =============================================================================
# Tests for CESMFVReader
# =============================================================================


class TestCESMFVReader:
    """Tests for CESM-FV model reader."""

    def test_name_property(self) -> None:
        """Test reader name property."""
        reader = CESMFVReader()
        assert reader.name == "cesm_fv"

    def test_variable_mapping(self) -> None:
        """Test variable mapping contains expected entries."""
        reader = CESMFVReader()
        mapping = reader.get_variable_mapping()
        assert "ozone" in mapping
        assert mapping["ozone"] == "O3"
        assert "temperature" in mapping
        assert mapping["temperature"] == "T"

    def test_standardize_dataset(self, cesm_fv_dataset: xr.Dataset) -> None:
        """Test that CESM-FV dimensions are standardized."""
        reader = CESMFVReader()
        standardized = reader._standardize_dataset(cesm_fv_dataset)

        # Check lev renamed to z
        assert "z" in standardized.dims


# =============================================================================
# Tests for CESMSEReader
# =============================================================================


class TestCESMSEReader:
    """Tests for CESM-SE model reader."""

    def test_name_property(self) -> None:
        """Test reader name property."""
        reader = CESMSEReader()
        assert reader.name == "cesm_se"

    def test_variable_mapping(self) -> None:
        """Test variable mapping returns CESM mapping."""
        reader = CESMSEReader()
        mapping = reader.get_variable_mapping()
        assert "ozone" in mapping


# =============================================================================
# Tests for GenericReader
# =============================================================================


class TestGenericReader:
    """Tests for generic model reader."""

    def test_name_property(self) -> None:
        """Test reader name property."""
        reader = GenericReader()
        assert reader.name == "generic"

    def test_variable_mapping_empty(self) -> None:
        """Test that generic reader has empty variable mapping."""
        reader = GenericReader()
        mapping = reader.get_variable_mapping()
        assert len(mapping) == 0

    def test_open_netcdf_file(self, temp_netcdf_file: Path) -> None:
        """Test opening a NetCDF file."""
        reader = GenericReader()
        ds = reader.open([temp_netcdf_file])

        assert "temperature" in ds.data_vars
        assert "time" in ds.dims
        assert "lat" in ds.dims
        assert "lon" in ds.dims

    def test_open_with_variable_selection(self, temp_netcdf_file: Path) -> None:
        """Test opening with variable selection."""
        reader = GenericReader()
        ds = reader.open([temp_netcdf_file], variables=["temperature"])

        assert "temperature" in ds.data_vars
        assert "precipitation" not in ds.data_vars

    def test_open_missing_files(self) -> None:
        """Test that opening missing files raises error."""
        reader = GenericReader()
        with pytest.raises(DataNotFoundError):
            reader.open(["/nonexistent/file.nc"])

    def test_detect_engine_netcdf(self, tmp_path: Path) -> None:
        """Test engine detection for NetCDF files."""
        reader = GenericReader()
        nc_path = tmp_path / "test.nc"
        assert reader._detect_engine(nc_path) == "netcdf4"

    def test_detect_engine_grib(self, tmp_path: Path) -> None:
        """Test engine detection for grib files."""
        reader = GenericReader()
        grib_path = tmp_path / "test.grib2"
        assert reader._detect_engine(grib_path) == "cfgrib"

    def test_standardize_dimensions(self) -> None:
        """Test dimension standardization with aliases."""
        reader = GenericReader()

        # Create dataset with non-standard dimension names
        ds = xr.Dataset(
            {"temp": (["Time", "level", "latitude", "longitude"], np.random.rand(5, 10, 20, 30))},
            coords={
                "Time": pd.date_range("2024-01-01", periods=5),
                "level": np.arange(10),
                "latitude": np.linspace(-90, 90, 20),
                "longitude": np.linspace(-180, 180, 30),
            },
        )

        standardized = reader._standardize_dataset(ds)

        # Check that aliases were standardized
        assert "time" in standardized.dims
        assert "z" in standardized.dims
        assert "lat" in standardized.dims
        assert "lon" in standardized.dims


# =============================================================================
# Tests for the generic reader open() entry point
# =============================================================================


class TestGenericReaderOpen:
    """Generic reader ``open()`` returns a plain ``xr.Dataset``.

    The universal ``open_model`` convenience function was removed; sources now
    load through registered reader classes' ``open()``.
    """

    def test_open_returns_dataset(self, temp_netcdf_file: Path) -> None:
        """Opening a file with GenericReader yields a populated Dataset."""
        ds = GenericReader().open([temp_netcdf_file])

        assert isinstance(ds, xr.Dataset)
        assert len(ds.data_vars) > 0

    def test_open_missing_pattern(self, tmp_path: Path) -> None:
        """Test that a missing file path raises DataNotFoundError."""
        with pytest.raises(DataNotFoundError):
            GenericReader().open([tmp_path / "nonexistent.nc"])


# =============================================================================
# Integration tests
# =============================================================================


class TestModelReaderRoundTrip:
    """Round-trip tests for model readers (calls internal APIs directly)."""

    def test_full_workflow_generic(self, temp_netcdf_file: Path) -> None:
        """Test opening through the generic reader yields usable data."""
        ds = GenericReader().open([temp_netcdf_file])

        # Verify data is loaded as a Dataset.
        assert isinstance(ds, xr.Dataset)
        assert len(ds.data_vars) > 0

        # Time subsetting is a plain xarray operation on the returned Dataset.
        subset = ds.sel(time=slice(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05")))
        assert len(subset.time) == 5

    def test_reader_standardization(self, cmaq_dataset: xr.Dataset) -> None:
        """Test that reader standardizes coordinates properly."""
        reader = CMAQReader()
        standardized = reader._standardize_dataset(cmaq_dataset)

        # Verify lat/lon are available as coordinates
        assert "lat" in standardized.coords or "latitude" in standardized.coords
        assert "lon" in standardized.coords or "longitude" in standardized.coords
