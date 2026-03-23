"""Tests for io module (readers and writers)."""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.core.exceptions import DataFormatError, DataNotFoundError
from davinci_monet.io import (
    read_csv,
    read_csv_to_xarray,
    read_dataset,
    read_mfdataset,
    read_pickle,
    read_saved_analysis,
    write_csv,
    write_dataset,
    write_paired_data,
    write_pickle,
    write_statistics,
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
def sample_dataframe() -> pd.DataFrame:
    """Create a sample pandas DataFrame."""
    n_rows = 100

    np.random.seed(42)
    return pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=n_rows, freq="h"),
            "value": np.random.randn(n_rows),
            "category": np.random.choice(["A", "B", "C"], n_rows),
        }
    )


@pytest.fixture
def sample_paired_datasets() -> dict[str, xr.Dataset]:
    """Create sample paired datasets."""
    n_times = 50
    times = np.datetime64("2020-01-01") + np.arange(n_times) * np.timedelta64(1, "h")

    np.random.seed(42)

    return {
        "model1_obs1": xr.Dataset(
            {
                "o3_model": (["time"], np.random.randn(n_times)),
                "o3_obs": (["time"], np.random.randn(n_times)),
            },
            coords={"time": times},
        ),
        "model1_obs2": xr.Dataset(
            {
                "pm25_model": (["time"], np.random.randn(n_times)),
                "pm25_obs": (["time"], np.random.randn(n_times)),
            },
            coords={"time": times},
        ),
    }


@pytest.fixture
def sample_statistics() -> dict[str, dict[str, dict[str, float]]]:
    """Create sample statistics results."""
    return {
        "model1_obs1": {
            "o3": {
                "n": 100,
                "mean_bias": 2.5,
                "rmse": 5.0,
                "correlation": 0.85,
            },
            "no2": {
                "n": 100,
                "mean_bias": -1.0,
                "rmse": 3.0,
                "correlation": 0.92,
            },
        },
        "model1_obs2": {
            "pm25": {
                "n": 80,
                "mean_bias": 5.0,
                "rmse": 10.0,
                "correlation": 0.75,
            },
        },
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Reader Tests
# =============================================================================


class TestReadDataset:
    """Tests for read_dataset function."""

    def test_read_netcdf(self, sample_dataset: xr.Dataset, temp_dir: Path):
        """Test reading NetCDF file."""
        filepath = temp_dir / "test.nc"
        sample_dataset.to_netcdf(filepath)

        result = read_dataset(filepath)

        assert isinstance(result, xr.Dataset)
        assert "temperature" in result.data_vars
        assert result.dims["time"] == 10

    def test_read_pickle_format(self, sample_dataset: xr.Dataset, temp_dir: Path):
        """Test reading pickle file."""
        filepath = temp_dir / "test.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(sample_dataset, f)

        result = read_dataset(filepath)

        assert isinstance(result, xr.Dataset)
        assert "temperature" in result.data_vars

    def test_file_not_found(self):
        """Test error when file not found."""
        with pytest.raises(DataNotFoundError, match="File not found"):
            read_dataset("/nonexistent/path.nc")

    def test_custom_engine(self, sample_dataset: xr.Dataset, temp_dir: Path):
        """Test using custom engine."""
        filepath = temp_dir / "test.nc"
        sample_dataset.to_netcdf(filepath)

        result = read_dataset(filepath, engine="netcdf4")

        assert isinstance(result, xr.Dataset)


class TestReadMfDataset:
    """Tests for read_mfdataset function."""

    def test_read_multiple_files(self, temp_dir: Path):
        """Test reading multiple NetCDF files."""
        # Create multiple files with different time slices
        for i in range(3):
            times = np.datetime64("2020-01-01") + np.arange(10) * np.timedelta64(1, "h")
            times = times + np.timedelta64(i * 10, "h")

            ds = xr.Dataset(
                {"data": (["time"], np.arange(10))},
                coords={"time": times},
            )
            ds.to_netcdf(temp_dir / f"file_{i}.nc")

        result = read_mfdataset([str(temp_dir / "file_*.nc")])

        assert isinstance(result, xr.Dataset)
        assert result.dims["time"] == 30

    def test_no_matching_files(self):
        """Test error when no files match pattern."""
        with pytest.raises(DataNotFoundError, match="No files found"):
            read_mfdataset(["/nonexistent/*.nc"])


class TestReadPickle:
    """Tests for read_pickle function."""

    def test_read_dataset(self, sample_dataset: xr.Dataset, temp_dir: Path):
        """Test reading pickled Dataset."""
        filepath = temp_dir / "data.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(sample_dataset, f)

        result = read_pickle(filepath)

        assert isinstance(result, xr.Dataset)

    def test_read_dataframe(self, sample_dataframe: pd.DataFrame, temp_dir: Path):
        """Test reading pickled DataFrame."""
        filepath = temp_dir / "data.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(sample_dataframe, f)

        result = read_pickle(filepath)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100

    def test_read_any_object(self, temp_dir: Path):
        """Test reading any pickled object."""
        data = {"key": "value", "numbers": [1, 2, 3]}
        filepath = temp_dir / "data.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        result = read_pickle(filepath)

        assert result == data

    def test_file_not_found(self):
        """Test error when file not found."""
        with pytest.raises(DataNotFoundError, match="File not found"):
            read_pickle("/nonexistent/data.pkl")


class TestReadCsv:
    """Tests for read_csv function."""

    def test_read_simple_csv(self, sample_dataframe: pd.DataFrame, temp_dir: Path):
        """Test reading simple CSV file."""
        filepath = temp_dir / "data.csv"
        sample_dataframe.to_csv(filepath, index=False)

        result = read_csv(filepath)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100

    def test_parse_dates(self, temp_dir: Path):
        """Test date parsing in CSV."""
        df = pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=5),
                "value": [1, 2, 3, 4, 5],
            }
        )
        filepath = temp_dir / "data.csv"
        df.to_csv(filepath, index=False)

        result = read_csv(filepath, parse_dates=["time"])

        assert pd.api.types.is_datetime64_any_dtype(result["time"])

    def test_file_not_found(self):
        """Test error when file not found."""
        with pytest.raises(DataNotFoundError, match="File not found"):
            read_csv("/nonexistent/data.csv")


class TestReadCsvToXarray:
    """Tests for read_csv_to_xarray function."""

    def test_basic_conversion(self, temp_dir: Path):
        """Test basic CSV to xarray conversion."""
        df = pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=5),
                "temperature": [20, 21, 22, 23, 24],
            }
        )
        filepath = temp_dir / "data.csv"
        df.to_csv(filepath, index=False)

        result = read_csv_to_xarray(filepath, parse_dates=["time"])

        assert isinstance(result, xr.Dataset)
        assert "temperature" in result.data_vars

    def test_custom_index(self, temp_dir: Path):
        """Test with custom index columns."""
        df = pd.DataFrame(
            {
                "site": ["A", "B", "A", "B"],
                "time": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
                "value": [1, 2, 3, 4],
            }
        )
        filepath = temp_dir / "data.csv"
        df.to_csv(filepath, index=False)

        result = read_csv_to_xarray(filepath, index_columns=["site", "time"])

        assert isinstance(result, xr.Dataset)


class TestReadSavedAnalysis:
    """Tests for read_saved_analysis function."""

    def test_read_pickle_analysis(self, temp_dir: Path):
        """Test reading pickle analysis."""
        data = {"stats": {"o3": {"mean": 50}}}
        filepath = temp_dir / "analysis.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        result = read_saved_analysis(filepath)

        assert result == data

    def test_read_netcdf_analysis(self, sample_dataset: xr.Dataset, temp_dir: Path):
        """Test reading NetCDF analysis."""
        filepath = temp_dir / "analysis.nc"
        sample_dataset.to_netcdf(filepath)

        result = read_saved_analysis(filepath)

        assert "data" in result
        assert isinstance(result["data"], xr.Dataset)

    def test_auto_format_detection(self, temp_dir: Path):
        """Test automatic format detection."""
        data = {"key": "value"}

        # Test pickle
        pkl_path = temp_dir / "data.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

        result = read_saved_analysis(pkl_path, format="auto")
        assert result == data

    def test_file_not_found(self):
        """Test error when file not found."""
        with pytest.raises(DataNotFoundError, match="File not found"):
            read_saved_analysis("/nonexistent/analysis.pkl")


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

    def test_write_dataframe(self, sample_dataframe: pd.DataFrame, temp_dir: Path):
        """Test writing DataFrame to pickle."""
        filepath = temp_dir / "data.pkl"
        write_pickle(sample_dataframe, filepath)

        assert filepath.exists()

        with open(filepath, "rb") as f:
            result = pickle.load(f)
        assert isinstance(result, pd.DataFrame)

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


class TestWriteCsv:
    """Tests for write_csv function."""

    def test_write_csv(self, sample_dataframe: pd.DataFrame, temp_dir: Path):
        """Test writing CSV file."""
        filepath = temp_dir / "data.csv"
        write_csv(sample_dataframe, filepath)

        assert filepath.exists()

        result = pd.read_csv(filepath)
        assert len(result) == 100

    def test_without_index(self, sample_dataframe: pd.DataFrame, temp_dir: Path):
        """Test writing without index."""
        filepath = temp_dir / "data.csv"
        write_csv(sample_dataframe, filepath, index=False)

        result = pd.read_csv(filepath)
        assert "Unnamed: 0" not in result.columns

    def test_creates_parent_directory(self, sample_dataframe: pd.DataFrame, temp_dir: Path):
        """Test parent directory is created."""
        filepath = temp_dir / "subdir" / "data.csv"
        write_csv(sample_dataframe, filepath)

        assert filepath.exists()


class TestWritePairedData:
    """Tests for write_paired_data function."""

    def test_write_netcdf(self, sample_paired_datasets: dict[str, xr.Dataset], temp_dir: Path):
        """Test writing paired data as NetCDF."""
        output_dir = temp_dir / "paired"
        written_files = write_paired_data(sample_paired_datasets, output_dir)

        assert len(written_files) == 2
        assert all(f.endswith(".nc") for f in written_files)

        for f in written_files:
            assert Path(f).exists()

    def test_write_pickle(self, sample_paired_datasets: dict[str, xr.Dataset], temp_dir: Path):
        """Test writing paired data as pickle."""
        output_dir = temp_dir / "paired"
        written_files = write_paired_data(sample_paired_datasets, output_dir, format="pickle")

        assert len(written_files) == 2
        assert all(f.endswith(".pkl") for f in written_files)

    def test_with_prefix(self, sample_paired_datasets: dict[str, xr.Dataset], temp_dir: Path):
        """Test writing with filename prefix."""
        output_dir = temp_dir / "paired"
        written_files = write_paired_data(sample_paired_datasets, output_dir, prefix="run1")

        assert all("run1_" in f for f in written_files)

    def test_skip_none_datasets(self, temp_dir: Path):
        """Test None datasets are skipped."""
        datasets = {
            "valid": xr.Dataset({"data": (["x"], [1, 2, 3])}),
            "none": None,
        }

        output_dir = temp_dir / "paired"
        written_files = write_paired_data(datasets, output_dir)

        assert len(written_files) == 1


class TestWriteStatistics:
    """Tests for write_statistics function."""

    def test_write_csv(
        self, sample_statistics: dict[str, dict[str, dict[str, float]]], temp_dir: Path
    ):
        """Test writing statistics as CSV."""
        filepath = temp_dir / "stats.csv"
        write_statistics(sample_statistics, filepath, format="csv")

        assert filepath.exists()

        result = pd.read_csv(filepath)
        assert "pair" in result.columns
        assert "variable" in result.columns
        assert "mean_bias" in result.columns

    def test_write_json(
        self, sample_statistics: dict[str, dict[str, dict[str, float]]], temp_dir: Path
    ):
        """Test writing statistics as JSON."""
        filepath = temp_dir / "stats.json"
        write_statistics(sample_statistics, filepath, format="json")

        assert filepath.exists()

        with open(filepath) as f:
            result = json.load(f)

        assert "model1_obs1" in result

    def test_write_pickle(
        self, sample_statistics: dict[str, dict[str, dict[str, float]]], temp_dir: Path
    ):
        """Test writing statistics as pickle."""
        filepath = temp_dir / "stats.pkl"
        write_statistics(sample_statistics, filepath, format="pickle")

        assert filepath.exists()

        with open(filepath, "rb") as f:
            result = pickle.load(f)

        assert result == sample_statistics

    def test_unsupported_format(
        self, sample_statistics: dict[str, dict[str, dict[str, float]]], temp_dir: Path
    ):
        """Test error for unsupported format."""
        filepath = temp_dir / "stats.xyz"

        with pytest.raises(DataFormatError, match="Unsupported format"):
            write_statistics(sample_statistics, filepath, format="xyz")


# =============================================================================
# Round-trip Tests
# =============================================================================


class TestRoundTrip:
    """Tests for reading back written data."""

    def test_netcdf_roundtrip(self, sample_dataset: xr.Dataset, temp_dir: Path):
        """Test NetCDF write and read."""
        filepath = temp_dir / "roundtrip.nc"

        write_dataset(sample_dataset, filepath)
        result = read_dataset(filepath)

        xr.testing.assert_equal(result, sample_dataset)

    def test_pickle_roundtrip(self, sample_dataset: xr.Dataset, temp_dir: Path):
        """Test pickle write and read."""
        filepath = temp_dir / "roundtrip.pkl"

        write_pickle(sample_dataset, filepath)
        result = read_pickle(filepath)

        xr.testing.assert_equal(result, sample_dataset)

    def test_csv_roundtrip(self, temp_dir: Path):
        """Test CSV write and read."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4.0, 5.0, 6.0],
                "c": ["x", "y", "z"],
            }
        )
        filepath = temp_dir / "roundtrip.csv"

        write_csv(df, filepath, index=False)
        result = read_csv(filepath, parse_dates=False)

        pd.testing.assert_frame_equal(result, df)
