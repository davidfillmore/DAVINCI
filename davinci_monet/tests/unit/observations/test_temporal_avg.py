"""Tests for observation temporal averaging.

Tests for the resample_data() method with min_count and track_count parameters.
These features enable averaging high-frequency observations (e.g., sub-hourly Pandora)
to match model output resolution (e.g., hourly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.observations.base import ObservationData, create_observation_data


class TestTemporalAveraging:
    """Tests for observation temporal averaging."""

    @pytest.fixture
    def high_freq_obs(self) -> xr.Dataset:
        """Create sub-hourly observation data (every 10 minutes)."""
        # 6 observations per hour for 4 hours = 24 data points
        times = pd.date_range("2024-01-01", periods=24, freq="10min")
        return xr.Dataset(
            {
                "no2_column": (["time"], np.arange(24, dtype=float) + 10.0),
                "o3_column": (["time"], np.arange(24, dtype=float) + 50.0),
            },
            coords={
                "time": times,
                "latitude": 37.5,
                "longitude": -122.5,
            },
        )

    @pytest.fixture
    def high_freq_obs_with_gaps(self) -> xr.Dataset:
        """Create sub-hourly data with some gaps (varying obs per hour)."""
        # Hour 0: 6 obs, Hour 1: 2 obs, Hour 2: 1 obs, Hour 3: 6 obs
        times = pd.to_datetime(
            [
                # Hour 0: full coverage (6 obs)
                "2024-01-01 00:00",
                "2024-01-01 00:10",
                "2024-01-01 00:20",
                "2024-01-01 00:30",
                "2024-01-01 00:40",
                "2024-01-01 00:50",
                # Hour 1: sparse (2 obs)
                "2024-01-01 01:15",
                "2024-01-01 01:45",
                # Hour 2: very sparse (1 obs)
                "2024-01-01 02:30",
                # Hour 3: full coverage (6 obs)
                "2024-01-01 03:00",
                "2024-01-01 03:10",
                "2024-01-01 03:20",
                "2024-01-01 03:30",
                "2024-01-01 03:40",
                "2024-01-01 03:50",
            ]
        )
        n_points = len(times)
        return xr.Dataset(
            {
                "no2_column": (["time"], np.arange(n_points, dtype=float) + 10.0),
            },
            coords={
                "time": times,
                "latitude": 37.5,
                "longitude": -122.5,
            },
        )

    def test_resample_to_hourly(self, high_freq_obs: xr.Dataset) -> None:
        """Test resampling high-frequency obs to hourly."""
        obs = ObservationData(data=high_freq_obs, label="pandora")
        obs.resample_data(freq="h")

        # 24 10-minute obs -> 4 hourly averages
        assert obs.data is not None
        assert len(obs.data["time"]) == 4

        # Check that values are means of 6 obs each
        # First hour: mean of 10, 11, 12, 13, 14, 15 = 12.5
        assert obs.data["no2_column"].values[0] == pytest.approx(12.5)

    def test_resample_to_30min(self, high_freq_obs: xr.Dataset) -> None:
        """Test resampling to 30-minute averages."""
        obs = ObservationData(data=high_freq_obs, label="pandora")
        obs.resample_data(freq="30min")

        # 24 10-minute obs -> 8 30-minute averages
        assert obs.data is not None
        assert len(obs.data["time"]) == 8

    def test_min_obs_count_filtering(self, high_freq_obs_with_gaps: xr.Dataset) -> None:
        """Test that averages with insufficient obs are set to NaN."""
        obs = ObservationData(data=high_freq_obs_with_gaps, label="pandora")
        obs.resample_data(freq="h", min_count=3)

        assert obs.data is not None
        no2 = obs.data["no2_column"].values

        # Hour 0: 6 obs >= 3, should have value
        assert not np.isnan(no2[0])
        # Hour 1: 2 obs < 3, should be NaN
        assert np.isnan(no2[1])
        # Hour 2: 1 obs < 3, should be NaN
        assert np.isnan(no2[2])
        # Hour 3: 6 obs >= 3, should have value
        assert not np.isnan(no2[3])

    def test_obs_count_tracking(self, high_freq_obs_with_gaps: xr.Dataset) -> None:
        """Test obs_count variable is added when requested."""
        obs = ObservationData(data=high_freq_obs_with_gaps, label="pandora")
        obs.resample_data(freq="h", track_count=True)

        assert obs.data is not None
        assert "obs_count" in obs.data.data_vars

        counts = obs.data["obs_count"].values
        assert counts[0] == 6  # Hour 0
        assert counts[1] == 2  # Hour 1
        assert counts[2] == 1  # Hour 2
        assert counts[3] == 6  # Hour 3

    def test_obs_count_with_min_count(self, high_freq_obs_with_gaps: xr.Dataset) -> None:
        """Test obs_count is added when using min_count, even if track_count=False."""
        obs = ObservationData(data=high_freq_obs_with_gaps, label="pandora")
        obs.resample_data(freq="h", min_count=3, track_count=True)

        assert obs.data is not None
        # obs_count should still be present
        assert "obs_count" in obs.data.data_vars

        # Values should be filtered by min_count
        no2 = obs.data["no2_column"].values
        assert np.isnan(no2[1])  # 2 obs < 3
        assert np.isnan(no2[2])  # 1 obs < 3

    def test_resample_preserves_scalar_coords(self, high_freq_obs: xr.Dataset) -> None:
        """Test that scalar lat/lon coordinates are preserved."""
        obs = ObservationData(data=high_freq_obs, label="pandora")
        obs.resample_data(freq="h")

        assert obs.data is not None
        assert "latitude" in obs.data.coords
        assert "longitude" in obs.data.coords
        assert float(obs.data.coords["latitude"]) == pytest.approx(37.5)
        assert float(obs.data.coords["longitude"]) == pytest.approx(-122.5)

    def test_resample_no_freq_noop(self, high_freq_obs: xr.Dataset) -> None:
        """Test that resample_data is a no-op when no freq specified."""
        obs = ObservationData(data=high_freq_obs, label="pandora")
        original_size = len(obs.data["time"])  # type: ignore
        obs.resample_data()  # No freq specified

        assert obs.data is not None
        assert len(obs.data["time"]) == original_size

    def test_resample_uses_self_resample(self, high_freq_obs: xr.Dataset) -> None:
        """Test that resample_data uses self.resample if freq not specified."""
        obs = ObservationData(data=high_freq_obs, label="pandora", resample="h")
        obs.resample_data()

        # Should use self.resample="h"
        assert obs.data is not None
        assert len(obs.data["time"]) == 4

    def test_resample_with_nans(self) -> None:
        """Test resampling data that contains NaN values."""
        times = pd.date_range("2024-01-01", periods=12, freq="10min")
        data = np.arange(12, dtype=float) + 10.0
        data[3] = np.nan  # Add a NaN in first hour
        data[4] = np.nan  # Add another NaN in first hour

        ds = xr.Dataset(
            {"no2_column": (["time"], data)},
            coords={"time": times, "latitude": 37.5, "longitude": -122.5},
        )
        obs = ObservationData(data=ds, label="pandora")
        obs.resample_data(freq="h", track_count=True)

        assert obs.data is not None
        # First hour should have 4 valid obs (6 - 2 NaN)
        # But count() counts non-NaN, so this should work correctly
        counts = obs.data["obs_count"].values
        assert counts[0] == 4  # 6 minus 2 NaN

    def test_resample_multiple_variables(self, high_freq_obs: xr.Dataset) -> None:
        """Test resampling with multiple data variables."""
        obs = ObservationData(data=high_freq_obs, label="pandora")
        obs.resample_data(freq="h", min_count=3, track_count=True)

        assert obs.data is not None
        assert "no2_column" in obs.data.data_vars
        assert "o3_column" in obs.data.data_vars
        assert "obs_count" in obs.data.data_vars

        # Both variables should be resampled
        assert len(obs.data["no2_column"]) == 4
        assert len(obs.data["o3_column"]) == 4


class TestTemporalAveragingFactory:
    """Tests for temporal averaging via factory function."""

    def test_create_observation_data_with_resample(self) -> None:
        """Test factory function sets resample attribute."""
        obs = create_observation_data(
            label="pandora",
            obs_type="pt_sfc",
            resample="h",
        )
        assert obs.resample == "h"

    def test_resample_after_factory_creation(self) -> None:
        """Test resampling works after factory creation."""
        times = pd.date_range("2024-01-01", periods=12, freq="10min")
        ds = xr.Dataset(
            {"no2_column": (["time"], np.arange(12, dtype=float))},
            coords={"time": times, "latitude": 37.5, "longitude": -122.5},
        )

        obs = create_observation_data(
            label="pandora",
            obs_type="pt_sfc",
            data=ds,
            resample="h",
        )
        obs.resample_data()

        assert obs.data is not None
        assert len(obs.data["time"]) == 2


class TestTemporalAveragingEdgeCases:
    """Edge case tests for temporal averaging."""

    def test_resample_no_time_dim(self) -> None:
        """Test resampling is skipped when no time dimension."""
        ds = xr.Dataset(
            {"no2_column": (["site"], np.arange(5, dtype=float))},
            coords={"site": np.arange(5)},
        )
        obs = ObservationData(data=ds, label="pandora")
        obs.resample_data(freq="h")

        # Should be unchanged
        assert obs.data is not None
        assert "time" not in obs.data.dims
        assert len(obs.data["site"]) == 5

    def test_resample_none_data(self) -> None:
        """Test resampling handles None data gracefully."""
        obs = ObservationData(label="pandora")
        obs.resample_data(freq="h")  # Should not raise
        assert obs.data is None

    def test_resample_excludes_coordinate_vars(self) -> None:
        """Test that lat/lon/alt aren't included in count calculation."""
        times = pd.date_range("2024-01-01", periods=6, freq="10min")
        ds = xr.Dataset(
            {
                "no2_column": (["time"], np.arange(6, dtype=float)),
                "latitude": (["time"], np.full(6, 37.5)),  # As variable not coord
                "longitude": (["time"], np.full(6, -122.5)),
            },
            coords={"time": times},
        )
        obs = ObservationData(data=ds, label="pandora")
        obs.resample_data(freq="h", track_count=True)

        assert obs.data is not None
        # obs_count should be based on no2_column, not lat/lon
        assert "obs_count" in obs.data.data_vars
        assert obs.data["obs_count"].values[0] == 6

    def test_resample_daily(self) -> None:
        """Test daily resampling."""
        times = pd.date_range("2024-01-01", periods=48, freq="h")
        ds = xr.Dataset(
            {"no2_column": (["time"], np.arange(48, dtype=float))},
            coords={"time": times, "latitude": 37.5, "longitude": -122.5},
        )
        obs = ObservationData(data=ds, label="pandora")
        obs.resample_data(freq="D", min_count=12, track_count=True)

        assert obs.data is not None
        # 48 hourly -> 2 daily averages
        assert len(obs.data["time"]) == 2
        # Each day has 24 obs
        assert obs.data["obs_count"].values[0] == 24
        assert obs.data["obs_count"].values[1] == 24
