"""Tests for pre-built test scenarios."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
from davinci_monet.tests.synthetic.scenarios import (
    BiasScenario,
    MismatchScenario,
    PerfectMatchScenario,
    Scenario,
    create_scenario,
)


class TestPerfectMatchScenario:
    """Tests for PerfectMatchScenario."""

    def test_generate_returns_tuple(self) -> None:
        """Test generate returns dataset and geometry tuple."""
        scenario = PerfectMatchScenario(
            variables=["O3"],
            domain=Domain(n_lon=12, n_lat=6),
            time_config=TimeConfig(end="2024-01-01 06:00"),
        )
        dataset, geometry = scenario.generate()

        assert isinstance(dataset, xr.Dataset)
        assert isinstance(geometry, xr.Dataset)

    def test_point_geometry(self) -> None:
        """Test perfect match with point geometry."""
        scenario = PerfectMatchScenario(
            variables=["O3"],
            geometry=DataGeometry.POINT,
            n_geometry=10,
        )
        dataset, geometry = scenario.generate()

        assert "site" in geometry.dims
        assert len(geometry.site) == 10

    def test_track_geometry(self) -> None:
        """Test perfect match with track geometry."""
        scenario = PerfectMatchScenario(
            variables=["O3"],
            geometry=DataGeometry.TRACK,
            n_geometry=50,
        )
        dataset, geometry = scenario.generate()

        # Track has time dimension with coords
        assert "latitude" in geometry.coords
        assert "longitude" in geometry.coords

    def test_expected_statistics(self) -> None:
        """Test expected statistics for zero noise."""
        scenario = PerfectMatchScenario(
            variables=["O3"],
            noise_level=0.0,
        )
        stats = scenario.expected_statistics

        assert "O3" in stats
        assert stats["O3"]["MB"] == 0.0
        assert stats["O3"]["RMSE"] == 0.0
        assert stats["O3"]["R2"] == 1.0

    def test_with_noise(self) -> None:
        """Test that noise_level adds variability."""
        scenario = PerfectMatchScenario(
            variables=["O3"],
            geometry=DataGeometry.POINT,
            noise_level=0.1,
            n_geometry=20,
        )
        stats = scenario.expected_statistics

        # With noise, R2 should be less than 1
        assert stats["O3"]["R2"] < 1.0


class TestBiasScenario:
    """Tests for BiasScenario."""

    def test_additive_bias(self) -> None:
        """Test additive bias is applied."""
        scenario = BiasScenario(
            variables=["O3"],
            bias=10.0,
            relative_bias=False,
        )
        dataset, geometry = scenario.generate()

        # Dataset should have the bias added
        assert "O3" in dataset
        assert "O3" in geometry

    def test_expected_statistics_additive(self) -> None:
        """Test expected statistics for additive bias."""
        scenario = BiasScenario(
            variables=["O3"],
            bias=5.0,
            relative_bias=False,
        )
        stats = scenario.expected_statistics

        assert stats["O3"]["MB"] == 5.0

    def test_relative_bias(self) -> None:
        """Test relative (multiplicative) bias."""
        scenario = BiasScenario(
            variables=["O3"],
            bias=0.1,  # 10% bias
            relative_bias=True,
        )
        dataset, geometry = scenario.generate()

        # Dataset values should be scaled
        assert "O3" in dataset

    def test_negative_bias(self) -> None:
        """Test negative bias (dataset lower than geometry)."""
        scenario = BiasScenario(
            variables=["O3"],
            bias=-5.0,
            relative_bias=False,
        )
        stats = scenario.expected_statistics

        assert stats["O3"]["MB"] == -5.0


class TestMismatchScenario:
    """Tests for MismatchScenario."""

    def test_spatial_mismatch(self) -> None:
        """Test spatial mismatch scenario."""
        scenario = MismatchScenario(
            variables=["O3"],
            mismatch_type="spatial",
            overlap_fraction=0.5,
        )
        dataset, geometry = scenario.generate()

        # Both should be generated
        assert isinstance(dataset, xr.Dataset)
        assert isinstance(geometry, xr.Dataset)

    def test_temporal_mismatch(self) -> None:
        """Test temporal mismatch scenario."""
        scenario = MismatchScenario(
            variables=["O3"],
            mismatch_type="temporal",
            overlap_fraction=0.5,
        )
        dataset, geometry = scenario.generate()

        # Time ranges should be offset
        dataset_times = dataset.time.values
        geometry_times = geometry.time.values

        # Check they're not identical
        assert not np.array_equal(dataset_times, geometry_times)

    def test_both_mismatch(self) -> None:
        """Test combined spatial and temporal mismatch."""
        scenario = MismatchScenario(
            variables=["O3"],
            mismatch_type="both",
            overlap_fraction=0.3,
        )
        dataset, geometry = scenario.generate()

        assert isinstance(dataset, xr.Dataset)
        assert isinstance(geometry, xr.Dataset)

    def test_empty_expected_statistics(self) -> None:
        """Test expected statistics are empty for mismatch."""
        scenario = MismatchScenario(variables=["O3"])
        stats = scenario.expected_statistics

        # Mismatched data do not have fixed stats.
        assert stats["O3"] == {}


class TestScenarioFactory:
    """Tests for create_scenario factory function."""

    def test_create_perfect_match(self) -> None:
        """Test creating perfect match scenario."""
        scenario = create_scenario("perfect_match")
        assert isinstance(scenario, PerfectMatchScenario)

    def test_create_bias(self) -> None:
        """Test creating bias scenario."""
        scenario = create_scenario("bias", bias=5.0)
        assert isinstance(scenario, BiasScenario)
        assert scenario.bias == 5.0

    def test_create_mismatch(self) -> None:
        """Test creating mismatch scenario."""
        scenario = create_scenario("mismatch", mismatch_type="temporal")
        assert isinstance(scenario, MismatchScenario)
        assert scenario.mismatch_type == "temporal"

    def test_invalid_scenario_type(self) -> None:
        """Test invalid scenario type raises error."""
        with pytest.raises(ValueError, match="Unknown scenario type"):
            create_scenario("invalid_type")

    def test_geometry_parameter(self) -> None:
        """Test geometry parameter is passed correctly."""
        scenario = create_scenario("perfect_match", geometry=DataGeometry.TRACK)
        assert scenario.geometry == DataGeometry.TRACK


class TestScenarioReproducibility:
    """Tests for scenario reproducibility."""

    def test_same_seed_same_data(self) -> None:
        """Test same seed produces same data."""
        scenario1 = PerfectMatchScenario(variables=["O3"], seed=42)
        scenario2 = PerfectMatchScenario(variables=["O3"], seed=42)

        dataset1, geometry1 = scenario1.generate()
        dataset2, geometry2 = scenario2.generate()

        xr.testing.assert_equal(dataset1, dataset2)
        xr.testing.assert_equal(geometry1, geometry2)

    def test_different_seed_different_data(self) -> None:
        """Test different seeds produce different data."""
        scenario1 = PerfectMatchScenario(variables=["O3"], seed=42)
        scenario2 = PerfectMatchScenario(variables=["O3"], seed=43)

        dataset1, _ = scenario1.generate()
        dataset2, _ = scenario2.generate()

        assert not np.allclose(dataset1["O3"].values, dataset2["O3"].values)


class TestScenarioWorkflow:
    """Scenario workflow tests (calls scenario APIs directly)."""

    def test_scenario_workflow(self) -> None:
        """Test complete scenario workflow."""
        # Create scenario
        scenario = PerfectMatchScenario(
            variables=["O3", "PM25"],
            geometry=DataGeometry.POINT,
            n_geometry=15,
            domain=Domain(n_lon=18, n_lat=9),
            time_config=TimeConfig(end="2024-01-01 12:00"),
        )

        # Generate data
        dataset, geometry = scenario.generate()

        # Verify data is usable
        assert len(dataset.time) > 0
        assert len(geometry.site) == 15
        assert "O3" in dataset and "O3" in geometry
        assert "PM25" in dataset and "PM25" in geometry

        # Get expected stats
        expected = scenario.expected_statistics
        assert "O3" in expected
        assert "PM25" in expected
