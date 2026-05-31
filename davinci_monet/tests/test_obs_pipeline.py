"""Tests for observation-only pipeline stages and auto-detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.observations.base import ObservationData
from davinci_monet.pipeline.stages import (
    PipelineContext,
    StageStatus,
)

# =============================================================================
# Fixtures
# =============================================================================


def _create_obs_dataset() -> xr.Dataset:
    """Create a synthetic aircraft-like observation dataset."""
    n = 200
    rng = np.random.default_rng(42)

    times = np.datetime64("2012-05-18T14:00") + np.arange(n) * np.timedelta64(30, "s")
    lats = 36.0 + rng.uniform(-2, 2, n)
    lons = -97.0 + rng.uniform(-2, 2, n)
    alts = rng.uniform(500, 12000, n)
    o3 = rng.uniform(20, 120, n)
    co = rng.uniform(50, 300, n)

    return xr.Dataset(
        {
            "O3": ("time", o3, {"units": "ppbv", "long_name": "Ozone"}),
            "CO": ("time", co, {"units": "ppbv", "long_name": "Carbon Monoxide"}),
            "altitude": ("time", alts, {"units": "m"}),
            "latitude": ("time", lats),
            "longitude": ("time", lons),
        },
        coords={"time": times},
    )


@pytest.fixture
def obs_dataset() -> xr.Dataset:
    """Synthetic observation dataset."""
    return _create_obs_dataset()


@pytest.fixture
def obs_data(obs_dataset: xr.Dataset) -> ObservationData:
    """ObservationData wrapper around synthetic data."""
    obs = ObservationData(
        data=obs_dataset,
        label="dc8",
        obs_type="aircraft",
        _geometry=DataGeometry.TRACK,
    )
    return obs


@pytest.fixture
def obs_context(obs_data: ObservationData, tmp_path: Any) -> PipelineContext:
    """PipelineContext with observation data and obs-only config."""
    return PipelineContext(
        config={
            "obs": {
                "dc8": {
                    "obs_type": "aircraft",
                    "filename": "/fake/path.nc",
                }
            },
            "analysis": {
                "output_dir": str(tmp_path / "output"),
            },
            "plots": {
                "o3_histogram": {
                    "type": "obs_histogram",
                    "obs": "dc8",
                    "variable": "O3",
                    "title": "O3 Distribution",
                },
            },
        },
        observations={"dc8": obs_data},
    )


# =============================================================================
# TestObsPlottingStage
# =============================================================================


class TestObsPlottingStage:
    """Tests for ObsPlottingStage."""

    def test_stage_name(self) -> None:
        """Stage name should be 'obs_plotting'."""
        from davinci_monet.pipeline.stages import ObsPlottingStage

        stage = ObsPlottingStage()
        assert stage.name == "obs_plotting"

    def test_validate_with_obs(self, obs_context: PipelineContext) -> None:
        """validate() returns True when observations exist."""
        from davinci_monet.pipeline.stages import ObsPlottingStage

        stage = ObsPlottingStage()
        assert stage.validate(obs_context) is True

    def test_validate_without_obs(self) -> None:
        """validate() returns False when no observations."""
        from davinci_monet.pipeline.stages import ObsPlottingStage

        stage = ObsPlottingStage()
        ctx = PipelineContext(config={})
        assert stage.validate(ctx) is False

    def test_execute_creates_plots(self, obs_context: PipelineContext, tmp_path: Any) -> None:
        """execute() creates plot files in output dir."""
        from davinci_monet.pipeline.stages import ObsPlottingStage

        stage = ObsPlottingStage()
        result = stage.execute(obs_context)

        assert result.status == StageStatus.COMPLETED
        assert result.data["plot_count"] >= 1

        output_dir = tmp_path / "output"
        png_files = list(output_dir.glob("*.png"))
        assert len(png_files) >= 1
        assert any("o3_histogram" in f.name for f in png_files)


# =============================================================================
# TestObsStatisticsStage
# =============================================================================


class TestObsStatisticsStage:
    """Tests for ObsStatisticsStage."""

    def test_stage_name(self) -> None:
        """Stage name should be 'obs_statistics'."""
        from davinci_monet.pipeline.stages import ObsStatisticsStage

        stage = ObsStatisticsStage()
        assert stage.name == "obs_statistics"

    def test_validate_with_obs(self, obs_context: PipelineContext) -> None:
        """validate() returns True when observations exist."""
        from davinci_monet.pipeline.stages import ObsStatisticsStage

        stage = ObsStatisticsStage()
        assert stage.validate(obs_context) is True

    def test_execute_computes_stats(self, obs_context: PipelineContext) -> None:
        """execute() returns correct structure: obs_label -> var_name -> metric."""
        from davinci_monet.pipeline.stages import ObsStatisticsStage

        stage = ObsStatisticsStage()
        result = stage.execute(obs_context)

        assert result.status == StageStatus.COMPLETED
        # Top level keyed by obs label
        assert "dc8" in result.data
        dc8_stats = result.data["dc8"]
        # Should have stats for O3 and CO (at least)
        assert "O3" in dc8_stats
        o3_stats = dc8_stats["O3"]
        # Check expected metrics exist
        for metric in ["N", "mean", "median", "std", "min", "max", "p10", "p25", "p75", "p90"]:
            assert metric in o3_stats, f"Missing metric: {metric}"
        # N should match dataset size
        assert o3_stats["N"] == 200
        # Mean should be roughly between 20 and 120 (our random range)
        assert 20 <= o3_stats["mean"] <= 120


# =============================================================================
# TestObsOnlyPipelineDetection
# =============================================================================


class TestObsOnlyPipelineDetection:
    """Tests for obs-only pipeline auto-detection."""

    def test_create_obs_pipeline(self) -> None:
        """create_obs_pipeline() returns correct stage sequence."""
        from davinci_monet.pipeline.stages import create_obs_pipeline

        stages = create_obs_pipeline()
        stage_names = [s.name for s in stages]

        # Unified loading stage replaces LoadObservationsStage.
        assert "load_sources" in stage_names
        assert "obs_statistics" in stage_names
        assert "obs_plotting" in stage_names

        # Must NOT include the legacy model/pairing stages.
        assert "load_models" not in stage_names
        assert "load_observations" not in stage_names
        assert "pairing" not in stage_names

    def test_run_from_config_detects_obs_only(
        self, obs_data: ObservationData, tmp_path: Any
    ) -> None:
        """PipelineRunner.run_from_config with obs-only config uses obs pipeline."""
        from davinci_monet.pipeline.runner import PipelineRunner

        config: dict[str, Any] = {
            "obs": {
                "dc8": {
                    "obs_type": "aircraft",
                    "filename": "/fake/path.nc",
                }
            },
            "analysis": {
                "output_dir": str(tmp_path / "output"),
            },
        }

        runner = PipelineRunner(show_progress=False)
        # run_from_config should detect obs-only and swap in obs pipeline.
        # The LoadObservationsStage will fail because the file doesn't exist,
        # but the important thing is that it attempts load_observations
        # (not load_models), proving the auto-detection worked.
        result = runner.run_from_config(config)

        # The unified pipeline handles obs-only runs: load_sources loads, the
        # pairing/statistics/plotting stages skip (no pairs), and the obs-only
        # stages run. (Legacy load_models stage is gone entirely.)
        stage_names = [s.name for s in runner.stages]
        assert "load_models" not in stage_names
        assert "load_sources" in stage_names
        assert "obs_statistics" in stage_names
        assert "obs_plotting" in stage_names
