"""Tests for observation-only pipeline stages and auto-detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.pipeline.stages import PipelineContext, SourceData, StageStatus

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
def obs_data(obs_dataset: xr.Dataset) -> SourceData:
    """SourceData wrapper around synthetic data."""
    return SourceData(
        data=obs_dataset,
        label="dc8",
        source_type="aircraft",
        geometry=DataGeometry.TRACK,
        role="obs",
    )


@pytest.fixture
def obs_context(obs_data: SourceData, tmp_path: Any) -> PipelineContext:
    """PipelineContext with observation data and obs-only config."""
    return PipelineContext(
        config={
            "sources": {
                "dc8": {
                    "type": "aircraft",
                    "role": "obs",
                    "filename": "/fake/path.nc",
                }
            },
            "analysis": {
                "output_dir": str(tmp_path / "output"),
            },
            "plots": {
                "o3_histogram": {
                    "type": "histogram",
                    "obs": "dc8",
                    "variable": "O3",
                    "title": "O3 Distribution",
                },
            },
        },
        sources={"dc8": obs_data},
    )


# =============================================================================
# TestObsPlottingStage
# =============================================================================


class TestObsPlottingStage:
    """Obs-only plotting is now handled by the unified PlottingStage."""

    def test_stage_name(self) -> None:
        """The unified plotting stage is named 'plotting'."""
        from davinci_monet.pipeline.stages import PlottingStage

        assert PlottingStage().name == "plotting"

    def test_validate_with_obs(self, obs_context: PipelineContext) -> None:
        """validate() returns True when observations exist (obs-only run)."""
        from davinci_monet.pipeline.stages import PlottingStage

        assert PlottingStage().validate(obs_context) is True

    def test_validate_without_obs(self) -> None:
        """validate() returns False when there is neither paired nor obs data."""
        from davinci_monet.pipeline.stages import PlottingStage

        ctx = PipelineContext(config={})
        assert PlottingStage().validate(ctx) is False

    def test_execute_creates_plots(self, obs_context: PipelineContext, tmp_path: Any) -> None:
        """execute() creates obs-only plot files in the output dir."""
        from davinci_monet.pipeline.stages import PlottingStage

        result = PlottingStage().execute(obs_context)

        assert result.status == StageStatus.COMPLETED
        assert result.data["plot_count"] >= 1

        output_dir = tmp_path / "output"
        png_files = list(output_dir.glob("*.png"))
        assert len(png_files) >= 1
        assert any("o3_histogram" in f.name for f in png_files)
        # Single-source plots save a PDF alongside each PNG (comparison parity)
        pdf_files = list(output_dir.glob("*.pdf"))
        assert {f.stem for f in pdf_files} == {f.stem for f in png_files}


# =============================================================================
# TestObsStatisticsStage
# =============================================================================


class TestObsStatisticsStage:
    """Obs-only descriptive statistics are now handled by the unified StatisticsStage."""

    def test_stage_name(self) -> None:
        """The unified statistics stage is named 'statistics'."""
        from davinci_monet.pipeline.stages import StatisticsStage

        assert StatisticsStage().name == "statistics"

    def test_validate_with_obs(self, obs_context: PipelineContext) -> None:
        """validate() returns True when observations exist (obs-only run)."""
        from davinci_monet.pipeline.stages import StatisticsStage

        assert StatisticsStage().validate(obs_context) is True

    def test_execute_computes_stats(self, obs_context: PipelineContext) -> None:
        """execute() returns descriptive stats: obs_label -> var_name -> metric."""
        from davinci_monet.pipeline.stages import StatisticsStage

        result = StatisticsStage().execute(obs_context)

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

        # Unified loading stage replaces LoadObservationsStage; obs-only stats and
        # plotting are handled by the unified StatisticsStage/PlottingStage.
        assert "load_sources" in stage_names
        assert "statistics" in stage_names
        assert "plotting" in stage_names
        # The obs/paired stage fork is gone.
        assert "obs_statistics" not in stage_names
        assert "obs_plotting" not in stage_names

        # Must NOT include the legacy model/pairing stages.
        assert "load_models" not in stage_names
        assert "load_observations" not in stage_names
        assert "pairing" not in stage_names

    @pytest.mark.integration
    def test_run_from_config_detects_obs_only(self, obs_data: SourceData, tmp_path: Any) -> None:
        """PipelineRunner.run_from_config with obs-only config uses obs pipeline."""
        from davinci_monet.pipeline.runner import PipelineRunner

        config: dict[str, Any] = {
            "sources": {
                "dc8": {
                    "role": "obs",
                    "type": "aircraft",
                    "filename": "/fake/path.nc",
                }
            },
            "analysis": {
                "output_dir": str(tmp_path / "output"),
            },
        }

        runner = PipelineRunner(show_progress=False)
        # run_from_config should detect obs-only and swap in obs pipeline.
        # The load_sources stage will fail because the file doesn't exist, but
        # the stage list still proves the obs-only path was selected.
        result = runner.run_from_config(config)

        # The unified pipeline handles obs-only runs: load_sources loads, the
        # pairing stage skips (no pairs), and the unified statistics/plotting
        # stages run the obs-only path. (Legacy load_models stage is gone, and so
        # is the separate obs_statistics/obs_plotting fork.)
        stage_names = [s.name for s in runner.stages]
        assert "load_models" not in stage_names
        assert "load_sources" in stage_names
        assert "statistics" in stage_names
        assert "plotting" in stage_names
        assert "obs_statistics" not in stage_names
        assert "obs_plotting" not in stage_names
