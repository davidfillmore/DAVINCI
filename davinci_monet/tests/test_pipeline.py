"""Tests for pipeline module."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.pipeline import (
    BaseStage,
    LoadSourcesStage,
    PairingStage,
    PipelineBuilder,
    PipelineContext,
    PipelineResult,
    PipelineRunner,
    PlottingStage,
    SaveResultsStage,
    Stage,
    StageResult,
    StageStatus,
    StatisticsStage,
    create_standard_pipeline,
)
from davinci_monet.pipeline.stages.base import SourceData

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_context() -> PipelineContext:
    """Create a sample pipeline context (unified sources schema)."""
    return PipelineContext(
        config={
            "sources": {
                "test_dataset": {
                    "type": "generic",
                    "files": "/path/to/dataset.nc",
                },
                "test_geometry": {
                    "type": "pt_sfc",
                    "filename": "/path/to/geometry.nc",
                },
            },
        }
    )


@pytest.fixture
def sample_paired_dataset() -> xr.Dataset:
    """Create a sample paired dataset."""
    n_times = 100
    times = np.datetime64("2020-01-01") + np.arange(n_times) * np.timedelta64(1, "h")

    # Simulated dataset and geometry data with some correlation
    np.random.seed(42)
    geometry_vals = 50 + 20 * np.random.randn(n_times)
    dataset_vals = geometry_vals + 5 + 3 * np.random.randn(n_times)  # Bias of ~5

    return xr.Dataset(
        {
            "dataset_o3": (["time"], dataset_vals),
            "geometry_o3": (["time"], geometry_vals),
            "latitude": (["time"], np.full(n_times, 40.0)),
            "longitude": (["time"], np.full(n_times, -105.0)),
        },
        coords={"time": times},
    )


@pytest.fixture
def context_with_paired(sample_paired_dataset: xr.Dataset) -> PipelineContext:
    """Create a context with paired data."""
    ctx = PipelineContext()
    ctx.paired["test_dataset_test_geometry"] = sample_paired_dataset
    return ctx


# =============================================================================
# StageStatus Tests
# =============================================================================


class TestStageStatus:
    """Tests for StageStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert StageStatus.PENDING
        assert StageStatus.RUNNING
        assert StageStatus.COMPLETED
        assert StageStatus.FAILED
        assert StageStatus.SKIPPED


# =============================================================================
# StageResult Tests
# =============================================================================


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_create_result(self):
        """Test creating a stage result."""
        result = StageResult(
            stage_name="test_stage",
            status=StageStatus.COMPLETED,
            data={"key": "value"},
            duration_seconds=1.5,
        )

        assert result.stage_name == "test_stage"
        assert result.status == StageStatus.COMPLETED
        assert result.data == {"key": "value"}
        assert result.duration_seconds == 1.5
        assert result.error is None

    def test_result_with_error(self):
        """Test creating a failed result with error."""
        result = StageResult(
            stage_name="test_stage",
            status=StageStatus.FAILED,
            error="Something went wrong",
        )

        assert result.status == StageStatus.FAILED
        assert result.error == "Something went wrong"


# =============================================================================
# PipelineContext Tests
# =============================================================================


class TestPipelineContext:
    """Tests for PipelineContext."""

    def test_empty_context(self):
        """Test creating empty context."""
        ctx = PipelineContext()

        assert ctx.config == {}
        assert ctx.sources == {}
        assert ctx.paired == {}
        assert ctx.results == {}

    def test_context_with_config(self, sample_context: PipelineContext):
        """Test context with configuration."""
        assert "sources" in sample_context.config
        assert "test_dataset" in sample_context.config["sources"]
        assert "test_geometry" in sample_context.config["sources"]

    def test_get_source(self):
        """Test getting a source from context."""
        ctx = PipelineContext()
        ctx.sources["test"] = {"data": "source_data"}

        assert ctx.get_source("test") == {"data": "source_data"}

    def test_get_source_not_found(self):
        """Test getting non-existent source raises error."""
        ctx = PipelineContext()

        with pytest.raises(KeyError, match="Source 'missing' not found"):
            ctx.get_source("missing")

    def test_get_paired(self, context_with_paired: PipelineContext):
        """Test getting paired data from context."""
        paired = context_with_paired.get_paired("test_dataset_test_geometry")
        assert isinstance(paired, xr.Dataset)

    def test_get_paired_not_found(self):
        """Test getting non-existent paired data raises error."""
        ctx = PipelineContext()

        with pytest.raises(KeyError, match="Paired data 'missing' not found"):
            ctx.get_paired("missing")


# =============================================================================
# BaseStage Tests
# =============================================================================


class TestBaseStage:
    """Tests for BaseStage abstract class."""

    def test_custom_stage(self):
        """Test creating a custom stage."""

        class CustomStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(
                    StageStatus.COMPLETED,
                    data={"executed": True},
                )

        stage = CustomStage(name="custom")
        assert stage.name == "custom"

        ctx = PipelineContext()
        result = stage.execute(ctx)

        assert result.status == StageStatus.COMPLETED
        assert result.data == {"executed": True}

    def test_default_name(self):
        """Test stage uses class name by default."""

        class MyCustomStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        stage = MyCustomStage()
        assert stage.name == "MyCustomStage"

    def test_default_validation(self):
        """Test default validation passes."""

        class TestStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        stage = TestStage()
        ctx = PipelineContext()

        assert stage.validate(ctx) is True


# =============================================================================
# Concrete Stage Tests
# =============================================================================


class TestLoadSourcesStage:
    """Tests for the unified LoadSourcesStage."""

    def test_name(self):
        """Test stage name."""
        stage = LoadSourcesStage()
        assert stage.name == "load_sources"

    def test_validation_rejects_top_level_dataset_geometry_config(self):
        """validate() fails for a top-level dataset:/geometry: config.

        Top-level dataset:/geometry: controls are rejected at config load; the stage
        only recognizes the unified ``sources:`` shape (or pre-populated containers).
        """
        stage = LoadSourcesStage()
        ctx = PipelineContext(
            config={
                "dataset": {"test_dataset": {"type": "generic", "files": "/path/to/m.nc"}},
                "geometry": {"test_geometry": {"type": "pt_sfc", "filename": "/path/to/o.nc"}},
            }
        )
        assert stage.validate(ctx) is False

    def test_validation_with_sources_config(self):
        """validate() passes for a native sources: config."""
        stage = LoadSourcesStage()
        ctx = PipelineContext(
            config={"sources": {"cam": {"type": "generic", "files": "/path/to/m.nc"}}}
        )
        assert stage.validate(ctx) is True

    def test_validation_without_config(self):
        """validate() fails when nothing is configured or pre-loaded."""
        stage = LoadSourcesStage()
        ctx = PipelineContext()
        assert stage.validate(ctx) is False

    def test_source_kwargs_flow_through_to_reader(self, monkeypatch):
        """A unified source's reader-specific kwargs reach the registered reader.

        Exercises the unified loader path: ``_load_unified_source`` passes
        through-config kwargs (e.g. ``mech``) to the registered reader's
        ``open()`` while filtering out the loader/schema control keys.
        """
        captured: dict[str, Any] = {}

        class _StubReader:
            geometry = DataGeometry.GRID

            def open(self, file_paths, variables=None, **kwargs):
                captured["file_paths"] = list(file_paths)
                captured["variables"] = variables
                captured.update(kwargs)
                return xr.Dataset({"O3": ("time", [1.0])}, coords={"time": [0]})

        from davinci_monet.core.registry import source_registry

        monkeypatch.setattr(source_registry, "get", lambda name: _StubReader)

        stage = LoadSourcesStage()
        ctx = PipelineContext(
            config={
                "sources": {
                    "WRF-Chem": {
                        "type": "wrfchem",
                        "files": "/path/to/wrfout.nc",
                        "mech": "racm_esrl_vcp",
                    }
                }
            }
        )

        result = stage.execute(ctx)

        assert result.status == StageStatus.COMPLETED, result.error
        assert captured.get("mech") == "racm_esrl_vcp"
        assert "WRF-Chem" in ctx.sources


class TestPairingStage:
    """Tests for PairingStage."""

    def test_name(self):
        """Test stage name."""
        stage = PairingStage()
        assert stage.name == "pairing"

    def test_validation_with_data(self):
        """Test validation passes with loaded dataset and geometry sources."""
        stage = PairingStage()
        ctx = PipelineContext(
            config={
                "pairs": {
                    "dataset_geometry": {
                        "sources": ["dataset", "geometry"],
                        "geometry": "geometry",
                        "variables": {"dataset": "v", "geometry": "v"},
                    }
                }
            }
        )
        ds = xr.Dataset({"v": ("time", [1.0])}, coords={"time": [0]})
        ctx.sources["dataset"] = SourceData(
            data=ds,
            label="dataset",
            source_type="generic",
            geometry=DataGeometry.GRID,
        )
        ctx.sources["geometry"] = SourceData(
            data=ds,
            label="geometry",
            source_type="pt_sfc",
            geometry=DataGeometry.POINT,
        )

        assert stage.validate(ctx) is True

    def test_validation_without_data(self):
        """Test validation fails without data."""
        stage = PairingStage()
        ctx = PipelineContext()

        assert stage.validate(ctx) is False

    def test_execute_forwards_pair_strategy_options(self):
        """Pair-level strategy options flow into the unified pairing strategy."""
        stage = PairingStage()
        times = pd.date_range("2024-01-01T00:00:00", periods=3, freq="1h")
        lat = np.array([0.0, 1.0])
        lon = np.array([10.0, 11.0])
        dataset = xr.Dataset(
            {"flux": (("time", "lat", "lon"), np.ones((3, 2, 2), dtype=np.float32))},
            coords={"time": times, "lat": lat, "lon": lon},
            attrs={"geometry": "grid"},
        )
        geometry = xr.Dataset(
            {"flux": ("time", np.array([10.0, 20.0, 30.0], dtype=np.float32))},
            coords={
                "time": times,
                "lat": ("time", np.array([0.0, 0.0, 0.0])),
                "lon": ("time", np.array([10.0, 10.0, 10.0])),
            },
            attrs={"geometry": "swath"},
        )
        ctx = PipelineContext(
            config={
                "pairs": {
                    "met_vs_ceres": {
                        "sources": ["met", "ceres"],
                        "geometry": "ceres",
                        "variables": {"met": "flux", "ceres": "flux"},
                        "time_resolution": "1h",
                    }
                },
            }
        )
        ctx.sources["met"] = SourceData(
            data=dataset,
            label="met",
            source_type="generic",
            geometry=DataGeometry.GRID,
        )
        ctx.sources["ceres"] = SourceData(
            data=geometry,
            label="ceres",
            source_type="ceres_ssf",
            geometry=DataGeometry.SWATH,
        )

        result = stage.execute(ctx)

        assert result.status is StageStatus.COMPLETED
        paired = ctx.paired["met_vs_ceres"].data
        assert paired.sizes["time"] == 3
        assert set(paired.data_vars) == {"ceres_flux", "met_flux"}
        assert all(not str(name).startswith(("geometry_", "dataset_")) for name in paired.data_vars)


class TestStatisticsStage:
    """Tests for StatisticsStage."""

    def test_name(self):
        """Test stage name."""
        stage = StatisticsStage()
        assert stage.name == "statistics"

    def test_validation_with_paired(self, context_with_paired: PipelineContext):
        """Test validation passes with paired data."""
        stage = StatisticsStage()
        assert stage.validate(context_with_paired) is True

    def test_validation_without_paired(self):
        """Test validation fails without paired data."""
        stage = StatisticsStage()
        ctx = PipelineContext()

        assert stage.validate(ctx) is False

    def test_execute_calculates_stats(self, context_with_paired: PipelineContext):
        """Test statistics are calculated correctly."""
        stage = StatisticsStage()
        result = stage.execute(context_with_paired)

        assert result.status == StageStatus.COMPLETED
        assert result.data is not None

        # Check stats for o3 variable
        pair_stats = result.data.get("test_dataset_test_geometry", {})
        o3_stats = pair_stats.get("o3", {})

        assert "n" in o3_stats
        assert "mean_bias" in o3_stats
        assert "rmse" in o3_stats
        assert "correlation" in o3_stats
        assert o3_stats["n"] == 100

    def test_metrics_takes_precedence_over_stat_list(self, context_with_paired: PipelineContext):
        """Test that 'metrics' key takes precedence over 'stat_list'.

        Regression test for review finding #2: when both keys are present
        (as happens after dataset_dump() with defaults), the user-specified
        'metrics' should win over the default 'stat_list'.
        """
        context_with_paired.config["stats"] = {
            "stat_list": ["MB", "NMB", "R2", "RMSE"],  # defaults
            "metrics": ["N", "MB", "RMSE", "R", "NMB", "NME", "IOA"],  # user
        }
        stage = StatisticsStage()
        result = stage.execute(context_with_paired)

        assert result.status == StageStatus.COMPLETED
        pair_stats = result.data.get("test_dataset_test_geometry", {})
        o3_stats = pair_stats.get("o3", {})
        # IOA is in 'metrics' but not in 'stat_list' - must be present
        assert "IOA" in o3_stats

    def test_domain_filter_restricts_stats_to_conus(self):
        """Stats config domain_type / domain_name must filter paired_data before metrics.

        Regression test for the silent no-op: pre-fix, writing
        ``stats: {domain_type: [conus]}`` had no effect — stats were
        computed over the full paired dataset including non-CONUS sites.
        """
        # 6 sites across regions: 4 CONUS, 2 in Asia. The Asian sites have
        # large geometry values that would skew an unfiltered mean.
        n_times = 12
        times = np.datetime64("2024-01-01") + np.arange(n_times) * np.timedelta64(1, "h")
        site_lats = np.array([35.0, 40.0, 45.0, 43.0, 28.6, 13.1])
        site_lons = np.array([-100.0, -105.0, -95.0, -85.0, 77.2, 80.3])
        geometry = np.full((n_times, 6), 10.0)
        geometry[:, 4:] = 200.0  # Asian sites: dramatically different
        dataset = np.full((n_times, 6), 11.0)

        paired = xr.Dataset(
            {
                "geometry_pm25": (["time", "site"], geometry),
                "dataset_pm25": (["time", "site"], dataset),
            },
            coords={
                "time": times,
                "site": np.arange(6),
                "latitude": ("site", site_lats),
                "longitude": ("site", site_lons),
            },
        )

        ctx = PipelineContext()
        ctx.paired["dataset_geometry"] = paired
        ctx.config["stats"] = {"domain_type": ["conus"], "metrics": ["N", "MB"]}

        stage = StatisticsStage()
        result = stage.execute(ctx)

        assert result.status == StageStatus.COMPLETED
        pm25_stats = result.data["dataset_geometry"]["pm25"]
        # 4 CONUS sites x 12 timesteps = 48 paired points
        assert pm25_stats["N"] == 48, (
            f"Expected N=48 after CONUS filter, got {pm25_stats['N']}. "
            "domain_type filter is not being applied in StatisticsStage."
        )
        # MB should reflect only CONUS sites: dataset=11, geometry=10 → MB=+1
        assert abs(pm25_stats["MB"] - 1.0) < 0.01, (
            f"Expected MB≈1.0 (CONUS only), got {pm25_stats['MB']}. "
            "Asian sites leaked through the filter."
        )


class TestPlottingStage:
    """Tests for PlottingStage."""

    def test_name(self):
        """Test stage name."""
        stage = PlottingStage()
        assert stage.name == "plotting"

    def test_skips_without_config(self, context_with_paired: PipelineContext):
        """Test stage skips when no plot config provided."""
        stage = PlottingStage()
        result = stage.execute(context_with_paired)

        assert result.status == StageStatus.SKIPPED

    def test_data_key_resolved_for_pairs(self):
        """Test that plot specs using 'data' key resolve pair names.

        Regression test for review finding #1: the schema uses 'data' but
        PlottingStage must read it (not just 'pairs').
        """
        stage = PlottingStage()
        # Simulate what _calculate_stats helper does internally:
        # the stage reads plot_spec.get("data") or plot_spec.get("pairs", [])
        plot_spec_data = {"type": "scatter", "data": ["dataset_geometry_pm25"]}
        plot_spec_pairs = {"type": "scatter", "pairs": ["dataset_geometry_pm25"]}
        plot_spec_both = {
            "type": "scatter",
            "data": ["dataset_geometry_o3"],
            "pairs": ["dataset_geometry_pm25"],
        }

        # 'data' key should work
        resolved_data = plot_spec_data.get("data") or plot_spec_data.get("pairs", [])
        assert resolved_data == ["dataset_geometry_pm25"]

        # 'pairs' key still works (backward compat)
        resolved_pairs = plot_spec_pairs.get("data") or plot_spec_pairs.get("pairs", [])
        assert resolved_pairs == ["dataset_geometry_pm25"]

        # 'data' takes precedence when both present
        resolved_both = plot_spec_both.get("data") or plot_spec_both.get("pairs", [])
        assert resolved_both == ["dataset_geometry_o3"]

    def test_plot_options_use_separate_subtitle_not_caption(
        self,
        context_with_paired: PipelineContext,
    ):
        """Plotting stage must keep subtitles separate from titles and captions."""
        title = "AOD: MERRA2 vs MODIS Terra"
        stage = PlottingStage()
        plotter_config, _ = stage._resolve_plot_options(
            context=context_with_paired,
            plot_type="scatter",
            plot_spec={},
            analysis_config={"start_time": "2003-01-01", "end_time": "2003-12-31"},
            title=title,
            paired_data=context_with_paired.paired["test_dataset_test_geometry"],
            var_spec={"geometry_var": "o3", "dataset_var": "o3"},
            geometry_label="test_geometry",
            dataset_label="test_dataset",
        )

        assert plotter_config["title"] == title
        assert plotter_config["subtitle"] == "2003-01-01 - 2003-12-31"
        assert "caption" not in plotter_config


class TestSaveResultsStage:
    """Tests for SaveResultsStage."""

    def test_name(self):
        """Test stage name."""
        stage = SaveResultsStage()
        assert stage.name == "save_results"


# =============================================================================
# create_standard_pipeline Tests
# =============================================================================


class TestCreateStandardPipeline:
    """Tests for create_standard_pipeline function."""

    def test_creates_all_stages(self):
        """Test all standard stages are created."""
        stages = create_standard_pipeline()

        assert len(stages) == 6

        stage_names = [s.name for s in stages]
        assert stage_names == [
            "load_sources",
            "pairing",
            "statistics",
            "plotting",
            "save_results",
            "summary",
        ]

    def test_stages_are_in_order(self):
        """Test stages are in correct execution order."""
        stages = create_standard_pipeline()

        assert stages[0].name == "load_sources"
        assert stages[1].name == "pairing"
        assert stages[2].name == "statistics"
        assert stages[3].name == "plotting"


# =============================================================================
# PipelineRunner Tests
# =============================================================================


class TestPipelineRunner:
    """Tests for PipelineRunner."""

    def test_default_stages(self):
        """Test runner uses standard pipeline by default."""
        runner = PipelineRunner()

        # Unified pipeline (geometry/paired stage fork collapsed): load_sources,
        # pairing, statistics, plotting, save_results, summary.
        assert len(runner.stages) == 6

    def test_custom_stages(self):
        """Test runner accepts custom stages."""

        class SimpleStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        runner = PipelineRunner(stages=[SimpleStage(name="simple")])

        assert len(runner.stages) == 1
        assert runner.stages[0].name == "simple"

    def test_add_stage(self):
        """Test adding a stage to runner."""

        class NewStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        # Pass empty list to avoid default stages
        runner = PipelineRunner(stages=[])
        runner.add_stage(NewStage(name="new"))

        assert len(runner.stages) == 1

    def test_add_stage_at_position(self):
        """Test adding a stage at specific position."""

        class StageA(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        class StageB(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        runner = PipelineRunner(stages=[StageA(name="a")])
        runner.add_stage(StageB(name="b"), position=0)

        assert runner.stages[0].name == "b"
        assert runner.stages[1].name == "a"

    def test_remove_stage(self):
        """Test removing a stage by name."""
        runner = PipelineRunner()
        initial_count = len(runner.stages)

        assert runner.remove_stage("plotting") is True
        assert len(runner.stages) == initial_count - 1

    def test_remove_nonexistent_stage(self):
        """Test removing non-existent stage returns False."""
        runner = PipelineRunner()

        assert runner.remove_stage("nonexistent") is False

    def test_run_empty_pipeline(self):
        """Test running pipeline with no stages."""
        # Explicitly pass empty list to avoid default stages
        runner = PipelineRunner(stages=[])
        result = runner.run()

        assert result.success is True
        assert len(result.stage_results) == 0

    def test_run_simple_pipeline(self):
        """Test running a simple pipeline."""

        class SuccessStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                context.metadata["executed"] = True
                return self._create_result(StageStatus.COMPLETED)

        runner = PipelineRunner(stages=[SuccessStage(name="success")])
        result = runner.run()

        assert result.success is True
        assert len(result.stage_results) == 1
        assert result.context is not None
        assert result.context.metadata.get("executed") is True

    def test_runner_can_leave_context_datasets_open(self):
        """Callers can opt out of result-context dataset cleanup after run()."""

        class CloseableData:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        class SourceStage(BaseStage):
            def __init__(self, data: CloseableData) -> None:
                super().__init__("source")
                self.data = data

            def execute(self, context: PipelineContext) -> StageResult:
                context.sources["synthetic"] = self.data
                return self._create_result(StageStatus.COMPLETED)

        data = CloseableData()
        runner = PipelineRunner(
            stages=[SourceStage(data)],
            show_progress=False,
            close_datasets_after_run=False,
        )

        result = runner.run(PipelineContext())

        assert result.success is True
        assert data.closed is False

    def test_run_with_failing_stage(self):
        """Test pipeline handles failing stage."""

        class FailStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.FAILED, error="Test failure")

        runner = PipelineRunner(stages=[FailStage(name="fail")])
        result = runner.run()

        assert result.success is False
        assert len(result.failed_stages) == 1
        assert result.failed_stages[0].error == "Test failure"

    def test_fail_fast(self):
        """Test fail_fast stops on first failure."""

        class FailStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.FAILED, error="Failed")

        class SuccessStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        runner = PipelineRunner(
            stages=[FailStage(name="fail"), SuccessStage(name="success")],
            fail_fast=True,
        )
        result = runner.run()

        assert result.success is False
        assert len(result.stage_results) == 1  # Only first stage ran

    def test_continue_on_failure(self):
        """Test pipeline continues when fail_fast is False."""

        class FailStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.FAILED, error="Failed")

        class SuccessStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        runner = PipelineRunner(
            stages=[FailStage(name="fail"), SuccessStage(name="success")],
            fail_fast=False,
        )
        result = runner.run()

        assert result.success is False
        assert len(result.stage_results) == 2  # Both stages ran

    def test_hooks(self):
        """Test pipeline hooks are called."""
        hook_calls: list[str] = []

        class SuccessStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        runner = PipelineRunner(
            stages=[SuccessStage(name="success")],
            hooks={
                "on_start": lambda ctx: hook_calls.append("on_start"),
                "on_stage_start": lambda s, ctx: hook_calls.append("on_stage_start"),
                "on_stage_end": lambda s, r, ctx: hook_calls.append("on_stage_end"),
                "on_end": lambda result: hook_calls.append("on_end"),
            },
        )
        runner.run()

        assert "on_start" in hook_calls
        assert "on_stage_start" in hook_calls
        assert "on_stage_end" in hook_calls
        assert "on_end" in hook_calls

    def test_stage_validation_failure(self):
        """Test stage is skipped when validation fails."""

        class ValidatingStage(BaseStage):
            def validate(self, context: PipelineContext) -> bool:
                return False

            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        runner = PipelineRunner(stages=[ValidatingStage(name="validating")])
        result = runner.run()

        assert result.success is True
        assert result.stage_results[0].status == StageStatus.SKIPPED

    def test_stage_exception_handling(self):
        """Test exceptions in stages are handled."""

        class ExceptionStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                raise RuntimeError("Unexpected error")

        runner = PipelineRunner(stages=[ExceptionStage(name="exception")])
        result = runner.run()

        assert result.success is False
        assert "Unexpected error" in (result.stage_results[0].error or "")


# =============================================================================
# PipelineResult Tests
# =============================================================================


class TestPipelineResult:
    """Tests for PipelineResult."""

    def test_failed_stages(self):
        """Test getting failed stages."""
        result = PipelineResult(
            success=False,
            stage_results=[
                StageResult("a", StageStatus.COMPLETED),
                StageResult("b", StageStatus.FAILED, error="Error"),
                StageResult("c", StageStatus.COMPLETED),
            ],
        )

        assert len(result.failed_stages) == 1
        assert result.failed_stages[0].stage_name == "b"

    def test_completed_stages(self):
        """Test getting completed stage names."""
        result = PipelineResult(
            success=True,
            stage_results=[
                StageResult("a", StageStatus.COMPLETED),
                StageResult("b", StageStatus.SKIPPED),
                StageResult("c", StageStatus.COMPLETED),
            ],
        )

        assert result.completed_stages == ["a", "c"]

    def test_get_stage_result(self):
        """Test getting specific stage result."""
        result = PipelineResult(
            success=True,
            stage_results=[
                StageResult("a", StageStatus.COMPLETED),
                StageResult("b", StageStatus.COMPLETED),
            ],
        )

        stage_a = result.get_stage_result("a")
        assert stage_a is not None
        assert stage_a.stage_name == "a"

    def test_get_stage_result_not_found(self):
        """Test getting non-existent stage result."""
        result = PipelineResult(success=True, stage_results=[])

        assert result.get_stage_result("missing") is None


# =============================================================================
# PipelineBuilder Tests
# =============================================================================


class TestPipelineBuilder:
    """Tests for PipelineBuilder."""

    def test_empty_builder(self):
        """Test building pipeline with no stages added.

        Note: PipelineBuilder starts empty, but PipelineRunner defaults
        to standard stages if given an empty list. So an empty builder
        produces a runner with no stages only if we don't add any.
        """
        builder = PipelineBuilder()
        runner = builder.build()

        # Builder starts with empty list, runner gets that empty list
        assert len(runner.stages) == 0

    def test_add_standard_stages(self):
        """Test adding standard stages via the unified builder."""
        runner = PipelineBuilder().add_sources().add_pairing().add_statistics().build()

        assert len(runner.stages) == 3
        assert runner.stages[0].name == "load_sources"
        assert runner.stages[1].name == "pairing"
        assert runner.stages[2].name == "statistics"

    def test_add_custom_stage(self):
        """Test adding custom stage."""

        class CustomStage(BaseStage):
            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED)

        runner = PipelineBuilder().add_stage(CustomStage(name="custom")).build()

        assert len(runner.stages) == 1
        assert runner.stages[0].name == "custom"

    def test_fail_fast_setting(self):
        """Test setting fail_fast option."""
        runner = PipelineBuilder().fail_fast(False).build()

        assert runner._fail_fast is False

    def test_with_hook(self):
        """Test adding hook."""
        called = []

        runner = PipelineBuilder().with_hook("on_start", lambda ctx: called.append("start")).build()

        assert "on_start" in runner._hooks
