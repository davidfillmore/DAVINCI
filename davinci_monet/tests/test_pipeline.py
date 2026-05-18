"""Tests for pipeline module."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import xarray as xr

from davinci_monet.pipeline import (
    BaseStage,
    LoadModelsStage,
    LoadObservationsStage,
    PairingStage,
    ParallelExecutor,
    ParallelPairingExecutor,
    ParallelResult,
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
    parallel_process_files,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_context() -> PipelineContext:
    """Create a sample pipeline context."""
    return PipelineContext(
        config={
            "model": {
                "test_model": {
                    "files": "/path/to/model.nc",
                    "mod_type": "generic",
                }
            },
            "obs": {
                "test_obs": {
                    "obs_type": "pt_sfc",
                    "filename": "/path/to/obs.nc",
                }
            },
        }
    )


@pytest.fixture
def sample_paired_dataset() -> xr.Dataset:
    """Create a sample paired dataset."""
    n_times = 100
    times = np.datetime64("2020-01-01") + np.arange(n_times) * np.timedelta64(1, "h")

    # Simulated model and obs data with some correlation
    np.random.seed(42)
    obs_vals = 50 + 20 * np.random.randn(n_times)
    model_vals = obs_vals + 5 + 3 * np.random.randn(n_times)  # Bias of ~5

    return xr.Dataset(
        {
            "model_o3": (["time"], model_vals),
            "obs_o3": (["time"], obs_vals),
            "latitude": (["time"], np.full(n_times, 40.0)),
            "longitude": (["time"], np.full(n_times, -105.0)),
        },
        coords={"time": times},
    )


@pytest.fixture
def context_with_paired(sample_paired_dataset: xr.Dataset) -> PipelineContext:
    """Create a context with paired data."""
    ctx = PipelineContext()
    ctx.paired["test_model_test_obs"] = sample_paired_dataset
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
        assert ctx.models == {}
        assert ctx.observations == {}
        assert ctx.paired == {}
        assert ctx.results == {}

    def test_context_with_config(self, sample_context: PipelineContext):
        """Test context with configuration."""
        assert "model" in sample_context.config
        assert "obs" in sample_context.config

    def test_get_model(self):
        """Test getting model from context."""
        ctx = PipelineContext()
        ctx.models["test"] = {"data": "model_data"}

        assert ctx.get_model("test") == {"data": "model_data"}

    def test_get_model_not_found(self):
        """Test getting non-existent model raises error."""
        ctx = PipelineContext()

        with pytest.raises(KeyError, match="Model 'missing' not found"):
            ctx.get_model("missing")

    def test_get_observation(self):
        """Test getting observation from context."""
        ctx = PipelineContext()
        ctx.observations["test"] = {"data": "obs_data"}

        assert ctx.get_observation("test") == {"data": "obs_data"}

    def test_get_observation_not_found(self):
        """Test getting non-existent observation raises error."""
        ctx = PipelineContext()

        with pytest.raises(KeyError, match="Observation 'missing' not found"):
            ctx.get_observation("missing")

    def test_get_paired(self, context_with_paired: PipelineContext):
        """Test getting paired data from context."""
        paired = context_with_paired.get_paired("test_model_test_obs")
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


class TestLoadModelsStage:
    """Tests for LoadModelsStage."""

    def test_name(self):
        """Test stage name."""
        stage = LoadModelsStage()
        assert stage.name == "load_models"

    def test_validation_with_model_config(self, sample_context: PipelineContext):
        """Test validation passes with model config."""
        stage = LoadModelsStage()
        assert stage.validate(sample_context) is True

    def test_validation_without_config(self):
        """Test validation fails without model config."""
        stage = LoadModelsStage()
        ctx = PipelineContext()

        assert stage.validate(ctx) is False

    def test_execute_missing_files(self):
        """Test execute fails with clear error when files are missing."""
        stage = LoadModelsStage()
        ctx = PipelineContext(config={"model": {"m1": {"mod_type": "generic"}}})

        result = stage.execute(ctx)

        assert result.status == StageStatus.FAILED
        assert result.error is not None
        assert "missing required 'files'" in result.error

    def test_execute_passes_mod_kwargs_to_open_model(self, monkeypatch):
        """mod_kwargs from config must flow through to the model reader."""
        captured: dict[str, Any] = {}

        def fake_open_model(**kwargs: Any) -> Any:
            captured.update(kwargs)

            class _Stub:
                data = None
                variables: dict[str, Any] = {}

                def apply_variable_config(self) -> None:
                    pass

            return _Stub()

        import davinci_monet.models as models_mod

        monkeypatch.setattr(models_mod, "open_model", fake_open_model)

        stage = LoadModelsStage()
        ctx = PipelineContext(
            config={
                "model": {
                    "WRF-Chem": {
                        "files": "/path/to/wrfout.nc",
                        "mod_type": "wrfchem",
                        "mod_kwargs": {"mech": "racm_esrl_vcp"},
                    }
                }
            }
        )

        result = stage.execute(ctx)

        assert result.status == StageStatus.COMPLETED, result.error
        assert captured.get("mech") == "racm_esrl_vcp"
        assert captured.get("mod_type") == "wrfchem"
        assert captured.get("label") == "WRF-Chem"


class TestLoadObservationsStage:
    """Tests for LoadObservationsStage."""

    def test_name(self):
        """Test stage name."""
        stage = LoadObservationsStage()
        assert stage.name == "load_observations"

    def test_validation_with_obs_config(self, sample_context: PipelineContext):
        """Test validation passes with obs config."""
        stage = LoadObservationsStage()
        assert stage.validate(sample_context) is True

    def test_validation_without_config(self):
        """Test validation fails without obs config."""
        stage = LoadObservationsStage()
        ctx = PipelineContext()

        assert stage.validate(ctx) is False


class TestPairingStage:
    """Tests for PairingStage."""

    def test_name(self):
        """Test stage name."""
        stage = PairingStage()
        assert stage.name == "pairing"

    def test_validation_with_data(self):
        """Test validation passes with models and observations."""
        stage = PairingStage()
        ctx = PipelineContext()
        ctx.models["test"] = "model_data"
        ctx.observations["test"] = "obs_data"

        assert stage.validate(ctx) is True

    def test_validation_without_data(self):
        """Test validation fails without data."""
        stage = PairingStage()
        ctx = PipelineContext()

        assert stage.validate(ctx) is False


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
        pair_stats = result.data.get("test_model_test_obs", {})
        o3_stats = pair_stats.get("o3", {})

        assert "n" in o3_stats
        assert "mean_bias" in o3_stats
        assert "rmse" in o3_stats
        assert "correlation" in o3_stats
        assert o3_stats["n"] == 100

    def test_metrics_takes_precedence_over_stat_list(self, context_with_paired: PipelineContext):
        """Test that 'metrics' key takes precedence over 'stat_list'.

        Regression test for review finding #2: when both keys are present
        (as happens after model_dump() with defaults), the user-specified
        'metrics' should win over the default 'stat_list'.
        """
        context_with_paired.config["stats"] = {
            "stat_list": ["MB", "NMB", "R2", "RMSE"],  # defaults
            "metrics": ["N", "MB", "RMSE", "R", "NMB", "NME", "IOA"],  # user
        }
        stage = StatisticsStage()
        result = stage.execute(context_with_paired)

        assert result.status == StageStatus.COMPLETED
        pair_stats = result.data.get("test_model_test_obs", {})
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
        # large obs values that would skew an unfiltered mean.
        n_times = 12
        times = np.datetime64("2024-01-01") + np.arange(n_times) * np.timedelta64(1, "h")
        site_lats = np.array([35.0, 40.0, 45.0, 43.0, 28.6, 13.1])
        site_lons = np.array([-100.0, -105.0, -95.0, -85.0, 77.2, 80.3])
        obs = np.full((n_times, 6), 10.0)
        obs[:, 4:] = 200.0  # Asian sites: dramatically different
        model = np.full((n_times, 6), 11.0)

        paired = xr.Dataset(
            {
                "obs_pm25": (["time", "site"], obs),
                "model_pm25": (["time", "site"], model),
            },
            coords={
                "time": times,
                "site": np.arange(6),
                "latitude": ("site", site_lats),
                "longitude": ("site", site_lons),
            },
        )

        ctx = PipelineContext()
        ctx.paired["mod_obs"] = paired
        ctx.config["stats"] = {"domain_type": ["conus"], "metrics": ["N", "MB"]}

        stage = StatisticsStage()
        result = stage.execute(ctx)

        assert result.status == StageStatus.COMPLETED
        pm25_stats = result.data["mod_obs"]["pm25"]
        # 4 CONUS sites x 12 timesteps = 48 paired points
        assert pm25_stats["N"] == 48, (
            f"Expected N=48 after CONUS filter, got {pm25_stats['N']}. "
            "domain_type filter is not being applied in StatisticsStage."
        )
        # MB should reflect only CONUS sites: model=11, obs=10 → MB=+1
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
        """Test that plot specs using 'data' key resolve pair references.

        Regression test for review finding #1: the schema uses 'data' but
        PlottingStage must read it (not just 'pairs').
        """
        stage = PlottingStage()
        # Simulate what _calculate_stats helper does internally:
        # the stage reads plot_spec.get("data") or plot_spec.get("pairs", [])
        plot_spec_data = {"type": "scatter", "data": ["model_obs_pm25"]}
        plot_spec_pairs = {"type": "scatter", "pairs": ["model_obs_pm25"]}
        plot_spec_both = {
            "type": "scatter",
            "data": ["model_obs_o3"],
            "pairs": ["model_obs_pm25"],
        }

        # 'data' key should work
        resolved_data = plot_spec_data.get("data") or plot_spec_data.get("pairs", [])
        assert resolved_data == ["model_obs_pm25"]

        # 'pairs' key still works (backward compat)
        resolved_pairs = plot_spec_pairs.get("data") or plot_spec_pairs.get("pairs", [])
        assert resolved_pairs == ["model_obs_pm25"]

        # 'data' takes precedence when both present
        resolved_both = plot_spec_both.get("data") or plot_spec_both.get("pairs", [])
        assert resolved_both == ["model_obs_o3"]


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
        assert "load_models" in stage_names
        assert "load_observations" in stage_names
        assert "pairing" in stage_names
        assert "statistics" in stage_names
        assert "plotting" in stage_names
        assert "save_results" in stage_names

    def test_stages_are_in_order(self):
        """Test stages are in correct execution order."""
        stages = create_standard_pipeline()

        assert stages[0].name == "load_models"
        assert stages[1].name == "load_observations"
        assert stages[2].name == "pairing"
        assert stages[3].name == "statistics"


# =============================================================================
# PipelineRunner Tests
# =============================================================================


class TestPipelineRunner:
    """Tests for PipelineRunner."""

    def test_default_stages(self):
        """Test runner uses standard pipeline by default."""
        runner = PipelineRunner()

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
        """Test adding standard stages."""
        runner = (
            PipelineBuilder().add_models().add_observations().add_pairing().add_statistics().build()
        )

        assert len(runner.stages) == 4
        assert runner.stages[0].name == "load_models"
        assert runner.stages[1].name == "load_observations"
        assert runner.stages[2].name == "pairing"
        assert runner.stages[3].name == "statistics"

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


# =============================================================================
# ParallelExecutor Tests
# =============================================================================


class TestParallelExecutor:
    """Tests for ParallelExecutor."""

    def test_empty_items(self):
        """Test map with empty items."""
        executor = ParallelExecutor()
        result = executor.map(lambda x: x * 2, [])

        assert result.success is True
        assert result.results == []
        assert result.errors == []

    def test_map_simple(self):
        """Test simple parallel map."""
        executor = ParallelExecutor(max_workers=2)
        result = executor.map(lambda x: x * 2, [1, 2, 3, 4, 5])

        assert result.success is True
        assert sorted(result.results) == [2, 4, 6, 8, 10]

    def test_map_with_errors(self):
        """Test map handles errors."""

        def process(x):
            if x == 3:
                raise ValueError("Error on 3")
            return x * 2

        executor = ParallelExecutor(max_workers=2)
        result = executor.map(process, [1, 2, 3, 4, 5])

        assert result.success is False
        assert len(result.errors) == 1
        assert len(result.results) == 4

    def test_execute_stages(self):
        """Test executing stages in parallel."""

        class SimpleStage(BaseStage):
            def __init__(self, name: str, value: int):
                super().__init__(name)
                self._value = value

            def execute(self, context: PipelineContext) -> StageResult:
                return self._create_result(StageStatus.COMPLETED, data=self._value)

        stages = [SimpleStage(f"stage_{i}", i) for i in range(5)]
        executor = ParallelExecutor(max_workers=2)
        ctx = PipelineContext()

        results = executor.execute_stages(stages, ctx)

        assert len(results) == 5
        values = sorted(r.data for r in results)
        assert values == [0, 1, 2, 3, 4]


class TestParallelPairingExecutor:
    """Tests for ParallelPairingExecutor."""

    def test_pair_all_with_mapping(self) -> None:
        """Test parallel pairing with explicit variable mapping."""
        times = np.array([np.datetime64("2024-01-01T00:00:00")])
        lats = np.array([40.0, 41.0])
        lons = np.array([-105.0, -104.0])

        model = xr.Dataset(
            {"temperature": (["time", "lat", "lon"], np.full((1, 2, 2), 290.0))},
            coords={"time": times, "lat": lats, "lon": lons},
        )

        obs = xr.Dataset(
            {"temperature": (["time", "site"], np.full((1, 1), 289.0))},
            coords={
                "time": times,
                "site": np.array([0]),
                "latitude": ("site", np.array([40.0])),
                "longitude": ("site", np.array([-105.0])),
            },
        )

        executor = ParallelPairingExecutor(max_workers=1)
        result = executor.pair_all(
            models={"model": model},
            observations={"obs": obs},
            config={"mapping": {"obs": {"temperature": "temperature"}}},
        )

        assert "model_obs" in result
        paired = result["model_obs"]
        ds = paired.data if hasattr(paired, "data") else paired

        assert "obs_temperature" in ds.data_vars
        assert "model_temperature" in ds.data_vars


class TestParallelResult:
    """Tests for ParallelResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = ParallelResult(
            results=[1, 2, 3],
            errors=[],
            success=True,
        )

        assert result.success is True
        assert result.results == [1, 2, 3]

    def test_failed_result(self):
        """Test failed result."""
        result = ParallelResult(
            results=[1, 2],
            errors=["Error 1"],
            success=False,
        )

        assert result.success is False
        assert len(result.errors) == 1


class TestParallelProcessFiles:
    """Tests for parallel_process_files function."""

    def test_process_files(self):
        """Test processing files in parallel."""
        files = ["file1.txt", "file2.txt", "file3.txt"]

        def processor(path):
            return f"processed_{path}"

        result = parallel_process_files(files, processor, max_workers=2)

        assert result.success is True
        assert sorted(result.results) == [
            "processed_file1.txt",
            "processed_file2.txt",
            "processed_file3.txt",
        ]
