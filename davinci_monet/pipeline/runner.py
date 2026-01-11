"""Pipeline runner for orchestrating analysis workflows.

This module provides the PipelineRunner class that executes a sequence
of analysis stages, managing state and handling errors.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

from tqdm import tqdm

from davinci_monet.core.exceptions import PipelineError
from davinci_monet.pipeline.stages import (
    BaseStage,
    PipelineContext,
    Stage,
    StageResult,
    StageStatus,
    create_standard_pipeline,
)

logger = logging.getLogger(__name__)


class TeeWriter:
    """Write to multiple file-like objects simultaneously."""

    def __init__(self, *writers: TextIO) -> None:
        self.writers = writers

    def write(self, message: str) -> None:
        for writer in self.writers:
            writer.write(message)
            writer.flush()

    def flush(self) -> None:
        for writer in self.writers:
            writer.flush()


@dataclass
class PipelineResult:
    """Result of a complete pipeline execution.

    Attributes
    ----------
    success
        True if all stages completed successfully.
    stage_results
        Results from each stage in execution order.
    context
        Final pipeline context with all data.
    start_time
        Pipeline start time.
    end_time
        Pipeline end time.
    total_duration_seconds
        Total execution time.
    """

    success: bool
    stage_results: list[StageResult] = field(default_factory=list)
    context: PipelineContext | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    total_duration_seconds: float = 0.0

    @property
    def failed_stages(self) -> list[StageResult]:
        """Get list of failed stage results."""
        return [r for r in self.stage_results if r.status == StageStatus.FAILED]

    @property
    def completed_stages(self) -> list[str]:
        """Get names of completed stages."""
        return [
            r.stage_name
            for r in self.stage_results
            if r.status == StageStatus.COMPLETED
        ]

    def get_stage_result(self, stage_name: str) -> StageResult | None:
        """Get result for a specific stage."""
        for result in self.stage_results:
            if result.stage_name == stage_name:
                return result
        return None


class PipelineRunner:
    """Orchestrates execution of analysis pipeline stages.

    The runner manages the flow of data through stages, handles errors,
    and provides hooks for monitoring and logging.

    Parameters
    ----------
    stages
        List of stages to execute. If None, uses standard pipeline.
    fail_fast
        If True, stop on first stage failure.
    hooks
        Optional callback hooks for pipeline events.

    Examples
    --------
    >>> from davinci_monet.pipeline import PipelineRunner, PipelineContext
    >>> runner = PipelineRunner()
    >>> context = PipelineContext(config=my_config)
    >>> result = runner.run(context)
    >>> print(f"Success: {result.success}")
    """

    def __init__(
        self,
        stages: Sequence[Stage] | None = None,
        fail_fast: bool = True,
        hooks: dict[str, Callable[..., None]] | None = None,
        show_progress: bool = True,
    ) -> None:
        """Initialize pipeline runner.

        Parameters
        ----------
        stages
            Stages to execute. If None, uses standard pipeline.
        fail_fast
            Stop execution on first failure.
        hooks
            Event callbacks: on_start, on_stage_start, on_stage_end, on_end.
        show_progress
            Display progress bar and stage status to stdout.
        """
        self._stages = list(stages) if stages is not None else create_standard_pipeline()
        self._fail_fast = fail_fast
        self._hooks = hooks or {}
        self._show_progress = show_progress

    @property
    def stages(self) -> list[Stage]:
        """Get the list of stages."""
        return list(self._stages)

    def add_stage(self, stage: Stage, position: int | None = None) -> None:
        """Add a stage to the pipeline.

        Parameters
        ----------
        stage
            Stage to add.
        position
            Position to insert at. If None, appends to end.
        """
        if position is None:
            self._stages.append(stage)
        else:
            self._stages.insert(position, stage)

    def remove_stage(self, stage_name: str) -> bool:
        """Remove a stage by name.

        Parameters
        ----------
        stage_name
            Name of stage to remove.

        Returns
        -------
        bool
            True if stage was found and removed.
        """
        for i, stage in enumerate(self._stages):
            if stage.name == stage_name:
                self._stages.pop(i)
                return True
        return False

    def run(self, context: PipelineContext | None = None) -> PipelineResult:
        """Execute the pipeline.

        Parameters
        ----------
        context
            Pipeline context. If None, creates empty context.

        Returns
        -------
        PipelineResult
            Result of pipeline execution.
        """
        if context is None:
            context = PipelineContext()

        # Set up logging to file if log_dir is configured
        log_file: TextIO | None = None
        log_path: Path | None = None
        file_handler: logging.Handler | None = None

        analysis_config = context.config.get("analysis", {})
        log_dir = analysis_config.get("log_dir")

        if log_dir:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)

            # Create timestamped log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir_path / f"pipeline_{timestamp}.log"
            log_file = open(log_path, "w")

            # Add file handler to logger
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logging.getLogger("davinci_monet").addHandler(file_handler)

            # Write header to log file
            log_file.write(f"DAVINCI-MONET Pipeline Log\n")
            log_file.write(f"Started: {datetime.now().isoformat()}\n")
            log_file.write("=" * 70 + "\n\n")
            log_file.flush()

        # Set up progress callback
        def _make_progress_callback(
            log_f: TextIO | None, show_prog: bool
        ) -> Callable[[str], None]:
            """Create progress callback that writes to log file and/or stdout."""
            def callback(msg: str) -> None:
                # Always write to log file first
                if log_f:
                    log_f.write(msg + "\n")
                    log_f.flush()
                # Then try stdout (may fail with broken pipe)
                if show_prog:
                    try:
                        tqdm.write(msg, file=sys.stdout)
                    except BrokenPipeError:
                        pass
            return callback

        context.progress_callback = _make_progress_callback(log_file, self._show_progress)

        result = PipelineResult(
            success=True,
            context=context,
            start_time=datetime.now(),
        )

        start_time = time.time()
        self._call_hook("on_start", context)

        # Create progress bar if enabled
        stages_iter = self._stages
        if self._show_progress:
            stages_iter = tqdm(
                self._stages,
                desc="Pipeline",
                unit="stage",
                file=sys.stdout,
                leave=True,
            )

        try:
            for stage in stages_iter:
                # Update progress bar description
                if self._show_progress and hasattr(stages_iter, "set_description"):
                    stages_iter.set_description(f"Running {stage.name}")

                stage_result = self._execute_stage(stage, context)
                result.stage_results.append(stage_result)

                # Store result in context
                context.results[stage.name] = stage_result

                if stage_result.status == StageStatus.FAILED:
                    result.success = False
                    msg = f"  FAILED: {stage.name} - {stage_result.error}"
                    # Write to log file first (before stdout which may have broken pipe)
                    if log_file:
                        log_file.write(msg + "\n")
                        log_file.flush()
                    if self._show_progress:
                        try:
                            tqdm.write(msg, file=sys.stdout)
                        except BrokenPipeError:
                            pass
                    if self._fail_fast:
                        logger.error(
                            f"Pipeline failed at stage '{stage.name}': "
                            f"{stage_result.error}"
                        )
                        break
                elif stage_result.status == StageStatus.COMPLETED:
                    msg = f"  {stage.name}: {stage_result.duration_seconds:.1f}s"
                    # Write to log file first (before stdout which may have broken pipe)
                    if log_file:
                        log_file.write(msg + "\n")
                        log_file.flush()
                    if self._show_progress:
                        try:
                            tqdm.write(msg, file=sys.stdout)
                        except BrokenPipeError:
                            pass

        finally:
            # Clean up logging
            if file_handler:
                logging.getLogger("davinci_monet").removeHandler(file_handler)
                file_handler.close()

            if log_file:
                log_file.write("\n" + "=" * 70 + "\n")
                log_file.write(f"Finished: {datetime.now().isoformat()}\n")
                log_file.write(f"Duration: {time.time() - start_time:.1f}s\n")
                log_file.write(f"Success: {result.success}\n")
                log_file.close()

                if log_path and self._show_progress:
                    tqdm.write(f"  Log saved: {log_path}", file=sys.stdout)

        result.end_time = datetime.now()
        result.total_duration_seconds = time.time() - start_time

        self._call_hook("on_end", result)

        return result

    def run_from_config(
        self, config: dict[str, Any] | str
    ) -> PipelineResult:
        """Execute pipeline from configuration.

        Parameters
        ----------
        config
            Configuration dictionary or path to YAML file.

        Returns
        -------
        PipelineResult
            Result of pipeline execution.
        """
        if isinstance(config, str):
            from davinci_monet.config import load_config
            config = load_config(config).model_dump()

        context = PipelineContext(config=config)
        return self.run(context)

    def _execute_stage(
        self, stage: Stage, context: PipelineContext
    ) -> StageResult:
        """Execute a single stage.

        Parameters
        ----------
        stage
            Stage to execute.
        context
            Pipeline context.

        Returns
        -------
        StageResult
            Result of stage execution.
        """
        self._call_hook("on_stage_start", stage, context)

        start_time = time.time()

        try:
            # Validate stage
            if not stage.validate(context):
                logger.warning(f"Stage '{stage.name}' validation failed, skipping")
                result = StageResult(
                    stage_name=stage.name,
                    status=StageStatus.SKIPPED,
                    error="Validation failed",
                    duration_seconds=time.time() - start_time,
                )
            else:
                # Execute stage
                logger.info(f"Executing stage: {stage.name}")
                result = stage.execute(context)
                result.duration_seconds = time.time() - start_time
                logger.info(
                    f"Stage '{stage.name}' completed in "
                    f"{result.duration_seconds:.2f}s"
                )

        except Exception as e:
            logger.exception(f"Stage '{stage.name}' raised exception")
            result = StageResult(
                stage_name=stage.name,
                status=StageStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

        self._call_hook("on_stage_end", stage, result, context)

        return result

    def _call_hook(self, hook_name: str, *args: Any) -> None:
        """Call a hook if registered."""
        if hook_name in self._hooks:
            try:
                self._hooks[hook_name](*args)
            except Exception as e:
                logger.warning(f"Hook '{hook_name}' raised exception: {e}")


class PipelineBuilder:
    """Fluent builder for constructing pipelines.

    Examples
    --------
    >>> pipeline = (
    ...     PipelineBuilder()
    ...     .add_models()
    ...     .add_observations()
    ...     .add_pairing()
    ...     .add_statistics()
    ...     .build()
    ... )
    """

    def __init__(self) -> None:
        self._stages: list[Stage] = []
        self._fail_fast = True
        self._hooks: dict[str, Callable[..., None]] = {}
        self._show_progress = True

    def add_stage(self, stage: Stage) -> PipelineBuilder:
        """Add a custom stage."""
        self._stages.append(stage)
        return self

    def add_models(self) -> PipelineBuilder:
        """Add model loading stage."""
        from davinci_monet.pipeline.stages import LoadModelsStage
        self._stages.append(LoadModelsStage())
        return self

    def add_observations(self) -> PipelineBuilder:
        """Add observation loading stage."""
        from davinci_monet.pipeline.stages import LoadObservationsStage
        self._stages.append(LoadObservationsStage())
        return self

    def add_pairing(self) -> PipelineBuilder:
        """Add pairing stage."""
        from davinci_monet.pipeline.stages import PairingStage
        self._stages.append(PairingStage())
        return self

    def add_statistics(self) -> PipelineBuilder:
        """Add statistics stage."""
        from davinci_monet.pipeline.stages import StatisticsStage
        self._stages.append(StatisticsStage())
        return self

    def add_plotting(self) -> PipelineBuilder:
        """Add plotting stage."""
        from davinci_monet.pipeline.stages import PlottingStage
        self._stages.append(PlottingStage())
        return self

    def add_save(self) -> PipelineBuilder:
        """Add save results stage."""
        from davinci_monet.pipeline.stages import SaveResultsStage
        self._stages.append(SaveResultsStage())
        return self

    def fail_fast(self, enabled: bool = True) -> PipelineBuilder:
        """Set fail-fast mode."""
        self._fail_fast = enabled
        return self

    def with_hook(
        self, event: str, callback: Callable[..., None]
    ) -> PipelineBuilder:
        """Add an event hook."""
        self._hooks[event] = callback
        return self

    def show_progress(self, enabled: bool = True) -> PipelineBuilder:
        """Set progress display mode."""
        self._show_progress = enabled
        return self

    def build(self) -> PipelineRunner:
        """Build the pipeline runner."""
        return PipelineRunner(
            stages=self._stages,
            fail_fast=self._fail_fast,
            hooks=self._hooks,
            show_progress=self._show_progress,
        )


def run_analysis(
    config: dict[str, Any] | str,
    show_progress: bool = True,
) -> PipelineResult:
    """Convenience function to run a complete analysis.

    Parameters
    ----------
    config
        Configuration dictionary or path to YAML file.
    show_progress
        Display progress bar and stage timing to stdout.

    Returns
    -------
    PipelineResult
        Result of pipeline execution.

    Examples
    --------
    >>> result = run_analysis("config.yaml")
    >>> if result.success:
    ...     print("Analysis complete!")
    """
    runner = PipelineRunner(show_progress=show_progress)
    return runner.run_from_config(config)
