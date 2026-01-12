"""Pipeline runner for orchestrating analysis workflows.

This module provides the PipelineRunner class that executes a sequence
of analysis stages, managing state and handling errors.
"""

from __future__ import annotations

import logging
import re
import sys
import time
import traceback
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


@dataclass
class LogEntry:
    """A single log entry with timing information."""

    name: str
    category: str  # 'model', 'observation', 'pair', 'stage'
    start_time: float
    end_time: float | None = None
    status: str = "running"
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


class LogCollector:
    """Collects structured log data for Markdown report generation."""

    def __init__(self) -> None:
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.config_path: str | None = None
        self.entries: list[LogEntry] = []
        self._current_stage: LogEntry | None = None
        self._item_start_times: dict[str, float] = {}
        # Additional data extracted from context
        self.model_details: dict[str, dict[str, Any]] = {}
        self.obs_details: dict[str, dict[str, Any]] = {}
        self.pair_details: dict[str, dict[str, Any]] = {}
        self.statistics: dict[str, dict[str, Any]] = {}
        # Error tracking
        self.errors: list[dict[str, Any]] = []

    def start_pipeline(self, config_path: str | None = None) -> None:
        """Record pipeline start."""
        self.start_time = datetime.now()
        self.config_path = config_path

    def end_pipeline(self, success: bool) -> None:
        """Record pipeline end."""
        self.end_time = datetime.now()
        self.success = success

    def log_error(
        self,
        stage_name: str,
        error_type: str,
        error_message: str,
        traceback_str: str | None = None,
    ) -> None:
        """Record an error that occurred during pipeline execution.

        Parameters
        ----------
        stage_name
            Name of the stage where the error occurred.
        error_type
            The exception class name (e.g., 'ValueError', 'DataNotFoundError').
        error_message
            The error message string.
        traceback_str
            Optional formatted traceback string.
        """
        self.errors.append({
            "stage": stage_name,
            "error_type": error_type,
            "error_message": error_message,
            "traceback": traceback_str,
            "timestamp": datetime.now().isoformat(),
        })

    def start_stage(self, name: str) -> None:
        """Record stage start."""
        self._current_stage = LogEntry(
            name=name,
            category="stage",
            start_time=time.time(),
        )

    def end_stage(self, name: str, status: str, duration: float) -> None:
        """Record stage completion."""
        if self._current_stage and self._current_stage.name == name:
            self._current_stage.end_time = time.time()
            self._current_stage.status = status
            self.entries.append(self._current_stage)
            self._current_stage = None

    def log_item(self, message: str) -> None:
        """Parse and log a progress message."""
        # Parse patterns like "Loading model: cesm_asiaq (1/2)"
        model_match = re.match(r"\s*Loading model: (\S+)", message)
        obs_match = re.match(r"\s*Loading obs: (\S+)", message)
        pair_match = re.match(r"\s*Pairing: (\S+)", message)

        if model_match:
            name = model_match.group(1)
            self._start_item(name, "model")
        elif obs_match:
            name = obs_match.group(1)
            self._start_item(name, "observation")
        elif pair_match:
            name = pair_match.group(1)
            self._start_item(name, "pair")

    def _start_item(self, name: str, category: str) -> None:
        """Start timing an item and close previous in same category."""
        # Close any previous item in the same category
        for entry in self.entries:
            if entry.category == category and entry.end_time is None:
                entry.end_time = time.time()

        # Also check _item_start_times for any unclosed items
        for key, start in list(self._item_start_times.items()):
            if key.startswith(f"{category}:"):
                old_name = key.split(":", 1)[1]
                self.entries.append(
                    LogEntry(
                        name=old_name,
                        category=category,
                        start_time=start,
                        end_time=time.time(),
                        status="completed",
                    )
                )
                del self._item_start_times[key]

        self._item_start_times[f"{category}:{name}"] = time.time()

    def finalize_items(self) -> None:
        """Close any open items when stage ends."""
        now = time.time()
        for key, start in list(self._item_start_times.items()):
            category, name = key.split(":", 1)
            self.entries.append(
                LogEntry(
                    name=name,
                    category=category,
                    start_time=start,
                    end_time=now,
                    status="completed",
                )
            )
        self._item_start_times.clear()

    def extract_context_data(self, context: "PipelineContext") -> None:
        """Extract detailed data from pipeline context after execution."""
        # Extract model details
        for label, model_data in context.models.items():
            details: dict[str, Any] = {}
            if hasattr(model_data, "data") and model_data.data is not None:
                ds = model_data.data
                details["variables"] = len(ds.data_vars)
                if "time" in ds.sizes:
                    details["time_steps"] = ds.sizes["time"]
                # Calculate approximate size
                total_size = sum(
                    ds[v].size for v in ds.data_vars
                )
                details["data_points"] = total_size
            self.model_details[label] = details

        # Extract observation details
        for label, obs_data in context.observations.items():
            details = {}
            if hasattr(obs_data, "data") and obs_data.data is not None:
                ds = obs_data.data
                details["variables"] = len(ds.data_vars)
                if "time" in ds.sizes:
                    details["time_steps"] = ds.sizes["time"]
                elif "obs_time" in ds.sizes:
                    details["time_steps"] = ds.sizes["obs_time"]
                # Get observation type
                if hasattr(obs_data, "obs_type"):
                    details["type"] = obs_data.obs_type
                # Count sites/points
                if "site" in ds.sizes:
                    details["sites"] = ds.sizes["site"]
                elif "x" in ds.sizes:
                    details["points"] = ds.sizes["x"]
            self.obs_details[label] = details

        # Extract pair details
        for pair_key, paired_data in context.paired.items():
            details = {}
            ds = paired_data.data if hasattr(paired_data, "data") else paired_data
            if ds is not None:
                # Count paired points
                total_points = 1
                for dim in ds.sizes:
                    total_points *= ds.sizes[dim]
                details["paired_points"] = total_points
                # Count variables
                model_vars = [v for v in ds.data_vars if v.startswith("model_")]
                details["variables"] = len(model_vars)
            self.pair_details[pair_key] = details

        # Extract statistics
        stats_result = context.results.get("statistics")
        if stats_result and stats_result.data:
            self.statistics = stats_result.data

    def _format_number(self, n: int) -> str:
        """Format large numbers with K/M suffix."""
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        elif n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    def to_markdown(self) -> str:
        """Generate Markdown report."""
        lines: list[str] = []

        # Header
        lines.append("# DAVINCI-MONET Pipeline Log")
        lines.append("")

        # Metadata table
        lines.append("## Run Information")
        lines.append("")
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")
        if self.config_path:
            lines.append(f"| Config | `{self.config_path}` |")
        if self.start_time:
            lines.append(f"| Started | {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} |")
        if self.end_time:
            lines.append(f"| Finished | {self.end_time.strftime('%Y-%m-%d %H:%M:%S')} |")
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            lines.append(f"| Duration | {duration:.1f}s |")
        if hasattr(self, "success"):
            status = "Success" if self.success else "Failed"
            lines.append(f"| Status | **{status}** |")
        lines.append("")

        # Stage summary table
        stages = [e for e in self.entries if e.category == "stage"]
        if stages:
            lines.append("## Stage Summary")
            lines.append("")
            lines.append("| Stage | Status | Duration |")
            lines.append("|-------|--------|----------|")
            for entry in stages:
                status_icon = "✓" if entry.status == "completed" else "✗"
                lines.append(
                    f"| {entry.name} | {status_icon} {entry.status.title()} | {entry.duration:.1f}s |"
                )
            lines.append("")

        # Models table with details
        models = [e for e in self.entries if e.category == "model"]
        if models:
            lines.append("## Models Loaded")
            lines.append("")
            lines.append("| Model | Variables | Time Steps | Data Points | Duration |")
            lines.append("|-------|-----------|------------|-------------|----------|")
            for entry in models:
                details = self.model_details.get(entry.name, {})
                vars_count = details.get("variables", "-")
                time_steps = details.get("time_steps", "-")
                data_points = details.get("data_points")
                data_str = self._format_number(data_points) if data_points else "-"
                lines.append(
                    f"| {entry.name} | {vars_count} | {time_steps} | {data_str} | {entry.duration:.1f}s |"
                )
            lines.append("")

        # Observations table with details
        observations = [e for e in self.entries if e.category == "observation"]
        if observations:
            lines.append("## Observations Loaded")
            lines.append("")
            lines.append("| Observation | Type | Variables | Records | Duration |")
            lines.append("|-------------|------|-----------|---------|----------|")
            for entry in observations:
                details = self.obs_details.get(entry.name, {})
                obs_type = details.get("type", "-")
                vars_count = details.get("variables", "-")
                # Get record count (sites, points, or time steps)
                records = (
                    details.get("sites")
                    or details.get("points")
                    or details.get("time_steps")
                    or "-"
                )
                if isinstance(records, int):
                    records = self._format_number(records)
                lines.append(
                    f"| {entry.name} | {obs_type} | {vars_count} | {records} | {entry.duration:.1f}s |"
                )
            lines.append("")

        # Pairings table with details
        pairs = [e for e in self.entries if e.category == "pair"]
        if pairs:
            lines.append("## Pairings")
            lines.append("")
            lines.append("| Pair | Variables | Paired Points | Duration |")
            lines.append("|------|-----------|---------------|----------|")
            for entry in pairs:
                details = self.pair_details.get(entry.name, {})
                vars_count = details.get("variables", "-")
                paired_points = details.get("paired_points")
                points_str = self._format_number(paired_points) if paired_points else "-"
                lines.append(
                    f"| {entry.name} | {vars_count} | {points_str} | {entry.duration:.1f}s |"
                )
            lines.append("")

        # Statistics summary
        if self.statistics:
            lines.append("## Statistics Summary")
            lines.append("")
            for pair_key, pair_stats in self.statistics.items():
                if not pair_stats:
                    continue
                lines.append(f"### {pair_key}")
                lines.append("")
                lines.append("| Variable | N | Mean Obs | Mean Model | MB | RMSE | R |")
                lines.append("|----------|---|----------|------------|-----|------|---|")
                for var_name, stats in pair_stats.items():
                    if not isinstance(stats, dict):
                        continue
                    n = stats.get("n", "-")
                    obs_mean = stats.get("obs_mean")
                    model_mean = stats.get("model_mean")
                    mb = stats.get("mean_bias")
                    rmse = stats.get("rmse")
                    r = stats.get("correlation")
                    # Format values
                    obs_str = f"{obs_mean:.2f}" if obs_mean is not None else "-"
                    model_str = f"{model_mean:.2f}" if model_mean is not None else "-"
                    mb_str = f"{mb:+.2f}" if mb is not None else "-"
                    rmse_str = f"{rmse:.2f}" if rmse is not None else "-"
                    r_str = f"{r:.2f}" if r is not None and not (isinstance(r, float) and r != r) else "-"
                    lines.append(
                        f"| {var_name} | {n} | {obs_str} | {model_str} | {mb_str} | {rmse_str} | {r_str} |"
                    )
                lines.append("")

        # Errors section (if any)
        if self.errors:
            lines.append("## Errors")
            lines.append("")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"### Error {i}: {error['error_type']} in `{error['stage']}`")
                lines.append("")
                lines.append(f"**Time:** {error['timestamp']}")
                lines.append("")
                lines.append(f"**Message:** {error['error_message']}")
                lines.append("")
                if error.get("traceback"):
                    lines.append("**Traceback:**")
                    lines.append("```")
                    lines.append(error["traceback"])
                    lines.append("```")
                    lines.append("")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("*Generated by DAVINCI-MONET*")
        lines.append("")

        return "\n".join(lines)


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


class ProgressFormatter:
    """Formats pipeline progress output with nice ASCII styling.

    Uses box-drawing characters and status icons for clear, readable output.
    """

    # Box drawing characters
    BOX_TL = "┌"
    BOX_TR = "┐"
    BOX_BL = "└"
    BOX_BR = "┘"
    BOX_H = "─"
    BOX_V = "│"
    BOX_T = "┬"
    BOX_B = "┴"

    # Status icons
    ICON_OK = "✓"
    ICON_FAIL = "✗"
    ICON_ARROW = "→"
    ICON_BULLET = "•"
    ICON_SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    # Width for formatting
    WIDTH = 72

    def __init__(self, show_output: bool = True) -> None:
        self.show_output = show_output
        self._current_stage: str | None = None
        self._stage_start: float | None = None
        self._lines: list[str] = []  # For log file

    def _emit(self, line: str) -> None:
        """Output a line and store for logging."""
        self._lines.append(line)
        if self.show_output:
            try:
                print(line, file=sys.stdout, flush=True)
            except BrokenPipeError:
                pass

    def header(self, config_path: str | None = None) -> None:
        """Print pipeline header."""
        self._emit("")
        self._emit(self.BOX_TL + self.BOX_H * (self.WIDTH - 2) + self.BOX_TR)
        title = "DAVINCI-MONET Pipeline"
        padding = (self.WIDTH - 2 - len(title)) // 2
        self._emit(self.BOX_V + " " * padding + title + " " * (self.WIDTH - 2 - padding - len(title)) + self.BOX_V)
        self._emit(self.BOX_BL + self.BOX_H * (self.WIDTH - 2) + self.BOX_BR)
        self._emit("")
        if config_path:
            # Truncate path if too long
            max_path_len = self.WIDTH - 10
            display_path = config_path
            if len(config_path) > max_path_len:
                display_path = "..." + config_path[-(max_path_len - 3):]
            self._emit(f"  Config: {display_path}")
            self._emit("")

    def stage_start(self, name: str) -> None:
        """Print stage start."""
        self._current_stage = name
        self._stage_start = time.time()
        self._emit(self.BOX_TL + self.BOX_H + f" {name} " + self.BOX_H * (self.WIDTH - len(name) - 5))

    def stage_end(self, name: str, success: bool, duration: float) -> None:
        """Print stage end."""
        icon = self.ICON_OK if success else self.ICON_FAIL
        status = "completed" if success else "FAILED"
        self._emit(self.BOX_BL + self.BOX_H + f" {icon} {name} {status} ({duration:.1f}s)")
        self._emit("")
        self._current_stage = None

    def item_start(self, category: str, name: str, index: int, total: int) -> None:
        """Print item start (model, observation, pair)."""
        self._emit(f"{self.BOX_V}  {self.ICON_ARROW} {name} ({index}/{total})")

    def step(self, message: str) -> None:
        """Print a step within an item."""
        self._emit(f"{self.BOX_V}      {self.ICON_BULLET} {message}")

    def item_done(self, summary: str) -> None:
        """Print item completion with summary."""
        self._emit(f"{self.BOX_V}      {self.ICON_OK} {summary}")

    def item_fail(self, error: str) -> None:
        """Print item failure."""
        # Truncate error if too long
        max_len = self.WIDTH - 12
        if len(error) > max_len:
            error = error[:max_len - 3] + "..."
        self._emit(f"{self.BOX_V}      {self.ICON_FAIL} {error}")

    def footer(self, success: bool, duration: float, log_path: Path | None = None) -> None:
        """Print pipeline footer."""
        self._emit(self.BOX_TL + self.BOX_H * (self.WIDTH - 2) + self.BOX_TR)
        if success:
            msg = f"{self.ICON_OK} Pipeline completed successfully in {duration:.1f}s"
        else:
            msg = f"{self.ICON_FAIL} Pipeline failed after {duration:.1f}s"
        padding = (self.WIDTH - 2 - len(msg)) // 2
        self._emit(self.BOX_V + " " * padding + msg + " " * (self.WIDTH - 2 - padding - len(msg)) + self.BOX_V)
        self._emit(self.BOX_BL + self.BOX_H * (self.WIDTH - 2) + self.BOX_BR)
        if log_path:
            self._emit(f"  Log: {log_path}")
        self._emit("")

    def get_log_lines(self) -> list[str]:
        """Get all output lines for logging."""
        return self._lines.copy()


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

        # Set up logging and formatting
        log_path: Path | None = None
        log_collector: LogCollector | None = None
        formatter = ProgressFormatter(show_output=self._show_progress)

        analysis_config = context.config.get("analysis", {})
        log_dir = analysis_config.get("log_dir")
        config_path = context.metadata.get("config_path")

        if log_dir:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)

            # Create timestamped log file with .md extension
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir_path / f"pipeline_{timestamp}.md"

            # Initialize log collector
            log_collector = LogCollector()
            log_collector.start_pipeline(config_path=config_path)

        # Print header
        formatter.header(config_path=config_path)

        # Set up progress callback that uses formatter and collects data
        def _make_progress_callback(
            fmt: ProgressFormatter,
            collector: LogCollector | None,
        ) -> Callable[[str], None]:
            """Create progress callback for formatted output and log collection."""
            def callback(msg: str) -> None:
                # Collect structured data for Markdown log
                if collector:
                    collector.log_item(msg)
                # Parse message type and format appropriately
                if msg.strip().startswith("Loading model:"):
                    match = re.match(r"\s*Loading model: (\S+) \((\d+)/(\d+)\)", msg)
                    if match:
                        name, idx, total = match.groups()
                        fmt.item_start("model", name, int(idx), int(total))
                elif msg.strip().startswith("Loading obs:"):
                    match = re.match(r"\s*Loading obs: (\S+) \((\d+)/(\d+)\)", msg)
                    if match:
                        name, idx, total = match.groups()
                        fmt.item_start("obs", name, int(idx), int(total))
                elif msg.strip().startswith("Pairing:"):
                    match = re.match(r"\s*Pairing: (\S+) \((\d+)/(\d+)\)", msg)
                    if match:
                        name, idx, total = match.groups()
                        fmt.item_start("pair", name, int(idx), int(total))
                elif msg.strip().startswith("Stats:"):
                    match = re.match(r"\s*Stats: (\S+) \((\d+)/(\d+)\)", msg)
                    if match:
                        name, idx, total = match.groups()
                        fmt.item_start("stats", name, int(idx), int(total))
                elif msg.strip().startswith("Plot:"):
                    match = re.match(r"\s*Plot: (\S+) \((\d+)/(\d+)\)", msg)
                    if match:
                        name, idx, total = match.groups()
                        fmt.item_start("plot", name, int(idx), int(total))
                elif msg.strip().startswith("step:"):
                    # Step messages from stages
                    step_msg = msg.strip()[5:].strip()
                    fmt.step(step_msg)
                elif msg.strip().startswith("done:"):
                    # Completion messages from stages
                    done_msg = msg.strip()[5:].strip()
                    fmt.item_done(done_msg)
            return callback

        context.progress_callback = _make_progress_callback(formatter, log_collector)

        result = PipelineResult(
            success=True,
            context=context,
            start_time=datetime.now(),
        )

        start_time = time.time()
        self._call_hook("on_start", context)

        try:
            for stage in self._stages:
                # Start stage in formatter and collector
                formatter.stage_start(stage.name)
                if log_collector:
                    log_collector.start_stage(stage.name)

                stage_result = self._execute_stage(stage, context)
                result.stage_results.append(stage_result)

                # Store result in context
                context.results[stage.name] = stage_result

                # Finalize any open items before ending stage
                if log_collector:
                    log_collector.finalize_items()

                if stage_result.status == StageStatus.FAILED:
                    result.success = False
                    formatter.stage_end(stage.name, False, stage_result.duration_seconds)
                    if log_collector:
                        log_collector.end_stage(stage.name, "failed", stage_result.duration_seconds)
                        # Log the error with traceback for the Markdown report
                        if stage_result.error:
                            log_collector.log_error(
                                stage_name=stage.name,
                                error_type=stage_result.error_type or "Exception",
                                error_message=stage_result.error,
                                traceback_str=stage_result.traceback_str,
                            )
                    if self._fail_fast:
                        logger.error(
                            f"Pipeline failed at stage '{stage.name}': "
                            f"{stage_result.error}"
                        )
                        break
                elif stage_result.status == StageStatus.COMPLETED:
                    formatter.stage_end(stage.name, True, stage_result.duration_seconds)
                    if log_collector:
                        log_collector.end_stage(stage.name, "completed", stage_result.duration_seconds)

        finally:
            # Print footer
            total_duration = time.time() - start_time
            formatter.footer(result.success, total_duration, log_path)

            # Write Markdown log file
            if log_collector and log_path:
                log_collector.end_pipeline(result.success)
                # Extract detailed data from context for the report
                log_collector.extract_context_data(context)
                try:
                    log_path.write_text(log_collector.to_markdown())
                except Exception as e:
                    logger.warning(f"Failed to write log file: {e}")

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
        config_path: str | None = None
        if isinstance(config, str):
            config_path = config
            from davinci_monet.config import load_config
            config = load_config(config).model_dump()

        context = PipelineContext(config=config)
        if config_path:
            context.metadata["config_path"] = config_path
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
            tb_str = traceback.format_exc()
            result = StageResult(
                stage_name=stage.name,
                status=StageStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
                traceback_str=tb_str,
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
