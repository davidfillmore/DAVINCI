"""Pipeline stage definitions.

This module provides the Stage protocol and concrete stage implementations
for the analysis pipeline. Each stage is a composable unit of work that
transforms data through the analysis workflow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import xarray as xr

from davinci_monet.core.exceptions import PipelineError
from davinci_monet.core.protocols import DataGeometry


class StageStatus(Enum):
    """Status of a pipeline stage."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class StageResult:
    """Result of a pipeline stage execution.

    Attributes
    ----------
    stage_name
        Name of the stage that produced this result.
    status
        Execution status.
    data
        Output data from the stage.
    metadata
        Additional metadata about the execution.
    error
        Error message if the stage failed.
    error_type
        Exception class name if the stage failed (e.g., 'ValueError').
    traceback_str
        Full traceback string if the stage failed with an exception.
    duration_seconds
        Execution time in seconds.
    """

    stage_name: str
    status: StageStatus
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_type: str | None = None
    traceback_str: str | None = None
    duration_seconds: float = 0.0


@runtime_checkable
class Stage(Protocol):
    """Protocol for pipeline stages.

    A stage is a single unit of work in the analysis pipeline.
    Stages can be composed and chained together.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        ...

    def execute(self, context: PipelineContext) -> StageResult:
        """Execute the stage.

        Parameters
        ----------
        context
            Pipeline context containing configuration and data.

        Returns
        -------
        StageResult
            Result of stage execution.
        """
        ...

    def validate(self, context: PipelineContext) -> bool:
        """Validate that the stage can run with the given context.

        Parameters
        ----------
        context
            Pipeline context to validate.

        Returns
        -------
        bool
            True if validation passes.
        """
        ...


@dataclass
class PipelineContext:
    """Context passed between pipeline stages.

    Contains configuration, data, and state that flows through the pipeline.

    Attributes
    ----------
    config
        Configuration dictionary from YAML or programmatic setup.
    models
        Dictionary of loaded model data.
    observations
        Dictionary of loaded observation data.
    paired
        Dictionary of paired model-observation data.
    results
        Results from completed stages.
    metadata
        Pipeline metadata (start time, etc.).
    progress_callback
        Optional callback for reporting progress within stages.
        Called with a message string to display progress updates.
    """

    config: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    observations: dict[str, Any] = field(default_factory=dict)
    paired: dict[str, Any] = field(default_factory=dict)
    results: dict[str, StageResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    progress_callback: Callable[[str], None] | None = None

    def log_progress(self, message: str) -> None:
        """Log a progress message if callback is set."""
        if self.progress_callback:
            self.progress_callback(message)

    def get_model(self, label: str) -> Any:
        """Get a model by label."""
        if label not in self.models:
            raise KeyError(f"Model '{label}' not found in context")
        return self.models[label]

    def get_observation(self, label: str) -> Any:
        """Get an observation by label."""
        if label not in self.observations:
            raise KeyError(f"Observation '{label}' not found in context")
        return self.observations[label]

    def get_paired(self, key: str) -> Any:
        """Get paired data by key."""
        if key not in self.paired:
            raise KeyError(f"Paired data '{key}' not found in context")
        return self.paired[key]


class BaseStage(ABC):
    """Abstract base class for pipeline stages.

    Provides common functionality for stage implementations.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialize stage.

        Parameters
        ----------
        name
            Optional custom name. If None, uses class name.
        """
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        """Stage name."""
        return self._name

    def validate(self, context: PipelineContext) -> bool:
        """Default validation - always passes.

        Override in subclasses for specific validation.
        """
        return True

    @abstractmethod
    def execute(self, context: PipelineContext) -> StageResult:
        """Execute the stage."""
        ...

    def _create_result(
        self,
        status: StageStatus,
        data: Any = None,
        error: str | None = None,
        duration: float = 0.0,
        **metadata: Any,
    ) -> StageResult:
        """Create a stage result."""
        return StageResult(
            stage_name=self.name,
            status=status,
            data=data,
            error=error,
            duration_seconds=duration,
            metadata=metadata,
        )


def _format_size(n: int) -> str:
    """Format large numbers with K/M/B suffix."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m{secs:.0f}s"


class LoadModelsStage(BaseStage):
    """Stage for loading model data.

    Reads model configuration and loads model files into the context.
    """

    def __init__(self) -> None:
        super().__init__("load_models")

    def validate(self, context: PipelineContext) -> bool:
        """Validate that model config exists."""
        return "model" in context.config or "models" in context.config

    def execute(self, context: PipelineContext) -> StageResult:
        """Load model data from configuration."""
        import time
        from glob import glob

        from davinci_monet.models import open_model

        start = time.time()
        model_config = context.config.get("model") or context.config.get("models", {})
        total_models = len(model_config)
        debug = context.config.get("analysis", {}).get("debug", False)

        loaded_count = 0
        for label, config in model_config.items():
            try:
                context.log_progress(f"    Loading model: {label} ({loaded_count + 1}/{total_models})")

                files = config.get("files", config.get("filename"))
                mod_type = config.get("mod_type", "generic")
                variables = config.get("variables")

                # Count files for progress message
                t0 = time.time()
                if isinstance(files, str) and ("*" in files or "?" in files):
                    file_list = glob(files)
                    n_files = len(file_list)
                    if debug:
                        context.log_progress(f"      [TIMING] glob: {_format_duration(time.time() - t0)}")
                    context.log_progress(f"step: Opening {n_files} files...")
                else:
                    context.log_progress(f"step: Opening dataset...")

                if isinstance(variables, dict):
                    var_list = list(variables.keys())
                else:
                    var_list = variables

                t0 = time.time()
                model_data = open_model(
                    files=files,
                    mod_type=mod_type,
                    variables=var_list,
                    label=label,
                )
                if debug:
                    context.log_progress(f"      [TIMING] open_model: {_format_duration(time.time() - t0)}")

                # Apply unit scaling, units, and display_name if configured
                if isinstance(variables, dict):
                    has_conversions = any(
                        isinstance(vc, dict) and "unit_scale" in vc
                        for vc in variables.values()
                    )
                    if has_conversions:
                        context.log_progress("step: Applying unit conversions...")

                    t0 = time.time()
                    for var_name, var_config in variables.items():
                        if isinstance(var_config, dict) and var_name in model_data.data.data_vars:
                            if "unit_scale" in var_config:
                                scale = var_config["unit_scale"]
                                method = var_config.get("unit_scale_method", "*")
                                model_data.apply_unit_scale(var_name, scale, method)
                            if "units" in var_config:
                                model_data.data[var_name].attrs["units"] = var_config["units"]
                            if var_config.get("display_name"):  # Only set if not None/empty
                                model_data.data[var_name].attrs["display_name"] = var_config["display_name"]
                    if debug and has_conversions:
                        context.log_progress(f"      [TIMING] unit_conversions: {_format_duration(time.time() - t0)}")

                context.models[label] = model_data
                loaded_count += 1

                # Summary message
                ds = model_data.data
                n_vars = len(ds.data_vars)
                n_times = ds.sizes.get("time", 0)
                context.log_progress(f"done: {n_vars} vars, {n_times} times")

            except Exception as e:
                return self._create_result(
                    StageStatus.FAILED,
                    error=f"Failed to load model '{label}': {e}",
                    duration=time.time() - start,
                )

        return self._create_result(
            StageStatus.COMPLETED,
            data={"loaded_models": list(context.models.keys())},
            duration=time.time() - start,
            count=loaded_count,
        )


class LoadObservationsStage(BaseStage):
    """Stage for loading observation data.

    Reads observation configuration and loads observation files into the context.
    """

    def __init__(self) -> None:
        super().__init__("load_observations")

    def validate(self, context: PipelineContext) -> bool:
        """Validate that observation config exists."""
        return "obs" in context.config or "observations" in context.config

    def execute(self, context: PipelineContext) -> StageResult:
        """Load observation data from configuration."""
        import time
        from glob import glob
        from pathlib import Path

        import xarray as xr

        from davinci_monet.observations import create_observation_data

        start = time.time()
        obs_config = context.config.get("obs") or context.config.get("observations", {})
        total_obs = len(obs_config)
        debug = context.config.get("analysis", {}).get("debug", False)

        # Get analysis time range for filtering
        analysis_config = context.config.get("analysis", {})
        analysis_start = analysis_config.get("start_time")
        analysis_end = analysis_config.get("end_time")

        # Use current working directory for relative paths
        base_path = Path.cwd()

        loaded_count = 0
        for label, config in obs_config.items():
            try:
                context.log_progress(f"    Loading obs: {label} ({loaded_count + 1}/{total_obs})")

                obs_type = config.get("obs_type", "pt_sfc")
                filename = config.get("filename")
                variables = config.get("variables", {})

                # Load data from file
                data = None
                if filename:
                    file_path = Path(filename)
                    # Expand user home directory first (before checking absolute)
                    file_path = file_path.expanduser()

                    # Handle relative paths
                    if not file_path.is_absolute():
                        file_path = base_path / file_path

                    # Handle glob patterns
                    if "*" in str(file_path) or "?" in str(file_path):
                        t0 = time.time()
                        files = sorted(glob(str(file_path)))
                        if debug:
                            context.log_progress(f"      [TIMING] glob: {_format_duration(time.time() - t0)}")

                        # Pre-filter files by date in filename (if analysis time range specified)
                        if files and analysis_start and analysis_end:
                            original_count = len(files)
                            files = self._filter_files_by_date(files, analysis_start, analysis_end)
                            if len(files) < original_count:
                                context.log_progress(
                                    f"step: Filtered {original_count} -> {len(files)} files by date"
                                )
                            if not files:
                                context.log_progress(
                                    f"done: No files in analysis date range, skipping"
                                )
                                loaded_count += 1
                                continue  # Skip this observation

                        if files:
                            n_files = len(files)
                            # Check if ICARTT files (.ict extension)
                            if files[0].endswith(".ict"):
                                context.log_progress(f"step: Reading {n_files} ICARTT files...")
                                t0 = time.time()
                                data = self._load_icartt_files(files)
                                if debug:
                                    context.log_progress(f"      [TIMING] load_icartt: {_format_duration(time.time() - t0)}")
                            else:
                                context.log_progress(f"step: Opening {n_files} files...")
                                t0 = time.time()
                                data = xr.open_mfdataset(files, combine="by_coords", parallel=True)
                                if debug:
                                    context.log_progress(f"      [TIMING] open_mfdataset: {_format_duration(time.time() - t0)}")
                    elif file_path.exists():
                        context.log_progress("step: Opening dataset...")
                        t0 = time.time()
                        if str(file_path).endswith(".ict"):
                            data = self._load_icartt_files([str(file_path)])
                        elif label == "aeronet" or "aeronet" in str(file_path).lower():
                            # Use AERONET reader for proper dimension handling
                            from davinci_monet.observations.surface.aeronet import AERONETReader
                            reader = AERONETReader()
                            data = reader.open([str(file_path)])
                        else:
                            data = xr.open_dataset(str(file_path))
                        if debug:
                            context.log_progress(f"      [TIMING] open_dataset: {_format_duration(time.time() - t0)}")

                # Filter by analysis time range if specified
                if data is not None and "time" in data.dims and analysis_start and analysis_end:
                    t0 = time.time()
                    original_size = data.sizes.get("time", 0)
                    data = self._filter_by_time(data, analysis_start, analysis_end)
                    filtered_size = data.sizes.get("time", 0)
                    if debug:
                        context.log_progress(
                            f"      [TIMING] time_filter: {_format_duration(time.time() - t0)} "
                            f"({original_size} -> {filtered_size} times)"
                        )
                    elif filtered_size < original_size:
                        context.log_progress(
                            f"step: Filtered to analysis period ({filtered_size} times)"
                        )

                t0 = time.time()
                obs_data = create_observation_data(
                    label=label,
                    obs_type=obs_type,
                    data=data,
                    filename=filename,
                    variables=variables,
                )
                if debug:
                    context.log_progress(f"      [TIMING] create_observation_data: {_format_duration(time.time() - t0)}")

                # Apply temporal averaging if configured
                resample_freq = config.get("resample")
                if resample_freq:
                    min_count = config.get("min_obs_count")
                    track_count = config.get("track_obs_count", False)
                    original_times = obs_data.data.sizes.get("time", 0) if obs_data.data is not None else 0

                    t0 = time.time()
                    obs_data.resample_data(
                        freq=resample_freq,
                        min_count=min_count,
                        track_count=track_count,
                    )
                    new_times = obs_data.data.sizes.get("time", 0) if obs_data.data is not None else 0

                    if debug:
                        context.log_progress(
                            f"      [TIMING] resample ({resample_freq}): {_format_duration(time.time() - t0)} "
                            f"({original_times} -> {new_times} times)"
                        )
                    else:
                        context.log_progress(
                            f"step: Resampled to {resample_freq} ({original_times} -> {new_times} times)"
                        )

                # Apply unit scaling, units, and display_name if configured
                if isinstance(variables, dict):
                    for var_name, var_config in variables.items():
                        if isinstance(var_config, dict) and var_name in obs_data.data.data_vars:
                            if "unit_scale" in var_config:
                                scale = var_config["unit_scale"]
                                method = var_config.get("unit_scale_method", "*")
                                obs_data.apply_unit_scale(var_name, scale, method)
                            if "units" in var_config:
                                obs_data.data[var_name].attrs["units"] = var_config["units"]
                            if var_config.get("display_name"):  # Only set if not None/empty
                                obs_data.data[var_name].attrs["display_name"] = var_config["display_name"]

                context.observations[label] = obs_data
                loaded_count += 1

                # Summary message
                ds = obs_data.data
                n_vars = len(ds.data_vars)
                # Get record count (sites, points, or time steps)
                n_records = (
                    ds.sizes.get("site")
                    or ds.sizes.get("x")
                    or ds.sizes.get("time")
                    or 0
                )
                context.log_progress(f"done: {n_vars} vars, {_format_size(n_records)} records")

            except Exception as e:
                return self._create_result(
                    StageStatus.FAILED,
                    error=f"Failed to load observation '{label}': {e}",
                    duration=time.time() - start,
                )

        return self._create_result(
            StageStatus.COMPLETED,
            data={"loaded_observations": list(context.observations.keys())},
            duration=time.time() - start,
            count=loaded_count,
        )

    def _load_icartt_files(self, files: list[str]) -> "xr.Dataset":
        """Load ICARTT format files using the specialized reader.

        Parameters
        ----------
        files
            List of ICARTT file paths.

        Returns
        -------
        xr.Dataset
            Combined dataset from all files.
        """
        from davinci_monet.observations.aircraft.icartt import ICARTTReader

        reader = ICARTTReader()
        return reader.open(files)

    def _filter_files_by_date(
        self,
        files: list[str],
        start_time: str,
        end_time: str,
    ) -> list[str]:
        """Filter file list by dates extracted from filenames.

        Looks for YYYYMMDD patterns in filenames and keeps only files
        within the analysis date range.

        Parameters
        ----------
        files
            List of file paths.
        start_time
            Start of analysis period (ISO format string).
        end_time
            End of analysis period (ISO format string).

        Returns
        -------
        list[str]
            Filtered file list.
        """
        import re
        import pandas as pd

        t_start = pd.Timestamp(start_time).date()
        t_end = pd.Timestamp(end_time).date()

        # Pattern to match YYYYMMDD in filename
        date_pattern = re.compile(r"(\d{4})(\d{2})(\d{2})")

        filtered = []
        for f in files:
            # Search for date pattern in filename (not full path)
            filename = f.split("/")[-1]
            match = date_pattern.search(filename)
            if match:
                try:
                    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    file_date = pd.Timestamp(year=year, month=month, day=day).date()
                    if t_start <= file_date <= t_end:
                        filtered.append(f)
                except ValueError:
                    # Invalid date, include file to be safe
                    filtered.append(f)
            else:
                # No date in filename, include file
                filtered.append(f)

        return filtered

    def _filter_by_time(
        self,
        data: "xr.Dataset",
        start_time: str,
        end_time: str,
    ) -> "xr.Dataset":
        """Filter dataset to analysis time range.

        Parameters
        ----------
        data
            Dataset with time dimension.
        start_time
            Start of analysis period (ISO format string).
        end_time
            End of analysis period (ISO format string).

        Returns
        -------
        xr.Dataset
            Dataset filtered to time range.
        """
        import pandas as pd

        # Parse time bounds - add 1 day to end to make it inclusive
        t_start = pd.Timestamp(start_time)
        t_end = pd.Timestamp(end_time) + pd.Timedelta(days=1)

        # Use sel with slice for efficient time filtering
        return data.sel(time=slice(t_start, t_end))


class PairingStage(BaseStage):
    """Stage for pairing model and observation data.

    Uses the pairing engine to match model output with observations.
    """

    def __init__(self) -> None:
        super().__init__("pairing")

    def validate(self, context: PipelineContext) -> bool:
        """Validate that models and observations are loaded."""
        return bool(context.models) and bool(context.observations)

    def execute(self, context: PipelineContext) -> StageResult:
        """Pair model and observation data.

        Uses two-phase execution to avoid GIL contention between Dask-backed
        and eager (non-Dask) models. Dask models spawn many threads during
        compute() which can block eager model pairings if run simultaneously.
        """
        import os
        import time

        from davinci_monet.pairing import PairingEngine, PairingConfig

        start = time.time()

        # Get pairing configuration
        pairing_config_dict = context.config.get("pairing", {})

        # Build list of pairs to process, separating Dask-backed from eager models
        # Each tuple includes an index for consistent ordering in output messages
        dask_pairs: list[tuple[int, str, Any, str, Any, dict, dict]] = []
        eager_pairs: list[tuple[int, str, Any, str, Any, dict, dict]] = []
        pair_index = 0

        for model_label, model_data in context.models.items():
            model_config = context.config.get("model", {}).get(model_label, {})
            mapping = model_config.get("mapping", {})

            # Check if model is Dask-backed
            model_ds = model_data.data if hasattr(model_data, "data") else model_data
            is_dask = self._is_dask_backed(model_ds)

            for obs_label, obs_data in context.observations.items():
                if mapping and obs_label not in mapping:
                    continue

                var_mapping = mapping.get(obs_label, {})
                if not var_mapping:
                    continue

                pair_index += 1
                pair_tuple = (
                    pair_index, model_label, model_data,
                    obs_label, obs_data,
                    model_config, var_mapping
                )

                if is_dask:
                    dask_pairs.append(pair_tuple)
                else:
                    eager_pairs.append(pair_tuple)

        total_pairs = len(dask_pairs) + len(eager_pairs)
        if total_pairs == 0:
            return self._create_result(
                StageStatus.COMPLETED,
                data={"paired_keys": []},
                duration=time.time() - start,
                count=0,
            )

        debug = context.config.get("analysis", {}).get("debug", False)

        def pair_single(args: tuple) -> tuple[int, str, Any, str | None, float]:
            """Process a single model-obs pair. Returns (index, pair_key, paired_ds, error, duration)."""
            import time as time_mod
            pair_start = time_mod.time()
            idx, model_label, model_data, obs_label, obs_data, model_config, var_mapping = args
            pair_key = f"{model_label}_{obs_label}"

            try:
                obs_vars = list(var_mapping.keys())
                model_vars = list(var_mapping.values())

                model_ds = model_data.data if hasattr(model_data, "data") else model_data
                obs_ds = obs_data.data if hasattr(obs_data, "data") else obs_data

                if model_ds is None or obs_ds is None:
                    return (idx, pair_key, None, "Model or obs data is None", time_mod.time() - pair_start)

                radius = model_config.get("radius_of_influence", 12000.0)
                pairing_cfg = PairingConfig(
                    radius_of_influence=radius,
                    time_tolerance=pairing_config_dict.get("time_tolerance", "1h"),
                )

                engine = PairingEngine()
                paired_ds = engine.pair(
                    model_ds,
                    obs_ds,
                    obs_vars=obs_vars,
                    model_vars=model_vars,
                    config=pairing_cfg,
                )

                return (idx, pair_key, paired_ds, None, time_mod.time() - pair_start)

            except Exception as e:
                return (idx, pair_key, None, str(e), time_mod.time() - pair_start)

        paired_count = 0

        def run_phase(pairs: list, phase_name: str) -> None:
            """Run a phase of pairs in parallel."""
            nonlocal paired_count
            from concurrent.futures import ThreadPoolExecutor, as_completed

            if not pairs:
                return

            # Run pairs in parallel
            max_workers = min(len(pairs), os.cpu_count() or 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                # Submit each pair and log start message
                for args in pairs:
                    idx, model_label, _, obs_label, _, _, _ = args
                    pair_key = f"{model_label}_{obs_label}"
                    context.log_progress(f"    parallel_started: {pair_key}")
                    futures[executor.submit(pair_single, args)] = args

                # Collect results as they complete
                for future in as_completed(futures):
                    idx, pair_key, paired_ds, error, pair_duration = future.result()

                    if error:
                        context.metadata.setdefault("pairing_errors", []).append(
                            f"{pair_key}: {error}"
                        )
                        if debug:
                            context.log_progress(f"      [TIMING] {pair_key} failed: {_format_duration(pair_duration)}")
                        # Still count as "completed" for progress display
                        context.log_progress(f"    parallel_completed: {pair_key} - FAILED")
                    elif paired_ds is not None:
                        context.paired[pair_key] = paired_ds
                        paired_count += 1

                        # Summary message
                        paired_data = paired_ds.data if hasattr(paired_ds, "data") else paired_ds
                        n_vars = len(paired_data.data_vars) // 2  # model_ and obs_ vars
                        n_points = paired_data.sizes.get("time", paired_data.sizes.get("x", 0))
                        timing_str = f" [{_format_duration(pair_duration)}]" if debug else ""
                        context.log_progress(
                            f"    parallel_completed: {pair_key} - "
                            f"{n_vars} vars, {_format_size(n_points)} points{timing_str}"
                        )

        # Build loading message for parallel mode display
        # Group Dask pairs by model to show what's being loaded
        loading_msg = ""
        if dask_pairs:
            # Extract unique model(s) and their obs from Dask pairs
            dask_models: dict[str, list[str]] = {}
            for _, model_label, _, obs_label, _, _, _ in dask_pairs:
                dask_models.setdefault(model_label, []).append(obs_label)

            # Format: "loading model → obs1, obs2, obs3"
            parts = []
            for model_name, obs_names in dask_models.items():
                obs_str = ", ".join(obs_names)
                parts.append(f"loading {model_name} → {obs_str}")
            loading_msg = "; ".join(parts)

        # Enter parallel mode for progress display
        if loading_msg:
            context.log_progress(f"    parallel_start: {total_pairs} | {loading_msg}")
        else:
            context.log_progress(f"    parallel_start: {total_pairs}")

        # Phase 1: Process Dask-backed model pairs in parallel
        # These share the same Dask scheduler so can run together efficiently
        run_phase(dask_pairs, "dask")

        # Phase 2: Process eager (non-Dask) model pairs in parallel
        # These run after Dask compute() completes, avoiding GIL contention
        run_phase(eager_pairs, "eager")

        # Exit parallel mode
        context.log_progress("    parallel_end")

        return self._create_result(
            StageStatus.COMPLETED,
            data={"paired_keys": list(context.paired.keys())},
            duration=time.time() - start,
            count=paired_count,
        )

    def _is_dask_backed(self, ds: xr.Dataset) -> bool:
        """Check if a dataset has Dask-backed arrays.

        Parameters
        ----------
        ds
            xarray Dataset to check.

        Returns
        -------
        bool
            True if any data variable has Dask chunks.
        """
        for var in ds.data_vars:
            if ds[var].chunks is not None:
                return True
        return False


class StatisticsStage(BaseStage):
    """Stage for calculating statistics on paired data."""

    def __init__(self) -> None:
        super().__init__("statistics")

    def validate(self, context: PipelineContext) -> bool:
        """Validate that paired data exists."""
        return bool(context.paired)

    def execute(self, context: PipelineContext) -> StageResult:
        """Calculate statistics on paired data."""
        import time

        start = time.time()
        stats_results: dict[str, Any] = {}

        stats_config = context.config.get("stats", {})
        total_pairs = len(context.paired)
        pair_count = 0

        for pair_key, paired_obj in context.paired.items():
            try:
                pair_count += 1
                context.log_progress(f"    Stats: {pair_key} ({pair_count}/{total_pairs})")
                context.log_progress("step: Computing metrics...")

                # Handle PairedData objects
                paired_data = paired_obj.data if hasattr(paired_obj, "data") else paired_obj
                # Calculate basic statistics
                pair_stats = self._calculate_stats(paired_data, stats_config)
                stats_results[pair_key] = pair_stats

                # Summary
                n_metrics = sum(len(v) for v in pair_stats.values())
                n_vars = len(pair_stats)
                context.log_progress(f"done: {n_vars} vars, {n_metrics} metrics")

            except Exception as e:
                context.metadata.setdefault("stats_errors", []).append(
                    f"{pair_key}: {e}"
                )

        return self._create_result(
            StageStatus.COMPLETED,
            data=stats_results,
            duration=time.time() - start,
        )

    def _calculate_stats(
        self, paired_data: xr.Dataset, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate statistics for a paired dataset."""
        import numpy as np

        stats: dict[str, Any] = {}

        # Find model and obs variable pairs (prefix format: model_*, obs_*)
        model_vars = [v for v in paired_data.data_vars if v.startswith("model_")]

        for model_var in model_vars:
            base_name = model_var.replace("model_", "", 1)
            obs_var = f"obs_{base_name}"

            if obs_var not in paired_data:
                continue

            model_vals = paired_data[model_var].values.flatten()
            obs_vals = paired_data[obs_var].values.flatten()

            # Remove NaNs
            mask = ~(np.isnan(model_vals) | np.isnan(obs_vals))
            model_vals = model_vals[mask]
            obs_vals = obs_vals[mask]

            if len(model_vals) == 0:
                continue

            # Calculate metrics
            diff = model_vals - obs_vals
            stats[base_name] = {
                "n": len(model_vals),
                "mean_bias": float(np.mean(diff)),
                "rmse": float(np.sqrt(np.mean(diff**2))),
                "correlation": float(np.corrcoef(model_vals, obs_vals)[0, 1])
                if len(model_vals) > 1
                else np.nan,
                "model_mean": float(np.mean(model_vals)),
                "obs_mean": float(np.mean(obs_vals)),
            }

        return stats


class PlottingStage(BaseStage):
    """Stage for generating plots from paired data."""

    def __init__(self) -> None:
        super().__init__("plotting")

    def validate(self, context: PipelineContext) -> bool:
        """Validate that paired data exists."""
        return bool(context.paired)

    def execute(self, context: PipelineContext) -> StageResult:
        """Generate plots from paired data."""
        import time
        from pathlib import Path

        import matplotlib.pyplot as plt

        from davinci_monet.plots import get_plotter
        from davinci_monet.plots.base import format_plot_title

        start = time.time()
        plots_generated: list[str] = []

        plot_config = context.config.get("plots", {})

        # Plotting is optional - if no config, skip
        if not plot_config:
            return self._create_result(
                StageStatus.SKIPPED,
                data={"message": "No plot configuration found"},
                duration=time.time() - start,
            )

        # Get output directory
        analysis_config = context.config.get("analysis", {})
        output_dir = Path(analysis_config.get("output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get pairs config for variable mapping
        pairs_config = context.config.get("pairs", {})
        model_config = context.config.get("model", {})
        total_plots = len(plot_config)
        plot_count = 0

        for plot_name, plot_spec in plot_config.items():
            try:
                plot_count += 1
                plot_type = plot_spec.get("type", "scatter")
                plot_pairs = plot_spec.get("pairs", [])
                title = format_plot_title(plot_spec.get("title", plot_name))

                context.log_progress(f"    Plot: {plot_name} ({plot_count}/{total_plots})")
                context.log_progress(f"step: Rendering {plot_type}...")

                for pair_name in plot_pairs:
                    # Get pair configuration
                    pair_spec = pairs_config.get(pair_name, {})
                    model_label = pair_spec.get("model", "")
                    obs_label = pair_spec.get("obs", "")
                    var_spec = pair_spec.get("variable", {})
                    obs_var = var_spec.get("obs_var", "")

                    # Find paired data
                    pair_key = f"{model_label}_{obs_label}"
                    if pair_key not in context.paired:
                        continue

                    paired_obj = context.paired[pair_key]
                    paired_data = paired_obj.data if hasattr(paired_obj, "data") else paired_obj

                    # Variable names in paired dataset use obs_var with prefixes
                    obs_var_name = f"obs_{obs_var}"
                    model_var_name = f"model_{obs_var}"

                    if obs_var_name not in paired_data or model_var_name not in paired_data:
                        continue

                    # Get plotter config from model variable settings
                    model_var = var_spec.get("model_var", "")
                    var_config = model_config.get(model_label, {}).get("variables", {}).get(model_var, {})
                    vmin = var_config.get("vmin_plot")
                    vmax = var_config.get("vmax_plot")
                    vdiff = var_config.get("vdiff_plot")

                    # Build plotter config
                    plotter_config = {"title": title}
                    if plot_type == "spatial_bias":
                        plotter_config["vmin"] = -vdiff if vdiff else None
                        plotter_config["vmax"] = vdiff if vdiff else None
                    elif plot_type == "timeseries":
                        # Timeseries uses smart auto-scaling from data range
                        # Don't pass vmin/vmax unless explicitly set in plot_spec
                        if "vmin" in plot_spec:
                            plotter_config["vmin"] = plot_spec["vmin"]
                        if "vmax" in plot_spec:
                            plotter_config["vmax"] = plot_spec["vmax"]
                    else:
                        plotter_config["vmin"] = vmin
                        plotter_config["vmax"] = vmax

                    # Extract additional plot options from plot_spec
                    plot_options: dict[str, Any] = {}
                    for opt_key in ["show_site_labels", "show_individual_sites",
                                    "show_uncertainty", "uncertainty_type",
                                    "resample", "aggregate_dim", "label_sites",
                                    "site_label_var", "city_labels",
                                    "show_density", "density_cmap", "alpha"]:
                        if opt_key in plot_spec:
                            plot_options[opt_key] = plot_spec[opt_key]

                    # Add city_labels from analysis config for spatial plots and 3D track maps
                    if (plot_type.startswith("spatial") or plot_type == "track_map_3d") and "city_labels" not in plot_options:
                        city_labels = analysis_config.get("city_labels")
                        if city_labels:
                            plot_options["city_labels"] = city_labels

                    # Get plotter and generate plot
                    plotter = get_plotter(plot_type, config=plotter_config)
                    fig = plotter.plot(paired_data, obs_var_name, model_var_name, **plot_options)

                    # Save plot
                    output_path = output_dir / f"{plot_name}.png"
                    plotter.save(fig, output_path, dpi=300)
                    plots_generated.append(str(output_path))

                    # Also save PDF
                    pdf_path = output_dir / f"{plot_name}.pdf"
                    plotter.save(fig, pdf_path)
                    plots_generated.append(str(pdf_path))

                    plt.close(fig)

                    context.log_progress(f"done: saved PNG + PDF")

            except Exception as e:
                context.metadata.setdefault("plot_errors", []).append(
                    f"{plot_name}: {e}"
                )

        return self._create_result(
            StageStatus.COMPLETED,
            data={"plots_generated": plots_generated},
            duration=time.time() - start,
        )


class SaveResultsStage(BaseStage):
    """Stage for saving analysis results."""

    def __init__(self) -> None:
        super().__init__("save_results")

    def execute(self, context: PipelineContext) -> StageResult:
        """Save analysis results to files."""
        import time
        from pathlib import Path

        import pandas as pd

        start = time.time()
        saved_files: list[str] = []

        # Get output directory from analysis config
        analysis_config = context.config.get("analysis", {})
        output_dir = Path(analysis_config.get("output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save statistics summary from statistics stage
        stats_result = context.results.get("statistics")
        if stats_result and stats_result.data:
            context.log_progress("step: Writing statistics CSV...")
            rows = []
            for pair_key, pair_stats in stats_result.data.items():
                for var_name, var_stats in pair_stats.items():
                    row = {"Variable": var_name}
                    row["N"] = var_stats.get("n", 0)
                    row["Mean_Obs"] = var_stats.get("obs_mean", float("nan"))
                    row["Mean_Model"] = var_stats.get("model_mean", float("nan"))
                    row["MB"] = var_stats.get("mean_bias", float("nan"))
                    row["RMSE"] = var_stats.get("rmse", float("nan"))
                    row["R"] = var_stats.get("correlation", float("nan"))
                    # Calculate NMB and NME
                    obs_mean = var_stats.get("obs_mean", 0)
                    if obs_mean != 0:
                        row["NMB_%"] = (var_stats.get("mean_bias", 0) / obs_mean) * 100
                        row["NME_%"] = (var_stats.get("rmse", 0) / abs(obs_mean)) * 100
                    else:
                        row["NMB_%"] = float("nan")
                        row["NME_%"] = float("nan")
                    row["IOA"] = var_stats.get("ioa", float("nan"))
                    rows.append(row)

            if rows:
                df = pd.DataFrame(rows)
                df = df.set_index("Variable")
                stats_file = output_dir / "statistics_summary.csv"
                df.to_csv(stats_file)
                saved_files.append(str(stats_file))
                context.log_progress(f"done: {len(rows)} rows saved")

        return self._create_result(
            StageStatus.COMPLETED,
            data={"saved_files": saved_files},
            duration=time.time() - start,
        )


# Convenience function to create a standard analysis pipeline
def create_standard_pipeline() -> list[BaseStage]:
    """Create a standard analysis pipeline with all stages.

    Returns
    -------
    list[BaseStage]
        List of stages for a complete analysis.
    """
    return [
        LoadModelsStage(),
        LoadObservationsStage(),
        PairingStage(),
        StatisticsStage(),
        PlottingStage(),
        SaveResultsStage(),
    ]
