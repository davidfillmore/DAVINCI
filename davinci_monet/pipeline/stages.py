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

from davinci_monet.core.exceptions import DataFormatError, PipelineError, write_error_log
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
    # Unified data-source view (Phase 3). Models and observations both register
    # here keyed by label; distinguished only by the dataset's ``role`` attr.
    sources: dict[str, Any] = field(default_factory=dict)
    paired: dict[str, Any] = field(default_factory=dict)
    results: dict[str, StageResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    progress_callback: Callable[[str], None] | None = None

    def log_progress(self, message: str) -> None:
        """Log a progress message if callback is set."""
        if self.progress_callback:
            self.progress_callback(message)

    def get_model(self, label: str) -> Any:
        """Get a model by label.

        Deprecated shim over the unified ``sources`` store: resolves from
        ``models`` first (back-compat), then falls back to ``sources``.
        """
        if label in self.models:
            return self.models[label]
        if label in self.sources:
            return self.sources[label]
        raise KeyError(f"Model '{label}' not found in context")

    def get_observation(self, label: str) -> Any:
        """Get an observation by label.

        Deprecated shim over the unified ``sources`` store: resolves from
        ``observations`` first (back-compat), then falls back to ``sources``.
        """
        if label in self.observations:
            return self.observations[label]
        if label in self.sources:
            return self.sources[label]
        raise KeyError(f"Observation '{label}' not found in context")

    def get_source(self, label: str) -> Any:
        """Get a data source (model or observation) by label.

        Part of the unified data-source abstraction (Phase 3). Sources are
        populated by :class:`LoadSourcesStage`.
        """
        if label not in self.sources:
            raise KeyError(f"Source '{label}' not found in context")
        return self.sources[label]

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


def tag_paired_roles(data: Any) -> None:
    """Tag each paired variable with its ``role`` based on the model_/obs_ prefix.

    Additive metadata (Phase 6): lets the paired output self-describe source roles
    so plot styling can resolve colors by role. Does not rename variables and does
    not overwrite a pre-existing ``role`` attr.
    """
    if data is None or not hasattr(data, "data_vars"):
        return
    for name in data.data_vars:
        lname = str(name).lower()
        if lname.startswith("model_"):
            data[name].attrs.setdefault("role", "model")
        elif lname.startswith("obs_"):
            data[name].attrs.setdefault("role", "obs")


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

        from davinci_monet.core.exceptions import ConfigurationError
        from davinci_monet.models import open_model

        start = time.time()
        model_config = context.config.get("model") or context.config.get("models", {})
        total_models = len(model_config)
        debug = context.config.get("analysis", {}).get("debug", False)

        def _normalize_var_configs(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
            normalized: dict[str, dict[str, Any]] = {}
            for name, cfg in raw.items():
                if hasattr(cfg, "model_dump"):
                    normalized[name] = cfg.model_dump()
                elif isinstance(cfg, dict):
                    normalized[name] = dict(cfg)
                elif hasattr(cfg, "__dict__"):
                    normalized[name] = dict(cfg.__dict__)
                else:
                    normalized[name] = {}
            return normalized

        loaded_count = 0
        for label, config in model_config.items():
            try:
                context.log_progress(
                    f"    Loading model: {label} ({loaded_count + 1}/{total_models})"
                )

                files = config.get("files", config.get("filename"))
                if files is None or (isinstance(files, str) and not files.strip()):
                    raise ConfigurationError(
                        f"Model '{label}' is missing required 'files' (or 'filename') setting."
                    )
                mod_type = config.get("mod_type", "generic")
                variables = config.get("variables")
                mod_kwargs = config.get("mod_kwargs") or {}

                # Count files for progress message
                t0 = time.time()
                if isinstance(files, str) and ("*" in files or "?" in files):
                    file_list = glob(files)
                    n_files = len(file_list)
                    if debug:
                        context.log_progress(
                            f"      [TIMING] glob: {_format_duration(time.time() - t0)}"
                        )
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
                    **mod_kwargs,
                )
                if debug:
                    context.log_progress(
                        f"      [TIMING] open_model: {_format_duration(time.time() - t0)}"
                    )

                # Apply variable configuration (scaling, masking, renaming) and metadata
                if isinstance(variables, dict):
                    model_data.variables = _normalize_var_configs(variables)

                    t0 = time.time()
                    model_data.apply_variable_config()
                    if debug:
                        context.log_progress(
                            f"      [TIMING] variable_config: {_format_duration(time.time() - t0)}"
                        )

                    for var_name, var_config in model_data.variables.items():
                        if model_data.data is not None and var_name in model_data.data.data_vars:
                            units = (
                                var_config.get("units") if isinstance(var_config, dict) else None
                            )
                            display = (
                                var_config.get("display_name")
                                if isinstance(var_config, dict)
                                else None
                            )
                            if units:
                                model_data.data[var_name].attrs["units"] = units
                            if display:
                                model_data.data[var_name].attrs["display_name"] = display

                context.models[label] = model_data
                loaded_count += 1

                # Summary message
                ds = model_data.data
                n_vars = len(ds.data_vars) if ds is not None else 0
                n_times = ds.sizes.get("time", 0) if ds is not None else 0
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
        end_time_has_time = analysis_config.get("_end_time_has_time")

        # Use current working directory for relative paths
        base_path = Path.cwd()

        def _normalize_var_configs(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
            normalized: dict[str, dict[str, Any]] = {}
            for name, cfg in raw.items():
                if hasattr(cfg, "model_dump"):
                    normalized[name] = cfg.model_dump()
                elif isinstance(cfg, dict):
                    normalized[name] = dict(cfg)
                elif hasattr(cfg, "__dict__"):
                    normalized[name] = dict(cfg.__dict__)
                else:
                    normalized[name] = {}
            return normalized

        loaded_count = 0
        for label, config in obs_config.items():
            try:
                context.log_progress(f"    Loading obs: {label} ({loaded_count + 1}/{total_obs})")

                obs_type = config.get("obs_type", "pt_sfc")
                filename = config.get("filename")
                variables = config.get("variables", {})
                normalized_variables = (
                    _normalize_var_configs(variables) if isinstance(variables, dict) else variables
                )

                # Load data from file
                data = None
                sat_type = config.get("sat_type")

                # --- MODIS L2 satellite swath: dedicated loader ---
                if sat_type == "modis_l2":
                    data = self._load_modis_l2(label, config, context)
                    if data is not None:
                        # After gridding, this is a grid product
                        obs_type = "sat_grid_clm"
                        # Variable renaming: the gridded dataset uses the
                        # source SDS names.  Apply rename from variable config.
                        for var_name, var_cfg in (normalized_variables or {}).items():
                            rename_to = var_cfg.get("rename") if isinstance(var_cfg, dict) else None
                            if rename_to and var_name in data.data_vars:
                                data = data.rename({var_name: rename_to})

                if data is None and filename:
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
                            context.log_progress(
                                f"      [TIMING] glob: {_format_duration(time.time() - t0)}"
                            )

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
                                    context.log_progress(
                                        f"      [TIMING] load_icartt: {_format_duration(time.time() - t0)}"
                                    )
                            elif obs_type == "lma":
                                context.log_progress(f"step: Reading {n_files} LMA files...")
                                t0 = time.time()
                                data = self._load_lma_files(files)
                                if debug:
                                    context.log_progress(
                                        f"      [TIMING] load_lma: {_format_duration(time.time() - t0)}"
                                    )
                            else:
                                context.log_progress(f"step: Opening {n_files} files...")
                                t0 = time.time()
                                try:
                                    data = xr.open_mfdataset(
                                        files, combine="by_coords", parallel=True
                                    )
                                except Exception as e:
                                    log_dir = context.config.get("analysis", {}).get("log_dir")
                                    error_file = write_error_log(
                                        e, f"Opening observation files for '{label}'", log_dir
                                    )
                                    msg = f"Failed to open observation files for '{label}': {e}"
                                    if error_file:
                                        msg += f" (details: {error_file})"
                                    raise DataFormatError(msg) from e
                                if debug:
                                    context.log_progress(
                                        f"      [TIMING] open_mfdataset: {_format_duration(time.time() - t0)}"
                                    )
                    elif file_path.exists():
                        context.log_progress("step: Opening dataset...")
                        t0 = time.time()
                        if str(file_path).endswith(".ict"):
                            data = self._load_icartt_files([str(file_path)])
                        elif obs_type == "lma":
                            data = self._load_lma_files([str(file_path)])
                        elif label == "aeronet" or "aeronet" in str(file_path).lower():
                            # Use AERONET reader for proper dimension handling
                            from davinci_monet.observations.surface.aeronet import AERONETReader

                            reader = AERONETReader()
                            data = reader.open([str(file_path)])
                        else:
                            try:
                                data = xr.open_dataset(str(file_path))
                            except Exception as e:
                                log_dir = context.config.get("analysis", {}).get("log_dir")
                                error_file = write_error_log(
                                    e, f"Opening observation file '{file_path}'", log_dir
                                )
                                msg = f"Failed to open observation file '{file_path}': {e}"
                                if error_file:
                                    msg += f" (details: {error_file})"
                                raise DataFormatError(msg) from e
                        if debug:
                            context.log_progress(
                                f"      [TIMING] open_dataset: {_format_duration(time.time() - t0)}"
                            )

                # Filter by analysis time range if specified
                if data is not None and "time" in data.dims and analysis_start and analysis_end:
                    t0 = time.time()
                    original_size = data.sizes.get("time", 0)
                    data = self._filter_by_time(
                        data,
                        analysis_start,
                        analysis_end,
                        end_time_has_time=end_time_has_time,
                    )
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
                    variables=(
                        normalized_variables
                        if isinstance(normalized_variables, dict)
                        else variables
                    ),
                )
                if debug:
                    context.log_progress(
                        f"      [TIMING] create_observation_data: {_format_duration(time.time() - t0)}"
                    )

                # Apply temporal averaging if configured
                resample_freq = config.get("resample")
                if resample_freq:
                    min_count = config.get("min_obs_count")
                    track_count = config.get("track_obs_count", False)
                    original_times = (
                        obs_data.data.sizes.get("time", 0) if obs_data.data is not None else 0
                    )

                    t0 = time.time()
                    obs_data.resample_data(
                        freq=resample_freq,
                        min_count=min_count,
                        track_count=track_count,
                    )
                    new_times = (
                        obs_data.data.sizes.get("time", 0) if obs_data.data is not None else 0
                    )

                    if debug:
                        context.log_progress(
                            f"      [TIMING] resample ({resample_freq}): {_format_duration(time.time() - t0)} "
                            f"({original_times} -> {new_times} times)"
                        )
                    else:
                        context.log_progress(
                            f"step: Resampled to {resample_freq} ({original_times} -> {new_times} times)"
                        )

                # Apply variable configuration (scaling, masking, renaming) and metadata
                if isinstance(normalized_variables, dict):
                    obs_data.variables = normalized_variables
                    t0 = time.time()
                    obs_data.apply_variable_config()
                    if debug:
                        context.log_progress(
                            f"      [TIMING] variable_config: {_format_duration(time.time() - t0)}"
                        )

                    for var_name, var_config in obs_data.variables.items():
                        if obs_data.data is not None and var_name in obs_data.data.data_vars:
                            units = (
                                var_config.get("units") if isinstance(var_config, dict) else None
                            )
                            display = (
                                var_config.get("display_name")
                                if isinstance(var_config, dict)
                                else None
                            )
                            if units:
                                obs_data.data[var_name].attrs["units"] = units
                            if display:
                                obs_data.data[var_name].attrs["display_name"] = display

                context.observations[label] = obs_data
                loaded_count += 1

                # Summary message
                ds = obs_data.data
                n_vars = len(ds.data_vars) if ds is not None else 0
                # Get record count (sites, points, or time steps)
                n_records = (
                    (ds.sizes.get("site") or ds.sizes.get("x") or ds.sizes.get("time") or 0)
                    if ds is not None
                    else 0
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

    def _load_lma_files(self, files: list[str]) -> "xr.Dataset":
        """Load LMA NetCDF files using the specialized reader.

        Parameters
        ----------
        files
            List of LMA file paths.

        Returns
        -------
        xr.Dataset
            Combined dataset with standardized coordinates.
        """
        from davinci_monet.observations.lightning.lma import LMAReader

        reader = LMAReader()
        return reader.open(files)

    def _load_modis_l2(
        self,
        label: str,
        config: dict[str, Any],
        context: "PipelineContext",
    ) -> "xr.Dataset | None":
        """Load MODIS L2 swath data, bin onto model grid, return gridded dataset.

        Handles binned-file caching: if ``load_binned`` is set and the
        cached file exists, loads it directly.  Otherwise reads HDF4
        granules via monetio, bins them, and optionally saves the result.

        Parameters
        ----------
        label
            Observation label (e.g. ``"terra_modis"``).
        config
            Observation configuration dict from YAML.
        context
            Pipeline context (must have models loaded for grid_source).

        Returns
        -------
        xr.Dataset or None
            Gridded dataset with dims ``(time, lon, lat)``, or None on
            failure.
        """
        import time as time_mod
        from pathlib import Path

        import numpy as np
        import xarray as xr

        from davinci_monet.observations.satellite.modis_l2 import (
            MODISL2Reader,
            build_modis_variable_dict,
            subset_modis_l2_files,
        )

        debug = context.config.get("analysis", {}).get("debug", False)

        # --- Check for cached binned file ---
        load_binned = config.get("load_binned", False)
        binned_file = config.get("binned_file")
        if load_binned and binned_file:
            binned_path = Path(binned_file).expanduser()
            if binned_path.exists():
                context.log_progress(f"step: Loading cached binned data from {binned_path.name}")
                t0 = time_mod.time()
                data = xr.open_dataset(str(binned_path))
                if debug:
                    context.log_progress(
                        f"      [TIMING] load_binned: {_format_duration(time_mod.time() - t0)}"
                    )
                context.log_progress(
                    f"done: Loaded cached grid "
                    f"({data.sizes.get('time', 0)} times, "
                    f"{data.sizes.get('lon', 0)}x{data.sizes.get('lat', 0)} grid)"
                )
                return data

        # --- Get target grid from model ---
        grid_source = config.get("grid_source")
        if not grid_source:
            context.log_progress("done: No grid_source specified, skipping MODIS")
            return None

        model_obj = context.models.get(grid_source)
        if model_obj is None:
            context.log_progress(f"done: grid_source model '{grid_source}' not loaded, skipping")
            return None

        model_ds = model_obj.data if hasattr(model_obj, "data") else model_obj
        # Extract lat/lon from model
        lat_centers = model_ds["lat"].values.astype(np.float64)
        lon_centers = model_ds["lon"].values.astype(np.float64)

        # --- Subset files by time ---
        filename = config.get("filename", "")
        analysis_config = context.config.get("analysis", {})
        start_time = str(analysis_config.get("start_time", ""))
        end_time = str(analysis_config.get("end_time", ""))

        context.log_progress("step: Subsetting MODIS files by time...")
        t0 = time_mod.time()
        files = subset_modis_l2_files(str(filename), start_time, end_time)
        if debug:
            context.log_progress(
                f"      [TIMING] subset_files: {_format_duration(time_mod.time() - t0)}"
            )

        if not files:
            context.log_progress("done: No MODIS files in analysis window")
            return None

        context.log_progress(f"step: Found {len(files)} granule files")

        # --- Build variable_dict for monetio ---
        variables = config.get("variables", {})
        # Normalize variable configs (may be VariableConfig objects)
        norm_vars: dict[str, dict[str, Any]] = {}
        for name, cfg in variables.items():
            if hasattr(cfg, "model_dump"):
                norm_vars[name] = cfg.model_dump()
            elif isinstance(cfg, dict):
                norm_vars[name] = dict(cfg)
            else:
                norm_vars[name] = {}
        variable_dict = build_modis_variable_dict(norm_vars)

        # --- Read and grid ---
        time_resolution = config.get("time_resolution", "1D")
        min_obs_count = config.get("min_obs_count", 1) or 1

        reader = MODISL2Reader()
        t0 = time_mod.time()
        data = reader.read_and_grid(
            files=files,
            variable_dict=variable_dict,
            lat_centers=lat_centers,
            lon_centers=lon_centers,
            start_time=start_time,
            end_time=end_time,
            time_resolution=time_resolution,
            min_obs_count=min_obs_count,
            debug=debug,
            progress_callback=context.log_progress,
        )
        if debug:
            context.log_progress(
                f"      [TIMING] read_and_grid: {_format_duration(time_mod.time() - t0)}"
            )

        # Store grid_source in attrs for downstream reference
        data.attrs["grid_source"] = grid_source

        # --- Save binned cache ---
        save_binned = config.get("save_binned", False)
        if save_binned and binned_file:
            binned_path = Path(binned_file).expanduser()
            binned_path.parent.mkdir(parents=True, exist_ok=True)
            context.log_progress(f"step: Saving binned data to {binned_path.name}")
            t0 = time_mod.time()
            data.to_netcdf(str(binned_path))
            if debug:
                context.log_progress(
                    f"      [TIMING] save_binned: {_format_duration(time_mod.time() - t0)}"
                )

        return data

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
        end_time_has_time: bool | None = None,
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

        t_start = pd.Timestamp(start_time)
        t_end = pd.Timestamp(end_time)

        if end_time_has_time is None and isinstance(end_time, str):
            import re

            end_time_has_time = bool(re.search(r"[T ]\d{2}:\d{2}", end_time))

        # If end_time is date-only, include the full day; otherwise honor exact timestamp.
        if end_time_has_time is False:
            t_end = t_end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

        # Use sel with slice for efficient time filtering
        return data.sel(time=slice(t_start, t_end))


class LoadSourcesStage(BaseStage):
    """Unified data-source loading stage (Phase 3, additive).

    Loads model and observation configuration via the existing
    :class:`LoadModelsStage` / :class:`LoadObservationsStage` loaders, then
    exposes everything through ``context.sources`` keyed by label, tagging each
    dataset's ``attrs`` with ``role`` (``"model"``/``"obs"``), ``source_label``,
    and ``geometry``. Results are mirrored in ``context.models`` /
    ``context.observations`` so existing accessors keep working.

    This is purely additive: it does not yet replace the legacy load stages in
    the default pipelines (that happens in a later phase). It can be used in a
    custom stage list, and it tolerates a context whose ``models`` /
    ``observations`` are already populated (in which case it simply unifies and
    tags them).
    """

    def __init__(self) -> None:
        super().__init__("load_sources")

    def validate(self, context: PipelineContext) -> bool:
        """Validate that at least one source (model or obs) is configured."""
        cfg = context.config
        has_config = any(k in cfg for k in ("model", "models", "obs", "observations"))
        # Already-populated containers are also a valid starting point.
        return has_config or bool(context.models) or bool(context.observations)

    def execute(self, context: PipelineContext) -> StageResult:
        """Load and unify all data sources into ``context.sources``."""
        import time

        start = time.time()

        # Delegate to the existing loaders when their config is present. They
        # populate context.models / context.observations as usual.
        if "model" in context.config or "models" in context.config:
            sub = LoadModelsStage().execute(context)
            if sub.status is StageStatus.FAILED:
                return self._create_result(
                    StageStatus.FAILED, error=sub.error, duration=time.time() - start
                )
        if "obs" in context.config or "observations" in context.config:
            sub = LoadObservationsStage().execute(context)
            if sub.status is StageStatus.FAILED:
                return self._create_result(
                    StageStatus.FAILED, error=sub.error, duration=time.time() - start
                )

        # Unify into the single source view, tagging role / source_label / geometry.
        for label, obj in context.models.items():
            self._register(context, label, obj, role="model")
        for label, obj in context.observations.items():
            self._register(context, label, obj, role="obs")

        return self._create_result(
            StageStatus.COMPLETED,
            data={"loaded_sources": list(context.sources.keys())},
            duration=time.time() - start,
            count=len(context.sources),
        )

    @staticmethod
    def _register(context: PipelineContext, label: str, obj: Any, role: str) -> None:
        """Register a loaded container in context.sources and tag its dataset."""
        context.sources[label] = obj
        data = getattr(obj, "data", None)
        if data is None:
            return
        try:
            geometry = obj.geometry
            geometry_name = geometry.name.lower() if hasattr(geometry, "name") else str(geometry)
        except Exception:
            geometry_name = "unknown"
        data.attrs["role"] = role
        data.attrs["source_label"] = label
        data.attrs["geometry"] = geometry_name


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
        import platform
        import subprocess
        import time

        from davinci_monet.pairing import PairingConfig, PairingEngine

        start = time.time()

        # Get pairing configuration
        pairing_config_dict = context.config.get("pairing", {})

        # Build list of pairs to process, separating Dask-backed from eager models
        # Each tuple includes an index for consistent ordering in output messages
        dask_pairs: list[tuple[int, str, Any, str, Any, dict, dict, int | None]] = []
        eager_pairs: list[tuple[int, str, Any, str, Any, dict, dict, int | None]] = []
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
                    pair_index,
                    model_label,
                    model_data,
                    obs_label,
                    obs_data,
                    model_config,
                    var_mapping,
                    None,
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
        cpu_count = os.cpu_count() or 4

        def _get_ram_gb() -> int | None:
            try:
                if platform.system() == "Darwin":
                    result = subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        return int(result.stdout.strip()) // (1024**3)
                elif platform.system() == "Linux":
                    with open("/proc/meminfo") as f:
                        for line in f:
                            if line.startswith("MemTotal"):
                                kb = int(line.split()[1])
                                return kb // (1024**2)
            except Exception:
                return None
            return None

        ram_gb = _get_ram_gb()
        dask_pair_workers = pairing_config_dict.get("dask_pair_workers")
        if dask_pair_workers is None:
            dask_pair_workers = 1  # Default to serial pairs for Dask-backed data
        dask_pair_workers = max(1, int(dask_pair_workers))
        dask_num_workers = pairing_config_dict.get("dask_num_workers")
        if dask_num_workers is None:
            dask_num_workers = max(1, cpu_count // dask_pair_workers)
            if ram_gb is not None:
                if ram_gb <= 16:
                    dask_num_workers = min(dask_num_workers, 4)
                elif ram_gb <= 32:
                    dask_num_workers = min(dask_num_workers, 6)
            dask_num_workers = min(32, dask_num_workers)
        dask_num_workers = max(1, int(dask_num_workers))
        eager_pair_workers = pairing_config_dict.get("max_workers")
        if eager_pair_workers is None:
            eager_pair_workers = min(len(eager_pairs), max(1, cpu_count // 2)) if eager_pairs else 1
            if ram_gb is not None and ram_gb <= 16:
                eager_pair_workers = min(eager_pair_workers, 4)
        eager_pair_workers = max(1, int(eager_pair_workers))
        if debug and dask_pairs:
            context.log_progress(
                f"step: Dask pairing workers={dask_pair_workers}, "
                f"dask_num_workers={dask_num_workers}, "
                f"eager_workers={eager_pair_workers}"
            )

        def pair_single(args: tuple) -> tuple[int, str, Any, str | None, float]:
            """Process a single model-obs pair. Returns (index, pair_key, paired_ds, error, duration)."""
            import time as time_mod

            pair_start = time_mod.time()
            (
                idx,
                model_label,
                model_data,
                obs_label,
                obs_data,
                model_config,
                var_mapping,
                dask_workers,
            ) = args
            pair_key = f"{model_label}_{obs_label}"

            try:
                obs_vars = list(var_mapping.keys())
                model_vars = list(var_mapping.values())

                model_ds = model_data.data if hasattr(model_data, "data") else model_data
                obs_ds = obs_data.data if hasattr(obs_data, "data") else obs_data

                if model_ds is None or obs_ds is None:
                    return (
                        idx,
                        pair_key,
                        None,
                        "Model or obs data is None",
                        time_mod.time() - pair_start,
                    )

                radius = model_config.get("radius_of_influence", 12000.0)
                pairing_cfg = PairingConfig(
                    radius_of_influence=radius,
                    time_tolerance=pairing_config_dict.get("time_tolerance", "1h"),
                    time_method=pairing_config_dict.get("time_method", "nearest"),
                )

                engine = PairingEngine()
                if dask_workers is not None:
                    import dask

                    with dask.config.set(scheduler="threads", num_workers=dask_workers):
                        paired_ds = engine.pair(
                            model_ds,
                            obs_ds,
                            obs_vars=obs_vars,
                            model_vars=model_vars,
                            config=pairing_cfg,
                            dask_num_workers=dask_workers,
                        )
                else:
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

        def run_phase(pairs: list, phase_name: str, max_workers: int) -> None:
            """Run a phase of pairs in parallel."""
            nonlocal paired_count
            from concurrent.futures import ThreadPoolExecutor, as_completed

            if not pairs:
                return
            max_workers = min(max_workers, len(pairs))

            # Run pairs in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                # Submit each pair and log start message
                for args in pairs:
                    idx, model_label, _, obs_label, _, _, _, _ = args
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
                            context.log_progress(
                                f"      [TIMING] {pair_key} failed: {_format_duration(pair_duration)}"
                            )
                        # Still count as "completed" for progress display
                        context.log_progress(f"    parallel_completed: {pair_key} - FAILED")
                    elif paired_ds is not None:
                        context.paired[pair_key] = paired_ds
                        paired_count += 1

                        # Tag role metadata on the paired variables (additive).
                        tag_paired_roles(
                            paired_ds.data if hasattr(paired_ds, "data") else paired_ds
                        )

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
            for _, model_label, _, obs_label, _, _, _, _ in dask_pairs:
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

        # Phase 1: Process Dask-backed model pairs (default serial to avoid oversubscription)
        for i, args in enumerate(dask_pairs):
            dask_pairs[i] = (*args[:-1], dask_num_workers)
        run_phase(dask_pairs, "dask", max_workers=dask_pair_workers)

        # Phase 2: Process eager (non-Dask) model pairs in parallel
        # These run after Dask compute() completes, avoiding GIL contention
        run_phase(eager_pairs, "eager", max_workers=eager_pair_workers)

        # Exit parallel mode
        context.log_progress("    parallel_end")

        # Warn if pairing produced no data (transient Dask/HDF5 issue)
        if total_pairs > 0 and paired_count == 0:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Pairing completed but produced no data ({total_pairs} pairs attempted). "
                "This may be a transient HDF5/Dask issue. Try: "
                "DASK_NUM_WORKERS=1 HDF5_USE_FILE_LOCKING=FALSE davinci-monet run ..."
            )
            context.log_progress(
                "    warning: No paired data produced - statistics/plotting will be skipped"
            )

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

                # Apply global stats-config domain filter if requested.
                from davinci_monet.util.domain import filter_paired_by_domain

                paired_data = filter_paired_by_domain(
                    paired_data,
                    stats_config.get("domain_type"),
                    stats_config.get("domain_name"),
                )

                # Calculate basic statistics
                pair_stats = self._calculate_stats(paired_data, stats_config)
                stats_results[pair_key] = pair_stats

                # Summary
                n_metrics = sum(len(v) for v in pair_stats.values())
                n_vars = len(pair_stats)
                context.log_progress(f"done: {n_vars} vars, {n_metrics} metrics")

            except Exception as e:
                context.metadata.setdefault("stats_errors", []).append(f"{pair_key}: {e}")
                context.log_progress(f"warning: stats failed for {pair_key}: {e}")

        return self._create_result(
            StageStatus.COMPLETED,
            data=stats_results,
            duration=time.time() - start,
        )

    def _calculate_stats(self, paired_data: xr.Dataset, config: dict[str, Any]) -> dict[str, Any]:
        """Calculate statistics for a paired dataset."""
        import numpy as np

        from davinci_monet.stats import StatisticsCalculator, StatisticsConfig
        from davinci_monet.stats.metrics import STANDARD_METRICS

        stats: dict[str, Any] = {}

        metrics = config.get("metrics") or config.get("stat_list")
        if isinstance(metrics, str):
            metrics = [metrics]
        round_precision = config.get("round_output", 3)
        include_counts = config.get("include_counts", True)
        remove_nan = config.get("remove_nan", True)
        min_samples = config.get("min_samples", 3)

        calc_config = StatisticsConfig(
            metrics=list(metrics) if metrics else list(STANDARD_METRICS),
            round_precision=round_precision,
            include_counts=include_counts,
            remove_nan=remove_nan,
            min_samples=min_samples,
        )
        calculator = StatisticsCalculator(calc_config)

        # Find model and obs variable pairs (prefix format: model_*, obs_*)
        model_vars = [v for v in paired_data.data_vars if str(v).startswith("model_")]

        for model_var in model_vars:
            base_name = str(model_var).replace("model_", "", 1)
            obs_var = f"obs_{base_name}"

            if obs_var not in paired_data:
                continue

            df = calculator.compute(
                paired_data,
                obs_var=obs_var,
                model_var=str(model_var),
                metrics=list(metrics) if metrics else None,
            )

            if df.empty:
                continue

            row = df.iloc[0].to_dict()
            # Normalize numpy scalars to Python types
            for key, value in list(row.items()):
                if isinstance(value, (np.floating, np.integer)):
                    row[key] = float(value)

            # Add legacy keys for backward compatibility
            legacy_map = {
                "n": "N",
                "mean_bias": "MB",
                "rmse": "RMSE",
                "correlation": "R",
                "model_mean": "MP",
                "obs_mean": "MO",
            }
            for legacy_key, metric_key in legacy_map.items():
                if metric_key in row and legacy_key not in row:
                    row[legacy_key] = row[metric_key]

            stats[base_name] = row

        # Per-flight statistics (if flight coord exists and enabled)
        per_flight = config.get("per_flight", False)
        if per_flight and "flight" in paired_data.coords:
            stats["_per_flight"] = self._calculate_per_flight_stats(paired_data)

        return stats

    def _calculate_per_flight_stats(self, paired_data: xr.Dataset) -> list[dict[str, Any]]:
        """Calculate statistics for each flight.

        Parameters
        ----------
        paired_data : xr.Dataset
            Paired dataset with model_* and obs_* variables and flight coordinate.

        Returns
        -------
        list[dict[str, Any]]
            List of dictionaries with per-flight statistics.
        """
        import numpy as np

        flights = np.unique(paired_data["flight"].values)
        flight_stats: list[dict[str, Any]] = []

        model_vars = [v for v in paired_data.data_vars if str(v).startswith("model_")]

        for flight in flights:
            mask = paired_data["flight"].values == flight
            flight_data = paired_data.isel(time=mask)

            for model_var in model_vars:
                base_name = str(model_var).replace("model_", "", 1)
                obs_var = f"obs_{base_name}"
                if obs_var not in flight_data:
                    continue

                model_vals = flight_data[model_var].values.flatten()
                obs_vals = flight_data[obs_var].values.flatten()

                # Remove NaNs
                valid = ~(np.isnan(model_vals) | np.isnan(obs_vals))
                if valid.sum() < 3:
                    continue

                m, o = model_vals[valid], obs_vals[valid]
                diff = m - o

                row: dict[str, Any] = {
                    "flight": str(flight),
                    "variable": base_name,
                    "N": len(m),
                    "MO": float(np.mean(o)),
                    "MP": float(np.mean(m)),
                    "MB": float(np.mean(diff)),
                    "RMSE": float(np.sqrt(np.mean(diff**2))),
                    "R": float(np.corrcoef(o, m)[0, 1]) if len(m) > 1 else np.nan,
                }
                # NMB/NME
                if row["MO"] != 0:
                    row["NMB_%"] = (row["MB"] / row["MO"]) * 100
                    row["NME_%"] = (float(np.mean(np.abs(diff))) / row["MO"]) * 100
                else:
                    row["NMB_%"] = np.nan
                    row["NME_%"] = np.nan
                flight_stats.append(row)

        return flight_stats


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
        output_dir_str = analysis_config.get("output_dir") or "."
        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get pairs config for variable mapping
        pairs_config = context.config.get("pairs", {})
        model_config = context.config.get("model", {})
        total_plots = len(plot_config)
        plot_count = 0
        file_index = 0  # Global counter for ordering files in preview

        for plot_name, plot_spec in plot_config.items():
            try:
                plot_count += 1
                plot_type = plot_spec.get("type", "scatter")
                plot_pairs = plot_spec.get("data") or plot_spec.get("pairs", [])
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

                    # Apply per-plot domain filter if requested. domain_type/
                    # domain_name in the plot spec restrict paired_data to a
                    # named lat/lon extent (e.g. 'conus', or 'epa_region' + R5).
                    # When the type is None / "all" or unrecognised, the helper
                    # returns the dataset unchanged.
                    from davinci_monet.util.domain import filter_paired_by_domain

                    paired_data = filter_paired_by_domain(
                        paired_data,
                        plot_spec.get("domain_type"),
                        plot_spec.get("domain_name"),
                    )

                    # Variable names in paired dataset use obs_var with prefixes
                    obs_var_name = f"obs_{obs_var}"
                    model_var_name = f"model_{obs_var}"

                    if obs_var_name not in paired_data or model_var_name not in paired_data:
                        continue

                    # Get plotter config from model variable settings
                    model_var = var_spec.get("model_var", "")
                    var_config = (
                        model_config.get(model_label, {}).get("variables", {}).get(model_var, {})
                    )
                    vmin = var_config.get("vmin_plot")
                    vmax = var_config.get("vmax_plot")
                    vdiff = var_config.get("vdiff_plot")
                    nlevels = var_config.get("nlevels_plot")

                    # Build plotter config
                    plotter_config: dict[str, Any] = {"title": title}
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
                    for opt_key in [
                        "show_site_labels",
                        "show_individual_sites",
                        "show_uncertainty",
                        "uncertainty_type",
                        "resample",
                        "aggregate_dim",
                        "label_sites",
                        "site_label_var",
                        "city_labels",
                        "show_density",
                        "density_cmap",
                        "alpha",
                        # spatial_overlay obs marker sizing
                        "marker_size",
                        "obs_edgecolor",
                        "obs_linewidth",
                        # spatial plotter rendering mode
                        "plot_type",
                        "cmap",
                        # track_map_3d options
                        "show_surface_map",
                        "surface_map_resolution",
                        "land_color",
                        "ocean_color",
                        "show_var",
                        "elev",
                        "azim",
                        "show_coastlines",
                        "show_borders",
                        "show_projection",
                        "alt_scale",
                    ]:
                        if opt_key in plot_spec:
                            plot_options[opt_key] = plot_spec[opt_key]

                    # Add city_labels from analysis config for spatial plots and 3D track maps
                    if (
                        plot_type.startswith("spatial") or plot_type == "track_map_3d"
                    ) and "city_labels" not in plot_options:
                        city_labels = analysis_config.get("city_labels")
                        if city_labels:
                            plot_options["city_labels"] = city_labels

                    # Forward nlevels_plot to spatial plotters as n_levels so
                    # configs can pick contour counts that produce nice round
                    # tick values (e.g. 21 levels over 0-1 -> step 0.05).
                    if plot_type.startswith("spatial") and nlevels is not None:
                        plot_options.setdefault("n_levels", nlevels)

                    # spatial_overlay needs the raw gridded model field for the
                    # contour layer; the paired dataset only carries model values
                    # interpolated to obs sites.
                    if plot_type == "spatial_overlay":
                        if "model_field" not in plot_options:
                            model_obj = context.models.get(model_label)
                            if model_obj is not None:
                                model_ds = (
                                    model_obj.data if hasattr(model_obj, "data") else model_obj
                                )
                                if (
                                    model_ds is not None
                                    and model_var in getattr(model_ds, "data_vars", {})
                                ):
                                    plot_options["model_field"] = model_ds[model_var]
                        # Observation readers differ on coord naming
                        # (`latitude`/`longitude` vs `lat`/`lon`). Pick whichever
                        # the paired dataset actually carries.
                        if "lat_var" not in plot_options:
                            for cand in ("latitude", "lat"):
                                if cand in paired_data.coords or cand in paired_data:
                                    plot_options["lat_var"] = cand
                                    break
                        if "lon_var" not in plot_options:
                            for cand in ("longitude", "lon"):
                                if cand in paired_data.coords or cand in paired_data:
                                    plot_options["lon_var"] = cand
                                    break

                    # Build subtitle: "<Model> vs <Obs> · <date>"; for
                    # snapshot-style plots (spatial_overlay) also show the
                    # specific timestamp being rendered.
                    start_time = analysis_config.get("start_time", "")
                    end_time = analysis_config.get("end_time", "")
                    date_str = ""
                    if start_time:
                        start_date = str(start_time).split(" ")[0]
                        end_date = str(end_time).split(" ")[0] if end_time else start_date
                        date_str = start_date if start_date == end_date else f"{start_date} → {end_date}"
                    snapshot_str = ""
                    if plot_type == "spatial_overlay" and "model_field" in plot_options:
                        mf = plot_options["model_field"]
                        time_idx = plot_options.get("time_index", 0)
                        if "time" in mf.dims and len(mf["time"]) > time_idx:
                            ts = mf["time"].values[time_idx]
                            try:
                                import pandas as pd
                                snapshot_str = pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M UTC")
                            except Exception:
                                snapshot_str = str(ts)[:16] + " UTC"
                    when = snapshot_str or date_str
                    # Prefer explicit display_name on the model/obs config (e.g.
                    # "AirNow", "AERONET"); fall back to the YAML key when not
                    # set so plot text reads cleanly regardless of casing.
                    obs_config = context.config.get("obs", {})
                    model_display = (
                        model_config.get(model_label, {}).get("display_name") or model_label
                    )
                    obs_display = (
                        obs_config.get(obs_label, {}).get("display_name") or obs_label
                    )
                    parts = [p for p in (model_display, obs_display) if p]
                    subtitle = ""
                    if parts:
                        subtitle = " vs ".join(parts)
                        if when:
                            subtitle = f"{subtitle} · {when}"
                    elif when:
                        subtitle = when
                    if subtitle:
                        plotter_config["title"] = f"{title}\n{subtitle}"

                    # Get plotter
                    plotter = get_plotter(plot_type, config=plotter_config)

                    # Create subdirectory by observation dataset
                    obs_output_dir = output_dir / obs_label
                    obs_output_dir.mkdir(parents=True, exist_ok=True)

                    # Check for per-flight splitting
                    split_by_flight = plot_spec.get("split_by_flight", False)
                    # Check for per-site splitting
                    split_by_site = plot_spec.get("split_by_site", False)

                    if split_by_flight and hasattr(plotter, "plot_per_flight"):
                        # Generate separate plot for each flight
                        flight_coord = plot_spec.get("flight_coord", "flight")
                        min_points = plot_spec.get("min_points", 10)

                        flight_count = 0
                        for flight_id, fig in plotter.plot_per_flight(
                            paired_data,
                            obs_var_name,
                            model_var_name,
                            flight_coord=flight_coord,
                            min_points=min_points,
                            **plot_options,
                        ):
                            # Save plot with flight ID first for grouping by flight in slideshows
                            output_path = (
                                obs_output_dir / f"{flight_id}_{file_index:02d}_{plot_name}.png"
                            )
                            plotter.save(fig, output_path, dpi=300)
                            plots_generated.append(str(output_path))

                            pdf_path = (
                                obs_output_dir / f"{flight_id}_{file_index:02d}_{plot_name}.pdf"
                            )
                            plotter.save(fig, pdf_path)
                            plots_generated.append(str(pdf_path))

                            plt.close(fig)
                            flight_count += 1
                            file_index += 1

                        context.log_progress(f"done: saved {flight_count} flights to {obs_label}/")

                    elif split_by_site and hasattr(plotter, "plot_per_site"):
                        # Generate separate plot for each site
                        site_dim = plot_spec.get("site_dim", "site")
                        min_points = plot_spec.get("min_points", 20)

                        site_count = 0
                        for site_id, fig in plotter.plot_per_site(
                            paired_data,
                            obs_var_name,
                            model_var_name,
                            site_dim=site_dim,
                            min_points=min_points,
                            **plot_options,
                        ):
                            output_path = (
                                obs_output_dir / f"site_{site_id}_{file_index:02d}_{plot_name}.png"
                            )
                            plotter.save(fig, output_path, dpi=300)
                            plots_generated.append(str(output_path))

                            pdf_path = (
                                obs_output_dir / f"site_{site_id}_{file_index:02d}_{plot_name}.pdf"
                            )
                            plotter.save(fig, pdf_path)
                            plots_generated.append(str(pdf_path))

                            plt.close(fig)
                            site_count += 1
                            file_index += 1

                        context.log_progress(f"done: saved {site_count} sites to {obs_label}/")
                    else:
                        # Generate single plot (original behavior)
                        fig = plotter.plot(
                            paired_data, obs_var_name, model_var_name, **plot_options
                        )

                        # Save plot (prefixed for ordering)
                        output_path = obs_output_dir / f"{file_index:02d}_{plot_name}.png"
                        plotter.save(fig, output_path, dpi=300)
                        plots_generated.append(str(output_path))

                        # Also save PDF
                        pdf_path = obs_output_dir / f"{file_index:02d}_{plot_name}.pdf"
                        plotter.save(fig, pdf_path)
                        plots_generated.append(str(pdf_path))

                        plt.close(fig)
                        file_index += 1

                        context.log_progress(f"done: saved to {obs_label}/")

            except Exception as e:
                context.metadata.setdefault("plot_errors", []).append(f"{plot_name}: {e}")
                context.log_progress(f"warning: plot failed for {plot_name}: {e}")

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
        import math
        import time
        from pathlib import Path

        import pandas as pd

        start = time.time()
        saved_files: list[str] = []

        # Get output directory from analysis config
        analysis_config = context.config.get("analysis", {})
        output_dir_str = analysis_config.get("output_dir") or "."
        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save statistics summary from statistics stage
        stats_result = context.results.get("statistics")
        if stats_result and stats_result.data:
            context.log_progress("step: Writing statistics CSV...")
            rows = []

            def _get_metric(stats: dict[str, Any], *keys: str, default: Any = float("nan")) -> Any:
                for key in keys:
                    if key in stats and stats[key] is not None:
                        return stats[key]
                return default

            for pair_key, pair_stats in stats_result.data.items():
                for var_name, var_stats in pair_stats.items():
                    # Skip internal keys like _per_flight
                    if var_name.startswith("_"):
                        continue
                    row = {"Variable": var_name}
                    row["N"] = _get_metric(var_stats, "N", "n", default=0)
                    row["Mean_Obs"] = _get_metric(var_stats, "MO", "obs_mean")
                    row["Mean_Model"] = _get_metric(var_stats, "MP", "model_mean")
                    row["MB"] = _get_metric(var_stats, "MB", "mean_bias")
                    row["RMSE"] = _get_metric(var_stats, "RMSE", "rmse")
                    row["R"] = _get_metric(var_stats, "R", "correlation")
                    row["IOA"] = _get_metric(var_stats, "IOA", "ioa")

                    # Prefer computed NMB/NME if present; otherwise derive as fallback
                    nmb = _get_metric(var_stats, "NMB", default=float("nan"))
                    nme = _get_metric(var_stats, "NME", default=float("nan"))
                    obs_mean = row["Mean_Obs"]

                    if isinstance(nmb, (int, float)) and not math.isnan(float(nmb)):
                        row["NMB_%"] = nmb
                    elif (
                        isinstance(obs_mean, (int, float))
                        and obs_mean not in (0, -0.0)
                        and not math.isnan(float(obs_mean))
                    ):
                        row["NMB_%"] = (
                            (row["MB"] / obs_mean) * 100
                            if isinstance(row["MB"], (int, float))
                            else float("nan")
                        )
                    else:
                        row["NMB_%"] = float("nan")

                    if isinstance(nme, (int, float)) and not math.isnan(float(nme)):
                        row["NME_%"] = nme
                    else:
                        # No correct fallback: NME requires per-point |mod-obs|,
                        # which can't be derived from RMSE or other summary scalars.
                        row["NME_%"] = float("nan")

                    rows.append(row)

            if rows:
                df = pd.DataFrame(rows)
                df = df.set_index("Variable")
                stats_file = output_dir / "statistics_summary.csv"
                df.to_csv(stats_file)
                saved_files.append(str(stats_file))
                context.log_progress(f"done: {len(rows)} rows saved")

            # Save per-flight statistics if available
            all_flight_stats: list[dict] = []
            for pair_key, pair_stats in stats_result.data.items():
                if "_per_flight" in pair_stats:
                    all_flight_stats.extend(pair_stats["_per_flight"])

            if all_flight_stats:
                context.log_progress("step: Writing per-flight statistics CSV...")
                flight_df = pd.DataFrame(all_flight_stats)
                # Reorder columns
                cols = [
                    "variable",
                    "flight",
                    "N",
                    "MO",
                    "MP",
                    "MB",
                    "RMSE",
                    "R",
                    "NMB_%",
                    "NME_%",
                ]
                cols = [c for c in cols if c in flight_df.columns]
                flight_df = flight_df[cols]
                flight_df = flight_df.sort_values(["variable", "flight"])

                flight_stats_file = output_dir / "statistics_per_flight.csv"
                flight_df.to_csv(flight_stats_file, index=False)
                saved_files.append(str(flight_stats_file))
                context.log_progress(f"done: {len(flight_df)} rows")

        return self._create_result(
            StageStatus.COMPLETED,
            data={"saved_files": saved_files},
            duration=time.time() - start,
        )


class ObsPlottingStage(BaseStage):
    """Pipeline stage for observation-only plots."""

    def __init__(self) -> None:
        super().__init__("obs_plotting")

    def validate(self, context: PipelineContext) -> bool:
        # Obs-only plotting: active when sources are loaded but no cross-source
        # pairs exist. In a paired run the standard PlottingStage handles output.
        return bool(context.observations) and not bool(context.paired)

    def execute(self, context: PipelineContext) -> StageResult:
        """Execute observation-only plotting.

        When datasets contain a ``flight`` coordinate with multiple flights,
        generates one plot per flight automatically.  The per-flight title
        appends the flight date to the configured title.
        """
        import logging
        import time
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np

        from davinci_monet.plots.registry import get_plotter

        _logger = logging.getLogger(__name__)

        # Keys to exclude when forwarding plot_spec to plotter kwargs
        _SCHEMA_KEYS = {
            "type",
            "obs",
            "variable",
            "fig_kwargs",
            "default_plot_kwargs",
            "text_kwargs",
            "domain_type",
            "domain_name",
            "data",
            "data_proc",
        }

        start = time.time()
        plots_config = context.config.get("plots", {})
        output_dir = Path(context.config.get("analysis", {}).get("output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_count = 0
        plots_generated: list[str] = []
        errors: list[str] = []

        for plot_name, plot_spec in plots_config.items():
            plot_type = plot_spec.get("type", "")
            if not plot_type.startswith("obs_"):
                continue

            obs_label = plot_spec.get("obs", "")
            variable = plot_spec.get("variable", "")

            if obs_label not in context.observations:
                errors.append(f"Observation '{obs_label}' not found for plot '{plot_name}'")
                continue

            obs_data = context.observations[obs_label]
            ds = obs_data.data if hasattr(obs_data, "data") else obs_data

            if variable not in ds.data_vars:
                errors.append(f"Variable '{variable}' not in '{obs_label}' for plot '{plot_name}'")
                continue

            plotter = get_plotter(plot_type)
            plot_kwargs = {k: v for k, v in plot_spec.items() if k not in _SCHEMA_KEYS}
            base_title = plot_kwargs.get("title", f"{variable} {plot_type}")

            # Determine flight subsets
            has_flights = "flight" in ds.coords
            if has_flights:
                flight_ids = sorted(set(np.unique(ds["flight"].values).tolist()))
            else:
                flight_ids = [None]

            for fid in flight_ids:
                if fid is not None:
                    mask = ds["flight"].values == fid
                    subset = ds.isel(time=mask)
                    suffix = f"_{fid}"
                    flight_kwargs = {**plot_kwargs, "title": f"{base_title} — {fid}"}
                else:
                    subset = ds
                    suffix = ""
                    flight_kwargs = plot_kwargs

                # Skip flights with no valid data for this variable
                vals = subset[variable].values
                if not np.any(np.isfinite(vals)):
                    continue

                # Provide obs_datasets for renderers needing cross-dataset access
                if "flight_tracks" in plot_spec:
                    flight_kwargs["obs_datasets"] = {
                        label: (od.data if hasattr(od, "data") else od)
                        for label, od in context.observations.items()
                    }

                try:
                    result = plotter.plot(subset, variable, **flight_kwargs)

                    # Multi-figure support (e.g., hourly LMA density maps)
                    if isinstance(result, list):
                        for fig, fig_suffix in result:
                            out_path = output_dir / f"{plot_name}{suffix}{fig_suffix}.png"
                            plotter.save(fig, out_path)
                            plt.close(fig)
                            plot_count += 1
                            plots_generated.append(str(out_path))
                            _logger.info(f"Saved obs plot: {out_path}")
                    else:
                        fig = result
                        out_path = output_dir / f"{plot_name}{suffix}.png"
                        plotter.save(fig, out_path)
                        plt.close(fig)
                        plot_count += 1
                        plots_generated.append(str(out_path))
                        _logger.info(f"Saved obs plot: {out_path}")
                except Exception as e:
                    label = f"'{plot_name}' (flight {fid})" if fid else f"'{plot_name}'"
                    errors.append(f"Plot {label} failed: {e}")
                    _logger.warning(f"Obs plot {label} failed: {e}")

        message = f"Generated {plot_count} obs-only plots"
        if errors:
            message += f" ({len(errors)} errors)"

        return self._create_result(
            StageStatus.COMPLETED if plot_count > 0 or not plots_config else StageStatus.SKIPPED,
            data={"plot_count": plot_count, "plots_generated": plots_generated, "errors": errors},
            duration=time.time() - start,
        )


class ObsStatisticsStage(BaseStage):
    """Pipeline stage for observation-only descriptive statistics."""

    def __init__(self) -> None:
        super().__init__("obs_statistics")

    def validate(self, context: PipelineContext) -> bool:
        # Obs-only statistics: active when sources are loaded but no cross-source
        # pairs exist. In a paired run the standard StatisticsStage handles stats.
        return bool(context.observations) and not bool(context.paired)

    def execute(self, context: PipelineContext) -> StageResult:
        """Compute descriptive statistics for all observation variables."""
        import time

        import numpy as np

        start = time.time()
        all_stats: dict[str, dict[str, dict[str, float]]] = {}

        for obs_label, obs_data in context.observations.items():
            ds = obs_data.data if hasattr(obs_data, "data") else obs_data
            obs_stats: dict[str, dict[str, float]] = {}

            for var_name in ds.data_vars:
                values = ds[var_name].values.flatten()
                values = values[np.isfinite(values)]
                if len(values) < 1:
                    continue
                obs_stats[var_name] = {
                    "N": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "p10": float(np.percentile(values, 10)),
                    "p25": float(np.percentile(values, 25)),
                    "p75": float(np.percentile(values, 75)),
                    "p90": float(np.percentile(values, 90)),
                }
            all_stats[obs_label] = obs_stats

        return self._create_result(
            StageStatus.COMPLETED,
            data=all_stats,
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
        LoadSourcesStage(),
        PairingStage(),
        StatisticsStage(),
        PlottingStage(),
        ObsStatisticsStage(),
        ObsPlottingStage(),
        SaveResultsStage(),
    ]


def create_obs_pipeline() -> list[BaseStage]:
    """Create an observation-only pipeline (no model/pairing stages).

    Returns
    -------
    list[BaseStage]
        List of stages for obs-only analysis.
    """
    return [
        LoadSourcesStage(),
        ObsStatisticsStage(),
        ObsPlottingStage(),
        SaveResultsStage(),
    ]
