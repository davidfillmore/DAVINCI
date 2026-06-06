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

from davinci_monet.core.base import iter_paired_variable_pairs
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
        Dictionary of paired source data.
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
    # Unified data-source view. Models and observations both register here
    # keyed by label; legacy role-specific dicts are kept in sync where known.
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

    def iter_sources(self) -> list[tuple[str, Any]]:
        """Return loaded sources in insertion order."""
        return list(self.sources.items())

    def get_source_dataset(self, label: str) -> xr.Dataset:
        """Return the xarray Dataset for a source label."""
        source = self.get_source(label)
        data = source.data if hasattr(source, "data") else source
        if not isinstance(data, xr.Dataset):
            raise KeyError(f"Source '{label}' does not contain an xarray Dataset")
        return data

    def get_source_role(self, label: str) -> str | None:
        """Return optional source role metadata for a source label."""
        source = self.get_source(label)
        role = getattr(source, "role", None)
        if role:
            return str(role)
        data = source.data if hasattr(source, "data") else source
        if isinstance(data, xr.Dataset):
            raw = data.attrs.get("role")
            return str(raw) if raw else None
        return None

    def get_paired(self, key: str) -> Any:
        """Get paired data by key."""
        if key not in self.paired:
            raise KeyError(f"Paired data '{key}' not found in context")
        return self.paired[key]


@dataclass
class SourceData:
    """Container for a unified data source loaded from ``sources:`` config."""

    data: xr.Dataset
    label: str
    source_type: str
    geometry: DataGeometry
    role: str | None = None
    variables: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourcePairJob:
    """Concrete source pair to process."""

    index: int
    pair_key: str
    reference_label: str
    reference_obj: Any
    comparand_label: str
    comparand_obj: Any
    reference_var: str
    comparand_var: str
    reference_role: str | None
    comparand_role: str | None
    radius_of_influence: float


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


def tag_paired_roles(
    data: Any,
    *,
    reference_label: str | None = None,
    comparand_label: str | None = None,
    reference_role: str | None = "obs",
    comparand_role: str | None = "model",
) -> None:
    """Tag each paired variable with its ``role`` and rename it by source label.

    Each paired variable gets a source ``role`` attr plus a ``pair_role`` attr
    (``reference``/``comparand``). For legacy model-vs-obs pairings these values
    line up with ``obs``/``model`` respectively; same-role source pairs keep
    their source role while ``pair_role`` preserves reference/comparand semantics
    for statistics and plotting.

    When the source labels are supplied (the pipeline path), the variable is
    *renamed* to ``<comparand_label>_<v>`` (model/comparand side) or
    ``<reference_label>_<v>`` (obs/reference side), the legacy prefix is dropped,
    and both ``role`` and ``source_label`` attrs are set — the renderer rewire R-5
    clean break to source-label-only naming. Pairing maps ``obs`` -> reference and
    ``model`` -> comparand.

    Without labels (the low-level engine/strategy API and untagged data) the
    legacy names are kept and only the ``role`` attr is set. A rename is also
    skipped when it would collide with an existing variable or re-enter the
    reserved ``model_``/``obs_`` namespace (e.g. a label like ``model_foo``), so
    such variables keep their legacy name. A pre-existing ``role`` attr is never
    overwritten.
    """
    if data is None or not hasattr(data, "data_vars"):
        return
    # Snapshot the names first: variables are renamed while iterating.
    for name in list(data.data_vars):
        lname = str(name).lower()
        if lname.startswith("model_"):
            role = comparand_role or "model"
            pair_role = "comparand"
            label = comparand_label
            canonical = str(name)[len("model_") :]
        elif lname.startswith("obs_"):
            role = reference_role or "obs"
            pair_role = "reference"
            label = reference_label
            canonical = str(name)[len("obs_") :]
        else:
            continue

        var = data[name]
        var.attrs.setdefault("role", role)
        var.attrs.setdefault("pair_role", pair_role)
        if not label:
            continue
        var.attrs.setdefault("source_label", label)

        # Rename to the source-label name, dropping the legacy prefix. Skip when
        # the new name would collide or re-enter the reserved model_/obs_
        # namespace (a pathological label like ``model_foo``); such a variable
        # keeps its legacy name (and its source_label attr).
        new_name = f"{label}_{canonical}"
        new_l = new_name.lower()
        if (
            new_name != name
            and new_name not in data.data_vars
            and not new_l.startswith("model_")
            and not new_l.startswith("obs_")
        ):
            data[new_name] = var
            data[new_name].attrs["role"] = var.attrs["role"]
            data[new_name].attrs["pair_role"] = var.attrs["pair_role"]
            data[new_name].attrs["source_label"] = var.attrs["source_label"]
            del data[name]


def tag_source_roles(
    data: Any,
    *,
    role: str | None,
    source_label: str,
) -> Any:
    """Tag each variable of a single-source dataset with ``role``/``source_label``.

    Per-variable counterpart of :func:`tag_paired_roles` for *unpaired* sources:
    a single obs (or model) source carries its label/role only at the dataset
    level today, so the unified series resolver
    (:func:`~davinci_monet.core.base.iter_canonical_variable_series`) cannot see
    its variables. This sets the attrs per data_var (idempotent via
    ``setdefault``, never overwriting a pre-existing ``role``) so a single source
    becomes a 1-series plot under the unified renderer. Returns ``data``.
    """
    if data is None or not hasattr(data, "data_vars"):
        return data
    for name in data.data_vars:
        var = data[name]
        if role is not None:
            var.attrs.setdefault("role", role)
        var.attrs.setdefault("source_label", source_label)
    return data


def iter_single_source_datasets(
    context: PipelineContext,
) -> list[tuple[str, Any, xr.Dataset, str | None]]:
    """Return loaded single-source datasets from the unified source view.

    ``context.sources`` is canonical for the standard pipeline. The legacy
    role-specific dictionaries remain a compatibility fallback for tests and
    direct stage use that bypasses ``LoadSourcesStage``.
    """
    if context.sources:
        items = list(context.sources.items())
    else:
        seen: set[str] = set()
        items = []
        for label, obj in context.models.items():
            seen.add(label)
            items.append((label, obj))
        for label, obj in context.observations.items():
            if label not in seen:
                items.append((label, obj))

    sources: list[tuple[str, Any, xr.Dataset, str | None]] = []
    for label, obj in items:
        ds = obj.data if hasattr(obj, "data") else obj
        if not isinstance(ds, xr.Dataset):
            continue
        role = getattr(obj, "role", None) or ds.attrs.get("role")
        sources.append((str(label), obj, ds, str(role) if role else None))
    return sources


def resolve_paired_var_names(
    paired_data: Any,
    obs_var: str,
    obs_label: str,
    model_label: str,
) -> tuple[str, str]:
    """Resolve the (obs, model) variable names to plot from a paired dataset.

    Renderer rewire R-2: prefer the source-label aliases (``<label>_<var>``,
    e.g. ``airnow_o3`` / ``cam_o3``) added by :func:`tag_paired_roles`, falling
    back to the legacy ``obs_``/``model_`` prefixes when no alias is present
    (older paired data, or a label in the reserved namespace). obs is the
    reference and model the comparand; the pairing engine names both paired
    variables off the *obs* canonical name (``model_<obs_var>``), so both
    resolutions key off ``obs_var``.

    The returned names are always concrete strings (alias if present, else the
    legacy prefix); the caller is responsible for checking membership before
    plotting.
    """
    from davinci_monet.plots.base import resolve_source_variable

    obs_name = resolve_source_variable(paired_data, obs_var, obs_label) or f"obs_{obs_var}"
    model_name = resolve_source_variable(paired_data, obs_var, model_label) or f"model_{obs_var}"
    return obs_name, model_name


class LoadSourcesStage(BaseStage):
    """Standard data-source loading stage.

    Loads the unified ``sources:`` config directly through ``source_registry``.
    Legacy ``model:`` / ``obs:`` (and ``models:`` / ``observations:``) configs
    are auto-converted to the unified ``sources:`` form (via
    :func:`~davinci_monet.config.migration.migrate_to_sources`) and loaded
    through the same unified loop — the legacy per-role loader stages have been
    removed. Loaded datasets are tagged with ``role`` (``"model"``/``"obs"`` when
    known), ``source_label``, and ``geometry``. Results are mirrored in
    ``context.models`` / ``context.observations`` so existing accessors keep
    working.
    """

    def __init__(self) -> None:
        super().__init__("load_sources")

    def validate(self, context: PipelineContext) -> bool:
        """Validate that at least one source (model or obs) is configured."""
        cfg = context.config
        has_config = any(k in cfg for k in ("sources", "model", "models", "obs", "observations"))
        # Already-populated containers are also a valid starting point.
        return (
            has_config
            or bool(context.sources)
            or bool(context.models)
            or bool(context.observations)
        )

    def execute(self, context: PipelineContext) -> StageResult:
        """Load and unify all data sources into ``context.sources``.

        Native ``sources:`` configs load directly through ``source_registry``.
        Legacy ``model:``/``obs:`` (and ``models:``/``observations:``) configs are
        auto-converted to the unified ``sources:`` form via
        :func:`~davinci_monet.config.migration.migrate_to_sources` and loaded
        through the *same* unified loop — there is no separate legacy loader path.
        """
        import time

        start = time.time()

        if not context.config.get("sources") and any(
            k in context.config for k in ("model", "models", "obs", "observations")
        ):
            # Auto-convert a legacy control file to the unified sources schema in
            # place, then fall through to the unified loader below. Mutating
            # context.config keeps downstream stages (pairing/plotting) on the
            # unified `sources:`/`pairs:` shape.
            from davinci_monet.config.migration import migrate_to_sources

            try:
                context.config = migrate_to_sources(context.config)
            except Exception as e:
                return self._create_result(
                    StageStatus.FAILED,
                    error=f"Failed to convert legacy model/obs config to sources: {e}",
                    duration=time.time() - start,
                )

        if context.config.get("sources"):
            try:
                source_configs = context.config.get("sources", {})
                total_sources = len(source_configs)
                for index, (label, raw_config) in enumerate(source_configs.items(), start=1):
                    context.log_progress(f"    Loading source: {label} ({index}/{total_sources})")
                    source = self._load_unified_source(label, raw_config, context)
                    self._register_source(context, label, source)
                return self._create_result(
                    StageStatus.COMPLETED,
                    data={"loaded_sources": list(context.sources.keys())},
                    duration=time.time() - start,
                    count=len(context.sources),
                )
            except Exception as e:
                return self._create_result(
                    StageStatus.FAILED,
                    error=f"Failed to load sources: {e}",
                    duration=time.time() - start,
                )

        # No source config at all: any pre-populated containers (tests/direct
        # stage use) are unified into the single source view, tagging role /
        # source_label / geometry.
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
    def _as_dict(config: Any) -> dict[str, Any]:
        if hasattr(config, "model_dump"):
            result = config.model_dump(exclude_none=True)
            return dict(result)
        if isinstance(config, dict):
            return dict(config)
        if hasattr(config, "__dict__"):
            return dict(config.__dict__)
        return {}

    @staticmethod
    def _normalize_var_configs(raw: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(raw, dict):
            return {}
        normalized: dict[str, dict[str, Any]] = {}
        for name, cfg in raw.items():
            if hasattr(cfg, "model_dump"):
                normalized[str(name)] = dict(cfg.model_dump(exclude_none=True))
            elif isinstance(cfg, dict):
                normalized[str(name)] = dict(cfg)
            elif hasattr(cfg, "__dict__"):
                normalized[str(name)] = dict(cfg.__dict__)
            else:
                normalized[str(name)] = {}
        return normalized

    @staticmethod
    def _file_list(files: Any) -> list[str]:
        from glob import glob
        from pathlib import Path

        if files is None:
            return []
        if isinstance(files, (list, tuple)):
            values = [str(Path(str(item)).expanduser()) for item in files]
        else:
            values = [str(Path(str(files)).expanduser())]

        expanded: list[str] = []
        for value in values:
            if "*" in value or "?" in value:
                expanded.extend(sorted(glob(value)))
            else:
                expanded.append(value)
        return expanded

    @staticmethod
    def _data_geometry(value: Any) -> DataGeometry:
        if isinstance(value, DataGeometry):
            return value
        if isinstance(value, str):
            return DataGeometry[value.upper()]
        raise ValueError(f"Unsupported geometry value: {value!r}")

    def _load_unified_source(
        self,
        label: str,
        raw_config: Any,
        context: PipelineContext,
    ) -> SourceData:
        """Load one source directly through ``source_registry``."""
        import davinci_monet.models  # noqa: F401
        import davinci_monet.observations  # noqa: F401
        from davinci_monet.core.registry import ComponentNotFoundError, source_registry

        cfg = self._as_dict(raw_config)
        source_type = str(cfg.get("type") or "generic")
        role = cfg.get("role")
        variables = self._normalize_var_configs(cfg.get("variables", {}))
        variable_names = list(variables) or None
        file_paths = self._file_list(cfg.get("files") or cfg.get("filename"))

        analysis = context.config.get("analysis", {})
        time_range = None
        if analysis.get("start_time") and analysis.get("end_time"):
            time_range = (analysis["start_time"], analysis["end_time"])

        try:
            reader_cls = source_registry.get(source_type)
        except ComponentNotFoundError as e:
            available = ", ".join(source_registry.list())
            raise ValueError(
                f"Unknown source type '{source_type}' for source '{label}'. "
                f"Available source types: {available}"
            ) from e

        role = self._infer_source_role(role, reader_cls)
        reader = reader_cls()
        # Control keys consumed by the loader / schema, NOT forwarded to the
        # reader's open(). Covers the unified SourceConfig keys plus every legacy
        # model:/obs: ModelConfig/ObservationConfig field, since an auto-converted
        # legacy control file (model_dump) materializes those keys (e.g.
        # files_vert=None, mod_kwargs={}, projection=None) at the source level —
        # forwarding them to xr.open_dataset would raise. mod_kwargs is handled
        # separately below (it is the reader-options channel).
        passthrough_keys = {
            # unified + shared
            "type",
            "role",
            "files",
            "filename",
            "variables",
            "radius_of_influence",
            "mapping",
            "display_name",
            "resample",
            "min_obs_count",
            "track_obs_count",
            # legacy ModelConfig control keys
            "files_vert",
            "files_surf",
            "mod_type",
            "mod_kwargs",
            "projection",
            "plot_kwargs",
            "apply_ak",
            # legacy ObservationConfig control keys
            "obs_type",
            "sat_type",
            "use_airnow",
            "data_proc",
            "grid_source",
            "time_resolution",
            "save_binned",
            "load_binned",
            "binned_file",
        }
        reader_kwargs = {k: v for k, v in cfg.items() if k not in passthrough_keys}
        # mod_kwargs is the legacy reader-options channel: flatten it so its
        # entries reach the reader's open() (preserving LoadModelsStage parity).
        mod_kwargs = cfg.get("mod_kwargs")
        if isinstance(mod_kwargs, dict):
            reader_kwargs.update(mod_kwargs)
        # Drop None-valued kwargs (schema defaults) so they never leak into the
        # reader / xarray as unexpected keyword arguments.
        reader_kwargs = {k: v for k, v in reader_kwargs.items() if v is not None}
        import inspect

        open_kwargs = dict(reader_kwargs)
        if "time_range" in inspect.signature(reader.open).parameters:
            open_kwargs["time_range"] = time_range
        if "progress_callback" in inspect.signature(reader.open).parameters:

            def _per_file_progress(i: int, total: int, name: str) -> None:
                # Right-align the counter to the width of `total` so it stays a
                # fixed width as files tick, and put the (often long) file name
                # on its own indented line so it is never truncated off the end
                # of the status line by the terminal width.
                width = len(str(total))
                context.log_progress(f"step: loading {label} [{i:>{width}}/{total}]\n      {name}")

            open_kwargs["progress_callback"] = _per_file_progress
        data = reader.open(file_paths, variables=variable_names, **open_kwargs)
        if time_range and "time" in data:
            start_time, end_time = time_range
            data = data.sel(time=slice(start_time, end_time))
        if variables:
            data = self._apply_variable_config(data, variables)

        resample_freq = cfg.get("resample")
        if resample_freq:
            from davinci_monet.observations.base import resample_dataset

            data = resample_dataset(
                data,
                str(resample_freq),
                min_count=cfg.get("min_obs_count"),
                track_count=bool(cfg.get("track_obs_count")),
            )

        geometry = self._data_geometry(getattr(reader, "geometry"))
        source = SourceData(
            data=data,
            label=label,
            source_type=source_type,
            geometry=geometry,
            role=role,
            variables=variables,
            config=cfg,
        )
        return source

    @staticmethod
    def _infer_source_role(explicit_role: Any, reader_cls: type[Any]) -> str | None:
        if explicit_role:
            return str(explicit_role)
        module = getattr(reader_cls, "__module__", "")
        if module.startswith("davinci_monet.observations"):
            return "obs"
        if module.startswith("davinci_monet.models"):
            return "model"
        return None

    @staticmethod
    def _apply_variable_config(
        data: xr.Dataset,
        variables: dict[str, dict[str, Any]],
    ) -> xr.Dataset:
        """Apply common variable scaling, masking, renaming, and metadata."""
        ds = data
        for var_name, cfg in list(variables.items()):
            source = cfg.get("source_name")
            if source and source in ds and var_name not in ds:
                ds = ds.rename({source: var_name})
            if var_name not in ds:
                continue
            arr = ds[var_name]
            if "unit_scale" in cfg:
                scale = cfg["unit_scale"]
                method = cfg.get("unit_scale_method", "*")
                if method == "*":
                    arr = arr * scale
                elif method == "/":
                    arr = arr / scale
                elif method == "+":
                    arr = arr + scale
                elif method == "-":
                    arr = arr - scale
            nan_value = cfg.get("nan_value")
            if nan_value is not None:
                arr = arr.where(arr != nan_value)
            min_val = cfg.get("obs_min")
            if min_val is not None:
                arr = arr.where(arr >= min_val)
            max_val = cfg.get("obs_max")
            if max_val is not None:
                arr = arr.where(arr <= max_val)
            if cfg.get("units"):
                arr.attrs["units"] = cfg["units"]
            if cfg.get("display_name"):
                arr.attrs["display_name"] = cfg["display_name"]
            ds[var_name] = arr
            rename_to = cfg.get("rename")
            if rename_to and rename_to != var_name:
                ds = ds.rename({var_name: rename_to})

        return ds

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

    @staticmethod
    def _register_source(context: PipelineContext, label: str, obj: SourceData) -> None:
        """Register a loaded source.

        ``context.sources`` is canonical. Role-specific dictionaries are
        compatibility mirrors for legacy callers and must not be required by
        standard stages.
        """
        context.sources[label] = obj
        if obj.role == "model":
            context.models[label] = obj
        elif obj.role == "obs":
            context.observations[label] = obj
        obj.data.attrs["source_label"] = label
        obj.data.attrs["geometry"] = obj.geometry.name.lower()
        if obj.role:
            obj.data.attrs["role"] = obj.role


class PairingStage(BaseStage):
    """Stage for pairing model and observation data.

    Uses the pairing engine to match model output with observations.
    """

    def __init__(self) -> None:
        super().__init__("pairing")

    def validate(self, context: PipelineContext) -> bool:
        """Validate that models and observations are loaded."""
        pairs_config = context.config.get("pairs")
        if isinstance(pairs_config, dict) and any(
            isinstance(pair, dict) and bool(pair.get("sources")) for pair in pairs_config.values()
        ):
            return True
        if pairs_config and len(context.sources) >= 2:
            return True
        return bool(context.models) and bool(context.observations)

    def execute(self, context: PipelineContext) -> StageResult:
        """Pair source data through the unified role-neutral engine.

        Jobs come from the explicit ``pairs:`` block when present, otherwise
        from implicit auto-pairing synthesized from role-tagged sources and the
        model source's ``mapping:``. Either way they run through
        :meth:`_execute_source_pair_jobs` / ``engine.pair_sources``.
        """
        import time

        start = time.time()

        # Get pairing configuration
        pairing_config_dict = context.config.get("pairing", {})

        source_jobs, source_pair_errors = self._build_source_pair_jobs(context)
        if source_pair_errors:
            context.metadata.setdefault("pairing_errors", []).extend(source_pair_errors)
            return self._create_result(
                StageStatus.FAILED,
                data={"paired_keys": []},
                error="Invalid source pair configuration: " + "; ".join(source_pair_errors),
                duration=time.time() - start,
                count=0,
            )
        if not source_jobs:
            # No explicit ``pairs:`` block: synthesize jobs from role-tagged
            # sources using each model source's ``mapping:`` (implicit auto-
            # pairing). These flow through the same unified executor as ``pairs:``
            # jobs, so all pairing goes through ``engine.pair_sources``.
            source_jobs = self._build_implicit_pair_jobs(context)

        if not source_jobs:
            return self._create_result(
                StageStatus.COMPLETED,
                data={"paired_keys": []},
                duration=time.time() - start,
                count=0,
            )

        return self._execute_source_pair_jobs(
            context,
            source_jobs,
            pairing_config_dict,
            start,
        )

    def _build_source_pair_jobs(
        self, context: PipelineContext
    ) -> tuple[list[SourcePairJob], list[str]]:
        """Build role-neutral pair jobs from the explicit ``pairs:`` block.

        Each pair must be in unified form (``sources: [a, b]``). Implicit
        auto-pairing (no ``pairs:`` block) is handled separately by
        :meth:`_build_implicit_pair_jobs`.
        """
        from davinci_monet.pairing.direction import resolve_pair_direction

        jobs: list[SourcePairJob] = []
        errors: list[str] = []
        pair_index = 0
        pairs_config = context.config.get("pairs")

        if not isinstance(pairs_config, dict):
            return [], errors

        for pair_name, raw_pair in pairs_config.items():
            if not isinstance(raw_pair, dict):
                errors.append(f"Pair '{pair_name}' must be a mapping")
                continue
            if raw_pair.get("sources"):
                srcs = raw_pair.get("sources") or []
                if len(srcs) != 2:
                    errors.append(f"Pair '{pair_name}' must list exactly two sources; got {srcs!r}")
                    continue
                a_label, b_label = str(srcs[0]), str(srcs[1])
                if a_label not in context.sources or b_label not in context.sources:
                    missing = [
                        label for label in (a_label, b_label) if label not in context.sources
                    ]
                    errors.append(
                        f"Pair '{pair_name}' references unknown source(s): " f"{', '.join(missing)}"
                    )
                    continue
                a_obj = context.sources[a_label]
                b_obj = context.sources[b_label]
                a_geom = self._source_geometry(a_obj)
                b_geom = self._source_geometry(b_obj)
                explicit_ref = raw_pair.get("reference")
                explicit_pos = None
                if explicit_ref is not None:
                    if explicit_ref == a_label:
                        explicit_pos = "a"
                    elif explicit_ref == b_label:
                        explicit_pos = "b"
                    else:
                        errors.append(
                            f"Pair '{pair_name}' references unknown source "
                            f"'{explicit_ref}'. Expected one of {srcs}."
                        )
                        continue
                ref_geom, comp_geom = resolve_pair_direction(
                    a_geom, b_geom, explicit_reference=explicit_pos
                )
                if explicit_pos == "b" or (
                    explicit_pos is None and ref_geom is b_geom and comp_geom is a_geom
                ):
                    reference_label, reference_obj = b_label, b_obj
                    comparand_label, comparand_obj = a_label, a_obj
                else:
                    reference_label, reference_obj = a_label, a_obj
                    comparand_label, comparand_obj = b_label, b_obj
                vmap = raw_pair.get("variables") or {}
                reference_var = vmap.get(reference_label)
                comparand_var = vmap.get(comparand_label)
                if not reference_var or not comparand_var:
                    missing = [
                        label
                        for label, value in (
                            (reference_label, reference_var),
                            (comparand_label, comparand_var),
                        )
                        if not value
                    ]
                    errors.append(
                        f"Pair '{pair_name}' missing variable mapping for source(s): "
                        f"{', '.join(missing)}"
                    )
                    continue
                pair_index += 1
                jobs.append(
                    SourcePairJob(
                        index=pair_index,
                        pair_key=str(pair_name),
                        reference_label=reference_label,
                        reference_obj=reference_obj,
                        comparand_label=comparand_label,
                        comparand_obj=comparand_obj,
                        reference_var=str(reference_var),
                        comparand_var=str(comparand_var),
                        reference_role=self._source_role(reference_obj),
                        comparand_role=self._source_role(comparand_obj),
                        radius_of_influence=self._pair_radius(raw_pair, comparand_obj),
                    )
                )
            else:
                errors.append(
                    f"Pair '{pair_name}' must declare 'sources: [a, b]'. The legacy "
                    "'model'/'obs'/'variable' pair shape is no longer accepted; run "
                    "'davinci-monet migrate-config' to convert the control file."
                )
        return jobs, errors

    def _build_implicit_pair_jobs(self, context: PipelineContext) -> list[SourcePairJob]:
        """Synthesize pair jobs when no explicit ``pairs:`` block is configured.

        Implicit auto-pairing: every model-role source is paired against every
        obs-role source named in that model's ``mapping:`` (``{obs_label: {obs_var:
        model_var}}``), reference = obs, comparand = model. The pair key matches
        the historical ``<model_label>_<obs_label>`` form so existing single-block
        configs (e.g. ``plots: {data: [cmaq_airnow]}``) keep resolving. Jobs run
        through the same unified executor (``engine.pair_sources``) as ``pairs:``
        jobs — there is no separate legacy ``engine.pair`` loop.

        ``mapping:`` is read from the source's loaded config (``SourceData.config``)
        and, for pre-populated ``context.models`` that bypassed
        :class:`LoadSourcesStage`, falls back to the legacy ``model:`` config block.
        """
        jobs: list[SourcePairJob] = []
        pair_index = 0
        legacy_model_cfg = context.config.get("model", {})
        if not isinstance(legacy_model_cfg, dict):
            legacy_model_cfg = {}

        for model_label, model_obj in self._model_role_sources(context):
            mapping = self._source_mapping(model_obj)
            if not mapping:
                legacy_entry = legacy_model_cfg.get(model_label, {})
                if isinstance(legacy_entry, dict):
                    mapping = legacy_entry.get("mapping", {}) or {}
            if not isinstance(mapping, dict) or not mapping:
                continue

            for obs_label, var_mapping in mapping.items():
                if not isinstance(var_mapping, dict) or not var_mapping:
                    continue
                obs_obj = self._lookup_source(context, str(obs_label))
                if obs_obj is None:
                    continue
                # mapping is {obs_var: model_var}; one job per variable pair.
                for obs_var, model_var in var_mapping.items():
                    pair_index += 1
                    jobs.append(
                        SourcePairJob(
                            index=pair_index,
                            pair_key=f"{model_label}_{obs_label}",
                            reference_label=str(obs_label),
                            reference_obj=obs_obj,
                            comparand_label=str(model_label),
                            comparand_obj=model_obj,
                            reference_var=str(obs_var),
                            comparand_var=str(model_var),
                            reference_role=self._source_role(obs_obj) or "obs",
                            comparand_role=self._source_role(model_obj) or "model",
                            radius_of_influence=self._pair_radius({}, model_obj),
                        )
                    )
        return jobs

    @staticmethod
    def _model_role_sources(context: PipelineContext) -> list[tuple[str, Any]]:
        """Return (label, obj) for model-role sources, preferring ``sources``.

        Falls back to the legacy ``context.models`` mirror for direct stage use
        that bypassed :class:`LoadSourcesStage`.
        """
        items: list[tuple[str, Any]] = []
        if context.sources:
            for label, obj in context.sources.items():
                if PairingStage._source_role(obj) == "model":
                    items.append((str(label), obj))
        if not items:
            items = [(str(label), obj) for label, obj in context.models.items()]
        return items

    @staticmethod
    def _lookup_source(context: PipelineContext, label: str) -> Any:
        """Resolve a source by label from ``sources`` then the obs mirror."""
        if label in context.sources:
            return context.sources[label]
        if label in context.observations:
            return context.observations[label]
        return None

    @staticmethod
    def _source_mapping(obj: Any) -> dict[str, Any]:
        """Return a source's ``mapping:`` config, if any."""
        cfg = getattr(obj, "config", None)
        if isinstance(cfg, dict):
            mapping = cfg.get("mapping")
            if isinstance(mapping, dict):
                return mapping
        return {}

    @staticmethod
    def _source_dataset(obj: Any) -> xr.Dataset | None:
        data = obj.data if hasattr(obj, "data") else obj
        return data if isinstance(data, xr.Dataset) else None

    @staticmethod
    def _source_role(obj: Any) -> str | None:
        role = getattr(obj, "role", None)
        if role is None:
            data = PairingStage._source_dataset(obj)
            role = data.attrs.get("role") if data is not None else None
        return str(role) if role else None

    @staticmethod
    def _source_geometry(obj: Any) -> DataGeometry:
        geom = getattr(obj, "geometry", None)
        if isinstance(geom, DataGeometry):
            return geom
        data = PairingStage._source_dataset(obj)
        if data is not None and "geometry" in data.attrs:
            raw = data.attrs["geometry"]
            if isinstance(raw, DataGeometry):
                return raw
            if isinstance(raw, str):
                return DataGeometry[raw.upper()]
        from davinci_monet.pairing import PairingEngine

        if data is None:
            raise PipelineError("Cannot determine source geometry without a dataset")
        return PairingEngine()._detect_geometry(data)

    @staticmethod
    def _pair_radius(pair_spec: dict[str, Any], comparand_obj: Any) -> float:
        if pair_spec.get("radius_of_influence") is not None:
            return float(pair_spec["radius_of_influence"])
        cfg = getattr(comparand_obj, "config", None)
        if isinstance(cfg, dict) and cfg.get("radius_of_influence") is not None:
            return float(cfg["radius_of_influence"])
        return float(getattr(comparand_obj, "radius_of_influence", 12000.0))

    def _execute_source_pair_jobs(
        self,
        context: PipelineContext,
        jobs: list[SourcePairJob],
        pairing_config_dict: dict[str, Any],
        start: float,
    ) -> StageResult:
        """Execute source-pair jobs through the role-neutral engine."""
        import time

        from davinci_monet.pairing import PairingConfig, PairingEngine

        debug = context.config.get("analysis", {}).get("debug", False)
        paired_count = 0
        execution_errors: list[str] = []
        context.log_progress(f"    parallel_start: {len(jobs)}")

        for job in jobs:
            pair_start = time.time()
            context.log_progress(f"    parallel_started: {job.pair_key}")
            ref_ds = self._source_dataset(job.reference_obj)
            comp_ds = self._source_dataset(job.comparand_obj)
            if ref_ds is None or comp_ds is None:
                execution_errors.append(f"{job.pair_key}: reference or comparand data is None")
                context.log_progress(f"    parallel_completed: {job.pair_key} - FAILED")
                continue

            try:
                pairing_cfg = PairingConfig(
                    radius_of_influence=job.radius_of_influence,
                    time_tolerance=pairing_config_dict.get("time_tolerance", "1h"),
                    time_method=pairing_config_dict.get("time_method", "nearest"),
                )
                engine = PairingEngine()
                paired_obj = engine.pair_sources(
                    reference=ref_ds,
                    comparand=comp_ds,
                    reference_vars=[job.reference_var],
                    comparand_vars=[job.comparand_var],
                    reference_geometry=self._source_geometry(job.reference_obj),
                    comparand_geometry=self._source_geometry(job.comparand_obj),
                    config=pairing_cfg,
                    reference_label=job.reference_label,
                    comparand_label=job.comparand_label,
                )
                paired_data = paired_obj.data
                tag_paired_roles(
                    paired_data,
                    reference_label=job.reference_label,
                    comparand_label=job.comparand_label,
                    reference_role=job.reference_role,
                    comparand_role=job.comparand_role,
                )
                context.paired[job.pair_key] = paired_obj
                paired_count += 1
                n_vars = len(iter_paired_variable_pairs(paired_data))
                n_points = paired_data.sizes.get("time", paired_data.sizes.get("x", 0))
                timing_str = f" [{_format_duration(time.time() - pair_start)}]" if debug else ""
                context.log_progress(
                    f"    parallel_completed: {job.pair_key} - "
                    f"{n_vars} vars, {_format_size(n_points)} points{timing_str}"
                )
            except Exception as e:
                execution_errors.append(f"{job.pair_key}: {e}")
                context.log_progress(f"    parallel_completed: {job.pair_key} - FAILED")

        context.log_progress("    parallel_end")
        if execution_errors:
            context.metadata.setdefault("pairing_errors", []).extend(execution_errors)
            return self._create_result(
                StageStatus.FAILED,
                data={"paired_keys": list(context.paired.keys())},
                error="Source pair execution failed: " + "; ".join(execution_errors),
                duration=time.time() - start,
                count=paired_count,
            )
        return self._create_result(
            StageStatus.COMPLETED,
            data={"paired_keys": list(context.paired.keys())},
            duration=time.time() - start,
            count=paired_count,
        )


class StatisticsStage(BaseStage):
    """Stage for calculating statistics on paired or single-source data."""

    def __init__(self) -> None:
        super().__init__("statistics")

    def validate(self, context: PipelineContext) -> bool:
        """Run for paired comparisons or descriptive stats on loaded sources."""
        return bool(context.paired) or bool(iter_single_source_datasets(context))

    def _execute_descriptive(
        self,
        context: PipelineContext,
        sources: list[tuple[str, Any, xr.Dataset, str | None]] | None = None,
    ) -> StageResult:
        """Descriptive stats for unpaired sources."""
        import time

        import numpy as np

        context.metadata["statistics_kind"] = "descriptive"
        start = time.time()
        all_stats: dict[str, dict[str, dict[str, float]]] = {}
        source_items = sources if sources is not None else iter_single_source_datasets(context)
        for source_label, _source_obj, ds, _role in source_items:
            source_stats: dict[str, dict[str, float]] = {}
            for var_name in ds.data_vars:
                var_key = str(var_name)
                values = ds[var_key].values.flatten()
                values = values[np.isfinite(values)]
                if len(values) < 1:
                    continue
                source_stats[var_key] = {
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
            all_stats[source_label] = source_stats
        return self._create_result(
            StageStatus.COMPLETED,
            data=all_stats,
            duration=time.time() - start,
        )

    def execute(self, context: PipelineContext) -> StageResult:
        """Comparison statistics on paired data, or descriptive statistics on
        unpaired sources."""
        import time

        single_sources = iter_single_source_datasets(context)
        if not context.paired and single_sources:
            return self._execute_descriptive(context, single_sources)

        context.metadata["statistics_kind"] = "comparison"
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

        # Pair (obs, model) variables by role/source-label, matched on canonical
        # name; stats are keyed by the canonical name (R-5).
        for obs_var, model_var, base_name in iter_paired_variable_pairs(paired_data):
            df = calculator.compute(
                paired_data,
                reference_var=obs_var,
                comparand_var=model_var,
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
            Paired dataset with role-tagged source-label variables and a flight coordinate.

        Returns
        -------
        list[dict[str, Any]]
            List of dictionaries with per-flight statistics.
        """
        import numpy as np

        flights = np.unique(paired_data["flight"].values)
        flight_stats: list[dict[str, Any]] = []

        var_pairs = iter_paired_variable_pairs(paired_data)

        for flight in flights:
            mask = paired_data["flight"].values == flight
            flight_data = paired_data.isel(time=mask)

            for obs_var, model_var, base_name in var_pairs:
                if obs_var not in flight_data or model_var not in flight_data:
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
    """Stage for generating plots from paired or single-source data."""

    def __init__(self) -> None:
        super().__init__("plotting")

    @staticmethod
    def _config_dict(value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return dict(value.model_dump(exclude_none=True))
        if isinstance(value, dict):
            return value
        return {}

    @classmethod
    def _source_var_config(
        cls,
        config: dict[str, Any],
        source_label: str,
        variable_name: str,
    ) -> dict[str, Any]:
        """Return variable plot config for a source label from unified or legacy config."""
        for block in ("sources", "model", "obs"):
            block_cfg = config.get(block, {})
            if not isinstance(block_cfg, dict):
                continue
            source_cfg = cls._config_dict(block_cfg.get(source_label, {}))
            variables = cls._config_dict(source_cfg.get("variables", {}))
            if variable_name in variables:
                return cls._config_dict(variables[variable_name])
        return {}

    def validate(self, context: PipelineContext) -> bool:
        """Run for paired comparisons or single-source plots."""
        return bool(context.paired) or bool(iter_single_source_datasets(context))

    def _execute_single_source(self, context: PipelineContext) -> StageResult:
        """Single-source plotting.

        Renders each plot spec against its single configured source, auto-splitting
        on a ``flight`` coord with >1 flight. Multi-figure renderers (e.g. hourly
        LMA density) return a list of ``(fig, suffix)``.
        """
        import logging
        import time
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np

        from davinci_monet.plots.base import build_series
        from davinci_monet.plots.registry import get_plotter

        _logger = logging.getLogger(__name__)
        _SCHEMA_KEYS = {
            "type",
            "obs",
            "source",
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
        source_map = {
            label: (obj, ds, role) for label, obj, ds, role in iter_single_source_datasets(context)
        }

        for plot_name, plot_spec in plots_config.items():
            plot_type = plot_spec.get("type", "")
            # Single-source specs carry either canonical ``source:`` or the
            # deprecated ``obs:`` key.
            if not plot_type or ("source" not in plot_spec and "obs" not in plot_spec):
                continue

            source_label = str(plot_spec.get("source") or plot_spec.get("obs") or "")
            variable = plot_spec.get("variable", "")

            if source_label not in source_map:
                errors.append(f"Source '{source_label}' not found for plot '{plot_name}'")
                continue

            _source_obj, ds, role = source_map[source_label]

            if variable not in ds.data_vars:
                errors.append(
                    f"Variable '{variable}' not in source '{source_label}' for plot '{plot_name}'"
                )
                continue

            plotter = get_plotter(plot_type)
            plot_kwargs = {k: v for k, v in plot_spec.items() if k not in _SCHEMA_KEYS}
            base_title = plot_kwargs.get("title", f"{variable} {plot_type}")

            has_flights = "flight" in ds.coords
            flight_ids = (
                sorted(set(np.unique(ds["flight"].values).tolist())) if has_flights else [None]
            )

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

                vals = subset[variable].values
                if not np.any(np.isfinite(vals)):
                    continue

                if "flight_tracks" in plot_spec:
                    flight_kwargs["obs_datasets"] = {
                        label: source_ds for label, (_obj, source_ds, _role) in source_map.items()
                    }

                try:
                    # Tag the single source so build_series picks up role +
                    # source label, then render through the unified contract.
                    tag_source_roles(subset, role=role, source_label=source_label)
                    result = plotter.render(build_series(subset, variable), **flight_kwargs)
                    if isinstance(result, list):
                        for fig, fig_suffix in result:
                            out_path = output_dir / f"{plot_name}{suffix}{fig_suffix}.png"
                            plotter.save(fig, out_path)
                            plt.close(fig)
                            plot_count += 1
                            plots_generated.append(str(out_path))
                            _logger.info(f"Saved source plot: {out_path}")
                    else:
                        fig = result
                        out_path = output_dir / f"{plot_name}{suffix}.png"
                        plotter.save(fig, out_path)
                        plt.close(fig)
                        plot_count += 1
                        plots_generated.append(str(out_path))
                        _logger.info(f"Saved source plot: {out_path}")
                except Exception as e:
                    label = f"'{plot_name}' (flight {fid})" if fid else f"'{plot_name}'"
                    errors.append(f"Plot {label} failed: {e}")
                    _logger.warning(f"Source plot {label} failed: {e}")

        return self._create_result(
            StageStatus.COMPLETED if plot_count > 0 or not plots_config else StageStatus.SKIPPED,
            data={"plot_count": plot_count, "plots_generated": plots_generated, "errors": errors},
            duration=time.time() - start,
        )

    def execute(self, context: PipelineContext) -> StageResult:
        """Generate comparison plots from paired data, or plots from unpaired sources."""
        if not context.paired and iter_single_source_datasets(context):
            return self._execute_single_source(context)

        import time
        from pathlib import Path

        import matplotlib.pyplot as plt

        from davinci_monet.plots import get_plotter
        from davinci_monet.plots.base import build_series, format_plot_title

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
                    if not isinstance(pair_spec, dict):
                        pair_spec = {}

                    sources = [str(src) for src in pair_spec.get("sources", [])]
                    model_label = ""
                    obs_label = ""
                    var_spec: dict[str, Any] = {}

                    # Resolve the paired-data key. Unified ``sources:`` pairs and
                    # implicitly auto-paired pairs are both keyed by the pair name
                    # in ``context.paired``; for unified pairs we also recover the
                    # reference/comparand labels from the spec.
                    pair_key = pair_name
                    if "sources" in pair_spec:
                        reference_label = pair_spec.get("reference")
                        if reference_label is not None and str(reference_label) in sources:
                            obs_label = str(reference_label)
                            model_label = next(
                                (src for src in sources if src != obs_label),
                                "",
                            )
                    if pair_key not in context.paired:
                        continue

                    paired_obj = context.paired[pair_key]
                    paired_data = paired_obj.data if hasattr(paired_obj, "data") else paired_obj

                    if "sources" in pair_spec:
                        pair_vars = iter_paired_variable_pairs(paired_data)
                        if not pair_vars:
                            continue
                        fallback_obs_name, fallback_model_name, fallback_var = pair_vars[0]
                        if not obs_label:
                            obs_label = str(
                                paired_data[fallback_obs_name].attrs.get(
                                    "source_label", "reference"
                                )
                            )
                        if not model_label:
                            model_label = str(
                                paired_data[fallback_model_name].attrs.get(
                                    "source_label", "comparand"
                                )
                            )
                        source_vars = pair_spec.get("variables") or {}
                        obs_var = str(source_vars.get(obs_label) or fallback_var)
                        model_var = str(source_vars.get(model_label) or obs_var)
                        var_spec = {"obs_var": obs_var, "model_var": model_var}
                        obs_var_name, model_var_name = resolve_paired_var_names(
                            paired_data, obs_var, obs_label, model_label
                        )
                    else:
                        # No (or empty) pair spec: a plot referencing a pair key in
                        # context.paired directly (implicit auto-pairing or a
                        # pre-paired key). Recover labels/vars from the paired
                        # data's role-tagged source-label attrs.
                        pair_vars = iter_paired_variable_pairs(paired_data)
                        if not pair_vars:
                            continue
                        obs_var_name, model_var_name, obs_var = pair_vars[0]
                        obs_label = str(
                            paired_data[obs_var_name].attrs.get("source_label", "reference")
                        )
                        model_label = str(
                            paired_data[model_var_name].attrs.get("source_label", "comparand")
                        )
                        var_spec = {"obs_var": obs_var, "model_var": obs_var}

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

                    if obs_var_name not in paired_data or model_var_name not in paired_data:
                        continue

                    # Get plotter config from source variable settings. The
                    # comparand side wins for comparison-specific plot limits;
                    # reference settings are a fallback for same-source-role runs.
                    model_var = var_spec.get("model_var", "")
                    obs_var = var_spec.get("obs_var", "")
                    var_config = self._source_var_config(context.config, model_label, model_var)
                    if not var_config:
                        var_config = self._source_var_config(context.config, obs_label, obs_var)
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
                    # contour layer; the paired dataset usually carries sampled
                    # values at reference locations only. Keep the renderer option
                    # name `model_field` for compatibility.
                    if plot_type == "spatial_overlay":
                        if "model_field" not in plot_options:
                            source_obj = (
                                context.sources.get(model_label)
                                or context.sources.get(obs_label)
                                or context.models.get(model_label)
                                or context.observations.get(obs_label)
                            )
                            if source_obj is not None:
                                source_ds = (
                                    source_obj.data if hasattr(source_obj, "data") else source_obj
                                )
                                source_vars = getattr(source_ds, "data_vars", {})
                                field_var = model_var if model_var in source_vars else obs_var
                                if source_ds is not None and field_var in source_vars:
                                    plot_options["model_field"] = source_ds[field_var]
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

                    # Build subtitle: date range (or snapshot timestamp for
                    # spatial_overlay plots).  The source-pair prefix ("Model vs
                    # Obs") has been removed — plot titles already name the
                    # sources, and the separator rendered as a missing-glyph box
                    # in the Poppins font.
                    start_time = analysis_config.get("start_time", "")
                    end_time = analysis_config.get("end_time", "")
                    date_str = ""
                    if start_time:
                        start_date = str(start_time).split(" ")[0]
                        end_date = str(end_time).split(" ")[0] if end_time else start_date
                        date_str = (
                            start_date if start_date == end_date else f"{start_date} - {end_date}"
                        )
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
                    subtitle = when
                    # Forward per-plot axis label overrides to the plotter config
                    # so renderers like scatter can display source-named labels
                    # (e.g. "MODIS Terra AOD") instead of "Observed AOD (550 nm)".
                    label_aliases = {
                        "reference_label": "obs_label",
                        "comparand_label": "model_label",
                        "obs_label": "obs_label",
                        "model_label": "model_label",
                    }
                    for input_key, plotter_key in label_aliases.items():
                        if input_key in plot_spec:
                            plotter_config[plotter_key] = plot_spec[input_key]
                    if subtitle:
                        plotter_config["title"] = f"{title}\n{subtitle}"

                    # Get plotter
                    plotter = get_plotter(plot_type, config=plotter_config)

                    # Create subdirectory by reference source.
                    reference_output_dir = output_dir / obs_label
                    reference_output_dir.mkdir(parents=True, exist_ok=True)

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
                                reference_output_dir
                                / f"{flight_id}_{file_index:02d}_{plot_name}.png"
                            )
                            plotter.save(fig, output_path, dpi=300)
                            plots_generated.append(str(output_path))

                            pdf_path = (
                                reference_output_dir
                                / f"{flight_id}_{file_index:02d}_{plot_name}.pdf"
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
                                reference_output_dir
                                / f"site_{site_id}_{file_index:02d}_{plot_name}.png"
                            )
                            plotter.save(fig, output_path, dpi=300)
                            plots_generated.append(str(output_path))

                            pdf_path = (
                                reference_output_dir
                                / f"site_{site_id}_{file_index:02d}_{plot_name}.pdf"
                            )
                            plotter.save(fig, pdf_path)
                            plots_generated.append(str(pdf_path))

                            plt.close(fig)
                            site_count += 1
                            file_index += 1

                        context.log_progress(f"done: saved {site_count} sites to {obs_label}/")
                    else:
                        # Generate single plot via the unified render contract.
                        fig = plotter.render(
                            build_series(paired_data, obs_var_name, model_var_name), **plot_options
                        )

                        # Save plot (prefixed for ordering)
                        output_path = reference_output_dir / f"{file_index:02d}_{plot_name}.png"
                        plotter.save(fig, output_path, dpi=300)
                        plots_generated.append(str(output_path))

                        # Also save PDF
                        pdf_path = reference_output_dir / f"{file_index:02d}_{plot_name}.pdf"
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

        # Save statistics from the statistics stage. Comparison stats go to
        # statistics_summary.csv for comparison stats; single-source descriptive
        # stats go to a separate statistics_descriptive.csv.
        stats_result = context.results.get("statistics")
        stats_kind = context.metadata.get("statistics_kind", "comparison")
        if stats_result and stats_result.data and stats_kind == "descriptive":
            context.log_progress("step: Writing descriptive statistics CSV...")
            desc_rows = []
            for source_label, var_stats in stats_result.data.items():
                for var_name, var_metrics in var_stats.items():
                    if var_name.startswith("_"):
                        continue
                    desc_rows.append({"Variable": var_name, "Source": source_label, **var_metrics})
            if desc_rows:
                desc_df = pd.DataFrame(desc_rows).set_index("Variable")
                desc_file = output_dir / "statistics_descriptive.csv"
                desc_df.to_csv(desc_file)
                saved_files.append(str(desc_file))
                context.log_progress(f"done: {len(desc_rows)} rows saved")
        elif stats_result and stats_result.data:
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
                    mean_reference = _get_metric(var_stats, "MO", "obs_mean")
                    mean_comparand = _get_metric(var_stats, "MP", "model_mean")
                    row["Mean_Reference"] = mean_reference
                    row["Mean_Comparand"] = mean_comparand
                    row["Mean_Obs"] = mean_reference
                    row["Mean_Model"] = mean_comparand
                    row["MB"] = _get_metric(var_stats, "MB", "mean_bias")
                    row["RMSE"] = _get_metric(var_stats, "RMSE", "rmse")
                    row["R"] = _get_metric(var_stats, "R", "correlation")
                    row["IOA"] = _get_metric(var_stats, "IOA", "ioa")

                    # Prefer computed NMB/NME if present; otherwise derive as fallback
                    nmb = _get_metric(var_stats, "NMB", default=float("nan"))
                    nme = _get_metric(var_stats, "NME", default=float("nan"))
                    obs_mean = row["Mean_Reference"]

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


class SummaryStage(BaseStage):
    """Optional final stage: AI summary of the analysis run via the Claude API.

    Always non-fatal. When ``summary.enabled`` is false the stage is skipped.
    Any failure (missing dependency/key, network/API error) logs a warning and
    returns SKIPPED so an otherwise-complete run is still reported successful.
    """

    def __init__(self) -> None:
        super().__init__("summary")

    def execute(self, context: PipelineContext) -> StageResult:
        import logging
        import time
        from pathlib import Path

        from davinci_monet.ai import collect_payload, extract_bullets, generate_summary
        from davinci_monet.config.schema import SummaryConfig

        start = time.time()
        logger = logging.getLogger(__name__)

        # The summary is a bonus produced after the analysis is already complete,
        # so this stage must never fail the run. Any error (config validation,
        # payload collection, the API call, or writing the file) degrades to
        # SKIPPED with a warning rather than propagating to the runner, which
        # would otherwise mark the whole run FAILED.
        try:
            cfg = SummaryConfig.model_validate(context.config.get("summary") or {})
            if not cfg.enabled:
                return self._create_result(
                    StageStatus.SKIPPED,
                    data={"skipped": "summary disabled"},
                    duration=time.time() - start,
                )

            payload = collect_payload(context, cfg)
            result = generate_summary(payload, cfg=cfg)

            output_dir = Path(context.config.get("analysis", {}).get("output_dir") or ".")
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / cfg.output_filename
            out_path.write_text(result.markdown)
        except Exception as exc:  # noqa: BLE001 - summary must never fail the run
            logger.warning("AI summary skipped: %s: %s", type(exc).__name__, exc)
            return self._create_result(
                StageStatus.SKIPPED,
                data={"skipped": f"{type(exc).__name__}: {exc}"},
                duration=time.time() - start,
            )

        # The brief is displayed by the runner at end of run (via
        # ProgressFormatter.print_summary, reading data["markdown"]). A raw
        # log_progress(markdown) here is swallowed by the prefix-matching
        # progress callback, so it is not used for display.
        context.log_progress(f"done: AI summary written ({result.images_sent} figures)")

        return self._create_result(
            StageStatus.COMPLETED,
            data={
                "summary_file": str(out_path),
                "markdown": result.markdown,
                "bullets": extract_bullets(result.markdown),
                "model": result.model,
                "usage": result.usage,
                "credits_remaining": result.credits_remaining,
                "images_sent": result.images_sent,
            },
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
        SaveResultsStage(),
        SummaryStage(),
    ]


def create_obs_pipeline() -> list[BaseStage]:
    """Create a single-source pipeline (no pairing stage).

    Returns
    -------
    list[BaseStage]
        List of stages for single-source analysis.
    """
    return [
        LoadSourcesStage(),
        StatisticsStage(),
        PlottingStage(),
        SaveResultsStage(),
        SummaryStage(),
    ]
