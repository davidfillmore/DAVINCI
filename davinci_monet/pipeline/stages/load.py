"""Source-loading stage.

Loads the unified ``sources:`` config through ``source_registry`` and unifies
any pre-populated containers into :attr:`PipelineContext.sources`.
"""

from __future__ import annotations

from typing import Any

import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    SourceData,
    StageResult,
    StageStatus,
)


class LoadSourcesStage(BaseStage):
    """Standard data-source loading stage.

    Loads the unified ``sources:`` config directly through ``source_registry``.
    Legacy ``model:`` / ``obs:`` configs are no longer accepted — they are
    rejected at config load (see
    :func:`~davinci_monet.config.parser.load_config`); convert them with
    ``davinci-monet migrate-config``. Loaded datasets are tagged with ``role``
    (``"model"``/``"obs"`` when known), ``source_label``, and ``geometry``.
    Results are mirrored in ``context.models`` / ``context.observations`` so
    existing accessors keep working.
    """

    def __init__(self) -> None:
        super().__init__("load_sources")

    def validate(self, context: PipelineContext) -> bool:
        """Validate that at least one source is configured."""
        cfg = context.config
        has_config = "sources" in cfg
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
        Legacy ``model:``/``obs:`` configs are rejected upstream at config load,
        so this stage only handles the unified ``sources:`` shape (plus any
        pre-populated containers for direct stage/unit-test use).
        """
        import time

        start = time.time()

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
        # reader's open(). Covers the unified SourceConfig keys plus the
        # historical model:/obs: control keys that a hand-migrated source config
        # may still carry (e.g. mod_kwargs, sat_type, grid_source); these are
        # filtered defensively so they never leak into xr.open_dataset as
        # unexpected kwargs. mod_kwargs is handled separately below (it is the
        # reader-options channel).
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
            # model-flavored reader/control keys
            "files_vert",
            "files_surf",
            "mod_type",
            "mod_kwargs",
            "projection",
            "plot_kwargs",
            "apply_ak",
            # obs-flavored reader/control keys
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
