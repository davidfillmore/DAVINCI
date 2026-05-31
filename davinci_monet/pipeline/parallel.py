"""Parallel execution utilities for pipeline stages.

This module provides utilities for executing pipeline stages in parallel,
using either threading or multiprocessing depending on the workload.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Generic, Sequence, TypeVar

from davinci_monet.pipeline.stages import PipelineContext, Stage, StageResult, StageStatus

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ParallelResult(Generic[T]):
    """Result of a parallel execution.

    Attributes
    ----------
    results
        List of individual results.
    errors
        List of any errors that occurred.
    success
        True if all tasks completed successfully.
    """

    results: list[T]
    errors: list[str]
    success: bool


class ParallelExecutor:
    """Execute tasks in parallel using thread or process pools.

    Parameters
    ----------
    max_workers
        Maximum number of parallel workers. If None, uses CPU count.
    use_processes
        If True, use processes instead of threads.
    timeout
        Timeout in seconds for each task.

    Examples
    --------
    >>> executor = ParallelExecutor(max_workers=4)
    >>> results = executor.map(process_file, file_list)
    """

    def __init__(
        self,
        max_workers: int | None = None,
        use_processes: bool = False,
        timeout: float | None = None,
    ) -> None:
        self._max_workers = max_workers
        self._use_processes = use_processes
        self._timeout = timeout

    def map(
        self,
        func: Callable[[T], Any],
        items: Sequence[T],
    ) -> ParallelResult[Any]:
        """Execute a function on multiple items in parallel.

        Parameters
        ----------
        func
            Function to execute on each item.
        items
            Items to process.

        Returns
        -------
        ParallelResult
            Combined results from all executions.
        """
        if not items:
            return ParallelResult(results=[], errors=[], success=True)

        executor_class = ProcessPoolExecutor if self._use_processes else ThreadPoolExecutor

        results: list[Any] = []
        errors: list[str] = []

        with executor_class(max_workers=self._max_workers) as executor:
            # Submit all tasks
            future_to_item: dict[Future[Any], T] = {
                executor.submit(func, item): item for item in items
            }

            # Collect results as they complete
            for future in as_completed(future_to_item, timeout=self._timeout):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    error_msg = f"Error processing {item}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

        return ParallelResult(
            results=results,
            errors=errors,
            success=len(errors) == 0,
        )

    def execute_stages(
        self,
        stages: Sequence[Stage],
        context: PipelineContext,
    ) -> list[StageResult]:
        """Execute multiple independent stages in parallel.

        Note: Stages must be independent and not modify shared state.

        Parameters
        ----------
        stages
            Stages to execute in parallel.
        context
            Pipeline context (must be thread-safe).

        Returns
        -------
        list[StageResult]
            Results from each stage.
        """

        def execute_stage(stage: Stage) -> StageResult:
            try:
                if not stage.validate(context):
                    return StageResult(
                        stage_name=stage.name,
                        status=StageStatus.SKIPPED,
                        error="Validation failed",
                    )
                return stage.execute(context)
            except Exception as e:
                return StageResult(
                    stage_name=stage.name,
                    status=StageStatus.FAILED,
                    error=str(e),
                )

        result = self.map(execute_stage, stages)
        return result.results


class ParallelPairingExecutor:
    """Execute model-observation pairing in parallel.

    Pairs multiple model-observation combinations concurrently.

    Parameters
    ----------
    max_workers
        Maximum number of parallel workers.

    Examples
    --------
    >>> executor = ParallelPairingExecutor(max_workers=4)
    >>> results = executor.pair_all(models, observations, config)
    """

    def __init__(self, max_workers: int | None = None) -> None:
        self._executor = ParallelExecutor(max_workers=max_workers, use_processes=False)

    def pair_all(
        self,
        models: dict[str, Any],
        observations: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Pair all model-observation combinations in parallel.

        Parameters
        ----------
        models
            Dictionary of model data.
        observations
            Dictionary of observation data.
        config
            Optional pairing configuration.

        Returns
        -------
        dict[str, Any]
            Dictionary of paired datasets keyed by "model_obs".
        """
        from davinci_monet.pairing import PairingConfig, PairingEngine

        config = config or {}
        pairs_to_process = []

        # Build list of pairs to process
        for model_label, model_data in models.items():
            for obs_label, obs_data in observations.items():
                pairs_to_process.append((model_label, model_data, obs_label, obs_data, config))

        def _is_var_mapping(value: Any) -> bool:
            return isinstance(value, dict) and all(
                isinstance(k, str) and isinstance(v, str) for k, v in value.items()
            )

        def _resolve_mapping(
            model_label: str, obs_label: str, cfg: dict[str, Any]
        ) -> dict[str, str] | None:
            # Prefer pipeline-style config: model -> <label> -> mapping -> obs_label
            model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else None
            if isinstance(model_cfg, dict) and model_label in model_cfg:
                mapping = model_cfg[model_label].get("mapping", {})
                if (
                    isinstance(mapping, dict)
                    and obs_label in mapping
                    and _is_var_mapping(mapping[obs_label])
                ):
                    return mapping[obs_label]

            mapping_cfg = cfg.get("mapping")
            if isinstance(mapping_cfg, dict):
                # model_label -> obs_label -> mapping
                if model_label in mapping_cfg and isinstance(mapping_cfg[model_label], dict):
                    model_map = mapping_cfg[model_label]
                    if obs_label in model_map and _is_var_mapping(model_map[obs_label]):
                        return model_map[obs_label]
                    if _is_var_mapping(model_map):
                        return model_map
                # obs_label -> mapping
                if obs_label in mapping_cfg and _is_var_mapping(mapping_cfg[obs_label]):
                    return mapping_cfg[obs_label]
                # direct mapping dict
                if _is_var_mapping(mapping_cfg):
                    return mapping_cfg

            return None

        def _cfg_get(cfg: dict[str, Any], key: str, default: Any) -> Any:
            pairing_cfg = cfg.get("pairing", {})
            if isinstance(pairing_cfg, dict) and key in pairing_cfg:
                return pairing_cfg[key]
            return cfg.get(key, default)

        def pair_single(args: tuple) -> tuple[str, Any]:
            model_label, model_data, obs_label, obs_data, cfg = args
            pair_key = f"{model_label}_{obs_label}"

            try:
                engine = PairingEngine()
                model_ds = model_data.data if hasattr(model_data, "data") else model_data
                obs_ds = obs_data.data if hasattr(obs_data, "data") else obs_data

                if model_ds is None or obs_ds is None:
                    return pair_key, None

                mapping = _resolve_mapping(model_label, obs_label, cfg)
                if mapping:
                    obs_vars = list(mapping.keys())
                    model_vars = list(mapping.values())
                else:
                    # Fallback to common variable names (identity mapping)
                    excluded = {
                        "lat",
                        "lon",
                        "latitude",
                        "longitude",
                        "time",
                        "x",
                        "y",
                        "z",
                        "lev",
                        "level",
                        "altitude",
                        "height",
                        "pressure",
                    }
                    obs_vars = sorted(
                        v
                        for v in obs_ds.data_vars
                        if v in model_ds.data_vars
                        and v not in excluded
                        and not v.startswith(("obs_", "model_"))
                    )
                    model_vars = list(obs_vars)

                if not obs_vars:
                    logger.warning(
                        f"Skipping {pair_key}: no variable mapping or common variables found"
                    )
                    return pair_key, None

                pairing_cfg = PairingConfig(
                    radius_of_influence=_cfg_get(cfg, "radius_of_influence", 1e6),
                    time_tolerance=_cfg_get(cfg, "time_tolerance", "1h"),
                    vertical_method=_cfg_get(cfg, "vertical_method", "nearest"),
                    horizontal_method=_cfg_get(cfg, "horizontal_method", "nearest"),
                    apply_averaging_kernel=_cfg_get(cfg, "apply_averaging_kernel", False),
                    require_overlap=_cfg_get(cfg, "require_overlap", True),
                )

                paired = engine.pair(
                    model_ds,
                    obs_ds,
                    obs_vars=obs_vars,
                    model_vars=model_vars,
                    config=pairing_cfg,
                )
                return pair_key, paired
            except Exception as e:
                logger.warning(f"Failed to pair {pair_key}: {e}")
                return pair_key, None

        result = self._executor.map(pair_single, pairs_to_process)

        # Convert results to dictionary
        paired_data = {}
        for pair_key, paired in result.results:
            if paired is not None:
                paired_data[pair_key] = paired

        return paired_data


def parallel_process_files(
    files: Sequence[str],
    processor: Callable[[str], Any],
    max_workers: int | None = None,
) -> ParallelResult[Any]:
    """Process multiple files in parallel.

    Parameters
    ----------
    files
        List of file paths.
    processor
        Function to process each file.
    max_workers
        Maximum parallel workers.

    Returns
    -------
    ParallelResult
        Combined results.

    Examples
    --------
    >>> def read_file(path):
    ...     return xr.open_dataset(path)
    >>> results = parallel_process_files(file_list, read_file, max_workers=4)
    """
    executor = ParallelExecutor(max_workers=max_workers)
    return executor.map(processor, files)
