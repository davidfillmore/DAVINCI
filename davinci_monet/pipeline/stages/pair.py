"""Pairing stage.

Builds and executes source-pair jobs from explicit ``pairs:`` config.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import xarray as xr

from davinci_monet.core.base import iter_paired_variable_pairs
from davinci_monet.core.exceptions import PipelineError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    SourcePairJob,
    StageResult,
    StageStatus,
)
from davinci_monet.pipeline.stages.helpers import (
    _format_duration,
    _format_size,
)


class PairingStage(BaseStage):
    """Stage for pairing datasets by geometry."""

    def __init__(self) -> None:
        super().__init__("pairing")

    def validate(self, context: PipelineContext) -> bool:
        """Validate that sources are loaded and pairable."""
        pairs_config = context.config.get("pairs")
        if isinstance(pairs_config, dict) and any(
            isinstance(pair, dict) and bool(pair.get("sources")) for pair in pairs_config.values()
        ):
            return True
        if pairs_config and len(context.sources) >= 2:
            return True
        return False

    def execute(self, context: PipelineContext) -> StageResult:
        """Pair source data through the pairing engine.

        Jobs come from the explicit ``pairs:`` block and run through
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
        """Build pair jobs from the explicit ``pairs:`` block.

        Each pair must be in unified form (``sources: [a, b]``).
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
                        f"Pair '{pair_name}' names unknown source(s): " f"{', '.join(missing)}"
                    )
                    continue
                a_obj = context.sources[a_label]
                b_obj = context.sources[b_label]
                a_geom = self._source_geometry(a_obj)
                b_geom = self._source_geometry(b_obj)
                explicit_geometry_label = raw_pair.get("geometry")
                explicit_pos = None
                if explicit_geometry_label is not None:
                    if explicit_geometry_label == a_label:
                        explicit_pos = "a"
                    elif explicit_geometry_label == b_label:
                        explicit_pos = "b"
                    else:
                        errors.append(
                            f"Pair '{pair_name}' names unknown source "
                            f"'{explicit_geometry_label}'. Expected one of {srcs}."
                        )
                        continue
                geometry_choice, dataset_choice = resolve_pair_direction(
                    a_geom, b_geom, explicit_geometry=explicit_pos
                )
                if explicit_pos == "b" or (
                    explicit_pos is None and geometry_choice is b_geom and dataset_choice is a_geom
                ):
                    geometry_label, geometry_obj = b_label, b_obj
                    dataset_label, dataset_obj = a_label, a_obj
                else:
                    geometry_label, geometry_obj = a_label, a_obj
                    dataset_label, dataset_obj = b_label, b_obj
                vmap = raw_pair.get("variables") or {}
                geometry_var = vmap.get(geometry_label)
                dataset_var = vmap.get(dataset_label)
                if not geometry_var or not dataset_var:
                    missing = [
                        label
                        for label, value in (
                            (geometry_label, geometry_var),
                            (dataset_label, dataset_var),
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
                        geometry_label=geometry_label,
                        geometry_obj=geometry_obj,
                        dataset_label=dataset_label,
                        dataset_obj=dataset_obj,
                        geometry_var=str(geometry_var),
                        dataset_var=str(dataset_var),
                        radius_of_influence=self._pair_radius(raw_pair, dataset_obj),
                        strategy_options=self._strategy_options(
                            pairing_config_dict=context.config.get("pairing", {}),
                            pair_spec=raw_pair,
                        ),
                    )
                )
            else:
                errors.append(f"Pair '{pair_name}' must declare 'sources: [a, b]' and 'variables'.")
        return jobs, errors

    @staticmethod
    def _lookup_source(context: PipelineContext, label: str) -> Any:
        """Resolve a source by label from the unified ``sources`` store."""
        return context.sources.get(label)

    @staticmethod
    def _source_dataset(obj: Any) -> xr.Dataset | None:
        data = obj.data if hasattr(obj, "data") else obj
        return data if isinstance(data, xr.Dataset) else None

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
    def _pair_radius(pair_spec: dict[str, Any], dataset_obj: Any) -> float:
        if pair_spec.get("radius_of_influence") is not None:
            return float(pair_spec["radius_of_influence"])
        cfg = getattr(dataset_obj, "config", None)
        if isinstance(cfg, dict) and cfg.get("radius_of_influence") is not None:
            return float(cfg["radius_of_influence"])
        return float(getattr(dataset_obj, "radius_of_influence", 12000.0))

    @staticmethod
    def _strategy_options(
        *,
        pairing_config_dict: dict[str, Any],
        pair_spec: dict[str, Any],
    ) -> dict[str, Any]:
        """Return strategy-specific options from global and pair config."""
        control_keys = {
            "sources",
            "geometry",
            "variables",
            "radius_of_influence",
            "time_tolerance",
            "time_method",
            "max_pair_workers",
            "dask_pair_workers",
        }
        options = {k: v for k, v in pairing_config_dict.items() if k not in control_keys}
        options.update({k: v for k, v in pair_spec.items() if k not in control_keys})
        return {k: v for k, v in options.items() if v is not None}

    @staticmethod
    def _is_dask_backed(ds: xr.Dataset | None) -> bool:
        """Return ``True`` if any data variable of ``ds`` is dask-chunked.

        A chunked (lazy) dataset triggers the per-strategy
        ``dask.config.set(scheduler="threads").compute()`` — i.e. its own thread
        pool — during extraction. Eager (numpy-backed, unchunked) datasets hit a
        cheap no-op ``.compute()`` that spawns no dask threads. This predicate is
        what the executor uses to keep dask-backed pairs off the overlapping fast
        path (see :meth:`_execute_source_pair_jobs`).
        """
        if ds is None:
            return False
        return any(
            getattr(getattr(var, "data", None), "chunks", None) is not None
            for var in ds.data_vars.values()
        )

    @staticmethod
    def _pair_worker_counts(
        pairing_config_dict: dict[str, Any], n_eager: int, cpu: int
    ) -> tuple[int, int]:
        """Compute bounded ``(eager_workers, dask_workers)`` thread counts.

        Eager: ``max_pair_workers`` if set, else ``min(n_eager, max(1, cpu // 2))``;
        clamped to ``[1, 8]``. Dask: ``dask_pair_workers`` (default ``1`` →
        serial), clamped to ``>= 1``. Pure/static so the clamping is unit-testable
        without running the executor.
        """
        eager_workers = pairing_config_dict.get("max_pair_workers")
        if eager_workers is None:
            eager_workers = min(n_eager, max(1, cpu // 2))
        eager_workers = max(1, min(8, int(eager_workers)))
        dask_workers = max(1, int(pairing_config_dict.get("dask_pair_workers", 1)))
        return eager_workers, dask_workers

    def _run_pair_job(
        self,
        context: PipelineContext,
        job: SourcePairJob,
        pairing_config_dict: dict[str, Any],
        debug: bool,
    ) -> tuple[SourcePairJob, Any, str | None]:
        """Pair a single job. Thread-safe worker: returns, never mutates context.

        Builds its own :class:`PairingConfig` and a local :class:`PairingEngine`
        (no shared engine across threads), runs ``engine.pair_sources``, and
        returns ``(job, paired_obj, None)`` on success or ``(job, None, err)`` on
        failure. It does **not** touch
        ``context.paired``/``context.metadata`` and does **not** call
        ``context.log_progress`` — all shared-state mutation and progress logging
        stays on the main thread.
        """
        from davinci_monet.pairing import PairingConfig, PairingEngine

        geometry_ds = self._source_dataset(job.geometry_obj)
        dataset_ds = self._source_dataset(job.dataset_obj)
        if geometry_ds is None or dataset_ds is None:
            return job, None, f"{job.pair_key}: geometry or dataset data is None"

        try:
            pairing_cfg = PairingConfig(
                radius_of_influence=job.radius_of_influence,
                time_tolerance=pairing_config_dict.get("time_tolerance", "1h"),
                time_method=pairing_config_dict.get("time_method", "nearest"),
            )
            engine = PairingEngine()
            paired_obj = engine.pair_sources(
                geometry_data=geometry_ds,
                dataset_data=dataset_ds,
                geometry_vars=[job.geometry_var],
                dataset_vars=[job.dataset_var],
                output_geometry=self._source_geometry(job.geometry_obj),
                dataset_geometry=self._source_geometry(job.dataset_obj),
                config=pairing_cfg,
                geometry_label=job.geometry_label,
                dataset_label=job.dataset_label,
                **job.strategy_options,
            )
            return job, paired_obj, None
        except Exception as e:
            return job, None, f"{job.pair_key}: {e}"

    def _execute_source_pair_jobs(
        self,
        context: PipelineContext,
        jobs: list[SourcePairJob],
        pairing_config_dict: dict[str, Any],
        start: float,
    ) -> StageResult:
        """Execute source-pair jobs through the pairing engine.

        Cross-pair concurrency is HDF5-safe and bounded. Jobs are partitioned by
        :meth:`_is_dask_backed`:

        * **eager jobs** (neither side dask-chunked) do a no-op ``.compute()`` and
          spawn no dask threads, so they overlap freely on a
          :class:`~concurrent.futures.ThreadPoolExecutor`.
        * **dask-backed jobs** (either side chunked) each open their own
          per-strategy dask thread pool during extraction. Overlapping them would
          nest thread pools → GIL/contention and the documented HDF5 segfaults, so
          they run **serially by default**.

        Thread-safety: worker threads (``_run_pair_job``) only read source data and
        mutate their own paired dataset attrs; they return results. Only the main
        thread writes ``context.paired``/``context.metadata`` and emits
        ``context.log_progress`` (as each result resolves), so the progress stream
        and shared state never race.

        Config knobs (under the ``pairing:`` block):

        * ``max_pair_workers`` (int, optional) — thread count for eager jobs.
          Default ``min(len(eager_jobs), max(1, cpu // 2))``, clamped to
          ``[1, 8]``.
        * ``dask_pair_workers`` (int, default ``1``) — thread count for
          dask-backed jobs. Default ``1`` keeps them serial (nested-pool safe);
          set higher only when each pair's own dask pool is constrained. Clamped
          to ``>= 1``.
        """
        import time

        debug = context.config.get("analysis", {}).get("debug", False)
        paired_count = 0
        execution_errors: list[str] = []
        context.log_progress(f"    parallel_start: {len(jobs)}")

        # Partition by dask-backing: only fully-eager jobs are safe to overlap.
        eager_jobs: list[SourcePairJob] = []
        dask_jobs: list[SourcePairJob] = []
        for job in jobs:
            ref_ds = self._source_dataset(job.geometry_obj)
            comp_ds = self._source_dataset(job.dataset_obj)
            if self._is_dask_backed(ref_ds) or self._is_dask_backed(comp_ds):
                dask_jobs.append(job)
            else:
                eager_jobs.append(job)

        # Bounded worker counts.
        cpu = os.cpu_count() or 1
        eager_workers, dask_workers = self._pair_worker_counts(
            pairing_config_dict, len(eager_jobs), cpu
        )

        def _record(
            job: SourcePairJob, paired_obj: Any, error: str | None, pair_start: float
        ) -> None:
            """Main-thread sink for a resolved job: write state, log, count."""
            nonlocal paired_count
            if error is not None or paired_obj is None:
                execution_errors.append(
                    error or f"{job.pair_key}: geometry or dataset data is None"
                )
                context.log_progress(f"    parallel_completed: {job.pair_key} - FAILED")
                return
            paired_data = paired_obj.data
            context.paired[job.pair_key] = paired_obj
            paired_count += 1
            n_vars = len(iter_paired_variable_pairs(paired_data))
            n_points = paired_data.sizes.get("time", paired_data.sizes.get("x", 0))
            timing_str = f" [{_format_duration(time.time() - pair_start)}]" if debug else ""
            context.log_progress(
                f"    parallel_completed: {job.pair_key} - "
                f"{n_vars} vars, {_format_size(n_points)} points{timing_str}"
            )

        # Eager jobs: overlap on a bounded thread pool.
        if eager_jobs:
            if eager_workers <= 1 or len(eager_jobs) <= 1:
                for job in eager_jobs:
                    pair_start = time.time()
                    context.log_progress(f"    parallel_started: {job.pair_key}")
                    j, paired_obj, error = self._run_pair_job(
                        context, job, pairing_config_dict, debug
                    )
                    _record(j, paired_obj, error, pair_start)
            else:
                with ThreadPoolExecutor(max_workers=eager_workers) as pool:
                    futures = {}
                    for job in eager_jobs:
                        context.log_progress(f"    parallel_started: {job.pair_key}")
                        futures[
                            pool.submit(
                                self._run_pair_job, context, job, pairing_config_dict, debug
                            )
                        ] = time.time()
                    for future in as_completed(futures):
                        pair_start = futures[future]
                        j, paired_obj, error = future.result()
                        _record(j, paired_obj, error, pair_start)

        # Dask-backed jobs: serial by default (nested dask pools are unsafe to
        # overlap); a bounded pool only if ``dask_pair_workers`` is raised.
        if dask_jobs:
            if dask_workers <= 1 or len(dask_jobs) <= 1:
                for job in dask_jobs:
                    pair_start = time.time()
                    context.log_progress(f"    parallel_started: {job.pair_key}")
                    j, paired_obj, error = self._run_pair_job(
                        context, job, pairing_config_dict, debug
                    )
                    _record(j, paired_obj, error, pair_start)
            else:
                with ThreadPoolExecutor(max_workers=dask_workers) as pool:
                    futures = {}
                    for job in dask_jobs:
                        context.log_progress(f"    parallel_started: {job.pair_key}")
                        futures[
                            pool.submit(
                                self._run_pair_job, context, job, pairing_config_dict, debug
                            )
                        ] = time.time()
                    for future in as_completed(futures):
                        pair_start = futures[future]
                        j, paired_obj, error = future.result()
                        _record(j, paired_obj, error, pair_start)

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
