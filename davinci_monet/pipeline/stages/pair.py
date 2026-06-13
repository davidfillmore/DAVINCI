"""Pairing stage.

Builds and executes source-pair jobs (explicit ``pairs:`` and implicit
auto-pairing) through the role-neutral pairing engine.
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
    tag_paired_roles,
)


class PairingStage(BaseStage):
    """Stage for pairing model and observation data.

    Uses the pairing engine to match model output with observations.
    """

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
        # Implicit auto-pairing needs at least one model-role and one obs-role
        # source available in the unified view.
        roles = {self._source_role(obj) for _, obj in context.sources.items()}
        return "model" in roles and "obs" in roles

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
                        strategy_options=self._strategy_options(
                            pairing_config_dict=context.config.get("pairing", {}),
                            pair_spec=raw_pair,
                        ),
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

        ``mapping:`` is read from each source's loaded config
        (``SourceData.config``).
        """
        jobs: list[SourcePairJob] = []
        pair_index = 0

        for model_label, model_obj in self._model_role_sources(context):
            mapping = self._source_mapping(model_obj)
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
                            strategy_options=self._strategy_options(
                                pairing_config_dict=context.config.get("pairing", {}),
                                pair_spec={},
                            ),
                        )
                    )
        return jobs

    @staticmethod
    def _model_role_sources(context: PipelineContext) -> list[tuple[str, Any]]:
        """Return (label, obj) for model-role sources from ``context.sources``."""
        return [
            (str(label), obj)
            for label, obj in context.sources.items()
            if PairingStage._source_role(obj) == "model"
        ]

    @staticmethod
    def _lookup_source(context: PipelineContext, label: str) -> Any:
        """Resolve a source by label from the unified ``sources`` store."""
        return context.sources.get(label)

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

    @staticmethod
    def _strategy_options(
        *,
        pairing_config_dict: dict[str, Any],
        pair_spec: dict[str, Any],
    ) -> dict[str, Any]:
        """Return strategy-specific options from global and pair config."""
        control_keys = {
            "sources",
            "reference",
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

        Builds its own :class:`PairingConfig` and a *local* :class:`PairingEngine`
        (no shared engine across threads), runs ``engine.pair_sources`` and
        :func:`tag_paired_roles` (which mutates only this job's own paired dataset
        attrs), and returns ``(job, paired_obj, None)`` on success or
        ``(job, None, "<pair_key>: <err>")`` on failure. It does **not** touch
        ``context.paired``/``context.metadata`` and does **not** call
        ``context.log_progress`` — all shared-state mutation and progress logging
        stays on the main thread.
        """
        from davinci_monet.pairing import PairingConfig, PairingEngine

        ref_ds = self._source_dataset(job.reference_obj)
        comp_ds = self._source_dataset(job.comparand_obj)
        if ref_ds is None or comp_ds is None:
            return job, None, f"{job.pair_key}: reference or comparand data is None"

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
                **job.strategy_options,
            )
            tag_paired_roles(
                paired_obj.data,
                reference_label=job.reference_label,
                comparand_label=job.comparand_label,
                reference_role=job.reference_role,
                comparand_role=job.comparand_role,
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
        """Execute source-pair jobs through the role-neutral engine.

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
            ref_ds = self._source_dataset(job.reference_obj)
            comp_ds = self._source_dataset(job.comparand_obj)
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
                    error or f"{job.pair_key}: reference or comparand data is None"
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
