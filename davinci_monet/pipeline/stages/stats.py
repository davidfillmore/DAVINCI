"""Statistics stage.

Computes comparison metrics on paired data or descriptive statistics on
unpaired single-source data.
"""

from __future__ import annotations

from typing import Any

import xarray as xr

from davinci_monet.core.base import iter_paired_variable_pairs
from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    StageResult,
    StageStatus,
)
from davinci_monet.pipeline.stages.helpers import iter_single_source_datasets


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
