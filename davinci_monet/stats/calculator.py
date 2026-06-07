"""Statistics calculator for paired source data.

This module provides the main interface for computing statistics
on paired reference/comparand datasets, with support for grouping and multiple
metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

from davinci_monet.stats.metrics import STANDARD_METRICS, get_metric, list_metrics

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class StatisticsConfig:
    """Configuration for statistics calculation.

    Attributes
    ----------
    metrics : list[str]
        List of metric names to calculate.
    round_precision : int
        Number of decimal places for rounding results.
    include_counts : bool
        If True, include sample count in output.
    remove_nan : bool
        If True, remove NaN values before calculation.
    min_samples : int
        Minimum number of samples required for calculation.
    """

    metrics: list[str] = field(default_factory=lambda: list(STANDARD_METRICS))
    round_precision: int = 3
    include_counts: bool = True
    remove_nan: bool = True
    min_samples: int = 3

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> StatisticsConfig:
        """Create config from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)


# =============================================================================
# Statistics Calculator
# =============================================================================


class StatisticsCalculator:
    """Calculator for computing statistics on paired data.

    Supports computing multiple metrics, grouping by dimensions,
    and outputting results as DataFrames.

    Parameters
    ----------
    config
        Statistics configuration.

    Examples
    --------
    >>> calc = StatisticsCalculator()
    >>> stats = calc.compute(paired_data, "obs_o3", "model_o3")
    >>> print(stats)
           N     MO     MP      MB    RMSE      R
    0  1000  45.2  47.1    1.9    5.3  0.85

    >>> # Group by site
    >>> stats = calc.compute(
    ...     paired_data, "obs_o3", "model_o3",
    ...     groupby="site"
    ... )
    """

    def __init__(self, config: StatisticsConfig | None = None) -> None:
        self.config = config or StatisticsConfig()

    def compute(
        self,
        paired_data: xr.Dataset,
        obs_var: str | None = None,
        model_var: str | None = None,
        metrics: Sequence[str] | None = None,
        groupby: str | Sequence[str] | None = None,
        *,
        reference_var: str | None = None,
        comparand_var: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute statistics for paired data.

        Parameters
        ----------
        paired_data
            Paired dataset with reference and comparand variables.
        obs_var
            Compatibility name for reference variable.
        model_var
            Compatibility name for comparand variable.
        metrics
            List of metric names. If None, uses config.metrics.
        groupby
            Optional dimension(s) to group by.
        reference_var
            Name of reference variable.
        comparand_var
            Name of comparand variable.
        **kwargs
            Additional options passed to individual metrics.

        Returns
        -------
        pd.DataFrame
            Statistics table with metrics as columns.
        """
        metrics = list(metrics) if metrics is not None else self.config.metrics
        reference_name = reference_var or obs_var
        comparand_name = comparand_var or model_var
        if reference_name is None or comparand_name is None:
            raise ValueError("Both reference_var and comparand_var are required")

        # Get data arrays
        obs_data = paired_data[reference_name]
        model_data = paired_data[comparand_name]

        if groupby is not None:
            return self._compute_grouped(obs_data, model_data, metrics, groupby, **kwargs)
        else:
            return self._compute_overall(obs_data, model_data, metrics, **kwargs)

    def _compute_overall(
        self,
        obs_data: xr.DataArray,
        model_data: xr.DataArray,
        metrics: list[str],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute overall statistics (no grouping).

        Parameters
        ----------
        obs_data, model_data
            Data arrays.
        metrics
            List of metric names.
        **kwargs
            Additional options.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with statistics.
        """
        obs = obs_data.values.flatten()
        mod = model_data.values.flatten()

        results = self._compute_metrics(obs, mod, metrics, **kwargs)
        df = pd.DataFrame([results])

        if self.config.round_precision is not None:
            df = df.round(self.config.round_precision)

        return df

    def _compute_grouped(
        self,
        obs_data: xr.DataArray,
        model_data: xr.DataArray,
        metrics: list[str],
        groupby: str | Sequence[str],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute statistics grouped by dimension(s).

        Parameters
        ----------
        obs_data, model_data
            Data arrays.
        metrics
            List of metric names.
        groupby
            Dimension(s) to group by.
        **kwargs
            Additional options.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per group.
        """
        if isinstance(groupby, str):
            groupby = [groupby]

        # Handle special groupby patterns (e.g., "time.month")
        parsed_groupby = []
        for g in groupby:
            if "." in g:
                dim, accessor = g.split(".", 1)
                # Create grouping coordinate
                if dim == "time":
                    if accessor == "month":
                        group_coord = obs_data.time.dt.month
                    elif accessor == "hour":
                        group_coord = obs_data.time.dt.hour
                    elif accessor == "dayofweek":
                        group_coord = obs_data.time.dt.dayofweek
                    elif accessor == "season":
                        group_coord = obs_data.time.dt.season
                    else:
                        raise ValueError(f"Unknown time accessor: {accessor}")
                    parsed_groupby.append((g, group_coord))
                else:
                    raise ValueError(f"Unknown groupby pattern: {g}")
            else:
                parsed_groupby.append((g, g))

        # Simple case: single dimension groupby
        if len(parsed_groupby) == 1:
            name, coord = parsed_groupby[0]
            return self._compute_single_groupby(
                obs_data, model_data, metrics, name, coord, **kwargs
            )

        # Multi-dimensional groupby
        return self._compute_multi_groupby(obs_data, model_data, metrics, parsed_groupby, **kwargs)

    def _compute_single_groupby(
        self,
        obs_data: xr.DataArray,
        model_data: xr.DataArray,
        metrics: list[str],
        name: str,
        coord: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute statistics for single-dimension groupby.

        Parameters
        ----------
        obs_data, model_data
            Data arrays.
        metrics
            List of metric names.
        name
            Groupby dimension name.
        coord
            Grouping coordinate.
        **kwargs
            Additional options.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per group.
        """
        import xarray as xr

        # Group the data
        if isinstance(coord, str):
            obs_grouped = obs_data.groupby(coord, squeeze=False)
            model_grouped = model_data.groupby(coord, squeeze=False)
        else:
            obs_grouped = obs_data.groupby(coord, squeeze=False)
            model_grouped = model_data.groupby(coord, squeeze=False)

        results = []
        for (obs_key, obs_group), (_, model_group) in zip(obs_grouped, model_grouped):
            obs = obs_group.values.flatten()
            mod = model_group.values.flatten()

            row = {name: obs_key}
            row.update(self._compute_metrics(obs, mod, metrics, **kwargs))
            results.append(row)

        df = pd.DataFrame(results)
        if self.config.round_precision is not None:
            # Only round numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].round(self.config.round_precision)

        return df

    def _compute_multi_groupby(
        self,
        obs_data: xr.DataArray,
        model_data: xr.DataArray,
        metrics: list[str],
        parsed_groupby: list[tuple[str, Any]],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute statistics for multi-dimension groupby.

        Parameters
        ----------
        obs_data, model_data
            Data arrays.
        metrics
            List of metric names.
        parsed_groupby
            List of (name, coord) tuples.
        **kwargs
            Additional options.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per group combination.
        """
        # For multi-dimensional groupby, we need a different approach
        # Stack all groupby dimensions and iterate

        # This is a simplified implementation - for now, convert to pandas
        # and use pandas groupby
        obs_df = obs_data.to_dataframe(name="obs").reset_index()
        model_df = model_data.to_dataframe(name="mod").reset_index()

        # Merge on common indices
        common_cols = list(set(obs_df.columns) & set(model_df.columns) - {"obs", "mod"})
        df = pd.merge(obs_df, model_df, on=common_cols)

        # Add groupby columns
        group_cols = []
        for name, coord in parsed_groupby:
            if "." in name:
                dim, accessor = name.split(".", 1)
                if dim == "time" and "time" in df.columns:
                    if accessor == "month":
                        df[name] = df["time"].dt.month
                    elif accessor == "hour":
                        df[name] = df["time"].dt.hour
                    elif accessor == "dayofweek":
                        df[name] = df["time"].dt.dayofweek
                    elif accessor == "season":
                        df[name] = df["time"].dt.month.map(
                            {
                                12: "DJF",
                                1: "DJF",
                                2: "DJF",
                                3: "MAM",
                                4: "MAM",
                                5: "MAM",
                                6: "JJA",
                                7: "JJA",
                                8: "JJA",
                                9: "SON",
                                10: "SON",
                                11: "SON",
                            }
                        )
            group_cols.append(name)

        # Group and compute
        results = []
        for group_keys, group_df in df.groupby(group_cols):
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)

            obs = group_df["obs"].values
            mod = group_df["mod"].values

            row = dict(zip(group_cols, group_keys))
            row.update(self._compute_metrics(obs, mod, metrics, **kwargs))
            results.append(row)

        result_df = pd.DataFrame(results)
        if self.config.round_precision is not None:
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns
            result_df[numeric_cols] = result_df[numeric_cols].round(self.config.round_precision)

        return result_df

    def _compute_metrics(
        self,
        obs: np.ndarray,
        mod: np.ndarray,
        metrics: list[str],
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute all requested metrics.

        Parameters
        ----------
        obs, mod
            Arrays of observation and model values.
        metrics
            List of metric names.
        **kwargs
            Additional options.

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to values.
        """
        # Remove NaN if configured
        if self.config.remove_nan:
            mask = np.isfinite(obs) & np.isfinite(mod)
            obs = obs[mask]
            mod = mod[mask]

        results = {}

        # Check minimum samples
        if len(obs) < self.config.min_samples:
            for metric_name in metrics:
                results[metric_name] = np.nan
            return results

        # Compute each metric
        for metric_name in metrics:
            try:
                metric = get_metric(metric_name)
                results[metric_name] = metric.compute(obs, mod, **kwargs)
            except Exception as exc:
                logger.warning(
                    "Metric '%s' raised an exception and will be set to NaN: %s",
                    metric_name,
                    exc,
                )
                results[metric_name] = np.nan

        return results

    def compute_summary(
        self,
        paired_data: xr.Dataset,
        obs_var: str | None = None,
        model_var: str | None = None,
        metrics: Sequence[str] | None = None,
        *,
        reference_var: str | None = None,
        comparand_var: str | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute summary statistics as a dictionary.

        Parameters
        ----------
        paired_data
            Paired dataset.
        obs_var
            Compatibility name for reference variable.
        model_var
            Compatibility name for comparand variable.
        metrics
            List of metric names.
        reference_var
            Reference variable name.
        comparand_var
            Comparand variable name.
        **kwargs
            Additional options.

        Returns
        -------
        dict[str, float]
            Dictionary of metric values.
        """
        df = self.compute(
            paired_data,
            obs_var,
            model_var,
            metrics=metrics,
            reference_var=reference_var,
            comparand_var=comparand_var,
            **kwargs,
        )
        return df.iloc[0].to_dict()


# =============================================================================
# Convenience Functions
# =============================================================================


def calculate_statistics(
    paired_data: xr.Dataset,
    obs_var: str | None = None,
    model_var: str | None = None,
    metrics: Sequence[str] | None = None,
    groupby: str | Sequence[str] | None = None,
    config: StatisticsConfig | dict[str, Any] | None = None,
    *,
    reference_var: str | None = None,
    comparand_var: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Calculate statistics for paired data.

    This is the main entry point for statistics calculation.

    Parameters
    ----------
    paired_data
        Paired dataset with reference and comparand variables.
    obs_var
        Compatibility name for reference variable.
    model_var
        Compatibility name for comparand variable.
    metrics
        List of metric names. If None, uses standard set.
    groupby
        Optional dimension(s) to group by.
    config
        Statistics configuration.
    reference_var
        Name of reference variable.
    comparand_var
        Name of comparand variable.
    **kwargs
        Additional options.

    Returns
    -------
    pd.DataFrame
        Statistics table.

    Examples
    --------
    >>> stats = calculate_statistics(
    ...     paired_data, "obs_o3", "model_o3",
    ...     metrics=["MB", "RMSE", "R"],
    ... )

    >>> # Group by site and month
    >>> stats = calculate_statistics(
    ...     paired_data, "obs_o3", "model_o3",
    ...     groupby=["site", "time.month"],
    ... )
    """
    if isinstance(config, dict):
        config = StatisticsConfig.from_dict(config)

    calc = StatisticsCalculator(config=config)
    return calc.compute(
        paired_data,
        obs_var,
        model_var,
        metrics=metrics,
        groupby=groupby,
        reference_var=reference_var,
        comparand_var=comparand_var,
        **kwargs,
    )


def quick_stats(
    obs: np.ndarray,
    mod: np.ndarray,
    metrics: Sequence[str] | None = None,
) -> dict[str, float]:
    """Quick statistics calculation from arrays.

    Parameters
    ----------
    obs
        Observation array.
    mod
        Model array.
    metrics
        List of metric names. If None, uses standard set.

    Returns
    -------
    dict[str, float]
        Dictionary of metric values.

    Examples
    --------
    >>> stats = quick_stats(obs_array, model_array)
    >>> print(f"RMSE: {stats['RMSE']:.2f}")
    """
    metrics = list(metrics) if metrics is not None else STANDARD_METRICS

    obs = np.asarray(obs).flatten()
    mod = np.asarray(mod).flatten()

    # Remove NaN
    mask = np.isfinite(obs) & np.isfinite(mod)
    obs = obs[mask]
    mod = mod[mask]

    results = {}
    for metric_name in metrics:
        try:
            metric = get_metric(metric_name)
            results[metric_name] = metric.compute(obs, mod)
        except Exception as exc:
            logger.warning(
                "Metric '%s' raised an exception and will be set to NaN: %s",
                metric_name,
                exc,
            )
            results[metric_name] = np.nan

    return results
