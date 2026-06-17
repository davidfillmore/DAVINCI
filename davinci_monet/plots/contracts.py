"""Canonical plot API contracts.

This module is the single source of truth for plot type arity and public
category metadata. Config validation, registry category helpers, and pipeline
dispatch must consume this module instead of re-declaring plot type sets.
"""

from __future__ import annotations

from enum import Enum

from davinci_monet.core.exceptions import PlottingError


class PlotArity(str, Enum):
    """Supported renderer input shapes."""

    SINGLE_SOURCE = "single_source"
    PAIRWISE = "pairwise"
    MULTI_SOURCE = "multi_source"


SINGLE_SOURCE_PLOTS = frozenset(
    {
        "spatial",
        "flight_track",
        "histogram",
        "lma_density",
        "vertical_profile",
        "eof_pattern",
        "eof_scree",
    }
)

PAIRWISE_PLOTS = frozenset(
    {
        "scatter",
        "spatial_bias",
        "spatial_overlay",
        "diurnal",
        "taylor",
        "boxplot",
        "curtain",
        "scorecard",
        "track_map_3d",
    }
)

MULTI_SOURCE_PLOTS = frozenset({"timeseries"})

TEMPORAL_PLOTS = frozenset({"timeseries", "diurnal"})
STATISTICAL_PLOTS = frozenset({"taylor", "boxplot", "scatter", "histogram", "eof_scree"})
SPATIAL_PLOTS = frozenset({"spatial", "spatial_bias", "spatial_overlay", "eof_pattern"})
SPECIALIZED_PLOTS = frozenset(
    {"curtain", "scorecard", "vertical_profile", "flight_track", "lma_density", "track_map_3d"}
)

ALL_PLOT_TYPES = TEMPORAL_PLOTS | STATISTICAL_PLOTS | SPATIAL_PLOTS | SPECIALIZED_PLOTS


def plot_arity(plot_type: str) -> PlotArity:
    """Return the canonical arity for a registered plot type."""
    if plot_type in SINGLE_SOURCE_PLOTS:
        return PlotArity.SINGLE_SOURCE
    if plot_type in PAIRWISE_PLOTS:
        return PlotArity.PAIRWISE
    if plot_type in MULTI_SOURCE_PLOTS:
        return PlotArity.MULTI_SOURCE
    raise PlottingError(f"Unknown plot type '{plot_type}'")


def validate_plot_shape(
    *,
    plot_name: str,
    plot_type: str,
    pairs: list[str],
    source: str | None,
    variable: str | None,
) -> list[str]:
    """Return config-shape validation errors for one plot spec."""
    arity = plot_arity(plot_type)
    has_pairs = bool(pairs)
    has_source = source is not None
    has_variable = variable is not None
    has_single = has_source or has_variable

    errors: list[str] = []
    if arity == PlotArity.SINGLE_SOURCE:
        if has_pairs:
            errors.append(
                f"plots.{plot_name}.pairs is invalid for single-source plot '{plot_type}'"
            )
        if not has_source:
            errors.append(
                f"plots.{plot_name}.source is required for single-source plot '{plot_type}'"
            )
        if not has_variable:
            errors.append(
                f"plots.{plot_name}.variable is required for single-source plot '{plot_type}'"
            )
    elif arity == PlotArity.PAIRWISE:
        if not has_pairs:
            errors.append(f"plots.{plot_name}.pairs is required for pairwise plot '{plot_type}'")
        if has_source:
            errors.append(f"plots.{plot_name}.source is invalid for pairwise plot '{plot_type}'")
        if has_variable:
            errors.append(f"plots.{plot_name}.variable is invalid for pairwise plot '{plot_type}'")
    elif arity == PlotArity.MULTI_SOURCE:
        if has_pairs and has_single:
            errors.append(
                f"plots.{plot_name} must use either pairs or source/variable for plot "
                f"'{plot_type}', not both"
            )
        if not has_pairs and not (has_source and has_variable):
            errors.append(
                f"plots.{plot_name} requires pairs or source+variable for plot '{plot_type}'"
            )
        if has_single and not (has_source and has_variable):
            errors.append(
                f"plots.{plot_name} source and variable must be provided together for plot "
                f"'{plot_type}'"
            )
    return errors


__all__ = [
    "PlotArity",
    "SINGLE_SOURCE_PLOTS",
    "PAIRWISE_PLOTS",
    "MULTI_SOURCE_PLOTS",
    "TEMPORAL_PLOTS",
    "STATISTICAL_PLOTS",
    "SPATIAL_PLOTS",
    "SPECIALIZED_PLOTS",
    "ALL_PLOT_TYPES",
    "plot_arity",
    "validate_plot_shape",
]
