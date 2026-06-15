"""Plot contract metadata is the single source of truth for plotting API shape."""

from __future__ import annotations

import pytest

import davinci_monet.plots as plots
from davinci_monet.plots.contracts import PlotArity, plot_arity, validate_plot_shape
from davinci_monet.plots.registry import (
    ALL_PLOT_TYPES,
    SPATIAL_PLOTS,
    SPECIALIZED_PLOTS,
    STATISTICAL_PLOTS,
    TEMPORAL_PLOTS,
    get_plot_category,
)


def test_contracts_cover_every_registered_plotter() -> None:
    assert set(plots.list_plotters()) == set(ALL_PLOT_TYPES)


def test_plot_arities_are_declared() -> None:
    assert plot_arity("spatial") == PlotArity.SINGLE_SOURCE
    assert plot_arity("scatter") == PlotArity.PAIRWISE
    assert plot_arity("timeseries") == PlotArity.MULTI_SOURCE


def test_plot_shape_validator_reports_wrong_arity() -> None:
    assert validate_plot_shape(
        plot_name="bad_spatial",
        plot_type="spatial",
        pairs=["pair_a"],
        source="obs",
        variable="O3",
    ) == ["plots.bad_spatial.pairs is invalid for single-source plot 'spatial'"]


@pytest.mark.parametrize(
    ("plot_type", "category"),
    [
        ("timeseries", "temporal"),
        ("scatter", "statistical"),
        ("spatial", "spatial"),
        ("flight_track", "specialized"),
    ],
)
def test_plot_categories_remain_public(plot_type: str, category: str) -> None:
    assert get_plot_category(plot_type) == category


def test_category_sets_are_disjoint() -> None:
    groups = [TEMPORAL_PLOTS, STATISTICAL_PLOTS, SPATIAL_PLOTS, SPECIALIZED_PLOTS]
    flattened = [item for group in groups for item in group]
    assert len(flattened) == len(set(flattened))
