"""Plot registry and factory functions for DAVINCI.

This module provides the plotting registry and convenience functions
for creating plotters by name.

Example usage:
    # Get a plotter by name
    plotter = get_plotter("timeseries")
    fig = plotter.plot(paired_data, "obs_o3", "model_o3")

    # List available plotters
    print(list_plotters())

    # Register a custom plotter
    @register_plotter("custom")
    class CustomPlotter(BasePlotter):
        ...
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from davinci_monet.core.registry import Registry, plotter_registry

if TYPE_CHECKING:
    from davinci_monet.plots.base import BasePlotter, PlotConfig

T = TypeVar("T")


def register_plotter(name: str, *, replace: bool = False) -> Callable[[type[T]], type[T]]:
    """Decorator to register a plotter class.

    Parameters
    ----------
    name
        Unique name for the plotter (e.g., 'timeseries', 'scatter').
    replace
        If True, allow replacing an existing registration.

    Returns
    -------
    Callable
        Decorator function.

    Examples
    --------
    >>> @register_plotter("my_plot")
    ... class MyPlotter(BasePlotter):
    ...     name = "my_plot"
    ...     def plot(self, paired_data, obs_var, model_var, **kwargs):
    ...         ...
    """
    return plotter_registry.register(name, replace=replace)


_warned_aliases: set[str] = set()


def register_alias(alias: str, target: str) -> None:
    """Register a deprecated plot-type ``alias`` that resolves to ``target``.

    Lets old ``type:`` strings (e.g. ``obs_timeseries``) keep working after a
    renderer is merged/renamed. Resolution emits a one-time ``LegacyConfigWarning``
    (see :func:`get_plotter_class`).
    """
    plotter_registry.register_alias(alias, target)


def get_plotter_class(name: str) -> type[BasePlotter]:
    """Get a plotter class by name (resolving deprecated aliases).

    Parameters
    ----------
    name
        Plotter name (or a registered deprecated alias).

    Returns
    -------
    type[BasePlotter]
        The plotter class.

    Raises
    ------
    ComponentNotFoundError
        If plotter is not registered.
    """
    if plotter_registry.is_alias(name) and name not in _warned_aliases:
        _warned_aliases.add(name)
        import warnings

        from davinci_monet.config.migration import LegacyConfigWarning

        target = plotter_registry.resolve(name)
        warnings.warn(
            f"Plot type '{name}' is deprecated; use '{target}'.",
            LegacyConfigWarning,
            stacklevel=2,
        )
    return plotter_registry.get(name)


def get_plotter(
    name: str,
    config: PlotConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> BasePlotter:
    """Get a configured plotter instance by name.

    Parameters
    ----------
    name
        Plotter name (e.g., 'timeseries', 'scatter', 'taylor').
    config
        Plot configuration. Can be PlotConfig or dict.
    **kwargs
        Additional arguments passed to plotter constructor.

    Returns
    -------
    BasePlotter
        Configured plotter instance.

    Examples
    --------
    >>> plotter = get_plotter("timeseries", config={"vmin": 0, "vmax": 100})
    >>> fig = plotter.plot(data, "obs_o3", "model_o3")
    """
    from davinci_monet.plots.base import PlotConfig

    plotter_cls = get_plotter_class(name)

    # Convert dict to PlotConfig if needed
    if isinstance(config, dict):
        config = PlotConfig.from_dict(config)

    return plotter_cls(config=config, **kwargs)


def list_plotters() -> list[str]:
    """List all registered plotter names.

    Returns
    -------
    list[str]
        Sorted list of plotter names.
    """
    return plotter_registry.list()


def has_plotter(name: str) -> bool:
    """Check if a plotter is registered.

    Parameters
    ----------
    name
        Plotter name to check.

    Returns
    -------
    bool
        True if plotter is registered.
    """
    return name in plotter_registry


# =============================================================================
# Plot Type Categories
# =============================================================================

# These are the standard plot type categories for reference
TEMPORAL_PLOTS = frozenset({"timeseries", "diurnal", "per_site_timeseries"})
STATISTICAL_PLOTS = frozenset({"taylor", "boxplot", "scatter"})
SPATIAL_PLOTS = frozenset({"spatial_bias", "spatial_overlay", "spatial_distribution"})
SPECIALIZED_PLOTS = frozenset({"curtain", "scorecard"})

ALL_PLOT_TYPES = TEMPORAL_PLOTS | STATISTICAL_PLOTS | SPATIAL_PLOTS | SPECIALIZED_PLOTS


def get_plot_category(name: str) -> str | None:
    """Get the category for a plot type.

    Parameters
    ----------
    name
        Plot type name.

    Returns
    -------
    str | None
        Category name, or None if not a standard type.
    """
    if name in TEMPORAL_PLOTS:
        return "temporal"
    if name in STATISTICAL_PLOTS:
        return "statistical"
    if name in SPATIAL_PLOTS:
        return "spatial"
    if name in SPECIALIZED_PLOTS:
        return "specialized"
    return None


# Re-export the registry for direct access
__all__ = [
    "plotter_registry",
    "register_plotter",
    "register_alias",
    "get_plotter",
    "get_plotter_class",
    "list_plotters",
    "has_plotter",
    "get_plot_category",
    "TEMPORAL_PLOTS",
    "STATISTICAL_PLOTS",
    "SPATIAL_PLOTS",
    "SPECIALIZED_PLOTS",
    "ALL_PLOT_TYPES",
]
