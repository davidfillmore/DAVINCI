"""Series and color helper functions for DAVINCI plots.

Provides:
- build_series: resolve var-args into an ordered list of PlotSeries
- series_colors: per-series colors under the unified, count-aware rule
- get_axis_color: plot color for a paired series by axis
- source_label: source label for a single-source dataset
- get_series_label: legend label for a paired series
- resolve_source_variable: resolve a variable name by source label
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from davinci_monet.core.base import (
    PlotSeries,
    paired_variable_axis,
)
from davinci_monet.plots.labels import canonical_variable_name, get_variable_label
from davinci_monet.plots.style import DATASET_A_COLOR, DATASET_B_COLOR, NCAR_PALETTE, NCAR_PRIMARY

if TYPE_CHECKING:
    import xarray as xr


def build_series(dataset: xr.Dataset, *var_args: Any) -> list[PlotSeries]:
    """Resolve facade var-args into an ordered list of :class:`PlotSeries`.

    Accepts the three call shapes the unified facade supports:

    - ``build_series(ds, x_var, y_var)`` → 2 series
    - ``build_series(ds, variable)`` → 1 series
    - ``build_series(ds, [v1, ..., vN])`` → N series

    A trailing positional ``matplotlib`` Axes (``plot(ds, var, ax)``) is
    ignored for series building. ``axis``/``source_label``/``canonical``
    are read from the dataset's attrs, with the ``x_``/``y_`` prefix
    fallback.
    """
    import matplotlib.axes

    args = list(var_args)
    if args and isinstance(args[-1], matplotlib.axes.Axes):
        args = args[:-1]
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        names = [str(n) for n in args[0]]
    else:
        names = [a for a in args if isinstance(a, str)]

    series: list[PlotSeries] = []
    for i, name in enumerate(names):
        # Prefer the per-variable source_label (paired/tagged data); fall back to
        # the dataset-level label that single-source geometry datasets carry.
        source_label = (
            dataset[name].attrs.get("source_label") if name in dataset.data_vars else None
        ) or dataset.attrs.get("source_label")
        series.append(
            PlotSeries(
                dataset=dataset,
                var_name=name,
                canonical=canonical_variable_name(dataset, name),
                axis=paired_variable_axis(dataset, name),
                source_label=str(source_label) if source_label else None,
                index=i,
            )
        )
    return series


def series_colors(
    series: list[PlotSeries],
    *,
    x_color: str | None = None,
    y_color: str | None = None,
) -> list[str]:
    """Per-series colors under the unified, count-aware rule.

    - **1 series** → ``NCAR_PRIMARY``.
    - **2 series** → geometry in ``x_color`` (gray) and dataset in
      ``y_color`` (blue), preserving today's comparison contrast.
    - **N > 2 series** → distinct ``NCAR_PALETTE`` colors cycled by ``index``.

    ``x_color``/``y_color`` let a caller pass the active ``StyleConfig``
    colors; they default to the module ``DATASET_A_COLOR``/``DATASET_B_COLOR``.
    """
    n = len(series)
    if n == 1:
        return [NCAR_PRIMARY]
    if n == 2:
        out: list[str] = []
        for s in series:
            is_dataset = s.axis == "y" or (s.axis is None and s.index == 1)
            out.append((y_color or DATASET_B_COLOR) if is_dataset else (x_color or DATASET_A_COLOR))
        return out
    return [NCAR_PALETTE[s.index % len(NCAR_PALETTE)] for s in series]


def get_axis_color(
    dataset: xr.Dataset,
    var_name: str,
    index: int = 0,
    *,
    x_color: str | None = None,
    y_color: str | None = None,
) -> str:
    """Plot color for a paired series by axis.

    ``x_color``/``y_color`` let a caller supply the active ``StyleConfig``
    colors so a customised style is honoured for the geometry/dataset axes.
    """
    axis = paired_variable_axis(dataset, var_name)
    if axis == "x":
        return x_color or DATASET_A_COLOR
    if axis == "y":
        return y_color or DATASET_B_COLOR
    return NCAR_PALETTE[index % len(NCAR_PALETTE)]


def source_label(dataset: xr.Dataset, default: str | None = None) -> str | None:
    """Source label for a single-source dataset.

    Single-source datasets carry their source label in the dataset-level ``attrs``
    (set by the loading stage), not per-variable. Returns it so a source plot
    can self-identify its source, or ``default`` when absent.
    """
    label = dataset.attrs.get("source_label")
    return str(label) if label else default


def get_series_label(
    dataset: xr.Dataset,
    var_name: str,
    custom_label: str | None = None,
) -> str:
    """Legend label for a paired series (renderer rewire R-3).

    Prefers an explicit ``custom_label``, then the variable's ``source_label``
    attr (the source's identity in a unified pair, e.g. ``airnow`` / ``cam``),
    and finally falls back to the standard variable label. Use this for the
    *series* legend; axis labels that name the variable should keep using
    :func:`get_variable_label`.
    """
    if custom_label:
        return custom_label
    if var_name in dataset:
        source_label = dataset[var_name].attrs.get("source_label")
        if source_label:
            return str(source_label)
    return get_variable_label(dataset, var_name)


def resolve_source_variable(
    dataset: xr.Dataset,
    canonical_var: str,
    source_label: str,
) -> str | None:
    """Resolve a variable name by source label (Phase 5, additive).

    Supports the unified source-label naming (``<source_label>_<canonical>``,
    e.g. ``cam_o3``) while falling back to the bare canonical name. Returns the
    matching variable name present in the dataset, or ``None`` if neither is
    found. Does not alter the existing ``x_``/``y_`` prefix handling.

    Parameters
    ----------
    dataset
        Dataset to search.
    canonical_var
        Canonical (unprefixed) variable name, e.g. ``"o3"``.
    source_label
        Source label used as a prefix, e.g. ``"cam"`` or ``"airnow"``.

    Returns
    -------
    str | None
        The resolved variable name, or ``None`` if absent.
    """
    for candidate in (f"{source_label}_{canonical_var}", canonical_var):
        if candidate in dataset.data_vars or candidate in dataset.coords:
            return candidate
    return None


__all__ = [
    "build_series",
    "series_colors",
    "get_axis_color",
    "source_label",
    "get_series_label",
    "resolve_source_variable",
]
