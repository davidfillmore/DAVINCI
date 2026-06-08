"""Series/role/color helper functions for DAVINCI plots.

Provides:
- build_series: resolve var-args into an ordered list of PlotSeries
- series_colors: per-series colors under the unified, count-aware rule
- get_role_color: plot color for a paired series by its source role
- dataset_source_label: source label for a single-source dataset
- get_series_label: legend label for a paired series
- resolve_source_variable: resolve a variable name by source label
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from davinci_monet.core.base import (
    PlotSeries,
    paired_variable_pair_role,
    paired_variable_role,
)
from davinci_monet.plots.labels import canonical_variable_name, get_variable_label
from davinci_monet.plots.style import MODEL_COLOR, NCAR_PALETTE, NCAR_PRIMARY, OBS_COLOR

if TYPE_CHECKING:
    import xarray as xr


def build_series(dataset: xr.Dataset, *var_args: Any) -> list[PlotSeries]:
    """Resolve facade var-args into an ordered list of :class:`PlotSeries`.

    Accepts the three call shapes the unified facade supports:

    - ``build_series(ds, obs_var, model_var)`` → 2 series
    - ``build_series(ds, variable)`` → 1 series
    - ``build_series(ds, [v1, ..., vN])`` → N series

    A trailing positional ``matplotlib`` Axes (legacy ``plot(ds, var, ax)``) is
    ignored for series building. ``role``/``pair_role``/``source_label``/
    ``canonical`` are read from the dataset's attrs, with the legacy
    ``obs_``/``model_`` prefix fallback.
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
        # the dataset-level label that single-source obs datasets carry.
        source_label = (
            dataset[name].attrs.get("source_label") if name in dataset.data_vars else None
        ) or dataset.attrs.get("source_label")
        series.append(
            PlotSeries(
                dataset=dataset,
                var_name=name,
                canonical=canonical_variable_name(dataset, name),
                role=paired_variable_role(dataset, name),
                pair_role=paired_variable_pair_role(dataset, name),
                source_label=str(source_label) if source_label else None,
                index=i,
            )
        )
    return series


def series_colors(
    series: list[PlotSeries],
    *,
    obs_color: str | None = None,
    model_color: str | None = None,
) -> list[str]:
    """Per-series colors under the unified, count-aware rule.

    - **1 series** → ``NCAR_PRIMARY`` (the single-source brand blue), or ``MODEL_COLOR``
      when the lone source is ``role == "model"``. This is what keeps a single
      source blue rather than the paired-reference gray that ``get_color_for_role``
      would assign.
    - **2 series** → reference in ``obs_color`` (gray) and comparand in
      ``model_color`` (blue), preserving today's comparison contrast.
    - **N > 2 series** → distinct ``NCAR_PALETTE`` colors cycled by ``index``.

    ``obs_color``/``model_color`` let a caller pass the active ``StyleConfig``
    colors; they default to the module ``OBS_COLOR``/``MODEL_COLOR``.
    """
    n = len(series)
    if n == 1:
        return [MODEL_COLOR if series[0].role == "model" else NCAR_PRIMARY]
    if n == 2:
        out: list[str] = []
        for s in series:
            is_model = s.role == "model" or s.pair_role == "comparand"
            out.append((model_color or MODEL_COLOR) if is_model else (obs_color or OBS_COLOR))
        return out
    return [NCAR_PALETTE[s.index % len(NCAR_PALETTE)] for s in series]


def get_role_color(
    dataset: xr.Dataset,
    var_name: str,
    index: int = 0,
    *,
    obs_color: str | None = None,
    model_color: str | None = None,
) -> str:
    """Plot color for a paired series, by its source role (renderer rewire R-3).

    Reads the variable's ``role`` attr (set by ``tag_paired_roles``): ``obs``
    renders in the neutral reference gray, ``model`` in NCAR blue (preserving
    the legacy model-vs-obs convention), and same-role / role-less series cycle the
    NCAR palette by ``index`` (their order in the plot).

    ``obs_color``/``model_color`` let a caller supply the active ``StyleConfig``
    colors so a customised style is honoured for the obs/model roles; when
    omitted the module's :func:`get_color_for_role` defaults are used.
    """
    from davinci_monet.plots.style import get_color_for_role

    role = dataset[var_name].attrs.get("role") if var_name in dataset else None
    # Fall back to the legacy prefix when no role attr is present, so renderers
    # called directly with model_/obs_ names (tests, examples, user scripts)
    # still get the obs gray / model blue convention rather than palette colors.
    if role is None:
        lname = str(var_name).lower()
        if lname.startswith("obs_"):
            role = "obs"
        elif lname.startswith("model_"):
            role = "model"
    if role == "obs" and obs_color is not None:
        return obs_color
    if role == "model" and model_color is not None:
        return model_color
    return get_color_for_role(role, index)


def dataset_source_label(dataset: xr.Dataset, default: str | None = None) -> str | None:
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
    and finally falls back to the standard variable label (the role-aware
    Observed/Modeled formatting). Use this for the *series* legend; axis labels
    that name the variable should keep using :func:`get_variable_label`.
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
    found. Does not alter the existing ``model_``/``obs_`` prefix handling.

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
    "get_role_color",
    "dataset_source_label",
    "get_series_label",
    "resolve_source_variable",
]
