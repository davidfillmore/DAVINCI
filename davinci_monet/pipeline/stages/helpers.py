"""Module-level helper functions shared across pipeline stages.

These are dataset-label and variable-resolution utilities plus small formatters
used by the concrete stages. They are kept free of stage subclasses to avoid
import cycles.
"""

from __future__ import annotations

from typing import Any

import xarray as xr

from davinci_monet.pipeline.stages.base import PipelineContext


def _format_size(n: int) -> str:
    """Format large numbers with K/M/B suffix."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m{secs:.0f}s"


def tag_source_label(
    data: Any,
    *,
    source_label: str,
) -> Any:
    """Attach a source label to each variable of a single-source dataset."""
    if data is None or not hasattr(data, "data_vars"):
        return data
    for name in data.data_vars:
        var = data[name]
        var.attrs.setdefault("source_label", source_label)
    return data


def iter_single_source_datasets(
    context: PipelineContext,
) -> list[tuple[str, Any, xr.Dataset]]:
    """Return loaded single-source datasets from the unified source view.

    ``context.sources`` is the canonical store for all loaded data sources
    keyed by label.
    """
    items = list(context.sources.items())

    sources: list[tuple[str, Any, xr.Dataset]] = []
    for label, obj in items:
        ds = obj.data if hasattr(obj, "data") else obj
        if not isinstance(ds, xr.Dataset):
            continue
        sources.append((str(label), obj, ds))
    return sources


def resolve_paired_var_names(
    paired_data: Any,
    x_var: str,
    x_source: str,
    y_source: str,
) -> tuple[str, str]:
    """Resolve the (geometry, dataset) variable names to plot from a paired dataset.

    Prefer the source-label aliases (``<label>_<var>``, e.g. ``airnow_o3`` /
    ``cam_o3``), falling back to the ``geometry_``/``dataset_`` prefixes. The
    returned names are concrete strings; the caller checks membership before
    plotting.
    """
    from davinci_monet.plots.base import resolve_source_variable

    geometry_name = resolve_source_variable(paired_data, x_var, x_source) or f"geometry_{x_var}"
    dataset_name = resolve_source_variable(paired_data, x_var, y_source) or f"dataset_{x_var}"
    return geometry_name, dataset_name
