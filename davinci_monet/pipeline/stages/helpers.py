"""Module-level helper functions shared across pipeline stages.

These are role-tagging and variable-resolution utilities plus small formatters
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


def tag_paired_roles(
    data: Any,
    *,
    reference_label: str | None = None,
    comparand_label: str | None = None,
    reference_role: str | None = "obs",
    comparand_role: str | None = "model",
) -> None:
    """Tag each paired variable with its ``role`` and rename it by source label.

    Each paired variable gets a source ``role`` attr plus a ``pair_role`` attr
    (``reference``/``comparand``). Same-role source pairs keep their source role
    while ``pair_role`` preserves reference/comparand semantics for statistics
    and plotting.

    The variable is renamed to ``<comparand_label>_<v>`` or
    ``<reference_label>_<v>``. When labels are omitted, ``comparand`` and
    ``reference`` are used as neutral defaults. A pre-existing ``role`` attr is
    never overwritten.
    """
    if data is None or not hasattr(data, "data_vars"):
        return
    reference_label = reference_label or "reference"
    comparand_label = comparand_label or "comparand"
    # Snapshot the names first: variables are renamed while iterating.
    for name in list(data.data_vars):
        existing_pair_role = data[name].attrs.get("pair_role")
        if existing_pair_role in {"reference", "comparand"}:
            role = reference_role if existing_pair_role == "reference" else comparand_role
            label = reference_label if existing_pair_role == "reference" else comparand_label
            if role is not None:
                data[name].attrs.setdefault("role", role)
            if label is not None:
                data[name].attrs.setdefault("source_label", label)
            continue

        lname = str(name).lower()
        if lname.startswith("model_"):
            role = comparand_role or "model"
            pair_role = "comparand"
            label = comparand_label
            canonical = str(name)[len("model_") :]
        elif lname.startswith("obs_"):
            role = reference_role or "obs"
            pair_role = "reference"
            label = reference_label
            canonical = str(name)[len("obs_") :]
        else:
            continue

        var = data[name]
        var.attrs.setdefault("role", role)
        var.attrs.setdefault("pair_role", pair_role)
        var.attrs.setdefault("source_label", label)

        # Rename to the source-label name unless it would collide with an
        # existing data variable.
        new_name = f"{label}_{canonical}"
        if new_name != name and new_name not in data.data_vars:
            data[new_name] = var
            data[new_name].attrs["role"] = var.attrs["role"]
            data[new_name].attrs["pair_role"] = var.attrs["pair_role"]
            data[new_name].attrs["source_label"] = var.attrs["source_label"]
            del data[name]


def tag_source_roles(
    data: Any,
    *,
    role: str | None,
    source_label: str,
) -> Any:
    """Tag each variable of a single-source dataset with ``role``/``source_label``.

    Per-variable counterpart of :func:`tag_paired_roles` for *unpaired* sources:
    a single obs (or model) source carries its label/role only at the dataset
    level today, so the unified series resolver
    (:func:`~davinci_monet.core.base.iter_canonical_variable_series`) cannot see
    its variables. This sets the attrs per data_var (idempotent via
    ``setdefault``, never overwriting a pre-existing ``role``) so a single source
    becomes a 1-series plot under the unified renderer. Returns ``data``.
    """
    if data is None or not hasattr(data, "data_vars"):
        return data
    for name in data.data_vars:
        var = data[name]
        if role is not None:
            var.attrs.setdefault("role", role)
        var.attrs.setdefault("source_label", source_label)
    return data


def iter_single_source_datasets(
    context: PipelineContext,
) -> list[tuple[str, Any, xr.Dataset, str | None]]:
    """Return loaded single-source datasets from the unified source view.

    ``context.sources`` is the canonical store for all loaded data sources
    (models and observations alike), keyed by label.
    """
    items = list(context.sources.items())

    sources: list[tuple[str, Any, xr.Dataset, str | None]] = []
    for label, obj in items:
        ds = obj.data if hasattr(obj, "data") else obj
        if not isinstance(ds, xr.Dataset):
            continue
        role = getattr(obj, "role", None) or ds.attrs.get("role")
        sources.append((str(label), obj, ds, str(role) if role else None))
    return sources


def resolve_paired_var_names(
    paired_data: Any,
    obs_var: str,
    obs_label: str,
    model_label: str,
) -> tuple[str, str]:
    """Resolve the (obs, model) variable names to plot from a paired dataset.

    Renderer rewire R-2: prefer the source-label aliases (``<label>_<var>``,
    e.g. ``airnow_o3`` / ``cam_o3``) added by :func:`tag_paired_roles`, falling
    back to the legacy ``obs_``/``model_`` prefixes when no alias is present
    (older paired data, or a label in the reserved namespace). obs is the
    reference and model the comparand; the pairing engine names both paired
    variables off the *obs* canonical name (``model_<obs_var>``), so both
    resolutions key off ``obs_var``.

    The returned names are always concrete strings (alias if present, else the
    legacy prefix); the caller is responsible for checking membership before
    plotting.
    """
    from davinci_monet.plots.base import resolve_source_variable

    obs_name = resolve_source_variable(paired_data, obs_var, obs_label) or f"obs_{obs_var}"
    model_name = resolve_source_variable(paired_data, obs_var, model_label) or f"model_{obs_var}"
    return obs_name, model_name
