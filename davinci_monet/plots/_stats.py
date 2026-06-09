"""Shared helper for renderer on-plot statistics annotations.

Renderers display small "stats boxes" (e.g. ``N``, ``MB``, ``RMSE``, ``R``,
``NMB``, ``NME``) on top of their figures. Historically each renderer
hand-rolled these computations inline with raw NumPy, duplicating the formulas
and risking inconsistent NaN handling. This module centralizes the computation
by delegating to the canonical metric registry in
:mod:`davinci_monet.stats.metrics`, so the annotation values stay consistent
with the statistics tables the pipeline writes.

The single entry point is :func:`annotation_metrics`, which performs a pairwise
NaN drop (matching the ``~isnan(obs) & ~isnan(model)`` behavior the renderers
already used) and then resolves each requested metric via ``get_metric`` and
calls its ``.compute(obs, mod)``.

This module imports from ``davinci_monet.stats`` (allowed direction); the stats
package never imports from ``davinci_monet.plots``, so no circular import is
introduced.
"""

from __future__ import annotations

import numpy as np

from davinci_monet.stats.metrics import get_metric

__all__ = ["annotation_metrics"]


def annotation_metrics(
    obs: np.ndarray,
    model: np.ndarray,
    metrics: list[str],
) -> dict[str, float]:
    """Compute on-plot annotation metrics via the central metric registry.

    The inputs are flattened and pairwise NaN-dropped (an index is kept only
    where *both* ``obs`` and ``model`` are non-NaN), reproducing the masking the
    renderers performed before computing their inline stats. Each requested
    metric is then resolved from the registry and computed on the cleaned
    arrays.

    Parameters
    ----------
    obs
        Reference (observation) values.
    model
        Comparand (model) values. Must broadcast to the same flattened length
        as ``obs``.
    metrics
        Registry metric names to compute (e.g. ``["N", "MB", "RMSE", "R"]``).

    Returns
    -------
    dict[str, float]
        Mapping of each requested metric name to its computed value. ``"N"`` is
        returned as a Python ``int``; all other metrics as ``float`` (possibly
        ``nan``).

    Notes
    -----
    The registry metrics re-apply a finite mask internally, so passing arrays
    that are already finite (as the renderers do) leaves the values unchanged.
    """
    obs_arr = np.asarray(obs).flatten()
    model_arr = np.asarray(model).flatten()

    valid_both = ~np.isnan(obs_arr) & ~np.isnan(model_arr)
    obs_valid = obs_arr[valid_both]
    model_valid = model_arr[valid_both]

    results: dict[str, float] = {}
    for name in metrics:
        value = get_metric(name).compute(obs_valid, model_valid)
        results[name] = int(value) if name == "N" else value
    return results
