"""Shared helper for renderer on-plot statistics annotations.

Renderers display small "stats boxes" (e.g. ``N``, ``MB``, ``RMSE``, ``R``,
``NMB``, ``NME``) on top of their figures. Earlier renderer implementations
hand-rolled these computations inline with raw NumPy, duplicating the formulas
and risking inconsistent NaN handling. This module centralizes the computation
by delegating to the canonical metric registry in
the davinci_monet.stats.metrics module, so the annotation values stay consistent
with the statistics tables the pipeline writes.

The single entry point is :func:`annotation_metrics`, which performs a pairwise
NaN drop (matching the ``~isnan(geometry) & ~isnan(dataset)`` behavior the renderers
already used) and then resolves each requested metric via ``get_metric`` and
calls its ``.compute(geometry, dataset)``.

This module imports from ``davinci_monet.stats`` (allowed direction); the stats
package never imports from ``davinci_monet.plots``, so no circular import is
introduced.
"""

from __future__ import annotations

import numpy as np

from davinci_monet.stats.metrics import get_metric

__all__ = ["annotation_metrics"]


def annotation_metrics(
    geometry: np.ndarray,
    dataset: np.ndarray,
    metrics: list[str],
) -> dict[str, float]:
    """Compute on-plot annotation metrics via the central metric registry.

    The inputs are flattened and pairwise NaN-dropped (an index is kept only
    where *both* ``geometry`` and ``dataset`` are non-NaN), reproducing the masking the
    renderers performed before computing their inline stats. Each requested
    metric is then resolved from the registry and computed on the cleaned
    arrays.

    Parameters
    ----------
    geometry
        Geometry (dataset) values.
    dataset
        Dataset (dataset) values. Must broadcast to the same flattened length
        as ``geometry``.
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
    geometry_arr = np.asarray(geometry).flatten()
    dataset_arr = np.asarray(dataset).flatten()

    valid_both = ~np.isnan(geometry_arr) & ~np.isnan(dataset_arr)
    geometry_valid = geometry_arr[valid_both]
    dataset_valid = dataset_arr[valid_both]

    results: dict[str, float] = {}
    for name in metrics:
        value = get_metric(name).compute(geometry_valid, dataset_valid)
        results[name] = int(value) if name == "N" else value
    return results
