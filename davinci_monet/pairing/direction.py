"""Pair-direction resolution for data-source pairing.

A binary pair has a *geometry* (whose geometry the dataset is sampled onto)
and a *dataset*. The default direction follows a geometry precedence rule:
irregular geometries (POINT/TRACK/PROFILE/SWATH) outrank GRID as the geometry,
so a GRID source is sampled onto them. When both sources share the same precedence tier (e.g. GRID vs GRID,
or two different irregular geometries) and no explicit geometry is given, the
first-listed source is used as the geometry and a :class:`PairDirectionWarning`
is emitted.
"""

from __future__ import annotations

import warnings

from davinci_monet.core.protocols import DataGeometry

__all__ = ["IRREGULAR_GEOMETRIES", "PairDirectionWarning", "resolve_pair_direction"]


class PairDirectionWarning(Warning):
    """Emitted when pair direction is ambiguous and defaulted to the first source.

    Intentionally not a ``UserWarning`` subclass so it is informational rather
    than escalated to an error by strict warning filters.
    """


#: Geometries that outrank GRID as the pairing geometry.
IRREGULAR_GEOMETRIES = frozenset(
    {DataGeometry.POINT, DataGeometry.TRACK, DataGeometry.PROFILE, DataGeometry.SWATH}
)


def _rank(geometry: DataGeometry) -> int:
    """Precedence rank; higher wins the geometry. GRID is lowest."""
    return 0 if geometry is DataGeometry.GRID else 1


def resolve_pair_direction(
    geom_a: DataGeometry,
    geom_b: DataGeometry,
    explicit_geometry: str | None = None,
) -> tuple[DataGeometry, DataGeometry]:
    """Resolve which source is the geometry and which is the dataset.

    Parameters
    ----------
    geom_a
        Geometry of the first-listed source.
    geom_b
        Geometry of the second-listed source.
    explicit_geometry
        Optional override naming which positional source is the geometry:
        ``"a"`` for ``geom_a`` or ``"b"`` for ``geom_b``. When given, precedence
        is bypassed and no warning is emitted.

    Returns
    -------
    tuple[DataGeometry, DataGeometry]
        ``(geometry, dataset_geometry)``.
    """
    if explicit_geometry is not None:
        if explicit_geometry == "a":
            return geom_a, geom_b
        if explicit_geometry == "b":
            return geom_b, geom_a
        raise ValueError(f"explicit_geometry must be 'a', 'b', or None; got {explicit_geometry!r}")

    rank_a, rank_b = _rank(geom_a), _rank(geom_b)
    if rank_a > rank_b:
        return geom_a, geom_b
    if rank_b > rank_a:
        return geom_b, geom_a

    # Same precedence tier (GRID/GRID or two irregular geometries): ambiguous.
    warnings.warn(
        f"Pair direction is ambiguous for geometries {geom_a.name} and {geom_b.name} "
        f"(same precedence tier) and no explicit geometry was given; defaulting to "
        f"the first-listed source ({geom_a.name}) as the geometry.",
        PairDirectionWarning,
        stacklevel=2,
    )
    return geom_a, geom_b
