"""Domain filtering utilities for paired datasets.

Translates ``domain_type`` / ``domain_name`` configuration keys into
lat/lon bounding-box filters that can be applied to paired datasets
before plotting or statistics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr


def _coerce_str(value: Any) -> str | None:
    """Coerce a domain_type/domain_name list-or-string config value to a single str.

    The pydantic schema declares these as ``list[str]``; YAML configs may use
    either ``domain_type: conus`` or ``domain_type: [conus]``. Multi-domain
    iteration is not yet supported — the first element wins.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    return str(value)


def filter_paired_by_domain(
    paired_data: xr.Dataset,
    domain_type: Any,
    domain_name: Any = None,
) -> xr.Dataset:
    """Filter paired_data to sites/points within a named domain's lat/lon extent.

    Parameters
    ----------
    paired_data
        Paired dataset with ``latitude``/``longitude`` (or ``lat``/``lon``)
        coordinates along its spatial dimension (e.g. ``site``, ``x``,
        ``time`` for tracks).
    domain_type
        Domain type string or single-element list (``"conus"``,
        ``"epa_region"``, ``"all"``, etc.). When ``None`` or ``"all"``, the
        dataset is returned unchanged.
    domain_name
        Specific domain name within the type (e.g. ``"R5"`` for EPA region 5).

    Returns
    -------
    xr.Dataset
        Filtered dataset, or the original if filtering is not applicable.

    Notes
    -----
    Returns the input unchanged when any of these hold:
        - ``domain_type`` is ``None`` or ``"all"``
        - the (domain_type, domain_name) pair is not in the named-domain table
        - the dataset has no lat/lon coords to filter on
        - lat/lon are 2-D (gridded geometry) — extent filtering on regular grids
          is currently not supported by this helper
    """
    dt = _coerce_str(domain_type)
    dn = _coerce_str(domain_name)

    if dt is None or dt == "all":
        return paired_data

    from davinci_monet.geography.domains import get_domain_extent

    extent = get_domain_extent(dt, dn)
    if extent is None:
        return paired_data

    lon_min, lon_max, lat_min, lat_max = extent

    lats = _get_coord(paired_data, ("latitude", "lat"))
    lons = _get_coord(paired_data, ("longitude", "lon"))
    if lats is None or lons is None:
        return paired_data

    # 2-D lat/lon (e.g. gridded geometry, swath) — not handled here. The caller
    # should rely on the dataset grid for that case.
    if lats.ndim != 1 or lons.ndim != 1:
        return paired_data

    # lats and lons must share a single spatial dim for the bbox mask to be 1-D.
    if lats.dims != lons.dims or len(lats.dims) != 1:
        return paired_data

    mask = (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)

    dim = mask.dims[0]
    keep = np.where(mask.values)[0]
    return paired_data.isel({dim: keep})


def _get_coord(ds: xr.Dataset, names: tuple[str, ...]) -> xr.DataArray | None:
    for name in names:
        if name in ds.coords:
            return ds.coords[name]
        if name in ds.data_vars:
            return ds[name]
    return None
