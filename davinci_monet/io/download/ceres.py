"""Stage CERES radiation-budget products to local disk via earthaccess.

Covers L2 SSF swath granules (per instrument) and L3 EBAF / SYN1deg gridded
products from the NASA Langley ASDC. Network access goes through the shared
:mod:`davinci_monet.io.download.earthdata` core, whose ``_login``/``_search``/
``_download`` functions tests monkeypatch directly; ``earthaccess`` is an
optional dependency (``pip install -e ".[reanalysis]"``) imported lazily.

CMR notes (verified 2026-06-12):

- Multiple editions share a short-name, so every spec pins a ``version``.
- EBAF ships as a single whole-record ~2 GB netCDF, so temporal bounds are
  optional for it and required everywhere else.
- SYN1deg Edition4B (Terra-Aqua-NOAA20) is a full-record reprocessing
  (2000-03 onward) that supersedes the Terra-Aqua-MODIS Edition4A
  collections, so only Edition4B is staged.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from davinci_monet.io.download import earthdata
from davinci_monet.io.download.earthdata import CollectionSpec

DEFAULT_ROOT = "/Volumes/Io"

# Friendly collection name -> (Earthdata short_name, Io subpath, edition).
CERES_COLLECTIONS: dict[str, CollectionSpec] = {
    # L2 Single Scanner Footprint (swath) — one entry per instrument.
    "ssf_terra-fm1": CollectionSpec(
        "CER_SSF_Terra-FM1-MODIS", Path("CERES/SSF/Terra-FM1"), "Edition4A"
    ),
    "ssf_aqua-fm3": CollectionSpec(
        "CER_SSF_Aqua-FM3-MODIS", Path("CERES/SSF/Aqua-FM3"), "Edition4A"
    ),
    "ssf_npp-fm5": CollectionSpec("CER_SSF_NPP-FM5-VIIRS", Path("CERES/SSF/NPP-FM5"), "Edition2A"),
    "ssf_noaa20-fm6": CollectionSpec(
        "CER_SSF_NOAA20-FM6-VIIRS", Path("CERES/SSF/NOAA20-FM6"), "Edition1C"
    ),
    # L3 EBAF — energy-balanced monthly means, one whole-record granule.
    "ebaf": CollectionSpec("CERES_EBAF", Path("CERES/EBAF"), "Edition4.2.1"),
    # L3 SYN1deg — Edition4B (Terra-Aqua-NOAA20) is a full-record reprocessing
    # (2000-03 onward) superseding the Terra-Aqua-MODIS Edition4A collections.
    "syn1deg_month": CollectionSpec(
        "CER_SYN1deg-Month_Terra-Aqua-NOAA20", Path("CERES/SYN1deg/month"), "Edition4B"
    ),
    "syn1deg_day": CollectionSpec(
        "CER_SYN1deg-Day_Terra-Aqua-NOAA20", Path("CERES/SYN1deg/day"), "Edition4B"
    ),
    "syn1deg_hour": CollectionSpec(
        "CER_SYN1deg-1Hour_Terra-Aqua-NOAA20", Path("CERES/SYN1deg/hour"), "Edition4B"
    ),
}

# Whole-record collections: a single granule spans the full record, so a
# temporal filter is unnecessary (and omitting it is the normal usage).
NO_TEMPORAL_OK = frozenset({"ebaf"})


def resolve_collection(collection: str) -> CollectionSpec:
    """Look up a collection spec, raising a helpful KeyError if unknown."""
    try:
        return CERES_COLLECTIONS[collection]
    except KeyError:
        known = ", ".join(sorted(CERES_COLLECTIONS))
        raise KeyError(f"Unknown CERES collection {collection!r}. Known: {known}") from None


def dest_dir(collection: str, root: str | Path = DEFAULT_ROOT) -> Path:
    """Return the staging directory for ``collection`` under ``root``."""
    return Path(root) / resolve_collection(collection).subpath


@dataclass(frozen=True)
class DryRunReport:
    """Summary of a dry-run search: granule count and total size in MB."""

    granules: int
    total_mb: float


def stage_ceres(
    collection: str,
    start: str | None = None,
    end: str | None = None,
    *,
    root: str | Path = DEFAULT_ROOT,
    dry_run: bool = False,
) -> DryRunReport | list[Path]:
    """Stage a CERES ``collection`` under ``root``.

    Parameters
    ----------
    collection
        Friendly collection name (see ``CERES_COLLECTIONS``).
    start, end
        Inclusive temporal bounds as ISO strings (e.g. ``"2023-07-01"``).
        Optional for whole-record collections (``ebaf``); required otherwise —
        an unbounded SSF search would match the entire 25-year record.
    root
        Staging root; the collection subpath is appended.
    dry_run
        If True, search only and report granule count + total size.

    Returns
    -------
    DryRunReport | list[Path]
        Count/size summary when ``dry_run``; otherwise the staged file paths.
    """
    spec = resolve_collection(collection)
    if (start is None) != (end is None):
        raise ValueError("provide both start and end, or neither")
    if start is None and collection not in NO_TEMPORAL_OK:
        raise ValueError(
            f"start and end are required for {collection!r}; only whole-record "
            f"collections ({', '.join(sorted(NO_TEMPORAL_OK))}) may omit them"
        )
    temporal = (start, end) if start is not None and end is not None else None

    result = earthdata.stage_collection(spec, temporal, Path(root) / spec.subpath, dry_run=dry_run)
    if dry_run:
        return DryRunReport(
            granules=len(result),
            total_mb=sum(earthdata.granule_size_mb(g) for g in result),
        )
    return [Path(p) for p in result]
