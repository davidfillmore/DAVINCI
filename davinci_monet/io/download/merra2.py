"""Stage MERRA-2 aerosol collections to local disk via earthaccess.

Network access is isolated in ``_login``/``_search``/``_download`` so the
rest of the module (and its tests) run offline. ``earthaccess`` is an
optional dependency (``pip install -e ".[reanalysis]"``) imported lazily.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

DEFAULT_ROOT = "/Volumes/Io"


@dataclass(frozen=True)
class CollectionSpec:
    """A MERRA-2 collection: Earthdata short-name + on-disk subpath."""

    short_name: str
    subpath: Path


# Friendly collection name -> (Earthdata short_name, Io subpath).
# Subpaths match the existing on-disk layout under the staging root.
MERRA2_COLLECTIONS: dict[str, CollectionSpec] = {
    "tavgM_2d_aer_Nx": CollectionSpec("M2TMNXAER", Path("MERRA2_tavgM/aer_Nx")),
    "inst3_3d_aer_Nv": CollectionSpec("M2I3NVAER", Path("MERRA2_inst3/aer_Nv")),
}


def resolve_collection(collection: str) -> CollectionSpec:
    """Look up a collection spec, raising a helpful KeyError if unknown."""
    try:
        return MERRA2_COLLECTIONS[collection]
    except KeyError:
        known = ", ".join(sorted(MERRA2_COLLECTIONS))
        raise KeyError(
            f"Unknown MERRA-2 collection {collection!r}. Known: {known}"
        ) from None


def dest_dir(collection: str, root: str | Path = DEFAULT_ROOT) -> Path:
    """Return the staging directory for ``collection`` under ``root``."""
    return Path(root) / resolve_collection(collection).subpath


def _login() -> Any:
    """Authenticate to NASA Earthdata (lazy earthaccess import)."""
    import earthaccess

    return earthaccess.login()


def _search(short_name: str, temporal: tuple[str, str]) -> Sequence[Any]:
    """Search GES DISC for granules of ``short_name`` over ``temporal``."""
    import earthaccess

    return earthaccess.search_data(short_name=short_name, temporal=temporal)


def _download(results: Sequence[Any], dest: str) -> list[Path]:
    """Download ``results`` into ``dest``; return local file paths."""
    import earthaccess

    return [Path(p) for p in earthaccess.download(results, dest)]


def stage_merra2(
    collection: str,
    start: str,
    end: str,
    *,
    root: str | Path = DEFAULT_ROOT,
    dry_run: bool = False,
) -> int | list[Path]:
    """Stage a MERRA-2 ``collection`` for ``[start, end]`` under ``root``.

    Parameters
    ----------
    collection
        Friendly collection name (see ``MERRA2_COLLECTIONS``).
    start, end
        Inclusive temporal bounds as ISO strings (e.g. ``"2003-01"``).
    root
        Staging root; the collection subpath is appended.
    dry_run
        If True, search only and return the granule count without downloading.

    Returns
    -------
    int | list[Path]
        Granule count when ``dry_run``; otherwise the staged file paths.
    """
    spec = resolve_collection(collection)
    _login()
    results = _search(spec.short_name, (start, end))
    if dry_run:
        return len(results)
    target = Path(root) / spec.subpath
    target.mkdir(parents=True, exist_ok=True)
    return _download(results, str(target))
