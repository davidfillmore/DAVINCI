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
