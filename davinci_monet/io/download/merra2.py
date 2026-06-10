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
# Subpaths follow the existing on-disk layout: MERRA2_<temporal>/<group>.
MERRA2_COLLECTIONS: dict[str, CollectionSpec] = {
    # Aerosol (GOCART) — the chemistry->climate bridge.
    "tavgM_2d_aer_Nx": CollectionSpec("M2TMNXAER", Path("MERRA2_tavgM/aer_Nx")),
    "inst3_3d_aer_Nv": CollectionSpec("M2I3NVAER", Path("MERRA2_inst3/aer_Nv")),
    # Meteorology — for ERA5-parallel climate evaluation.
    "tavg1_2d_slv_Nx": CollectionSpec("M2T1NXSLV", Path("MERRA2_tavg1/slv_Nx")),
    "inst3_3d_asm_Np": CollectionSpec("M2I3NPASM", Path("MERRA2_inst3/asm_Np")),
    # Clouds & radiation. 2D cloud fractions live alongside radiative fluxes
    # in rad_Nx; 3D cloud and radiation are separate collections.
    "tavg1_2d_rad_Nx": CollectionSpec("M2T1NXRAD", Path("MERRA2_tavg1/rad_Nx")),
    "tavg3_3d_cld_Np": CollectionSpec("M2T3NPCLD", Path("MERRA2_tavg3/cld_Np")),
    "tavg3_3d_rad_Np": CollectionSpec("M2T3NPRAD", Path("MERRA2_tavg3/rad_Np")),
}


def resolve_collection(collection: str) -> CollectionSpec:
    """Look up a collection spec, raising a helpful KeyError if unknown."""
    try:
        return MERRA2_COLLECTIONS[collection]
    except KeyError:
        known = ", ".join(sorted(MERRA2_COLLECTIONS))
        raise KeyError(f"Unknown MERRA-2 collection {collection!r}. Known: {known}") from None


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

    return [Path(p) for p in earthaccess.download(list(results), dest)]


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


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: stage a MERRA-2 collection. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="davinci-stage-merra2",
        description="Stage MERRA-2 aerosol collections to local disk.",
    )
    parser.add_argument("--collection", required=True, choices=sorted(MERRA2_COLLECTIONS))
    parser.add_argument("--start", required=True, help="ISO start, e.g. 2003-01")
    parser.add_argument("--end", required=True, help="ISO end, e.g. 2003-03")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Staging root dir")
    parser.add_argument("--dry-run", action="store_true", help="Search only; do not download")
    ns = parser.parse_args(argv)

    result = stage_merra2(ns.collection, ns.start, ns.end, root=ns.root, dry_run=ns.dry_run)
    if ns.dry_run:
        print(f"{result} granule(s) found for {ns.collection} " f"[{ns.start}..{ns.end}]")
    else:
        assert isinstance(result, list)
        print(f"Staged {len(result)} file(s) to " f"{dest_dir(ns.collection, ns.root)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
