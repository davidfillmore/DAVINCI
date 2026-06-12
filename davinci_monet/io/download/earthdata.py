"""Shared NASA Earthdata staging core used by the product downloaders.

Network access is isolated in ``_login``/``_search``/``_download`` so product
modules (and their tests) run offline. ``earthaccess`` is an optional
dependency (``pip install -e ".[reanalysis]"``) imported lazily.

``stage_collection`` accepts injectable hooks so product modules can route
network calls through their own monkeypatchable module attributes (MERRA-2
does this for backward compatibility); when no hooks are passed, the module
functions here are resolved at call time, so tests may monkeypatch them
directly on this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

LoginFn = Callable[[], Any]
SearchFn = Callable[["CollectionSpec", "tuple[str, str] | None"], Sequence[Any]]
DownloadFn = Callable[[Sequence[Any], str], "list[Path]"]


@dataclass(frozen=True)
class CollectionSpec:
    """An Earthdata collection: short-name + on-disk subpath + pinned version.

    ``version`` is required when multiple editions share a short-name in CMR
    (e.g. CERES editions); ``None`` searches across all versions.
    """

    short_name: str
    subpath: Path
    version: str | None = None


def _login() -> Any:
    """Authenticate to NASA Earthdata (lazy earthaccess import)."""
    import earthaccess

    return earthaccess.login()


def _search(
    short_name: str,
    temporal: tuple[str, str] | None = None,
    version: str | None = None,
) -> Sequence[Any]:
    """Search CMR for granules of ``short_name`` (lazy earthaccess import)."""
    import earthaccess

    kwargs: dict[str, Any] = {"short_name": short_name}
    if temporal is not None:
        kwargs["temporal"] = temporal
    if version is not None:
        kwargs["version"] = version
    return earthaccess.search_data(**kwargs)


def _download(results: Sequence[Any], dest: str) -> list[Path]:
    """Download ``results`` into ``dest``; return local file paths."""
    import earthaccess

    return [Path(p) for p in earthaccess.download(list(results), dest)]


# CMR SizeUnit -> MB conversion. earthaccess's DataGranule.size() sums the
# raw Size numbers ignoring SizeUnit, so granule_size_mb reads the UMM
# metadata itself and only falls back to size() when that's unavailable.
_SIZE_UNIT_TO_MB = {
    "KB": 1.0 / 1024.0,
    "MB": 1.0,
    "GB": 1024.0,
    "TB": 1024.0 * 1024.0,
    "PB": 1024.0**3,
}


def granule_size_mb(granule: Any) -> float:
    """Best-effort size of an earthaccess granule in MB (0.0 if unavailable).

    Prefers the UMM archive metadata, which carries explicit units; see
    ``_SIZE_UNIT_TO_MB`` for why ``DataGranule.size()`` is not trusted.
    """
    try:
        infos = granule["umm"]["DataGranule"]["ArchiveAndDistributionInformation"]
    except (KeyError, TypeError):
        infos = None
    if infos:
        total = 0.0
        found = False
        for info in infos:
            if not isinstance(info, dict):
                continue
            factor = _SIZE_UNIT_TO_MB.get(str(info.get("SizeUnit")))
            size = info.get("Size")
            if factor is None or not isinstance(size, (int, float)):
                continue
            total += float(size) * factor
            found = True
        if found:
            return total
    size_method = getattr(granule, "size", None)
    if not callable(size_method):
        return 0.0
    try:
        return float(size_method())
    except (TypeError, ValueError):
        return 0.0


def _default_search(spec: CollectionSpec, temporal: tuple[str, str] | None) -> Sequence[Any]:
    return _search(spec.short_name, temporal, spec.version)


def stage_collection(
    spec: CollectionSpec,
    temporal: tuple[str, str] | None,
    dest: Path,
    *,
    dry_run: bool = False,
    login: LoginFn | None = None,
    search: SearchFn | None = None,
    download: DownloadFn | None = None,
) -> Sequence[Any] | list[Path]:
    """Stage ``spec`` granules into ``dest``: login, search, mkdir, download.

    Returns the raw search results when ``dry_run`` (callers summarize),
    otherwise the downloaded file paths.
    """
    do_login = login if login is not None else _login
    do_search = search if search is not None else _default_search
    do_download = download if download is not None else _download

    do_login()
    results = do_search(spec, temporal)
    if dry_run:
        return results
    dest.mkdir(parents=True, exist_ok=True)
    return do_download(results, str(dest))
