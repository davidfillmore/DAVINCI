# CERES Downloaders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stage CERES L2 (SSF) and L3 (EBAF, SYN1deg) products to the Io drive via a `davinci-stage-ceres` CLI, with the generic earthaccess plumbing extracted into a shared `earthdata.py` core reused by the existing MERRA-2 downloader.

**Architecture:** A new `davinci_monet/io/download/earthdata.py` owns `CollectionSpec` (now with an optional `version` field), the lazy-import `_login`/`_search`/`_download` network seam, `granule_size_mb`, and a generic `stage_collection()` orchestrator with injectable hooks. `merra2.py` keeps its public API and existing tests unchanged by passing its own module-level network wrappers as hooks (its tests monkeypatch `merra2._login` etc.). `ceres.py` is a product table + `stage_ceres()` (EBAF temporal special-casing, size-aware dry-run) + CLI; its tests monkeypatch the `earthdata` module functions, which the core's default hooks resolve at call time.

**Tech Stack:** Python 3.11, `earthaccess` (lazy optional import, `.[reanalysis]` extra — already declared), pytest with `monkeypatch`, mypy strict, black/isort.

**Spec:** `docs/superpowers/specs/2026-06-12-ceres-downloader-design.md`

**Repo rules that apply:**
- Run everything in the `davinci` conda env:
  `source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci`
- Prefix pytest with `HDF5_USE_FILE_LOCKING=FALSE`.
- **No auto-commits**: the commit steps below execute only with the user's standing approval for this plan; if that wasn't given at execution start, pause and ask before the first commit.
- No network in tests; real staging is manual and dry-run-first.

---

### Task 1: Shared earthaccess core (`earthdata.py`)

**Files:**
- Create: `davinci_monet/io/download/earthdata.py`
- Test: `davinci_monet/tests/test_download_earthdata.py`

- [ ] **Step 1.1: Write the failing tests**

Create `davinci_monet/tests/test_download_earthdata.py`:

```python
"""Offline unit tests for the shared Earthdata staging core."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from davinci_monet.io.download import earthdata


class _FakeGranule:
    """Mimics earthaccess.DataGranule's size() -> MB accessor."""

    def __init__(self, mb: float) -> None:
        self._mb = mb

    def size(self) -> float:
        return self._mb


def test_collection_spec_version_defaults_to_none() -> None:
    spec = earthdata.CollectionSpec("M2TMNXAER", Path("MERRA2_tavgM/aer_Nx"))
    assert spec.version is None


def test_granule_size_mb_uses_size_method() -> None:
    assert earthdata.granule_size_mb(_FakeGranule(42.5)) == 42.5


def test_granule_size_mb_zero_when_unavailable() -> None:
    assert earthdata.granule_size_mb("just-a-string") == 0.0


def test_stage_collection_dry_run_returns_results_without_download(tmp_path: Path) -> None:
    spec = earthdata.CollectionSpec("X", Path("X/sub"), version="V1")
    calls: dict[str, Any] = {}

    results = earthdata.stage_collection(
        spec,
        ("2023-01", "2023-02"),
        tmp_path / "X/sub",
        dry_run=True,
        login=lambda: calls.setdefault("login", True),
        search=lambda s, t: ["g1", "g2"],
        download=lambda r, d: pytest.fail("download called in dry run"),
    )

    assert calls["login"] is True
    assert list(results) == ["g1", "g2"]
    assert not (tmp_path / "X/sub").exists()  # dry run creates nothing


def test_stage_collection_creates_dest_and_downloads(tmp_path: Path) -> None:
    spec = earthdata.CollectionSpec("X", Path("X/sub"))
    dest = tmp_path / "X/sub"
    seen: dict[str, Any] = {}

    def _fake_download(results: Any, d: str) -> list[Path]:
        seen["results"] = list(results)
        seen["dest"] = d
        return [Path(d) / "f.nc"]

    out = earthdata.stage_collection(
        spec,
        None,
        dest,
        login=lambda: None,
        search=lambda s, t: ["g1"],
        download=_fake_download,
    )

    assert dest.is_dir()  # created before download
    assert seen["results"] == ["g1"]
    assert Path(seen["dest"]) == dest
    assert out == [dest / "f.nc"]


def test_stage_collection_default_search_passes_spec_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default hooks resolve earthdata module functions at call time."""
    spec = earthdata.CollectionSpec("CERES_EBAF", Path("CERES/EBAF"), version="Edition4.2.1")
    calls: dict[str, Any] = {}

    monkeypatch.setattr(earthdata, "_login", lambda: calls.setdefault("login", True))

    def _fake_search(
        short_name: str,
        temporal: tuple[str, str] | None = None,
        version: str | None = None,
    ) -> list[str]:
        calls["search"] = (short_name, temporal, version)
        return ["g1"]

    monkeypatch.setattr(earthdata, "_search", _fake_search)

    results = earthdata.stage_collection(spec, None, tmp_path / "CERES/EBAF", dry_run=True)

    assert calls["login"] is True
    assert calls["search"] == ("CERES_EBAF", None, "Edition4.2.1")
    assert list(results) == ["g1"]
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_earthdata.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'davinci_monet.io.download.earthdata'`

- [ ] **Step 1.3: Implement `earthdata.py`**

Create `davinci_monet/io/download/earthdata.py`:

```python
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


def granule_size_mb(granule: Any) -> float:
    """Best-effort size of an earthaccess granule in MB (0.0 if unavailable)."""
    size = getattr(granule, "size", None)
    if not callable(size):
        return 0.0
    try:
        return float(size())
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
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_earthdata.py -v
```

Expected: 6 passed

- [ ] **Step 1.5: Commit**

```bash
git add davinci_monet/io/download/earthdata.py davinci_monet/tests/test_download_earthdata.py
git commit -m "feat(io): shared Earthdata staging core for product downloaders"
```

---

### Task 2: Refactor `merra2.py` onto the shared core (behavior unchanged)

**Files:**
- Modify: `davinci_monet/io/download/merra2.py` (replace `CollectionSpec` definition and `stage_merra2` body; keep `_login`/`_search`/`_download` as module attributes)
- Test: `davinci_monet/tests/test_download_merra2.py` (MUST NOT be edited — it is the regression gate)

- [ ] **Step 2.1: Refactor `merra2.py`**

The existing tests monkeypatch `merra2._login`, `merra2._search`, `merra2._download` and call `stage_merra2`. Therefore `merra2.py` must keep those three as module attributes, and `stage_merra2` must route through them (resolved from module globals at call time) by passing them as hooks to `stage_collection`.

Replace the module header/imports, delete the local `CollectionSpec` dataclass, change `_login`/`_search`/`_download` to delegate to `earthdata`, and rewrite the `stage_merra2` body. The result (full file, excluding the unchanged `MERRA2_COLLECTIONS` dict, `resolve_collection`, `dest_dir`, `main`, and `__main__` block):

```python
"""Stage MERRA-2 aerosol collections to local disk via earthaccess.

Network access is isolated in ``_login``/``_search``/``_download`` so the
rest of the module (and its tests) run offline. ``earthaccess`` is an
optional dependency (``pip install -e ".[reanalysis]"``) imported lazily
via the shared :mod:`davinci_monet.io.download.earthdata` core.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from davinci_monet.io.download import earthdata
from davinci_monet.io.download.earthdata import CollectionSpec, stage_collection

DEFAULT_ROOT = "/Volumes/Io"

# ... MERRA2_COLLECTIONS, resolve_collection, dest_dir unchanged ...


def _login() -> Any:
    """Authenticate to NASA Earthdata (delegates to the shared core)."""
    return earthdata._login()


def _search(short_name: str, temporal: tuple[str, str]) -> Sequence[Any]:
    """Search GES DISC for granules of ``short_name`` over ``temporal``."""
    return earthdata._search(short_name, temporal)


def _download(results: Sequence[Any], dest: str) -> list[Path]:
    """Download ``results`` into ``dest``; return local file paths."""
    return earthdata._download(results, dest)


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
    temporal = (start, end)
    # Hooks reference this module's network wrappers so tests can monkeypatch
    # merra2._login/_search/_download (resolved from module globals at call time).
    result = stage_collection(
        spec,
        temporal,
        Path(root) / spec.subpath,
        dry_run=dry_run,
        login=_login,
        search=lambda s, _t: _search(s.short_name, temporal),
        download=_download,
    )
    if dry_run:
        return len(result)
    return list(result)
```

Keep the existing `MERRA2_COLLECTIONS` dict (entries already construct `CollectionSpec(short_name, subpath)` — the new `version` field defaults to `None`, no edits needed), `resolve_collection`, `dest_dir`, `main`, and the `__main__` block exactly as they are.

Note the lambda subtlety: `login=_login` and `download=_download` are evaluated when `stage_merra2` runs — i.e. *after* a test's `monkeypatch.setattr` — so the patched functions are picked up. The `search` lambda likewise looks up `_search` in module globals at call time.

- [ ] **Step 2.2: Run the existing MERRA-2 tests unchanged**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_merra2.py davinci_monet/tests/test_download_earthdata.py -v
```

Expected: all pass (7 parametrized resolve cases + 5 behavior tests + 6 core tests). If any MERRA-2 test fails, fix `merra2.py` — do not touch the test file.

- [ ] **Step 2.3: Commit**

```bash
git add davinci_monet/io/download/merra2.py
git commit -m "refactor(io): delegate MERRA-2 staging to shared Earthdata core"
```

---

### Task 3: CERES collections table

**Files:**
- Create: `davinci_monet/io/download/ceres.py` (table + resolve/dest_dir only in this task)
- Test: `davinci_monet/tests/test_download_ceres.py`

- [ ] **Step 3.1: Write the failing tests**

Create `davinci_monet/tests/test_download_ceres.py`:

```python
"""Offline unit tests for the CERES staging helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from davinci_monet.io.download import ceres, earthdata


@pytest.mark.parametrize(
    "collection, short_name, version, subpath",
    [
        ("ssf_terra-fm1", "CER_SSF_Terra-FM1-MODIS", "Edition4A", "CERES/SSF/Terra-FM1"),
        ("ssf_aqua-fm3", "CER_SSF_Aqua-FM3-MODIS", "Edition4A", "CERES/SSF/Aqua-FM3"),
        ("ssf_npp-fm5", "CER_SSF_NPP-FM5-VIIRS", "Edition2A", "CERES/SSF/NPP-FM5"),
        ("ssf_noaa20-fm6", "CER_SSF_NOAA20-FM6-VIIRS", "Edition1C", "CERES/SSF/NOAA20-FM6"),
        ("ebaf", "CERES_EBAF", "Edition4.2.1", "CERES/EBAF"),
        (
            "syn1deg_month",
            "CER_SYN1deg-Month_Terra-Aqua-NOAA20",
            "Edition4B",
            "CERES/SYN1deg/month",
        ),
        ("syn1deg_day", "CER_SYN1deg-Day_Terra-Aqua-NOAA20", "Edition4B", "CERES/SYN1deg/day"),
        (
            "syn1deg_hour",
            "CER_SYN1deg-1Hour_Terra-Aqua-NOAA20",
            "Edition4B",
            "CERES/SYN1deg/hour",
        ),
    ],
)
def test_all_collections_resolve(
    collection: str, short_name: str, version: str, subpath: str
) -> None:
    spec = ceres.resolve_collection(collection)
    assert spec.short_name == short_name
    assert spec.version == version
    assert spec.subpath == Path(subpath)


def test_collection_table_is_exactly_the_documented_set() -> None:
    assert set(ceres.CERES_COLLECTIONS) == {
        "ssf_terra-fm1",
        "ssf_aqua-fm3",
        "ssf_npp-fm5",
        "ssf_noaa20-fm6",
        "ebaf",
        "syn1deg_month",
        "syn1deg_day",
        "syn1deg_hour",
    }


def test_every_collection_is_under_ceres_root() -> None:
    for spec in ceres.CERES_COLLECTIONS.values():
        assert spec.subpath.parts[0] == "CERES"
        assert spec.version is not None  # editions share short-names in CMR


def test_unknown_collection_raises_with_helpful_message() -> None:
    with pytest.raises(KeyError) as exc:
        ceres.resolve_collection("not_a_collection")
    assert "ebaf" in str(exc.value)


def test_destdir_joins_root_and_subpath() -> None:
    dest = ceres.dest_dir("ebaf", root="/Volumes/Io")
    assert dest == Path("/Volumes/Io/CERES/EBAF")
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_ceres.py -v
```

Expected: FAIL — `ImportError: cannot import name 'ceres'` (module does not exist yet)

- [ ] **Step 3.3: Implement the table**

Create `davinci_monet/io/download/ceres.py`:

```python
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
    "ssf_npp-fm5": CollectionSpec(
        "CER_SSF_NPP-FM5-VIIRS", Path("CERES/SSF/NPP-FM5"), "Edition2A"
    ),
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
```

(`argparse`, `dataclass`, `Sequence`, and the `earthdata` module import are used by Tasks 4-5; if the linter flags them as unused at this point, that resolves in the next task.)

- [ ] **Step 3.4: Run tests to verify they pass**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_ceres.py -v
```

Expected: 12 passed (8 parametrized + 4 others)

- [ ] **Step 3.5: Commit**

```bash
git add davinci_monet/io/download/ceres.py davinci_monet/tests/test_download_ceres.py
git commit -m "feat(io): CERES collections table (SSF, EBAF, SYN1deg)"
```

---

### Task 4: `stage_ceres()` — EBAF temporal rules + size-aware dry-run

**Files:**
- Modify: `davinci_monet/io/download/ceres.py` (append below `dest_dir`)
- Test: `davinci_monet/tests/test_download_ceres.py` (append)

- [ ] **Step 4.1: Write the failing tests**

Append to `davinci_monet/tests/test_download_ceres.py`:

```python
class _FakeGranule:
    """Mimics earthaccess.DataGranule's size() -> MB accessor."""

    def __init__(self, mb: float) -> None:
        self._mb = mb

    def size(self) -> float:
        return self._mb


def test_stage_ceres_dry_run_reports_count_and_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    monkeypatch.setattr(earthdata, "_login", lambda: calls.setdefault("login", True))

    def _fake_search(
        short_name: str,
        temporal: tuple[str, str] | None = None,
        version: str | None = None,
    ) -> list[_FakeGranule]:
        calls["search"] = (short_name, temporal, version)
        return [_FakeGranule(60.0), _FakeGranule(70.0)]

    monkeypatch.setattr(earthdata, "_search", _fake_search)
    monkeypatch.setattr(
        earthdata, "_download", lambda *a: pytest.fail("download called in dry run")
    )

    report = ceres.stage_ceres(
        "ssf_aqua-fm3", "2023-07-01", "2023-07-02", root=tmp_path, dry_run=True
    )

    assert calls["login"] is True
    assert calls["search"] == (
        "CER_SSF_Aqua-FM3-MODIS",
        ("2023-07-01", "2023-07-02"),
        "Edition4A",
    )
    assert isinstance(report, ceres.DryRunReport)
    assert report.granules == 2
    assert report.total_mb == pytest.approx(130.0)


def test_stage_ceres_ebaf_searches_without_temporal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(earthdata, "_login", lambda: None)
    seen: dict[str, Any] = {}

    def _fake_search(
        short_name: str,
        temporal: tuple[str, str] | None = None,
        version: str | None = None,
    ) -> list[_FakeGranule]:
        seen["args"] = (short_name, temporal, version)
        return [_FakeGranule(2000.0)]

    monkeypatch.setattr(earthdata, "_search", _fake_search)

    report = ceres.stage_ceres("ebaf", root=tmp_path, dry_run=True)

    assert seen["args"] == ("CERES_EBAF", None, "Edition4.2.1")
    assert isinstance(report, ceres.DryRunReport)
    assert report.granules == 1


def test_stage_ceres_requires_temporal_for_non_ebaf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(earthdata, "_login", lambda: pytest.fail("network touched"))

    with pytest.raises(ValueError, match="start and end are required"):
        ceres.stage_ceres("ssf_aqua-fm3", root=tmp_path, dry_run=True)


def test_stage_ceres_rejects_half_open_temporal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(earthdata, "_login", lambda: pytest.fail("network touched"))

    with pytest.raises(ValueError, match="both start and end"):
        ceres.stage_ceres("ebaf", start="2023-07-01", root=tmp_path, dry_run=True)


def test_stage_ceres_downloads_into_dest_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(earthdata, "_login", lambda: None)
    monkeypatch.setattr(
        earthdata,
        "_search",
        lambda short_name, temporal=None, version=None: ["g1"],
    )

    seen: dict[str, Any] = {}

    def _fake_download(results: Any, dest: str) -> list[Path]:
        seen["results"] = list(results)
        seen["dest"] = dest
        return [Path(dest) / "CER_SSF_Aqua-FM3-MODIS_Edition4A_407405.2023070100.hdf"]

    monkeypatch.setattr(earthdata, "_download", _fake_download)

    out = ceres.stage_ceres("ssf_aqua-fm3", "2023-07-01", "2023-07-01", root=tmp_path)

    expected_dir = tmp_path / "CERES/SSF/Aqua-FM3"
    assert expected_dir.is_dir()  # created before download
    assert seen["results"] == ["g1"]
    assert Path(seen["dest"]) == expected_dir
    assert isinstance(out, list)
    assert out[0].name == "CER_SSF_Aqua-FM3-MODIS_Edition4A_407405.2023070100.hdf"
```

- [ ] **Step 4.2: Run tests to verify they fail**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_ceres.py -v
```

Expected: the 5 new tests FAIL with `AttributeError: module ... has no attribute 'stage_ceres'` (and `DryRunReport`); the 14 Task 3 tests still pass.

- [ ] **Step 4.3: Implement `DryRunReport` and `stage_ceres`**

Append to `davinci_monet/io/download/ceres.py` (below `dest_dir`):

```python
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

    result = earthdata.stage_collection(
        spec, temporal, Path(root) / spec.subpath, dry_run=dry_run
    )
    if dry_run:
        return DryRunReport(
            granules=len(result),
            total_mb=sum(earthdata.granule_size_mb(g) for g in result),
        )
    return [Path(p) for p in result]
```

- [ ] **Step 4.4: Run tests to verify they pass**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_ceres.py -v
```

Expected: 17 passed

- [ ] **Step 4.5: Commit**

```bash
git add davinci_monet/io/download/ceres.py davinci_monet/tests/test_download_ceres.py
git commit -m "feat(io): stage_ceres with EBAF temporal rules and size-aware dry-run"
```

---

### Task 5: CLI, console script, package re-exports

**Files:**
- Modify: `davinci_monet/io/download/ceres.py` (append `main` + `__main__` block)
- Modify: `davinci_monet/io/download/__init__.py`
- Modify: `pyproject.toml:74` (add script entry after `davinci-stage-merra2`)
- Test: `davinci_monet/tests/test_download_ceres.py` (append)

- [ ] **Step 5.1: Write the failing tests**

Append to `davinci_monet/tests/test_download_ceres.py`:

```python
def test_main_dry_run_prints_count_and_size(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def _fake_stage(
        collection: str,
        start: str | None = None,
        end: str | None = None,
        *,
        root: Any,
        dry_run: bool,
    ) -> ceres.DryRunReport:
        captured.update(
            collection=collection, start=start, end=end, root=root, dry_run=dry_run
        )
        return ceres.DryRunReport(granules=48, total_mb=3100.0)

    monkeypatch.setattr(ceres, "stage_ceres", _fake_stage)

    rc = ceres.main(
        [
            "--collection",
            "ssf_aqua-fm3",
            "--start",
            "2023-07-01",
            "--end",
            "2023-07-02",
            "--root",
            str(tmp_path),
            "--dry-run",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert captured["collection"] == "ssf_aqua-fm3"
    assert captured["dry_run"] is True
    assert "48 granule(s)" in out
    assert "GB" in out  # size is part of the dry-run report
    assert "[2023-07-01..2023-07-02]" in out  # temporal window echoed


def test_main_ebaf_needs_no_temporal_flags(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    def _fake_stage(
        collection: str,
        start: str | None = None,
        end: str | None = None,
        *,
        root: Any,
        dry_run: bool,
    ) -> list[Path]:
        assert start is None and end is None
        return [Path(root) / "CERES/EBAF/CERES_EBAF_Edition4.2.1_200003-202512.nc"]

    monkeypatch.setattr(ceres, "stage_ceres", _fake_stage)

    rc = ceres.main(["--collection", "ebaf", "--root", str(tmp_path)])

    assert rc == 0
    assert "Staged 1 file(s)" in capsys.readouterr().out


def test_download_package_reexports_ceres() -> None:
    from davinci_monet.io import download

    assert download.stage_ceres is ceres.stage_ceres
    assert "ebaf" in download.CERES_COLLECTIONS


def test_main_missing_temporal_exits_cleanly(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as exc:
        ceres.main(["--collection", "ssf_aqua-fm3", "--root", str(tmp_path), "--dry-run"])

    assert exc.value.code == 2
    assert "start and end are required" in capsys.readouterr().err
```

- [ ] **Step 5.2: Run tests to verify they fail**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_ceres.py -v
```

Expected: 4 new tests FAIL (`no attribute 'main'`, `no attribute 'stage_ceres'` on the package); 17 prior tests pass.

- [ ] **Step 5.3: Implement `main` and the re-exports**

Append to `davinci_monet/io/download/ceres.py`:

```python
def main(argv: Sequence[str] | None = None) -> int:
    """CLI: stage a CERES collection. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="davinci-stage-ceres",
        description="Stage CERES radiation products (SSF, EBAF, SYN1deg) to local disk.",
    )
    parser.add_argument("--collection", required=True, choices=sorted(CERES_COLLECTIONS))
    parser.add_argument("--start", help="ISO start, e.g. 2023-07-01 (optional for ebaf)")
    parser.add_argument("--end", help="ISO end, e.g. 2023-07-02 (optional for ebaf)")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Staging root dir")
    parser.add_argument(
        "--dry-run", action="store_true", help="Search only; report granule count and size"
    )
    ns = parser.parse_args(argv)

    try:
        result = stage_ceres(ns.collection, ns.start, ns.end, root=ns.root, dry_run=ns.dry_run)
    except ValueError as exc:
        parser.error(str(exc))
    if isinstance(result, DryRunReport):
        gb = result.total_mb / 1024.0
        window = f" [{ns.start}..{ns.end}]" if ns.start else ""
        print(
            f"{result.granules} granule(s), ~{result.total_mb:,.0f} MB (~{gb:,.1f} GB) "
            f"for {ns.collection}{window}"
        )
    else:
        print(f"Staged {len(result)} file(s) to {dest_dir(ns.collection, ns.root)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

Replace `davinci_monet/io/download/__init__.py` content:

```python
"""Optional data-staging helpers for reanalysis and EOS data sources.

Public API is re-exported from the product modules (``merra2``, ``ceres``).
These helpers require optional extras (``pip install -e ".[reanalysis]"``)
that are imported lazily, so importing this package never requires the
network or ``earthaccess`` to be installed.
"""

from davinci_monet.io.download.ceres import CERES_COLLECTIONS, stage_ceres
from davinci_monet.io.download.merra2 import MERRA2_COLLECTIONS, stage_merra2

__all__ = ["CERES_COLLECTIONS", "MERRA2_COLLECTIONS", "stage_ceres", "stage_merra2"]
```

In `pyproject.toml`, add one line to `[project.scripts]` directly after the `davinci-stage-merra2` entry (line 74):

```toml
davinci-stage-ceres = "davinci_monet.io.download.ceres:main"
```

- [ ] **Step 5.4: Run tests to verify they pass**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_ceres.py -v
```

Expected: 21 passed

- [ ] **Step 5.5: Commit**

```bash
git add davinci_monet/io/download/ceres.py davinci_monet/io/download/__init__.py \
        davinci_monet/tests/test_download_ceres.py pyproject.toml
git commit -m "feat(cli): davinci-stage-ceres console script + package re-exports"
```

---

### Task 6: Full gates

**Files:** none new — verification only (plus any formatting diffs black/isort produce)

- [ ] **Step 6.1: Run the full test suite**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest
```

Expected: all tests pass (1,262 pre-existing + 21 CERES + 6 earthdata; some suites skip without real data — skips are fine, failures are not).

- [ ] **Step 6.2: Type-check**

```bash
mypy davinci_monet
```

Expected: `Success: no issues found`. Likely friction points if it complains: the `search` lambda in `merra2.stage_merra2` (annotate via a small named function if needed) and `Sequence[Any]` vs `list[Path]` returns in `stage_collection` callers.

- [ ] **Step 6.3: Format and import-sort**

```bash
black davinci_monet && isort davinci_monet
git diff --stat
```

Expected: no reformatting of the new files (they were written to style); if black/isort touch anything, re-run pytest before committing.

- [ ] **Step 6.4: Commit any formatting fallout**

```bash
git add -A davinci_monet
git commit -m "style: black/isort on CERES downloader modules"
```

(Skip this commit if `git diff` was empty.)

- [ ] **Step 6.5: Manual smoke (optional, requires network + Earthdata login)**

```bash
davinci-stage-ceres --collection ebaf --dry-run
davinci-stage-ceres --collection ssf_aqua-fm3 --start 2023-07-01 --end 2023-07-02 --dry-run
davinci-stage-ceres --collection syn1deg_month --start 2003-07-01 --end 2003-08-01 --dry-run
```

Expected: EBAF reports 1 granule ~2,000 MB; SSF reports ~24-48 granules at ~60-70 MB each; SYN1deg-Month reports ~50-100 MB total — note CMR may return more than one granule per month for the same collection (different production-strategy codes in the filename), so counts above the naive expectation are normal. Use historical dates (2023 or earlier) — CERES lags; do NOT use 2026 dates. Report results to the user before any non-dry-run staging.
