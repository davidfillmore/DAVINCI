# MERRA-2 Downloader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional `earthaccess`-based helper that stages MERRA-2 aerosol collections onto the Io drive, runnable as a function and a small CLI.

**Architecture:** A new `davinci_monet/io/download/` subpackage with a `merra2` module. A collection registry maps friendly names to Earthdata short-names and on-disk subpaths. Network access is funneled through three thin, monkeypatchable wrappers (`_login`, `_search`, `_download`) — mirroring the AI module's `_build_client` pattern — so unit tests run fully offline. `earthaccess` is an optional `[reanalysis]` extra, imported lazily inside the wrappers.

**Tech Stack:** Python 3.11+, `earthaccess`, `pytest`, `argparse`. No new pipeline stage — staging is a deliberate, separate step.

**Scope:** This is Phase 0 (scaffolding) + Phase 1 (MERRA-2 downloader) of the reanalysis spec (`docs/superpowers/specs/2026-06-10-reanalysis-sources-design.md`). Readers and ERA5/CAMS get their own plans.

---

## File Structure

- Create: `davinci_monet/io/download/__init__.py` — package; re-exports `stage_merra2`, `MERRA2_COLLECTIONS`.
- Create: `davinci_monet/io/download/merra2.py` — registry, wrappers, `stage_merra2`, CLI `main`.
- Create: `davinci_monet/tests/test_download_merra2.py` — offline unit tests (mocked wrappers).
- Modify: `pyproject.toml` — add `[project.optional-dependencies]` `reanalysis` extra and include it in `dev`; add console script.

---

### Task 1: Optional `reanalysis` extra + console script

**Files:**
- Modify: `pyproject.toml` (`[project.optional-dependencies]` and `[project.scripts]`)

- [ ] **Step 1: Add the extra and console script**

In `pyproject.toml`, under `[project.optional-dependencies]` add a `reanalysis` group and append its dep to `dev`:

```toml
reanalysis = [
    "earthaccess>=0.11",
]
```

Append `"earthaccess>=0.11",` to the existing `dev = [ ... ]` list.

Under `[project.scripts]`, add the staging entry beside the existing one:

```toml
[project.scripts]
davinci-monet = "davinci_monet.cli.app:app"
davinci-stage-merra2 = "davinci_monet.io.download.merra2:main"
```

- [ ] **Step 2: Verify it parses**

Run: `python -c "import tomllib,pathlib; tomllib.loads(pathlib.Path('pyproject.toml').read_text()); print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add optional [reanalysis] extra (earthaccess) + stage-merra2 script"
```

---

### Task 2: Collection registry + dest-path resolution

**Files:**
- Create: `davinci_monet/io/download/__init__.py`
- Create: `davinci_monet/io/download/merra2.py`
- Test: `davinci_monet/tests/test_download_merra2.py`

- [ ] **Step 1: Write the failing test**

Create `davinci_monet/tests/test_download_merra2.py`:

```python
"""Offline unit tests for the MERRA-2 staging helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from davinci_monet.io.download import merra2


def test_known_collection_resolves_shortname_and_destdir() -> None:
    spec = merra2.resolve_collection("tavgM_2d_aer_Nx")
    assert spec.short_name == "M2TMNXAER"
    assert spec.subpath == Path("MERRA2_tavgM/aer_Nx")


def test_destdir_joins_root_and_subpath() -> None:
    dest = merra2.dest_dir("tavgM_2d_aer_Nx", root="/Volumes/Io")
    assert dest == Path("/Volumes/Io/MERRA2_tavgM/aer_Nx")


def test_unknown_collection_raises_with_helpful_message() -> None:
    with pytest.raises(KeyError) as exc:
        merra2.resolve_collection("not_a_collection")
    assert "tavgM_2d_aer_Nx" in str(exc.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_merra2.py -q`
Expected: FAIL — `ModuleNotFoundError: davinci_monet.io.download`

- [ ] **Step 3: Write minimal implementation**

Create `davinci_monet/io/download/__init__.py`:

```python
"""Optional data-staging helpers for reanalysis sources."""

from davinci_monet.io.download.merra2 import MERRA2_COLLECTIONS, stage_merra2

__all__ = ["MERRA2_COLLECTIONS", "stage_merra2"]
```

Create `davinci_monet/io/download/merra2.py`:

```python
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
        raise KeyError(f"Unknown MERRA-2 collection {collection!r}. Known: {known}")


def dest_dir(collection: str, root: str | Path = DEFAULT_ROOT) -> Path:
    """Return the staging directory for ``collection`` under ``root``."""
    return Path(root) / resolve_collection(collection).subpath
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_merra2.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/io/download/__init__.py davinci_monet/io/download/merra2.py davinci_monet/tests/test_download_merra2.py
git commit -m "feat(download): MERRA-2 collection registry + dest-path resolution"
```

---

### Task 3: `stage_merra2` orchestration (offline-tested)

**Files:**
- Modify: `davinci_monet/io/download/merra2.py`
- Test: `davinci_monet/tests/test_download_merra2.py`

- [ ] **Step 1: Write the failing test**

Append to `davinci_monet/tests/test_download_merra2.py`:

```python
def test_stage_merra2_dry_run_searches_but_does_not_download(monkeypatch, tmp_path) -> None:
    calls: dict[str, Any] = {}

    monkeypatch.setattr(merra2, "_login", lambda: calls.setdefault("login", True))

    def _fake_search(short_name, temporal):
        calls["search"] = (short_name, temporal)
        return ["granule-1", "granule-2"]

    def _fake_download(results, dest):  # pragma: no cover - must NOT run
        calls["download"] = True
        return []

    monkeypatch.setattr(merra2, "_search", _fake_search)
    monkeypatch.setattr(merra2, "_download", _fake_download)

    planned = merra2.stage_merra2(
        "tavgM_2d_aer_Nx", "2003-01", "2003-03", root=tmp_path, dry_run=True
    )

    assert calls["login"] is True
    assert calls["search"] == ("M2TMNXAER", ("2003-01", "2003-03"))
    assert "download" not in calls
    assert planned == 2  # number of granules found


def test_stage_merra2_downloads_into_dest_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(merra2, "_login", lambda: None)
    monkeypatch.setattr(merra2, "_search", lambda short_name, temporal: ["g1"])

    seen: dict[str, Any] = {}

    def _fake_download(results, dest):
        seen["results"] = results
        seen["dest"] = dest
        return [Path(dest) / "MERRA2_400.tavgM_2d_aer_Nx.200301.nc4"]

    monkeypatch.setattr(merra2, "_download", _fake_download)

    out = merra2.stage_merra2("tavgM_2d_aer_Nx", "2003-01", "2003-01", root=tmp_path)

    expected_dir = tmp_path / "MERRA2_tavgM/aer_Nx"
    assert expected_dir.is_dir()  # created before download
    assert seen["results"] == ["g1"]
    assert Path(seen["dest"]) == expected_dir
    assert out[0].name == "MERRA2_400.tavgM_2d_aer_Nx.200301.nc4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_merra2.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute '_login'` / `stage_merra2`

- [ ] **Step 3: Write minimal implementation**

Append to `davinci_monet/io/download/merra2.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_merra2.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/io/download/merra2.py davinci_monet/tests/test_download_merra2.py
git commit -m "feat(download): stage_merra2 orchestration (login/search/download)"
```

---

### Task 4: CLI entry point

**Files:**
- Modify: `davinci_monet/io/download/merra2.py`
- Test: `davinci_monet/tests/test_download_merra2.py`

- [ ] **Step 1: Write the failing test**

Append to `davinci_monet/tests/test_download_merra2.py`:

```python
def test_main_dry_run_invokes_stage(monkeypatch, capsys, tmp_path) -> None:
    captured: dict[str, Any] = {}

    def _fake_stage(collection, start, end, *, root, dry_run):
        captured.update(
            collection=collection, start=start, end=end, root=root, dry_run=dry_run
        )
        return 7

    monkeypatch.setattr(merra2, "stage_merra2", _fake_stage)

    rc = merra2.main(
        [
            "--collection", "tavgM_2d_aer_Nx",
            "--start", "2003-01",
            "--end", "2003-03",
            "--root", str(tmp_path),
            "--dry-run",
        ]
    )

    assert rc == 0
    assert captured["collection"] == "tavgM_2d_aer_Nx"
    assert captured["dry_run"] is True
    assert captured["root"] == str(tmp_path)
    assert "7" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_merra2.py::test_main_dry_run_invokes_stage -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'main'`

- [ ] **Step 3: Write minimal implementation**

Append to `davinci_monet/io/download/merra2.py`:

```python
def main(argv: Sequence[str] | None = None) -> int:
    """CLI: stage a MERRA-2 collection. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="davinci-stage-merra2",
        description="Stage MERRA-2 aerosol collections to local disk.",
    )
    parser.add_argument(
        "--collection", required=True, choices=sorted(MERRA2_COLLECTIONS)
    )
    parser.add_argument("--start", required=True, help="ISO start, e.g. 2003-01")
    parser.add_argument("--end", required=True, help="ISO end, e.g. 2003-03")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Staging root dir")
    parser.add_argument(
        "--dry-run", action="store_true", help="Search only; do not download"
    )
    ns = parser.parse_args(argv)

    result = stage_merra2(
        ns.collection, ns.start, ns.end, root=ns.root, dry_run=ns.dry_run
    )
    if ns.dry_run:
        print(f"{result} granule(s) found for {ns.collection} [{ns.start}..{ns.end}]")
    else:
        print(f"Staged {len(result)} file(s) to {Path(ns.root) / resolve_collection(ns.collection).subpath}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_merra2.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/io/download/merra2.py davinci_monet/tests/test_download_merra2.py
git commit -m "feat(download): davinci-stage-merra2 CLI entry point"
```

---

### Task 5: Gate sweep + manual staging note

**Files:**
- (no code) verification only

- [ ] **Step 1: Full local gate sweep**

Run (in the `davinci` conda env):
```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_merra2.py -q
mypy davinci_monet/io/download
black --check davinci_monet/io/download davinci_monet/tests/test_download_merra2.py
isort --check davinci_monet/io/download davinci_monet/tests/test_download_merra2.py
```
Expected: pytest all pass; mypy clean; black/isort report no changes.

- [ ] **Step 2: Manual smoke (real network, optional, not a test)**

After `pip install -e ".[reanalysis]"` and an Earthdata `.netrc`:
```bash
davinci-stage-merra2 --collection tavgM_2d_aer_Nx --start 2003-01 --end 2003-01 --dry-run
```
Expected: prints a non-zero granule count. (Drop `--dry-run` to stage into `/Volumes/Io/MERRA2_tavgM/aer_Nx`.)

- [ ] **Step 3: No commit** — verification only.

---

## Self-Review

- **Spec coverage:** Implements spec Phase 0 (download package, `[reanalysis]` extra) and Phase 1 (MERRA-2 `earthaccess` downloader, Io layout). Readers, ERA5, CAMS are explicitly out of scope (separate plans) — matches the spec's per-phase plan note.
- **Placeholder scan:** No TBD/TODO; every code step shows complete code.
- **Type consistency:** `resolve_collection` → `CollectionSpec` (`.short_name`, `.subpath`) used consistently in `dest_dir`, `stage_merra2`, and `main`. Wrapper names `_login`/`_search`/`_download` match between implementation and the monkeypatched tests. `stage_merra2` signature (`collection, start, end, *, root, dry_run`) matches every caller (tests + `main`).
- **Offline guarantee:** all network funneled through three wrappers that lazily import `earthaccess`; every unit test monkeypatches them, so the suite needs neither `earthaccess` installed nor a network.
