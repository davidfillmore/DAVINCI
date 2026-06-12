# ERA5 Downloader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional `cdsapi`-based helper that stages ERA5 single-level and pressure-level NetCDF from the Copernicus CDS onto the Io drive, runnable as a function and a small CLI.

**Architecture:** Extend `davinci_monet/io/download/` with an `era5` module. A dataset registry maps friendly names to CDS dataset ids and on-disk subpaths; a friendly→CDS variable-name map and a date-range→request builder are pure (offline-testable) functions. Network access is funneled through two monkeypatchable wrappers (`_client`, `_retrieve`) — mirroring the MERRA-2 downloader's `_login`/`_search`/`_download` — so unit tests run fully offline. `cdsapi` is added to the optional `[reanalysis]` extra and imported lazily.

**Tech Stack:** Python 3.11+, `cdsapi`, `pytest`, `argparse`. NetCDF only (no GRIB). No new pipeline stage — staging is a deliberate, separate step.

**Scope:** The *downloader* half of spec Phase 3 (`docs/superpowers/specs/2026-06-10-reanalysis-sources-design.md`). The ERA5 *reader* is a separate plan (`2026-06-10-era5-reader.md`). This downloader stages the data that the reader's real-data smoke test (currently skipped) will then validate.

---

## Background: how CDS differs from GES DISC

- **Auth:** `cdsapi.Client()` reads `~/.cdsapirc` (url + Personal Access Token) or `CDSAPI_URL`/`CDSAPI_KEY` env vars. No interactive flow.
- **No cheap granule count:** unlike `earthaccess.search_data`, CDS has no listing call. So `--dry-run` returns the **request dict that would be submitted** (the analog of "what will happen"), without retrieving.
- **Async/queued retrieve:** `client.retrieve(dataset_id, request, target)` blocks until the queued job finishes, then writes `target`. cdsapi handles polling internally.
- **Request shape (current CDS API):** lists of `year`/`month`/`day`/`time`, `variable` (CDS *long* names), `data_format: "netcdf"`, `download_format: "unarchived"`, optional `pressure_level` and `area`.
- **Cartesian dates:** CDS expands `year × month × day` as a product. To avoid over-requesting nonexistent dates across month boundaries, this downloader restricts a daily request to a **single calendar month** (raises a clear error otherwise; call once per month for longer spans).
- **Variable names:** CDS requests use long names (`2m_temperature`) while the NetCDF output uses short names (`t2m`). The downloader maps friendly short names → CDS long names, passing unknown names through unchanged.

---

## File Structure

- Create: `davinci_monet/io/download/era5.py` — dataset registry, variable map, request builder, wrappers, `stage_era5`, CLI `main`.
- Modify: `davinci_monet/io/download/__init__.py` — re-export `stage_era5`, `ERA5_DATASETS`.
- Create: `davinci_monet/tests/test_download_era5.py` — offline unit tests (mocked wrappers).
- Modify: `pyproject.toml` — add `cdsapi` to the `[reanalysis]` extra (and to `dev`); add the `davinci-stage-era5` console script.

---

### Task 1: `cdsapi` dependency + console script

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add cdsapi to the extras and a console script**

In `pyproject.toml`, change the `reanalysis` extra from:

```toml
reanalysis = [
    "earthaccess>=0.11",
]
```

to:

```toml
reanalysis = [
    "earthaccess>=0.11",
    "cdsapi>=0.7",
]
```

Append `"cdsapi>=0.7",` to the `dev = [ ... ]` list (after the existing `"earthaccess>=0.11",`).

Under `[project.scripts]`, add beside the existing entries:

```toml
davinci-stage-era5 = "davinci_monet.io.download.era5:main"
```

- [ ] **Step 2: Verify it parses**

Run: `python -c "import tomllib,pathlib; tomllib.loads(pathlib.Path('pyproject.toml').read_text()); print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add cdsapi to [reanalysis] extra + stage-era5 script"
```

---

### Task 2: Dataset registry, variable map, dest-path resolution

**Files:**
- Create: `davinci_monet/io/download/era5.py`
- Modify: `davinci_monet/io/download/__init__.py`
- Test: `davinci_monet/tests/test_download_era5.py`

- [ ] **Step 1: Write the failing test**

Create `davinci_monet/tests/test_download_era5.py`:

```python
"""Offline unit tests for the ERA5 staging helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from davinci_monet.io.download import era5


def test_known_dataset_resolves_id_and_destdir() -> None:
    spec = era5.resolve_dataset("single_levels")
    assert spec.dataset_id == "reanalysis-era5-single-levels"
    assert spec.subpath == Path("ERA5/single")
    assert spec.has_levels is False

    pl = era5.resolve_dataset("pressure_levels")
    assert pl.dataset_id == "reanalysis-era5-pressure-levels"
    assert pl.subpath == Path("ERA5/pressure")
    assert pl.has_levels is True


def test_destdir_joins_root_and_subpath() -> None:
    assert era5.dest_dir("single_levels", root="/Volumes/Io") == Path(
        "/Volumes/Io/ERA5/single"
    )


def test_unknown_dataset_raises_with_helpful_message() -> None:
    with pytest.raises(KeyError) as exc:
        era5.resolve_dataset("nope")
    assert "single_levels" in str(exc.value)


def test_variable_mapping_short_to_cds_long() -> None:
    assert era5.cds_variable("t2m") == "2m_temperature"
    assert era5.cds_variable("z") == "geopotential"
    # Unknown / already-long names pass through unchanged.
    assert era5.cds_variable("2m_temperature") == "2m_temperature"
    assert era5.cds_variable("some_custom_var") == "some_custom_var"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_era5.py -q`
Expected: FAIL — `ImportError`/`AttributeError` (module/functions absent).

- [ ] **Step 3: Write minimal implementation**

Create `davinci_monet/io/download/era5.py`:

```python
"""Stage ERA5 single-level and pressure-level NetCDF from the Copernicus CDS.

Network access is isolated in ``_client``/``_retrieve`` so the rest of the
module (and its tests) run offline. ``cdsapi`` is an optional dependency
(``pip install -e ".[reanalysis]"``) imported lazily.
"""

from __future__ import annotations

import argparse
import calendar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

DEFAULT_ROOT = "/Volumes/Io"

# All 24 hourly analysis times, "HH:00".
ALL_HOURS: list[str] = [f"{h:02d}:00" for h in range(24)]


@dataclass(frozen=True)
class DatasetSpec:
    """An ERA5 CDS dataset: dataset id, on-disk subpath, has-levels flag."""

    dataset_id: str
    subpath: Path
    has_levels: bool


ERA5_DATASETS: dict[str, DatasetSpec] = {
    "single_levels": DatasetSpec(
        "reanalysis-era5-single-levels", Path("ERA5/single"), has_levels=False
    ),
    "pressure_levels": DatasetSpec(
        "reanalysis-era5-pressure-levels", Path("ERA5/pressure"), has_levels=True
    ),
}

# Friendly short name -> CDS long variable name. Unknown names pass through.
ERA5_VARIABLES: dict[str, str] = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "sp": "surface_pressure",
    "tp": "total_precipitation",
    "tcwv": "total_column_water_vapour",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "z": "geopotential",
    "q": "specific_humidity",
    "r": "relative_humidity",
}


def resolve_dataset(dataset: str) -> DatasetSpec:
    """Look up a dataset spec, raising a helpful KeyError if unknown."""
    try:
        return ERA5_DATASETS[dataset]
    except KeyError:
        known = ", ".join(sorted(ERA5_DATASETS))
        raise KeyError(
            f"Unknown ERA5 dataset {dataset!r}. Known: {known}"
        ) from None


def dest_dir(dataset: str, root: str | Path = DEFAULT_ROOT) -> Path:
    """Return the staging directory for ``dataset`` under ``root``."""
    return Path(root) / resolve_dataset(dataset).subpath


def cds_variable(name: str) -> str:
    """Map a friendly short name to its CDS long name (pass through if unknown)."""
    return ERA5_VARIABLES.get(name, name)
```

Modify `davinci_monet/io/download/__init__.py` — add to the existing re-export block:

```python
from davinci_monet.io.download.era5 import ERA5_DATASETS, stage_era5
```

and extend `__all__` to include `"ERA5_DATASETS"` and `"stage_era5"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_era5.py -q`
Expected: FAIL on the `__init__` import line until `stage_era5` exists (Task 4). To keep this task green in isolation, add `stage_era5` re-export in Task 4; for now import only what exists:

In `davinci_monet/io/download/__init__.py` add just:

```python
from davinci_monet.io.download.era5 import ERA5_DATASETS
```

(extend `__all__` with `"ERA5_DATASETS"`). The `stage_era5` re-export is added in Task 4 once defined.

Re-run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_era5.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/io/download/era5.py davinci_monet/io/download/__init__.py davinci_monet/tests/test_download_era5.py
git commit -m "feat(download): ERA5 dataset registry + variable map + dest-path resolution"
```

---

### Task 3: Date-range -> CDS request builder

**Files:**
- Modify: `davinci_monet/io/download/era5.py`
- Test: `davinci_monet/tests/test_download_era5.py`

- [ ] **Step 1: Write the failing test**

Append to `davinci_monet/tests/test_download_era5.py`:

```python
def test_expand_dates_single_month() -> None:
    years, months, days = era5.expand_dates("2026-04-01", "2026-04-03")
    assert years == ["2026"]
    assert months == ["04"]
    assert days == ["01", "02", "03"]


def test_expand_dates_rejects_cross_month() -> None:
    with pytest.raises(ValueError) as exc:
        era5.expand_dates("2026-04-28", "2026-05-02")
    assert "single calendar month" in str(exc.value)


def test_build_request_single_level() -> None:
    req = era5.build_request(
        "single_levels", ["t2m", "msl"], "2026-04-01", "2026-04-02"
    )
    assert req["variable"] == ["2m_temperature", "mean_sea_level_pressure"]
    assert req["year"] == ["2026"]
    assert req["month"] == ["04"]
    assert req["day"] == ["01", "02"]
    assert req["time"] == era5.ALL_HOURS
    assert req["data_format"] == "netcdf"
    assert req["download_format"] == "unarchived"
    assert "pressure_level" not in req  # single-level dataset


def test_build_request_pressure_level_requires_levels() -> None:
    with pytest.raises(ValueError) as exc:
        era5.build_request("pressure_levels", ["t"], "2026-04-01", "2026-04-01")
    assert "pressure_levels" in str(exc.value)

    req = era5.build_request(
        "pressure_levels",
        ["t", "z"],
        "2026-04-01",
        "2026-04-01",
        pressure_levels=[500, 850, 1000],
    )
    assert req["variable"] == ["temperature", "geopotential"]
    assert req["pressure_level"] == ["500", "850", "1000"]


def test_build_request_accepts_area_and_times() -> None:
    req = era5.build_request(
        "single_levels",
        ["t2m"],
        "2026-04-01",
        "2026-04-01",
        times=["00:00", "12:00"],
        area=[90, -180, -90, 180],
    )
    assert req["time"] == ["00:00", "12:00"]
    assert req["area"] == [90, -180, -90, 180]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_era5.py -q`
Expected: FAIL — `expand_dates`/`build_request` absent.

- [ ] **Step 3: Write minimal implementation**

Append to `davinci_monet/io/download/era5.py`:

```python
def expand_dates(start: str, end: str) -> tuple[list[str], list[str], list[str]]:
    """Expand an inclusive ``start``..``end`` (YYYY-MM-DD) into CDS lists.

    Restricted to a single calendar month so the CDS year x month x day
    cartesian product does not over-request nonexistent dates. For longer
    spans, call once per month.

    Returns
    -------
    (years, months, days)
        Each a list of zero-padded strings (a single year and month, and the
        inclusive day range).
    """
    sy, sm, sd = (int(p) for p in start.split("-"))
    ey, em, ed = (int(p) for p in end.split("-"))
    if (sy, sm) != (ey, em):
        raise ValueError(
            f"expand_dates spans more than a single calendar month "
            f"({start}..{end}); call once per month."
        )
    days = [f"{d:02d}" for d in range(sd, ed + 1)]
    return [f"{sy}"], [f"{sm:02d}"], days


def build_request(
    dataset: str,
    variables: Sequence[str],
    start: str,
    end: str,
    *,
    pressure_levels: Sequence[int] | None = None,
    times: Sequence[str] | None = None,
    area: Sequence[float] | None = None,
    product_type: str = "reanalysis",
    data_format: str = "netcdf",
) -> dict[str, Any]:
    """Build a CDS retrieve request dict for ``dataset``.

    Raises ValueError if a pressure-level dataset is requested without
    ``pressure_levels``.
    """
    spec = resolve_dataset(dataset)
    if spec.has_levels and not pressure_levels:
        raise ValueError(
            f"dataset {dataset!r} (pressure_levels) requires pressure_levels=[...]"
        )

    years, months, days = expand_dates(start, end)
    req: dict[str, Any] = {
        "product_type": [product_type],
        "variable": [cds_variable(v) for v in variables],
        "year": years,
        "month": months,
        "day": days,
        "time": list(times) if times is not None else ALL_HOURS,
        "data_format": data_format,
        "download_format": "unarchived",
    }
    if spec.has_levels:
        req["pressure_level"] = [str(p) for p in pressure_levels or []]
    if area is not None:
        req["area"] = list(area)
    return req
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_era5.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/io/download/era5.py davinci_monet/tests/test_download_era5.py
git commit -m "feat(download): ERA5 date-range -> CDS request builder (single-month, netcdf)"
```

---

### Task 4: `stage_era5` orchestration (offline-tested)

**Files:**
- Modify: `davinci_monet/io/download/era5.py`
- Modify: `davinci_monet/io/download/__init__.py`
- Test: `davinci_monet/tests/test_download_era5.py`

- [ ] **Step 1: Write the failing test**

Append to `davinci_monet/tests/test_download_era5.py`:

```python
def test_stage_era5_dry_run_builds_request_without_retrieve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        era5, "_client", lambda: calls.setdefault("client", True)
    )

    def _no_retrieve(dataset_id: str, request: Any, target: str) -> None:  # pragma: no cover
        calls["retrieve"] = True

    monkeypatch.setattr(era5, "_retrieve", _no_retrieve)

    req = era5.stage_era5(
        "single_levels",
        ["t2m"],
        "2026-04-01",
        "2026-04-02",
        root=tmp_path,
        dry_run=True,
    )

    assert isinstance(req, dict)
    assert req["variable"] == ["2m_temperature"]
    assert "client" not in calls  # dry-run does not authenticate
    assert "retrieve" not in calls


def test_stage_era5_retrieves_into_dest_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(era5, "_client", lambda: object())
    seen: dict[str, Any] = {}

    def _fake_retrieve(dataset_id: str, request: Any, target: str) -> None:
        seen["dataset_id"] = dataset_id
        seen["request"] = request
        seen["target"] = target
        Path(target).write_bytes(b"\x89HDF\r\n\x1a\n")  # pretend a netcdf landed

    monkeypatch.setattr(era5, "_retrieve", _fake_retrieve)

    out = era5.stage_era5(
        "pressure_levels",
        ["t", "z"],
        "2026-04-01",
        "2026-04-01",
        pressure_levels=[850, 1000],
        root=tmp_path,
    )

    expected_dir = tmp_path / "ERA5/pressure"
    assert expected_dir.is_dir()
    assert seen["dataset_id"] == "reanalysis-era5-pressure-levels"
    assert seen["request"]["pressure_level"] == ["850", "1000"]
    assert Path(seen["target"]).parent == expected_dir
    assert out.exists() and out.suffix == ".nc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_era5.py -q`
Expected: FAIL — `_client`/`_retrieve`/`stage_era5` absent.

- [ ] **Step 3: Write minimal implementation**

Append to `davinci_monet/io/download/era5.py`:

```python
def _client() -> Any:
    """Construct a CDS API client (lazy cdsapi import; reads ~/.cdsapirc)."""
    import cdsapi

    return cdsapi.Client()


def _retrieve(dataset_id: str, request: dict[str, Any], target: str) -> None:
    """Submit a CDS retrieve job and download to ``target`` (blocks)."""
    _client().retrieve(dataset_id, request, target)


def stage_era5(
    dataset: str,
    variables: Sequence[str],
    start: str,
    end: str,
    *,
    pressure_levels: Sequence[int] | None = None,
    times: Sequence[str] | None = None,
    area: Sequence[float] | None = None,
    root: str | Path = DEFAULT_ROOT,
    dry_run: bool = False,
) -> dict[str, Any] | Path:
    """Stage an ERA5 ``dataset`` for ``[start, end]`` (single month) under ``root``.

    Returns the request dict when ``dry_run``; otherwise the staged file path.
    """
    spec = resolve_dataset(dataset)
    request = build_request(
        dataset,
        variables,
        start,
        end,
        pressure_levels=pressure_levels,
        times=times,
        area=area,
    )
    if dry_run:
        return request

    target_dir = Path(root) / spec.subpath
    target_dir.mkdir(parents=True, exist_ok=True)
    years, months, _ = expand_dates(start, end)
    name = f"era5_{spec.subpath.name}_{years[0]}{months[0]}.nc"
    target = target_dir / name
    _retrieve(spec.dataset_id, request, str(target))
    return target
```

Note `_retrieve` calls `_client()` internally; the dry-run test asserts `_client` is *not* called because `stage_era5` returns before `_retrieve`. The retrieve test stubs `_retrieve` wholesale, so the real `_client` is never reached.

Now add the deferred re-export in `davinci_monet/io/download/__init__.py`:

```python
from davinci_monet.io.download.era5 import ERA5_DATASETS, stage_era5
```

(replace the Task 2 `ERA5_DATASETS`-only import; ensure `"stage_era5"` is in `__all__`).

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_era5.py -q`
Expected: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/io/download/era5.py davinci_monet/io/download/__init__.py davinci_monet/tests/test_download_era5.py
git commit -m "feat(download): stage_era5 orchestration (cdsapi client/retrieve, dry-run returns request)"
```

---

### Task 5: CLI entry point

**Files:**
- Modify: `davinci_monet/io/download/era5.py`
- Test: `davinci_monet/tests/test_download_era5.py`

- [ ] **Step 1: Write the failing test**

Append to `davinci_monet/tests/test_download_era5.py`:

```python
def test_main_dry_run_prints_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def _fake_stage(dataset, variables, start, end, **kw):
        captured.update(dataset=dataset, variables=variables, start=start, end=end, kw=kw)
        return {"variable": ["2m_temperature"]}

    monkeypatch.setattr(era5, "stage_era5", _fake_stage)

    rc = era5.main(
        [
            "--dataset", "single_levels",
            "--variables", "t2m", "msl",
            "--start", "2026-04-01",
            "--end", "2026-04-02",
            "--root", str(tmp_path),
            "--dry-run",
        ]
    )

    assert rc == 0
    assert captured["dataset"] == "single_levels"
    assert captured["variables"] == ["t2m", "msl"]
    assert captured["kw"]["dry_run"] is True
    assert "2m_temperature" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_era5.py::test_main_dry_run_prints_request -q`
Expected: FAIL — `main` absent.

- [ ] **Step 3: Write minimal implementation**

Append to `davinci_monet/io/download/era5.py`:

```python
def main(argv: Sequence[str] | None = None) -> int:
    """CLI: stage an ERA5 dataset. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="davinci-stage-era5",
        description="Stage ERA5 NetCDF from the Copernicus CDS.",
    )
    parser.add_argument("--dataset", required=True, choices=sorted(ERA5_DATASETS))
    parser.add_argument(
        "--variables", required=True, nargs="+", help="Short or CDS variable names"
    )
    parser.add_argument("--start", required=True, help="ISO date, e.g. 2026-04-01")
    parser.add_argument("--end", required=True, help="ISO date (same month as start)")
    parser.add_argument(
        "--pressure-levels",
        type=int,
        nargs="+",
        default=None,
        help="hPa levels (pressure_levels dataset only)",
    )
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Staging root dir")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the request; do not retrieve"
    )
    ns = parser.parse_args(argv)

    result = stage_era5(
        ns.dataset,
        ns.variables,
        ns.start,
        ns.end,
        pressure_levels=ns.pressure_levels,
        root=ns.root,
        dry_run=ns.dry_run,
    )
    if ns.dry_run:
        import json

        print(f"Would submit to CDS ({ns.dataset}):")
        print(json.dumps(result, indent=2))
    else:
        print(f"Staged {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_era5.py -q`
Expected: PASS (12 passed)

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/io/download/era5.py davinci_monet/tests/test_download_era5.py
git commit -m "feat(download): davinci-stage-era5 CLI entry point"
```

---

### Task 6: Gate sweep + manual staging note

**Files:** (verification only)

- [ ] **Step 1: Full local gate sweep**

Run (in the `davinci` conda env):
```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_download_era5.py -q
mypy davinci_monet/io/download/era5.py
black --check davinci_monet/io/download/era5.py davinci_monet/tests/test_download_era5.py
isort --check davinci_monet/io/download/era5.py davinci_monet/tests/test_download_era5.py
```
Expected: pytest all pass; mypy clean; black/isort report no changes (run without `--check` to fix, re-gate, amend).

- [ ] **Step 2: Manual smoke (real network, optional, not a test)**

After `pip install -e ".[reanalysis]"` and a `~/.cdsapirc` with a CDS Personal Access Token:
```bash
# Inspect the request without submitting:
davinci-stage-era5 --dataset single_levels --variables t2m msl \
  --start 2026-04-01 --end 2026-04-03 --dry-run

# Stage 3 days of hourly single-level data into /Volumes/Io/ERA5/single:
davinci-stage-era5 --dataset single_levels --variables t2m msl \
  --start 2026-04-01 --end 2026-04-03
```
Expected: dry-run prints the request JSON; the real run writes `era5_single_202604.nc`. Once staged, the ERA5 reader's `test_real_era5_file_opens` smoke (in the reader plan) will run instead of skip.

- [ ] **Step 3: No commit** — verification only.

---

## Self-Review

- **Spec coverage:** Implements the ERA5 downloader (spec Phase 3, downloader half): `cdsapi`/CDS, NetCDF-only, single-level + pressure-level, Io `ERA5/{single,pressure}` layout, optional `[reanalysis]` extra. Reader and CAMS are separate plans.
- **Placeholder scan:** No TBD/TODO; every code step shows complete code.
- **Type consistency:** `resolve_dataset`→`DatasetSpec` (`.dataset_id`, `.subpath`, `.has_levels`) used consistently in `dest_dir`/`build_request`/`stage_era5`/`main`. `expand_dates`/`build_request`/`cds_variable` signatures match all callers and tests. Wrapper names `_client`/`_retrieve` match between implementation and the monkeypatched tests.
- **Offline guarantee:** all network funneled through `_client`/`_retrieve` (lazy `cdsapi` import); every unit test monkeypatches them, so the suite needs neither `cdsapi` installed nor a network. `--dry-run` builds and returns the request without any client construction.
- **CDS-specific correctness:** single-month restriction documented and enforced (avoids the year×month×day cartesian over-request); friendly→CDS-long variable mapping with pass-through; `data_format: netcdf` / `download_format: unarchived`; pressure-level dataset requires `pressure_levels`.
- **Dependency between Task 2 and Task 4:** the `__init__` re-export of `stage_era5` is deferred to Task 4 (Task 2 re-exports only `ERA5_DATASETS`), so each task's package import succeeds — mirroring the fix applied during the MERRA-2 downloader build.
```
