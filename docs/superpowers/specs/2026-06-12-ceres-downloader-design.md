# CERES Downloaders (L2 SSF + L3 EBAF/SYN1deg) — Design

**Date:** 2026-06-12
**Status:** Approved
**Branch:** develop

## Context

DAVINCI is adding CERES radiation-budget datasets as part of the climate
refocus (see `2026-06-10-reanalysis-sources-design.md` for the sibling
reanalysis effort). This spec covers the **downloaders only** — staging
helpers that extend the established `io/download/` pattern. Readers for
these products are a follow-on effort with their own spec/plan.

CERES is distributed by the NASA Langley ASDC, an Earthdata DAAC, so the
downloaders use `earthaccess` — the same optional dependency
(`pip install -e ".[reanalysis]"`) and authentication path (Earthdata
login via `.netrc` or interactive) as the existing MERRA-2 downloader.

### Decisions already made (from brainstorming)

| Decision | Choice |
| --- | --- |
| Products | **SSF (L2 swath) + EBAF (L3 monthly gridded) + SYN1deg (L3 gridded)** |
| SSF platforms | **All**: Terra FM1, Aqua FM3, S-NPP FM5, NOAA-20 FM6 |
| SYN1deg resolutions | **All**: monthly, daily, hourly |
| Structure | **Mirror + shared core** — extract generic earthaccess plumbing into `io/download/earthdata.py`; `merra2.py` and new `ceres.py` become thin product tables over it |
| Staging location | `/Volumes/Io/CERES/...` (Io drive, parallel to existing MERRA-2 layout) |
| Downloads | Manual, deliberate step — never a pipeline stage |

### Data facts that shape the design (verified via CMR, 2026-06-12)

- **EBAF** is a *single* ~2 GB netCDF covering the entire record
  (`CERES_EBAF_Edition4.2.1_200003-202512.nc`). Temporal subsetting does
  not apply — it is a download-once file.
- **SSF L2** is hourly granules of ~60–70 MB (~1.5 GB/day/instrument),
  HDF4 for the MODIS-era editions.
- **SYN1deg** Edition4B (Terra-Aqua-**NOAA20**) is a *full-record
  reprocessing* (granules 2000-03 through 2025-12) that supersedes the
  Terra-Aqua-MODIS Edition4A collections (2002-07 through 2022-03). Only
  Edition4B is staged — carrying both eras would duplicate ~20 years of
  overlapping months. Monthly granules ~50 MB; hourly ~715 MB/day.
  (Corrected 2026-06-12 in code review; the original design wrongly assumed
  Edition4B was a post-2022 continuation.)
- Multiple editions share a short-name in CMR (e.g. `CER_SSF_NOAA20-FM6-VIIRS`
  has Edition1B and Edition1C), so unlike MERRA-2, **searches must pin a
  `version`** (e.g. `Edition4A`).

## Goals

1. Stage CERES SSF, EBAF, and SYN1deg granules to the Io drive in a
   consistent, reader-friendly layout via a `davinci-stage-ceres` CLI.
2. Extract a shared `earthdata.py` staging core so MERRA-2, CERES, and
   future Earthdata products do not duplicate earthaccess plumbing.
3. Keep MERRA-2 downloader behavior and its public API unchanged.
4. Size-aware dry-run so large orders (SSF, hourly SYN1deg) are visible
   before committing.

## Non-Goals (YAGNI)

- CERES **readers** (swath/grid `sources:` types) — separate follow-on spec.
- Spatial or variable subsetting (full granules only; the ASDC ordering
  tool exists for that and is out of scope).
- Other CERES products (CRS, FluxByCldTyp, SSF1deg, FLASHflux).
- The superseded SYN1deg Terra-Aqua-MODIS Edition4A collections (fully
  contained within Edition4B's record).
- A generic "download any Earthdata collection" CLI.

## Architecture

### Files

```
davinci_monet/io/download/earthdata.py   # NEW: shared earthaccess core
davinci_monet/io/download/merra2.py      # REFACTOR: delegates to core; API unchanged
davinci_monet/io/download/ceres.py       # NEW: CERES table + stage_ceres() + CLI
davinci_monet/tests/test_download_ceres.py
```

### Shared core (`earthdata.py`)

Moves the generic pieces out of `merra2.py`:

- `CollectionSpec(short_name, subpath, version=None)` — frozen dataclass;
  gains an optional `version` field. When set, it is passed to
  `earthaccess.search_data(version=...)`.
- `_login()`, `_search(short_name, temporal, version)`, `_download(results, dest)`
  — the network-isolation seam, lazy `earthaccess` imports. Tests mock here.
- `stage_collection(spec, start, end, *, root, dry_run)` — generic staging:
  login → search → (dry-run: return count/size) → mkdir → download.
  `start`/`end` of `None` searches without a temporal filter.

`merra2.py` keeps `MERRA2_COLLECTIONS`, `resolve_collection`, `dest_dir`,
`stage_merra2()`, and its CLI with identical signatures and behavior —
they delegate to the core. Existing MERRA-2 tests pass unchanged.

### CERES collections table (`ceres.py`)

`CERES_COLLECTIONS: dict[str, CollectionSpec]`, all under `/Volumes/Io/CERES/`:

| Friendly key | Earthdata short-name | Version | Io subpath |
| --- | --- | --- | --- |
| `ssf_terra-fm1` | `CER_SSF_Terra-FM1-MODIS` | Edition4A | `CERES/SSF/Terra-FM1` |
| `ssf_aqua-fm3` | `CER_SSF_Aqua-FM3-MODIS` | Edition4A | `CERES/SSF/Aqua-FM3` |
| `ssf_npp-fm5` | `CER_SSF_NPP-FM5-VIIRS` | Edition2A | `CERES/SSF/NPP-FM5` |
| `ssf_noaa20-fm6` | `CER_SSF_NOAA20-FM6-VIIRS` | Edition1C | `CERES/SSF/NOAA20-FM6` |
| `ebaf` | `CERES_EBAF` | Edition4.2.1 | `CERES/EBAF` |
| `syn1deg_month` | `CER_SYN1deg-Month_Terra-Aqua-NOAA20` | Edition4B | `CERES/SYN1deg/month` |
| `syn1deg_day` | `CER_SYN1deg-Day_Terra-Aqua-NOAA20` | Edition4B | `CERES/SYN1deg/day` |
| `syn1deg_hour` | `CER_SYN1deg-1Hour_Terra-Aqua-NOAA20` | Edition4B | `CERES/SYN1deg/hour` |

SYN1deg uses only Edition4B (Terra-Aqua-NOAA20), the full-record
reprocessing that supersedes the MODIS-era Edition4A collections.

### `stage_ceres()` + CLI

`stage_ceres(collection, start=None, end=None, *, root=DEFAULT_ROOT,
dry_run=False)` — same shape as `stage_merra2`, delegating to
`stage_collection`, plus two CERES-specific behaviors:

1. **EBAF temporal rules**: `start`/`end` are optional for `ebaf` (single
   whole-record granule; search runs without a temporal filter). For every
   other collection, missing `start`/`end` raises a clear `ValueError`
   *before* any network call — an unbounded SSF search would match the
   entire 25-year record.
2. **Size-aware dry-run**: `--dry-run` prints granule count *and* total
   size (MB/GB), summed from earthaccess result metadata.

CLI mirrors `davinci-stage-merra2`:

```
davinci-stage-ceres --collection ssf_aqua-fm3 --start 2023-07-01 --end 2023-07-02 [--root /Volumes/Io] [--dry-run]
davinci-stage-ceres --collection ebaf [--dry-run]    # no temporal needed
```

Registered as a console script in `pyproject.toml` `[project.scripts]`,
matching the existing entry:
`davinci-stage-ceres = "davinci_monet.io.download.ceres:main"`.

## Testing

All offline; network functions mocked at the shared-core seam, mirroring
`test_download_merra2.py`. Entry points exercised:

1. **Collections table integrity** — every entry has short-name, version,
   subpath under `CERES/`; unknown key raises `KeyError` listing valid names.
2. **`stage_ceres()` with mocked `_login`/`_search`/`_download`** — search
   called with short-name + version + temporal tuple; dest dir created
   under `root`; downloads land there; `dry_run=True` returns count/size
   without calling `_download`.
3. **EBAF temporal rules** — `ebaf` without start/end searches with no
   temporal filter; any other collection without start/end raises before
   any network call.
4. **CLI `main()`** — arg parsing, dry-run output includes total size,
   exit code 0.
5. **Refactor regression** — `test_download_merra2.py` passes unchanged,
   proving the shared-core extraction did not alter MERRA-2 behavior.

Gates: `pytest`, `mypy davinci_monet`, `black`, `isort` — run locally in
the `davinci` conda env (`HDF5_USE_FILE_LOCKING=FALSE`).

Real staging stays manual and dry-run-first. CERES lags more than MERRA-2
(EBAF currently ends 2025-12), so real smoke tests should use a safely
historical period such as 2023-07, not a recent month.
