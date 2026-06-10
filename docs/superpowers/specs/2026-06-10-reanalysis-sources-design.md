# Reanalysis Data Sources (MERRA-2, ERA5, CAMS) — Design

**Date:** 2026-06-10
**Status:** Draft for review
**Branch:** develop

## Context

DAVINCI is refocusing from atmospheric chemistry / air quality toward **climate**
in terms of datasets. Alongside the planned EOS observation work, we want to add
**reanalyses** as data sources. This spec covers three products:

- **MERRA-2** (NASA GMAO, via GES DISC) — meteorology, surface, and GOCART aerosols
- **ERA5** (ECMWF, via Copernicus CDS) — core meteorology + surface/fluxes
- **CAMS** (ECMWF atmospheric composition reanalysis / EAC4, via ADS) — O₃, CO, NO₂, AOD

All three are **gridded** products and slot into the post-unification `sources:`
architecture as new `type:` readers picked up by the existing GRID pairing
strategy. No new geometry and no pairing-engine changes are required.

### Decisions already made (from brainstorming)

| Decision | Choice |
| --- | --- |
| Role of reanalyses in pairing | **Comparand** — just another gridded source; no special reference precedence. `role:` stays styling-only metadata. |
| Data access | **Local-files-first readers** + **bundled optional downloaders**. |
| MERRA-2 downloader | **`earthaccess`** (Earthdata login + search + download). |
| ERA5 / CAMS downloader | **`cdsapi`** (CDS for ERA5, ADS for CAMS). |
| File format | **NetCDF only** to start (no GRIB / `cfgrib` / `eccodes` dependency). |
| Build sequence | **MERRA-2 first**, then ERA5, then CAMS. Downloaders implemented first so sample data can be staged. |
| Staging location | Existing external **Io drive**, pointing readers at the real on-disk layout. |

## Goals

1. Read MERRA-2, ERA5, and CAMS NetCDF as gridded `sources:` with correct
   coordinate standardization and (where applicable) vertical handling.
2. Provide optional downloader helpers that stage sample data to the Io drive in
   a consistent, reader-friendly layout.
3. Validate end-to-end through the **pipeline** against an existing obs reader
   (MERRA-2 AOD vs AERONET/MODIS) — no bespoke pairing/plot scripts.
4. Keep core toolkit dependencies light: `earthaccess`/`cdsapi` are optional
   extras, imported lazily.

## Non-Goals (YAGNI)

- GRIB support (NetCDF only for now).
- ERA5-Land, ensemble spread, model-level (`Nv`) ERA5 — pressure/single levels only.
- Reanalysis-as-preferred-reference precedence logic (comparand only).
- A generic "download any collection" CLI — downloaders target the specific
  collections in this spec.

## Existing Io Layout (observed 2026-06-10)

MERRA-2 aerosol data already exists on the Io drive:

```
/Volumes/Io/MERRA2_inst3/aer_Nv/MERRA2_300.inst3_3d_aer_Nv.YYYYMMDD.nc4   # 3-hourly 3D aerosol mixing ratios
/Volumes/Io/MERRA2_tavgM/aer_Nx/MERRA2_NNN.tavgM_2d_aer_Nx.YYYYMM.nc4      # monthly 2D aerosol diagnostics (AODs)
/Volumes/Io/Optics/MERRA2                                                  # optics tables (out of scope)
```

`tavgM_2d_aer_Nx` files: dims `time × lat(361) × lon(576)`, CF-standard `lat`/`lon`,
**no vertical dimension**. Key 550 nm AOD fields: `TOTEXTTAU` (total), `DUEXTTAU`
(dust), `SSEXTTAU` (sea salt), `BCEXTTAU`, `OCEXTTAU`, sulfate. AODs are
dimensionless → `unit_scale: 1.0`.

**Readers point at this existing layout.** The MERRA-2 downloader, when used,
extends the same tree. ERA5 and CAMS adopt a parallel layout:

```
/Volumes/Io/ERA5/pressure/era5_pl_YYYYMM.nc
/Volumes/Io/ERA5/single/era5_sfc_YYYYMM.nc
/Volumes/Io/CAMS/eac4_YYYYMM.nc
```

## Architecture

### Reader pattern (shared)

Each reader is a small class registered with `@source_registry.register("<type>")`,
exposing `name` and `geometry = DataGeometry.GRID`, and an `open(file_paths,
variables, **kwargs) -> xr.Dataset`. It reads NetCDF via `xr.open_mfdataset` and
standardizes dims/coords using the shared helpers in
`davinci_monet/io/reader_utils.py` (`validate_file_list`, `select_variables`,
`standardize_dims`, `retry_transient_open`). This mirrors `models/cesm.py`.

For 3D products, vertical standardization renames the level dim to `z` and relies
on the existing `surface_level_index()` auto-detection (pressure increasing with
index → surface at last index). 2D products (`tavgM_2d_aer_Nx`, ERA5 single-level,
CAMS surface diagnostics) have no vertical dim and skip this step.

### New files

```
davinci_monet/models/merra2.py        # MERRA2Reader  -> type: merra2
davinci_monet/models/era5.py          # ERA5Reader    -> type: era5
davinci_monet/models/cams.py          # CAMSReader    -> type: cams
davinci_monet/io/download/__init__.py
davinci_monet/io/download/merra2.py    # earthaccess staging helper
davinci_monet/io/download/era5.py      # cdsapi (CDS) staging helper
davinci_monet/io/download/cams.py      # cdsapi (ADS) staging helper
```

Example configs (one per product, `*.example.yaml`, env-var/absolute paths):

```
analyses/reanalysis/configs/merra2-aod-aeronet.example.yaml   # validation case
analyses/reanalysis/configs/era5-met.example.yaml
analyses/reanalysis/configs/cams-composition.example.yaml
```

### Downloaders

Optional, lazily-imported helpers. Each exposes a simple function (e.g.
`stage_merra2(collection, start, end, dest)`) and a thin CLI entry usable as a
script. They write into the Io layout above and **do not** run as a pipeline
stage — staging is a separate, deliberate step.

- **MERRA-2** (`earthaccess`): authenticate via Earthdata login (`.netrc` or
  interactive), search collection short-names (`M2TMNXAER` for `tavgM_2d_aer_Nx`,
  `M2I3NVAER` for `inst3_3d_aer_Nv`) over a date range, download to
  `/Volumes/Io/MERRA2_<class>/<short>/`.
- **ERA5** (`cdsapi`, CDS): request `reanalysis-era5-pressure-levels` and
  `reanalysis-era5-single-levels`, `format: netcdf`, write to `/Volumes/Io/ERA5/...`.
- **CAMS** (`cdsapi`, ADS): request `cams-global-reanalysis-eac4`,
  `format: netcdf`, write to `/Volumes/Io/CAMS/...`.

`earthaccess` and `cdsapi` go in an optional extra (e.g. `.[reanalysis]`) and are
imported inside the helper functions so the core package does not require them.

## Variable Scope (initial)

- **MERRA-2 `tavgM_2d_aer_Nx`** (validation-first): `TOTEXTTAU`, `DUEXTTAU`,
  `SSEXTTAU`, `BCEXTTAU`, `OCEXTTAU`, sulfate AOD — `unit_scale: 1.0`.
- **MERRA-2 `inst3_3d_aer_Nv`** (next): 3D aerosol mixing ratios (`DU`, `SS`,
  `BC`, `OC`, `SO4` bins) — vertical handling applies.
- **ERA5**: pressure-level `t`, `u`, `v`, `z`, `q`; single-level `2t`, `10u`,
  `10v`, `tp`, `msl`.
- **CAMS (EAC4)**: `go3` (O₃), `co`, `no2`, `aod550`.

## Data Flow

```
[Io NetCDF] -> Reader.open() -> standardize dims/coords (-> z, surface idx)
            -> sources: variables + unit_scale
            -> GRID pairing strategy (sampled onto obs/finer geometry)
            -> statistics -> plots -> save_results
```

Reanalyses are comparands. When paired with an irregular-geometry obs source
(point/track/profile/swath), geometry precedence makes the reanalysis the sampled
GRID source automatically. GRID-vs-GRID uses first-listed-as-reference (existing
behavior, unchanged).

## Validation / Testing

Per CLAUDE.md testing rules — integration tests run through
`PipelineRunner.run_from_config()`.

1. **Unit**: synthetic gridded `xr.Dataset` generators for each reader; assert
   dims standardized (`lat`/`lon`/`time`, `z` for 3D), variable selection, and
   surface-index detection for 3D.
2. **Integration (pipeline)**: a config pairing **MERRA-2 `TOTEXTTAU` vs
   AERONET AOD** (existing reader), run via the pipeline, asserting paired
   variables carry `role`/`source_label` attrs and stats/plots are produced.
   Synthetic obs + a small real (or synthetic) MERRA-2 monthly file.
3. **No external-network tests**: downloaders are exercised with mocked
   `earthaccess`/`cdsapi` clients; real staging is manual.
4. Gates: `pytest`, `mypy davinci_monet`, `black`, `isort` — run locally in the
   `davinci` conda env (`HDF5_USE_FILE_LOCKING=FALSE`).

## Implementation Sequence

Downloaders first so sample data can be staged, then readers, validated per
product before moving on.

1. **Phase 0 — scaffolding**: `io/download/` package, optional `[reanalysis]`
   extra, `analyses/reanalysis/` skeleton.
2. **Phase 1 — MERRA-2 downloader** (`earthaccess`): stage a sample of
   `tavgM_2d_aer_Nx` (already partly present) + verify `inst3_3d_aer_Nv`.
3. **Phase 2 — MERRA-2 reader** (`type: merra2`): 2D `tavgM_2d_aer_Nx` first,
   then 3D `inst3_3d_aer_Nv`. Validate via AOD-vs-AERONET pipeline config.
4. **Phase 3 — ERA5 downloader + reader** (`cdsapi`/CDS, NetCDF).
5. **Phase 4 — CAMS downloader + reader** (`cdsapi`/ADS, NetCDF).

Each phase is its own implementation plan (writing-plans), kept small and
verified before the next.

## Open Choices (confirm during spec review)

1. **`analyses/` directory name** — `analyses/reanalysis/` proposed; OK?
2. **MERRA-2 short-names** — `M2TMNXAER` (monthly 2D aer) / `M2I3NVAER`
   (3-hourly 3D aer) assumed for the downloader; confirm against your Earthdata
   access.
3. **Optional-extra name** — `.[reanalysis]` proposed (vs reusing `.[ai]`-style
   naming). OK?
4. **ERA5/CAMS Io layout** — `/Volumes/Io/ERA5/{pressure,single}` and
   `/Volumes/Io/CAMS/` proposed; OK or match an existing convention?
