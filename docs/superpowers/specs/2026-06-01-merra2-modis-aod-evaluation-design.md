# MERRA2 vs MODIS AOD Global Evaluation — Design

**Date:** 2026-06-01
**Status:** Design approved, pending spec review
**Branch target:** `develop` (planning only; no implementation started)

## Context

The existing ASIA-AQ analysis (`analyses/asia-aq/`) evaluates surface PM2.5/O3/NO2/CO
from a CAM/CAM-chem run against AirNow point monitors for Feb 2024. The user asked to
"update the ASIA-AQ analysis to use MERRA2 instead of CAM" using MERRA2 data on the Io
external drive.

Investigation of the Io drive showed this cannot be a literal CAM→MERRA2 swap:

- **MERRA2 here is aerosol-only.** `MERRA2_inst3/aer_Nv` (3-hourly, 3D, 72 levels) and
  `MERRA2_tavgM/aer_Nx` (monthly, 2D surface) carry aerosol species and AOD — **no
  O3/NO2/CO** (those need MERRA2-GMI, not on the drive). So the gas-phase pairs of the
  AirNow analysis are impossible with this data.
- **Temporal coverage.** `inst3/aer_Nv` is only 10 days of July 2008 (no overlapping
  ASIA-AQ obs). `tavgM/aer_Nx` is monthly means spanning 2000-01 → 2026-03.

Given those constraints, the chosen direction (confirmed with the user) is a **new,
distinct analysis**: a global, multi-year monthly comparison of **MERRA2 reanalysis AOD**
against **MODIS Terra and Aqua L3 monthly AOD**, playing to MERRA2's aerosol strengths.
The existing ASIA-AQ AirNow analysis is left intact.

### Decisions locked during brainstorming

1. **Comparison:** MERRA2 AOD vs MODIS L3 AOD (not AirNow surface, not PM2.5).
2. **Obs sources:** MODIS Terra (`MOD08_M3`) **and** Aqua (`MYD08_M3`), both at 550 nm.
3. **Period:** full multi-year monthly time series (configurable window).
4. **Domain:** global.
5. **MODIS reader:** **bootstrap a minimal vertical slice of the planned `modis_viirs`
   catalog reader** (see `2026-06-01-modis-viirs-catalog-readers-design.md`) rather than a
   throwaway dedicated reader. This analysis becomes that subsystem's first consumer.
6. **Placement:** new `analyses/merra2-aod/` directory; the existing ASIA-AQ AirNow
   config is unchanged.

### Verified facts (from data inspection)

- **MERRA2 `tavgM/aer_Nx`**: dims `(time:1, lat:361, lon:576)` per file (0.625°×0.5°);
  AOD variable `TOTEXTTAU` is dimensionless (550 nm); real `time` coord per file;
  315 monthly files 2000-01 → 2026-03.
- **MODIS `MOD08_M3` / `MYD08_M3`**: HDF4/HDF-EOS, opens via
  `xr.open_dataset(engine="netcdf4")`. AOD SDS `Aerosol_Optical_Depth_Land_Ocean_Mean_Mean`
  at **0.55 µm**, `scale_factor=0.001`, `add_offset=0.0`, `_FillValue=-9999`, dims
  `YDim:mod08`(180) × `XDim:mod08`(360) = **1°×1° global**. **No time coordinate in-file** —
  the month is encoded only in the filename (`*.A{YYYY}{DDD}.*`). `XDim`/`YDim` SDS hold the
  grid coordinate values. Selecting only the needed SDS avoids the file's 1144-variable /
  duplicate-dimension noise.
- **Coverage overlap:** MERRA2 2000–2026; MOD08 (Terra) from 2000-02; MYD08 (Aqua) from
  2002-07. Three-way overlap begins mid-2002.
- **Infrastructure present:** `GridStrategy` (grid-to-grid pairing with regridding) exists;
  `spatial_bias` / `scatter` / `timeseries` renderers exist (previously exercised on point
  data — see open item on GRID-paired support).

## Relationship to the modis_viirs catalog design

This spec implements a **vertical slice** of
`2026-06-01-modis-viirs-catalog-readers-design.md`, specifically the parts needed for L3
gridded atmosphere AOD:

- Catalog schema + registry/lookup (Phase 1 of that design, minimal).
- Variable metadata handling / scale-fill / `variables: "*"` discovery (Phase 2, minimal —
  enough for the AOD SDS).
- L3 regular-grid loading and grid-pairing integration (Phase 5, for MOD08_M3/MYD08_M3).

Out of scope here (deferred to the larger effort): L1/L2 readers, swath geolocation,
swath-to-grid gridding/cache generalization, projected-grid handling, QA-mask curation,
DAAC discovery/download, and non-atmosphere product families.

After this lands, the modis_viirs design doc should be updated to note that its
Phase 1/2/5 have a working initial slice. (Editing that doc is a follow-up, gated on user
approval — not done as part of this implementation.)

## Architecture

### Component 1 — minimal `modis_viirs` catalog reader

Following the approved catalog design's layout:

```text
davinci_monet/observations/satellite/
├── catalog/
│   ├── __init__.py
│   ├── schema.py        # pydantic product-entry + variable-entry schema
│   ├── registry.py      # load YAML catalog data, resolve product/alias → entry
│   └── data/
│       ├── modis_viirs_core.yaml        # shared identity fields (instrument/platform/DAAC)
│       └── modis_viirs_atmosphere.yaml  # MOD08_M3 + MYD08_M3 entries (AOD SDS, grid, time-parse)
└── modis_viirs.py       # MODISVIIRSReader, @source_registry.register("modis_viirs")
```

**Catalog entries** (MOD08_M3, MYD08_M3) declare: `product_id`, instrument (MODIS),
platform (Terra/Aqua), DAAC (LAADS), collection (061), level `L3`, geometry `GRID`, file
format HDF4, dimension aliases (`XDim`→`lon`, `YDim`→`lat`), filename time-parse rule
(`A{YYYY}{DDD}` → first-of-month), and the AOD variable entry
(`Aerosol_Optical_Depth_Land_Ocean_Mean_Mean`, display `aod_550nm`, units `1`, 550 nm,
scale/offset/fill as metadata).

**`MODISVIIRSReader` (L3 grid path only for this slice):**
1. Require `product`; look it up in the catalog (fail with close matches if unknown — per
   the catalog design's error handling). `level`, if given, must match catalog.
2. For each file: open via `engine="netcdf4"`, select cataloged variables (or discover all
   on `variables: "*"`; for this analysis the default is just the AOD SDS), apply CF
   scale/offset/fill (xarray `mask_and_scale` handles the AOD SDS automatically).
3. Rename `XDim`/`YDim` → `lon`/`lat`, attach the `XDim`/`YDim` coordinate values.
4. Parse the month from the filename, assign a `time` coord.
5. Sort by time and concatenate files → monthly GRID time series.
6. Rename cataloged variables to their **catalog display name** (e.g. the AOD SDS
   `Aerosol_Optical_Depth_Land_Ocean_Mean_Mean` → `aod_550nm`), so downstream configs and
   pairs reference the stable display name regardless of the verbose SDS name.
7. Set `geometry = GRID`; attach standard source/variable attrs from the catalog
   (`product_id`, `instrument`, `platform`, `level`, `daac`, `collection`, `catalog_status`,
   per-variable `units`/`scale_factor`/`add_offset`/`_FillValue`).

The source `variables:` list selects SDS by their **file (SDS) name**; after the rename in
step 6, pairs reference the **display name** (`aod_550nm`). Both names are documented in the
catalog entry.

### Component 2 — MERRA2 source (no new reader)

`type: generic` reading `MERRA2_tavgM/aer_Nx/*.nc4`, exposing `TOTEXTTAU`. Implementation
verifies the generic reader tags a 2D `(time, lat, lon)` field as GRID with no `lev`. If a
thin convenience registration is warranted it can be added, but generic is expected to
suffice.

### Component 3 — analysis directory

```text
analyses/merra2-aod/
├── configs/merra2-modis-aod.example.yaml   # env-var portable template
├── scripts/run_evaluation.py               # thin run_analysis wrapper
├── output/   (gitignored)
└── logs/     (gitignored)
```

## Config shape (unified `sources:` schema)

```yaml
analysis:
  start_time: "2003-01-01"   # default to a short window; user widens for full archive
  end_time:   "2003-12-31"
  output_dir: ${MERRA2_AOD_ANALYSIS}/output
  log_dir:    ${MERRA2_AOD_ANALYSIS}/logs
  style: { theme: ncar, context: default }

sources:
  merra2:
    type: generic
    role: model
    files: ${MERRA2_DATA}/aer_Nx/*.nc4
    variables:
      TOTEXTTAU: { units: "1", ylabel_plot: "AOD (550 nm)", vmin_plot: 0, vmax_plot: 1 }
  modis_terra:
    type: modis_viirs
    role: obs
    product: MOD08_M3
    files: ${MODIS_DATA}/MOD08_M3/*.hdf
    variables: [Aerosol_Optical_Depth_Land_Ocean_Mean_Mean]
  modis_aqua:
    type: modis_viirs
    role: obs
    product: MYD08_M3
    files: ${MODIS_DATA}/MYD08_M3/*.hdf
    variables: [Aerosol_Optical_Depth_Land_Ocean_Mean_Mean]

pairs:
  merra2_vs_terra:
    sources: [merra2, modis_terra]
    reference: modis_terra
    variables: { merra2: TOTEXTTAU, modis_terra: aod_550nm }
  merra2_vs_aqua:
    sources: [merra2, modis_aqua]
    reference: modis_aqua
    variables: { merra2: TOTEXTTAU, modis_aqua: aod_550nm }

plots:   # see Plots section
stats:
  output_table: true
  metrics: [N, MO, MP, MB, RMSE, R, NMB, NME, IOA]
```

- **Common grid:** regrid MERRA2 (0.625°×0.5°) **onto the coarser MODIS 1° grid** via
  `GridStrategy` (the MODIS source is the `reference`).
- **Time matching:** exact monthly alignment; each pair uses its own overlap window
  (Terra ≥2000-02, Aqua ≥2002-07).
- **Window default:** the example config defaults to a short (~1 year) window for a fast
  first run; the user widens `start_time`/`end_time` to sweep the full archive.

## Data flow

```text
MERRA2 aer_Nx (generic, GRID) ─┐
MOD08_M3 (modis_viirs, GRID) ──┼─ GridStrategy (regrid→1°, monthly align) ─ stats ─ plots ─ CSV
MYD08_M3 (modis_viirs, GRID) ──┘
```

## Plots (default set)

- **Spatial mean AOD maps** per source and **spatial bias maps** (MERRA2 − MODIS) over the
  window — `spatial_bias`.
- **Scatter density** of paired grid cells per pair (`show_density: true`).
- **Global-mean monthly time series** (MERRA2 vs Terra vs Aqua) — `timeseries` aggregating
  over `lat`/`lon`.

## Stats

`[N, MO, MP, MB, RMSE, R, NMB, NME, IOA]` per pair, written to CSV.

## Open items to resolve in the implementation plan

- **Renderer audit (required by repo policy):** confirm `spatial_bias` / `scatter` /
  `timeseries` handle GRID-geometry paired data; these were previously exercised on point
  data. Extend rather than reinvent if gaps exist.
- Confirm the `generic` reader tags MERRA2 2D fields as GRID (no `lev`).
- Reconcile lon/lat ordering between MERRA2 (lat ascending, lon −180..180) and MODIS
  (`YDim` likely descending) during regrid.
- Confirm MODIS `XDim`/`YDim` coordinate-value ranges (cell centers vs edges) for correct
  grid registration.

## Testing and validation

Per repo testing rules (integration tests must run through `PipelineRunner`):

- **Unit tests** for the modis_viirs slice: catalog schema validation; product/alias
  lookup + unknown-product error; filename→month parsing; scale/offset/fill application;
  `XDim`/`YDim`→`lon`/`lat` rename + coord attach; multi-file monthly concat; GRID geometry
  tag; `variables: "*"` discovery vs explicit-variable selection.
- **Integration test** through `PipelineRunner.run_from_config()` on a small synthetic
  pair of L3 GRID sources (MERRA2-like + MODIS-like) covering 2–3 months, asserting paired
  output, stats CSV, and generated plots.
- **Real-data smoke test** (skipped unless Io env vars are set) over a short window.

Validation commands:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate davinci
HDF5_USE_FILE_LOCKING=FALSE python -m pytest
mypy davinci_monet
black --check davinci_monet && isort --check davinci_monet
```

## Phasing

1. **Catalog foundation:** `catalog/schema.py`, `catalog/registry.py`, core +
   atmosphere YAML with MOD08_M3/MYD08_M3 AOD entries; schema/lookup unit tests.
2. **modis_viirs L3 reader:** `modis_viirs.py` L3-grid path; reader unit tests (synthetic
   HDF/netCDF L3 fixture).
3. **MERRA2 source verification:** confirm generic reader → GRID for `TOTEXTTAU`.
4. **Analysis dir + config:** `analyses/merra2-aod/` with config + run script;
   `.gitignore` for output/logs.
5. **Renderer audit + grid pairing wiring:** verify/extend renderers for GRID-paired data;
   confirm GridStrategy regrid path.
6. **Integration test + real-data smoke test.**
7. **Run, copy plots to the iCloud Claude folder, review.**

## Out of scope

- Gas-phase (O3/NO2/CO) evaluation — not in MERRA2 aerosol data.
- PM2.5 derivation from MERRA2 surface mass concentrations.
- The full modis_viirs catalog effort (L1/L2, swath, projected grids, QA masks, download).
- Modifying the existing ASIA-AQ AirNow analysis.
