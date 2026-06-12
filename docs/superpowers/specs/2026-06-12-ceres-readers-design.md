# CERES Readers (L2 SSF + L3 EBAF/SYN1deg) — Design

**Date:** 2026-06-12
**Status:** Approved
**Branch:** develop

## Context

Follow-on to the CERES downloaders
(`2026-06-12-ceres-downloader-design.md`). Sample data is staged on the Io
drive (see inventory below). This spec covers three reader types that plug
into the post-unification `sources:` architecture as observation readers.
No pairing-engine or strategy changes.

### Decisions already made (from brainstorming)

| Decision | Choice |
| --- | --- |
| Use case | **TOA + surface fluxes** first-class (all-sky + clear-sky, CRE derivable); cloud properties later |
| SSF formats | **Both** HDF4 (Terra/Aqua Edition4A) and netCDF (NOAA-20 Edition1C) |
| SSF pairing geometry | **SWATH + grid binning** — reader emits the natural 1-D footprint stream tagged `swath`; the existing `SwathGridStrategy` (registered default for SWATH) bins footprints onto a model-matched grid before grid-to-grid comparison |
| Module placement | `observations/satellite/` (CERES is an observation product; precedent: `modis_l2_aod`, `mopitt_l3_co`) |
| L3 variable naming | Native names via `variables:` config (no catalog — names are sane) |
| SSF variable naming | **Canonical catalog** (names differ wildly between editions) + `source_name` escape hatch |

### Staged sample data (verified 2026-06-12)

```
/Volumes/Io/CERES/EBAF/CERES_EBAF_Edition4.2.1_200003-202512.nc   # netCDF, (time=310, lat=180, lon=360) + ctime/sc climatology dims
/Volumes/Io/CERES/SSF/Terra-FM1/   CER_SSF_Terra-FM1-MODIS_Edition4A_*.2026040100..23      # HDF4, 1-D SDS (~67k footprints/hr)
/Volumes/Io/CERES/SSF/Aqua-FM3/    (same structure)
/Volumes/Io/CERES/SSF/NOAA20-FM6/  CER_SSF_NOAA20-FM6-VIIRS_Edition1C_*.nc                 # netCDF-4 with ~15 groups, Footprints dim (~105k/hr)
/Volumes/Io/CERES/SYN1deg/month/   *.202510..12   # HDF4, 2-D SDS (latitude=180, longitude=360)
/Volumes/Io/CERES/SYN1deg/day/     *.20251201..31 # HDF4, 2-D SDS per day-file
/Volumes/Io/CERES/SYN1deg/hour/    *.20251229..31 # HDF4, 3-D SDS (gmt_hr_index=24, latitude, longitude), one file per day
```

File-structure facts that shape the design:

- **SSF is a flat 1-D footprint stream** in both editions — no scanline×pixel
  structure. HDF4 keys are long SDS names with spaces; positions are
  *colatitude* (0–180) and 0–360 longitude; time is Julian day. The netCDF
  edition organizes the same parameters into groups (`Time_and_Position`,
  `TOA_and_Surface_Fluxes`, …) with epoch time and true latitude.
- **`bin_swath_to_grid` / `SwathGridStrategy` flatten obs arrays before
  binning**, so 1-D footprints pass through without fabricated 2-D dims.
- **SYN1deg HDF4 fields carry `_FillValue`, `valid_range`, and a `group`
  attribute**; hourly files add a `gmt_hr_index` dim. No explicit lat/lon
  coordinate values were confirmed in the SDS inventory — orientation is
  verified empirically (see Testing).
- **EBAF is CF-clean** `(time, lat, lon)` with 247 variables including
  `ctime`-dimensioned climatologies.

## Goals

1. `type: ceres_ebaf`, `type: ceres_syn1deg` (GRID) and `type: ceres_ssf`
   (SWATH) readers conforming to the `SourceReader` protocol
   (`core/protocols.py:62-110`), reusing `io/reader_utils.py` helpers.
2. SSF canonical variable catalog working identically over both editions.
3. End-to-end validation through `PipelineRunner.run_from_config` for all
   three types, including the SSF→`SwathGridStrategy` binning path.
4. Real-data smoke coverage against the staged Io samples.

## Non-Goals (YAGNI)

- Cloud-property variables as catalog entries (reachable via `source_name`).
- Per-footprint (un-binned) SSF pairing — `SwathStrategy` stays off the
  production path.
- CRS, FluxByCldTyp, SSF1deg, FLASHflux products.
- Angular-distribution/ADM handling, footprint PSF weighting.
- Downloader changes.

## Architecture

### Files

```
davinci_monet/observations/satellite/ceres_l3.py    # CERESEBAFReader (ceres_ebaf), CERESSYN1degReader (ceres_syn1deg)
davinci_monet/observations/satellite/ceres_ssf.py   # CERESSSFReader (ceres_ssf), dual-format
davinci_monet/tests/test_ceres_l3_readers.py
davinci_monet/tests/test_ceres_ssf_reader.py
davinci_monet/tests/integration/test_ceres_readers_pipeline.py
```

Each reader: `@source_registry.register("<type>")`, `name`/`geometry`
properties, `open(file_paths, variables=None, **kwargs) -> xr.Dataset`,
ending with `set_geometry_attr`. pyhdf is imported lazily inside methods
with the `modis_viirs.py:169-175` guard pattern.

### `ceres_ebaf` (GRID)

- `validate_file_list` → `retry_transient_open(xr.open_dataset)` (single
  whole-record file; `open_mfdataset` if several).
- `select_variables` on native names; drop `ctime`/`sc` dims when no
  requested variable uses them.
- Longitude normalized to [-180, 180) and sorted ascending (repo grid
  convention); lat already ascending.

### `ceres_syn1deg` (GRID)

- One assembler for all three cadences, per file:
  - parse time from filename tail — `YYYYMM` → month start, `YYYYMMDD` →
    day start; 3-D `gmt_hr_index` fields → day start + hour offsets.
  - read requested SDS via pyhdf; apply `_FillValue`/`valid_range`
    masking and scale/offset (the `modis_viirs._apply_hdf4_scale`
    pattern).
  - lat/lon centers from coordinate SDS when present, else documented 1°
    centers; orientation cross-checked in tests (below).
- Concat over files on `time`, sort, → `(time, lat, lon)`.
- Native variable names (`obs_all_toa_sw_reg`, …) + `source_name` escape.

### `ceres_ssf` (SWATH)

- Output: dims `(time,)` — one entry per footprint — with `lat`/`lon`
  coords on the same dim, requested variables as 1-D arrays, geometry attr
  `"swath"`. `SwathGridStrategy` flattens and bins by lat/lon/time values;
  per-footprint time drives temporal binning.
- Format sniffing per file: `.nc` suffix (or netCDF magic) → netCDF path,
  else HDF4 path. A multi-file open may mix editions only of the same
  format family; mixing is detected and rejected with a clear error.
- **HDF4 path** (Edition4A): pyhdf reads `Time of observation` (Julian day
  → `datetime64[ns]`), `Colatitude of CERES FOV at surface` →
  `lat = 90 − colat`, `Longitude of CERES FOV at surface` wrapped to
  [-180, 180); requested data SDS read flat, fill-masked.
- **netCDF path** (Edition1C): open `Time_and_Position` group for `time`,
  `instrument_fov_latitude`, `instrument_fov_longitude`; open the catalog
  group(s) for requested variables; merge on the `Footprints` dim renamed
  to `time`-indexed footprint stream; longitude wrapped to [-180, 180).

### SSF canonical catalog

| Canonical | HDF4 Edition4A SDS | netCDF Edition1C (`TOA_and_Surface_Fluxes/`) |
| --- | --- | --- |
| `toa_sw_up` | `CERES SW TOA flux - upwards` | `toa_shortwave_flux` |
| `toa_lw_up` | `CERES LW TOA flux - upwards` | `toa_longwave_flux` |
| `toa_solar_in` | `TOA Incoming Solar Radiation` | `toa_incoming_solar_radiation` |
| `sfc_sw_down` | `CERES downward SW surface flux - Model B` | `model_b_surface_shortwave_downward_flux` |
| `sfc_sw_down_clr` | `CERES downward SW surface flux - Model B, clearsky` | `model_b_clearsky_surface_shortwave_downward_flux` |
| `sfc_lw_down` | `CERES downward LW surface flux - Model B` | `model_b_surface_longwave_downward_flux` |
| `sfc_lw_down_clr` | `CERES downward LW surface flux - Model B, clearsky` | `model_b_clearsky_surface_longwave_downward_flux` |
| `sfc_sw_net` | `CERES net SW surface flux - Model B` | `model_b_surface_shortwave_net_flux` |
| `sfc_lw_net` | `CERES net LW surface flux - Model B` | `model_b_surface_longwave_net_flux` |

Surface fluxes standardize on **Model B** — the only parameterization with
all-sky + clear-sky coverage in both editions. A `variables:` entry whose
name is not in the catalog must provide `source_name:` (raw SDS name for
HDF4; `"Group/var"` path for netCDF); a catalog name used with an explicit
`source_name:` honors the override.

## Config Example

```yaml
sources:
  cam:
    type: cesm_fv
    role: model
    files: ${DATA}/cam/*.nc
    variables: { FLNT: { unit_scale: 1.0 } }
  ceres:
    type: ceres_ssf
    role: obs
    files: /Volumes/Io/CERES/SSF/NOAA20-FM6/*.nc
    variables: { toa_lw_up: { obs_min: 0, obs_max: 500 } }

pairs:
  cam_vs_ceres_olr:
    sources: [cam, ceres]
    reference: ceres
    variables: { cam: FLNT, ceres: toa_lw_up }
```

(SWATH outranks GRID as pairing reference by geometry precedence;
`SwathGridStrategy` defaults to `grid_mode: match_model`.)

## Testing

Per CLAUDE.md testing rules — integration means
`PipelineRunner.run_from_config`, and the test design below names the
entry points exercised.

1. **Unit — `ceres_ebaf`**: synthetic CF netCDF in `tmp_path` (monthly
   vars + one `ctime` climatology var, lon 0–360): variable selection,
   `ctime` dropped, lon normalized/sorted, `geometry == "grid"`, registry
   resolution.
2. **Unit — `ceres_syn1deg`**: synthetic HDF4 written via pyhdf (2-D
   month/day files and a 3-D `gmt_hr_index` file, with `_FillValue`/
   `valid_range` attrs): filename-time parsing for all three cadences,
   masking, multi-file concat on time, `geometry == "grid"`.
3. **Unit — `ceres_ssf`**: synthetic HDF4 (1-D SDS, colatitude, 0–360
   lon, Julian time) AND synthetic grouped netCDF (Footprints dim):
   canonical catalog resolves identically on both, colat→lat, lon wrap,
   Julian→datetime64 spot-check against a known timestamp, mixed-format
   rejection, `source_name` escape, `geometry == "swath"`.
4. **Integration (pipeline)**: three configs, each pairing synthetic CERES
   data against a synthetic `generic` gridded model through
   `PipelineRunner.run_from_config`: EBAF (GRID–GRID), SYN1deg
   (GRID–GRID), SSF (SWATH→`SwathGridStrategy` binning — the design's
   main risk, pinned here). Assert success, paired variables carry
   `role`/`source_label` attrs, stats/plots produced.
5. **Real-data smokes** (skipif `/Volumes/Io/CERES` absent): open each
   staged product; assert dims/coords/ranges — EBAF global-mean
   `toa_lw_all_mon` in 220–260 W m⁻²; SSF lat within ±90 and times inside
   the granule hour; **SYN1deg-vs-EBAF zonal-mean correlation for
   2025-12 > 0.9** (catches a flipped latitude axis, which would
   anti-correlate).
6. Gates: `pytest`, `mypy davinci_monet`, `black`, `isort` in the
   `davinci` conda env (`HDF5_USE_FILE_LOCKING=FALSE`).

## Implementation Sequence

Three phases, each its own implementation plan, verified before the next:

1. **Phase 1 — `ceres_ebaf`** (smallest; establishes the L3 test scaffold
   and the integration-config pattern).
2. **Phase 2 — `ceres_syn1deg`** (pyhdf grid assembly; reuses Phase 1
   scaffold).
3. **Phase 3 — `ceres_ssf`** (dual-format + catalog + binning
   integration; largest).
