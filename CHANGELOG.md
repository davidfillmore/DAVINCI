# Changelog

## 26.06 (2026-06-08)

**Breaking release.** Completes the dataset/geometry to unified data-source migration and
removes the `dataset:`/`geometry:` config blocks and runtime containers.

### Added

- **Derived-analysis layer** (`analyses:` config block) — single-source diagnostics
  whose outputs register as first-class pseudo-sources and run in dependency order
  after loading (derived sources are not pairable):
  - **EOF** decomposition of any 2-D/3-D gridded field — area/layer-mass-weighted
    anomalies via an in-repo SVD, North's-rule eigenvalue errors, optional varimax —
    producing spatial modes, unit-variance principal-component series, and explained
    variance.
  - **Wavelet** analysis (Torrence & Compo Morlet CWT via the new `pycwt` dependency)
    with AR(1) red-noise significance and a cone of influence, applied to any 1-D
    series: a station record, a domain mean, or an EOF principal component.
  - New single-source renderers `eof_pattern`, `eof_scree`, and `wavelet_scalogram`.

### Breaking changes

- `dataset:`/`geometry:` configuration is no longer accepted. Use the unified
  `sources:`/`pairs:` schema.
- Removed superseded Python APIs for separate dataset and geometry containers,
  loaders, registries, protocols, and accessors. Reader classes return `xr.Dataset`
  directly, and paired outputs use the geometry/dataset accessors.

### Fixed

- The production `SwathGridStrategy` is now the engine's SWATH handler. Swath L2 paired
  via `sources:` now uses the binned grid strategy; also fixed a
  per-scanline time-broadcast crash.
- The unified `sources:` loader now applies `resample`/`min_geometry_count`/`track_geometry_count`
  (previously silently dropped on the unified path).
- Consistent `geometry` attribute encoding across geometry readers (several wrote an
  integer the consumers silently ignored).
- Stats metric failures are logged rather than silently coerced to NaN;
  `PipelineResult.stage_errors` surfaces per-stage errors; HDF5 file locking is disabled
  by default to pre-empt the documented thread-safety segfaults.

### Changed

- One `render(series)` contract for all plot renderers; the pipeline routes both
  paired and single-source plotting through it.
- Pairing runs through a single `pair_sources` path with an HDF5-safe bounded cross-pair
  executor (eager pairs run bounded-parallel; dask-backed pairs serial by default; tune
  via `pairing.max_pair_workers` / `pairing.dask_pair_workers`).
- Shared reader plumbing (`io/reader_utils.py`) deduplicates the dataset and geometry
  readers.
- Large modules split for maintainability: `stages.py` → `pipeline/stages/` package;
  `runner.py` → `runner.py` + `display.py` + `reporting.py`; `plots/base.py` →
  `base.py` + `plot_config.py` + `labels.py` + `series.py`.
- The structured `logging/` subsystem is now activated at the CLI entry; dead `io/`
  helpers and stale compiled-bytecode directories removed.
- Official name expansion is now "Data Analysis and Visual Intelligence for
  Climate/Chemistry"; the JOSS paper title, package metadata, README, CLI, and logo
  updated to match.

### Quality

- 1544 tests passing; mypy clean (308 source files); black/isort clean — gates run locally
  in the `davinci` conda env. The CI workflow (`.github/workflows/ci.yml`) is configured to
  install the package and run the suite on a 3.11/3.12 matrix, but GitHub Actions is currently
  disabled for the repository, so it does not execute on push.

## 26.03 (2026-03-23)

Initial public release for JOSS submission. Calendar versioned (YY.MM).

### Core Features

- **Pipeline architecture**: Stage-based execution (load, pair, stats, plot, save) orchestrated by `PipelineRunner`
- **Unified pairing engine**: Geometry-aware strategies for point, track, profile, swath, and grid data
- **27 statistical metrics**: N, MB, RMSE, R, NMB, NME, IOA, and more with groupby support
- **14 plot types**: Time series, scatter, Taylor, boxplot, diurnal, spatial bias/distribution/overlay, curtain, scorecard, site/flight time series, 3D track map
- **Type-safe configuration**: Pydantic-validated YAML with environment variable expansion
- **CLI**: `davinci-monet run`, `validate`, and `get` commands via Typer

### Dataset Support

- CESM/CAM-chem (hybrid sigma-pressure coordinates)
- CMAQ
- WRF-Chem
- UFS-AQM
- Generic NetCDF

### Geometry Support

- Surface: AirNow, AQS, AERONET, OpenAQ
- Column: Pandora
- Sonde: Ozonesonde
- Aircraft: ICARTT
- Satellite L2: MODIS AOD (validated)
- Lightning: LMA
- In development: TROPOMI, TEMPO, MOPITT, OMPS, GOES (readers exist, need averaging kernel support and validation)

### Quality

- 1030 tests passing, 0 warnings
- CI via GitHub Actions (pytest with coverage gate, black, isort, mypy)
- Zero mypy errors across 156 source files
