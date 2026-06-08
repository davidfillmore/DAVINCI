# Changelog

## 26.06 (2026-06-08)

**Breaking release.** Completes the model/obs → unified data-source migration and
removes the legacy `model:`/`obs:` config and runtime. Convert existing control files
with `davinci-monet migrate-config <old.yaml> -o <new.yaml>`.

### Breaking changes

- Legacy `model:`/`obs:` configuration is no longer accepted: a config containing a
  non-empty `model:`/`obs:` block now raises `ConfigurationError` pointing to
  `migrate-config`. Use the unified `sources:`/`pairs:` schema.
- Removed legacy Python APIs: `ModelConfig`/`ObservationConfig`; the
  `ModelData`/`ObservationData` containers (+ `create_model_data`/`create_observation_data`
  and the `open_*` convenience functions — reader classes return `xr.Dataset` directly);
  `LoadModelsStage`/`LoadObservationsStage`; `PairingEngine.pair()` (use `pair_sources`);
  the `ModelReader`/`ObservationReader` protocols; the `model_registry`/`observation_registry`
  aliases (use `source_registry`); `expand_sources_to_legacy`; `PipelineContext.models`/
  `observations` + `get_model`/`get_observation`; and the `PairedData`
  `get_obs`/`get_model`/`model_variables`/`obs_variables` aliases (use the
  reference/comparand accessors).

### Fixed

- The production `SwathGridStrategy` is now the engine's SWATH handler — swath L2 paired
  via `sources:` was silently using a non-production per-pixel strategy; also fixed a
  per-scanline time-broadcast crash.
- The unified `sources:` loader now applies `resample`/`min_obs_count`/`track_obs_count`
  (previously silently dropped on the unified path).
- Consistent `geometry` attribute encoding across observation readers (several wrote an
  integer the consumers silently ignored).
- Stats metric failures are logged rather than silently coerced to NaN;
  `PipelineResult.stage_errors` surfaces per-stage errors; HDF5 file locking is disabled
  by default to pre-empt the documented thread-safety segfaults.

### Changed

- One `render(series)` contract for all 18 plot renderers; the pipeline routes both
  paired and single-source plotting through it.
- Pairing runs through a single `pair_sources` path with an HDF5-safe bounded cross-pair
  executor (eager pairs run bounded-parallel; dask-backed pairs serial by default; tune
  via `pairing.max_pair_workers` / `pairing.dask_pair_workers`).
- Shared reader plumbing (`io/reader_utils.py`) deduplicates the model and observation
  readers.
- Large modules split for maintainability: `stages.py` → `pipeline/stages/` package;
  `runner.py` → `runner.py` + `display.py` + `reporting.py`; `plots/base.py` →
  `base.py` + `plot_config.py` + `labels.py` + `series.py`.
- The structured `logging/` subsystem is now activated at the CLI entry; dead `io/`
  helpers and stale compiled-bytecode directories removed.

### Quality

- 1262 tests passing; mypy clean (225 source files); black/isort clean — gates run locally
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
- **Type-safe configuration**: Pydantic-validated YAML with environment variable expansion and backward compatibility with MELODIES-MONET configs
- **CLI**: `davinci-monet run`, `validate`, and `get` commands via Typer

### Model Support

- CESM/CAM-chem (hybrid sigma-pressure coordinates)
- CMAQ
- WRF-Chem
- UFS-AQM
- Generic NetCDF

### Observation Support

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
