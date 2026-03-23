# Changelog

## 1.0.0 (2026-03-23)

Initial public release for JOSS submission.

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
- Satellite L2: TROPOMI, MODIS, TEMPO
- Satellite L3: MOPITT, OMPS, GOES
- Lightning: LMA

### Quality

- 1030 tests passing, 0 warnings
- CI via GitHub Actions (pytest with coverage gate, black, isort, mypy)
- Zero mypy errors across 156 source files
