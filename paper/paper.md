---
title: 'DAVINCI-MONET: A Type-Safe, Geometry-Aware Toolkit for Atmospheric Chemistry Model Evaluation'
tags:
  - Python
  - atmospheric chemistry
  - model evaluation
  - air quality
  - xarray
  - satellite remote sensing
authors:
  - name: David Fillmore
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: National Center for Atmospheric Research, Boulder, CO, USA
    index: 1
date: 19 March 2026
bibliography: paper.bib
---

# Summary

DAVINCI-MONET (Data Analysis and Validation Infrastructure for Numerical
Chemistry Intercomparison -- Model and ObservatioN Evaluation Toolkit) is a
Python package for evaluating atmospheric chemistry and air quality models
against observations. It provides a config-driven pipeline that pairs model
output with surface, aircraft, sonde, satellite, and lightning observations
using geometry-aware strategies, then computes standard evaluation statistics
and generates publication-ready plots. DAVINCI-MONET supports both paired
model-versus-observation workflows and observation-only workflows for field
campaign characterization, all driven by a single YAML control file.

# Statement of need

Atmospheric chemistry model evaluation requires pairing model output with
observations that span fundamentally different data geometries: fixed surface
stations (points), aircraft flight tracks, vertical profiles, satellite swaths,
and gridded products. Existing evaluation workflows are often fragmented across
observation type, with geometry-specific pairing code duplicated or handled
ad hoc for each campaign or satellite product. This fragmentation makes
evaluation workflows difficult to reproduce, extend to new observation types,
or share across research groups.

DAVINCI-MONET addresses this by providing a unified, config-driven evaluation
runtime where the pairing strategy is selected automatically based on data
geometry rather than data source. A single YAML control file specifies model
files, observation files, variable mappings, plot types, and statistical
metrics. The pipeline validates the configuration, loads data, pairs
observations with model output, computes statistics, generates plots, and
writes structured logs -- all from one command:

```bash
davinci-monet run config.yaml
```

Target users are atmospheric chemistry model developers and analysis teams who
need reproducible evaluation across multiple observation types and campaigns.

# State of the field

DAVINCI-MONET is a ground-up refactor of MELODIES-MONET
[@melodies_monet], an evaluation toolkit developed at NOAA CSL.
While MELODIES-MONET established the concept of a Python-based model
evaluation framework backed by monetio and monet, DAVINCI-MONET differs
in several fundamental ways:

- **Architecture**: MELODIES-MONET uses a procedural workflow
  (`.open_models()` then `.open_obs()` then `.pair_data()`).
  DAVINCI-MONET replaces this with a composable, stage-based pipeline
  where each stage implements a common `Stage` Protocol and communicates
  through a shared `PipelineContext`.
- **Type safety**: DAVINCI-MONET enforces full mypy strict mode across
  the codebase, validates configuration with Pydantic schemas at parse
  time, and ships a `py.typed` marker for downstream type checking.
- **Geometry-driven pairing**: Rather than pairing by data source,
  DAVINCI-MONET auto-detects observation geometry (point, track,
  profile, swath, grid) from dataset structure and dispatches to
  specialized pairing strategies.
- **Observation-only mode**: When no model section is present in the
  configuration, the pipeline automatically switches to an
  observation-only stage sequence with dedicated plot renderers --
  a capability not available in MELODIES-MONET.
- **Satellite swath-to-grid binning**: DAVINCI-MONET includes
  numba-accelerated binning of satellite L2 swath pixels onto regular
  grids, with configurable grid modes and pixel-count tracking.
- **Testing**: The package includes over 900 tests with synthetic data
  fixtures covering all pairing strategies and plot types.

Other tools in the atmospheric evaluation space include the Model
Evaluation Tools (MET) framework [@met_framework] and the Atmospheric
Model Evaluation Tool (AMET) [@amet], which focus on meteorological
and air quality verification respectively. DAVINCI-MONET complements
these by providing a Python-native, xarray-first workflow that handles
the full range of observation geometries encountered in atmospheric
chemistry field campaigns.

# Software design

DAVINCI-MONET is organized into composable modules:

- **Configuration** (`config/`): Pydantic-validated YAML loading with
  environment variable expansion and a `validate` CLI command for
  pre-run checking.
- **Pipeline** (`pipeline/`): Stage-based execution with progress
  tracking and structured markdown logs. Standard mode runs six stages
  (load models, load observations, pairing, statistics, plotting, save
  results); observation-only mode runs a reduced four-stage sequence.
- **Pairing** (`pairing/`): A `PairingEngine` with five geometry-based
  strategies (point, track, profile, swath, grid) plus a
  numba-accelerated swath-to-grid strategy for satellite L2 products.
- **Statistics** (`stats/`): 27 evaluation metrics including bias, RMSE,
  correlation, index of agreement, and normalized metrics, with groupby
  support by time, site, or altitude.
- **Plotting** (`plots/`): 14 paired plot types and 5 observation-only
  plot types, registered via a plugin registry. A comprehensive gallery
  of all supported plot types is available in the repository
  documentation (`docs/gallery/`).
- **Observations** (`observations/`): Readers for surface networks
  (AirNow, AQS, AERONET, OpenAQ, Pandora), aircraft (ICARTT), sonde,
  satellite L2 (TROPOMI, TEMPO, MODIS), satellite L3 (MOPITT, OMPS,
  GOES), and lightning (LMA).
- **Models** (`models/`): Readers for CMAQ, WRF-Chem, UFS, CESM
  (finite volume and spectral element), and generic NetCDF.

Performance optimizations include time filtering at observation load
(reducing load time by three orders of magnitude for multi-month files),
configurable Dask concurrency for pairing, and numba JIT-compiled grid
binning for satellite data.

# Research impact

DAVINCI-MONET has been applied to three distinct evaluation workflows
that demonstrate its breadth:

- **ASIA-AQ**: Multi-observation paired evaluation of CESM/CAM-chem
  against four observation networks (AirNow surface, AERONET AOD,
  Pandora NO$_2$ columns, DC-8 aircraft) over East and Southeast Asia.
- **DC3**: Observation-only characterization of the Deep Convective
  Clouds and Chemistry field campaign, including DC-8 and GV aircraft
  trace gas profiles and Oklahoma Lightning Mapping Array flash density.
- **MODIS AOD**: Satellite swath-to-grid evaluation of Terra and Aqua
  MODIS L2 aerosol optical depth against two CAM6 model variants during
  the December 2019 Australian bushfire event.

Each workflow is fully reproducible from checked-in YAML configurations
and download scripts in the repository's `analyses/` directory. Example
outputs from all three workflows are shown in the repository's plot
gallery (`docs/gallery/`).

# AI usage disclosure

Claude (Anthropic) and Codex (OpenAI) were used as coding assistants
throughout DAVINCI-MONET development, including implementation,
test writing, documentation, and paper planning. All AI-generated
code was reviewed, tested, and integrated by the authors.

# Acknowledgements

This work was supported by the National Center for Atmospheric Research,
which is a major facility sponsored by the National Science Foundation
under Cooperative Agreement No. 1852977.

DAVINCI-MONET builds on the foundation established by MELODIES-MONET
and the monetio and monet packages developed at NOAA CSL.

# References
