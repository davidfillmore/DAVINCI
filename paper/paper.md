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
Python package for evaluating atmospheric chemistry and air quality model
output against observations. The package combines validated YAML
configuration, geometry-aware pairing, evaluation statistics, and plotting in a
single stage-based workflow built around xarray datasets. DAVINCI-MONET
supports paired model-versus-observation analyses, observation-only workflows
for field campaigns, and satellite swath-to-grid evaluation within one software
stack. It is intended for atmospheric chemistry model developers and analysis
teams who need reproducible, scriptable evaluation across multiple observation
types.

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
runtime in which pairing behavior is selected from dataset geometry rather than
from observation source alone. A single YAML control file specifies model and
observation inputs, variable mappings, plot requests, and statistical
configuration. The runtime validates the configuration, loads data, performs
pairing, computes statistics, generates plots, and writes structured logs from
one command:

```bash
davinci-monet run config.yaml
```

This design reduces the amount of campaign-specific glue code needed to compare
one model against many observation classes, or to characterize an observation
campaign even when no model fields are available. Target users include
atmospheric chemistry model developers, air quality analysis teams, and field
campaign scientists who need evaluation workflows that are easier to review,
share, and rerun.

# State of the field

DAVINCI-MONET builds on ideas explored in MELODIES-MONET, a predecessor
toolkit developed at NOAA CSL, but it is not a simple rename or thin wrapper.
The software recasts model evaluation as a typed, stage-based pipeline rather
than a procedural sequence of reader and pairing calls. It also moves pairing
logic toward geometry-driven dispatch, adds validated configuration with
Pydantic schemas, supports observation-only execution when no model is present,
and includes satellite swath-to-grid binning for Level 2 products. Together,
these changes make DAVINCI-MONET a distinct software contribution aimed at
modern, reproducible evaluation workflows.

Other tools in the atmospheric evaluation space include the Model Evaluation
Tools (MET) framework [@met_framework] and the Atmospheric Model Evaluation
Tool (AMET) [@amet]. Those projects address adjacent verification problems, but
DAVINCI-MONET emphasizes a Python-native, xarray-first workflow that unifies
surface, aircraft, profile, swath, and gridded observations within one package.
Its primary contribution is not a new evaluation metric, but a software design
that makes heterogeneous atmospheric chemistry evaluation workflows easier to
configure, extend, and reuse.

# Software design

DAVINCI-MONET is organized around a small number of composable subsystems. The
configuration layer loads YAML control files, expands environment variables,
and validates structure before runtime. The pipeline layer executes named
stages with a shared context, allowing standard paired runs and reduced
observation-only runs to share the same execution model. The pairing layer uses
a `PairingEngine` and geometry-specific strategies for point, track, profile,
swath, and grid data, with an additional swath-to-grid path for satellite Level
2 products. Statistics and plotting are handled by dedicated modules that can
operate on paired outputs, while observation-only rendering uses a separate
plotter interface tailored to single-dataset workflows.

Reader coverage includes surface networks such as AirNow, AQS, AERONET,
OpenAQ, and Pandora; aircraft data through ICARTT; ozonesondes; satellite
Level 2 and Level 3 products; and lightning observations from LMA networks.
Model readers support CMAQ, WRF-Chem, UFS, CESM, and generic NetCDF inputs.
The package also includes performance-oriented features such as observation
time filtering during load, configurable Dask concurrency during pairing, and
numba-accelerated grid binning for satellite workflows. These implementation
choices are intended to make large evaluation runs more practical without
changing the user-facing configuration model.

# Research impact

DAVINCI-MONET has been applied to three distinct evaluation workflows
that demonstrate the breadth of the software:

- **ASIA-AQ**: Multi-observation paired evaluation of CESM/CAM-chem
  against four observation networks (AirNow surface, AERONET AOD,
  Pandora NO$_2$ columns, DC-8 aircraft) over East and Southeast Asia.
- **DC3**: Observation-only characterization of the Deep Convective
  Clouds and Chemistry field campaign, including DC-8 and GV aircraft
  trace gas profiles and Oklahoma Lightning Mapping Array flash density
  [@barth_dc3].
- **MODIS AOD**: Satellite swath-to-grid evaluation of Terra and Aqua
  MODIS L2 aerosol optical depth against two CAM6 model variants during
  the December 2019 Australian bushfire event.

These workflows are represented in the repository by checked-in configurations,
analysis scripts, and example outputs. They are included here as evidence that
the same package can support distinct workflow classes rather than as new
scientific results produced for this paper. Some workflows depend on external
datasets, preprocessing steps, or credentials for data access, so DAVINCI-MONET
does not claim that every analysis is push-button reproducible in a fresh
environment. Instead, the repository makes the configuration, acquisition
paths, and workflow structure explicit, which is the level of transparency most
useful for software review and reuse.

# AI usage disclosure

Generative AI tools were used during both DAVINCI-MONET software development
and manuscript preparation. Interactive sessions with Anthropic Claude and
OpenAI Codex-family agents were used during software architecture discussion,
implementation, refactoring, test scaffolding, and documentation revision.
These sessions often included cross-model review, where output proposed by one
system was critiqued, revised, or stress-tested with assistance from another
before human acceptance.

The same tools were also used during paper planning, editorial revision, and
early manuscript drafting. Human authors made the primary architectural,
scientific, and design decisions; reviewed and edited the generated code and
text; inspected or ran the relevant tests and outputs; and take full
responsibility for the accuracy, originality, licensing compliance, and final
content of both the software and the paper.

# Acknowledgements

This work was supported by the National Center for Atmospheric Research,
which is a major facility sponsored by the National Science Foundation
under Cooperative Agreement No. 1852977.

DAVINCI-MONET builds on the foundation established by MELODIES-MONET
and the monetio and monet packages developed at NOAA CSL.

# References
