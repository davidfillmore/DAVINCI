# DAVINCI-MONET

**Data Analysis and Validation Infrastructure for Numerical Chemistry Intercomparison - Model and ObservatioN Evaluation Toolkit**

A modern, type-safe Python toolkit for evaluating atmospheric chemistry and air quality models against observations. DAVINCI-MONET is a Claude Code (Opus 4.5) AI assisted refactor of [MELODIES-MONET](https://github.com/NOAA-CSL/MELODIES-MONET) with improved architecture, full type hints, and comprehensive testing.

## Features

- **Unified Pairing Engine** - Single pairing system based on data geometry (point, track, profile, swath, grid)
- **Multiple Model Support** - CMAQ, WRF-Chem, UFS, CESM, and generic NetCDF
- **27 Statistical Metrics** - Bias, error, correlation, and agreement metrics with groupby support
- **Multiple Plot Types** - Time series, scatter, Taylor diagrams, spatial maps, 3D track, curtain, and more
- **Type-Safe Configuration** - Pydantic-validated YAML configs with backward compatibility
- **Full Test Coverage** - 792+ tests with synthetic data generation

### Supported Observations

| Type | Reader | Description | Variables |
|------|--------|-------------|-----------|
| **Surface** | AirNow | EPA real-time air quality | O3, PM2.5, NO2, CO |
| | AQS | EPA Air Quality System | O3, PM2.5, NO2, SO2, CO |
| | AERONET | Aerosol Robotic Network | AOD, Angstrom exponent |
| | OpenAQ | Global air quality platform | O3, PM2.5, NO2, SO2, CO |
| **Column** | Pandora | Ground-based spectrometers | Tropospheric NO2 column |
| **Sonde** | Ozonesonde | Balloon profiles | O3 vertical profiles |
| **Aircraft** | ICARTT | NASA/NOAA flight campaigns | Multiple trace gases |
| **Satellite L2** | TROPOMI | Polar; Sentinel-5P | NO2, O3, CO, HCHO, SO2 |
| | MODIS | Polar; Terra/Aqua | AOD |
| | TEMPO | Geo; hourly N. America | NO2, O3, HCHO |
| **Satellite L3** | MOPITT | Polar; Terra | CO |
| | OMPS | Polar; Suomi-NPP | Total O3 |
| | GOES | Geo; GOES-R/S | AOD |

<details>
<summary><strong>Observation Acronyms</strong></summary>

| Acronym | Full Name |
|---------|-----------|
| AirNow | EPA Real-Time Air Quality Index |
| AQS | EPA Air Quality System |
| AERONET | Aerosol Robotic Network |
| OpenAQ | Open Air Quality (global platform) |
| Pandora | Pandora Global Network (ground-based spectrometers) |
| ICARTT | International Consortium for Atmospheric Research on Transport and Transformation |
| TROPOMI | TROPOspheric Monitoring Instrument (Sentinel-5P) |
| MODIS | Moderate Resolution Imaging Spectroradiometer (Terra/Aqua) |
| TEMPO | Tropospheric Emissions: Monitoring of Pollution |
| MOPITT | Measurements of Pollution in the Troposphere (Terra) |
| OMPS | Ozone Mapping and Profiler Suite (Suomi-NPP) |
| GOES | Geostationary Operational Environmental Satellite |

</details>

<details>
<summary><strong>Model Acronyms</strong></summary>

| Acronym | Full Name |
|---------|-----------|
| CESM | Community Earth System Model |
| CAM-chem | Community Atmosphere Model with Chemistry |
| WRF-Chem | Weather Research and Forecasting model with Chemistry |
| CMAQ | Community Multiscale Air Quality model |
| UFS-AQM | Unified Forecast System - Air Quality Model |

</details>

## Quick Start

```bash
# Install from environment file
git clone https://github.com/NCAR/DAVINCI-MONET.git
cd DAVINCI-MONET
conda env create -f environment.yml
conda activate davinci-monet

# Run analysis
davinci-monet run config.yaml

# Validate config
davinci-monet validate config.yaml
```

## Minimal Example

```python
from davinci_monet.config import load_config
from davinci_monet.pipeline import PipelineRunner, PipelineContext

# Load and run
config = load_config("config.yaml")
runner = PipelineRunner()
result = runner.run(PipelineContext(config=config))

print(f"Success: {result.success}")
```

## Examples

The `examples/` directory contains complete working examples for different observation types:

| Example | Description | Plots |
|---------|-------------|-------|
| [`surface_evaluation.py`](examples/surface_evaluation.py) | Surface station evaluation | Scatter, diurnal cycle, spatial bias |
| [`aircraft_evaluation.py`](examples/aircraft_evaluation.py) | Aircraft track evaluation | Flight path, curtain, scatter, time series, vertical profile |
| [`satellite_evaluation.py`](examples/satellite_evaluation.py) | Satellite L2/L3 evaluation | Swath footprint, bias maps, scatter density, histograms |
| [`sonde_evaluation.py`](examples/sonde_evaluation.py) | Ozonesonde profile evaluation | Launch map, profiles, mean/bias, layer statistics |
| [`all_plot_types.py`](examples/all_plot_types.py) | All 10 plot types demo | Time series, diurnal, scatter, Taylor, box, spatial, curtain, scorecard |

Run any example:
```bash
python examples/aircraft_evaluation.py
```

Output is saved to `examples/output/<name>/` as both PNG (300 DPI) and PDF.

## Analyses

The `analyses/` directory contains real-world model evaluation studies:

| Analysis | Model | Observations | Period | Description |
|----------|-------|--------------|--------|-------------|
| [`ASIA-AQ`](analyses/asia-aq/) | CESM/CAM-chem | AirNow, AERONET, Pandora | Feb 2024 | NASA ASIA-AQ campaign evaluation |

See the [ASIA-AQ Analysis](../../wiki/ASIA-AQ-Analysis) wiki page for detailed documentation.

## Documentation

See the [Wiki](../../wiki) for full documentation:

- [Installation](../../wiki/Installation) - Setup and dependencies
- [Configuration](../../wiki/Configuration) - YAML configuration guide
- [CLI Reference](../../wiki/CLI-Reference) - Command-line interface
- [API Reference](../../wiki/API-Reference) - Python API documentation
- [Examples](../../wiki/Examples) - Detailed example walkthroughs
- [Migration Guide](../../wiki/Migration-Guide) - Migrating from MELODIES-MONET

## Architecture

```
davinci_monet/
├── config/       # Pydantic schemas, YAML parsing
├── models/       # Model readers (CMAQ, WRF-Chem, UFS, CESM)
├── observations/ # Observation handlers by type
├── pairing/      # Unified pairing engine + strategies
├── plots/        # Plotting system with registry
├── stats/        # Statistics calculation
├── pipeline/     # Execution orchestration
├── io/           # File readers/writers
└── cli/          # Command-line interface
```

## Data Flow

```
Model Files ──► Model Reader ──► xr.Dataset ──┐
                                              ├──► Pairing Engine ──► Paired Dataset
Obs Files ────► Obs Reader ───► xr.Dataset ──┘         │
                                                       ▼
                                              Statistics + Plots
```

## Requirements

- Python 3.10+
- Core: xarray, numpy, pandas, matplotlib, cartopy
- I/O: netCDF4, monet, monetio
- Config: pydantic, pyyaml
- CLI: typer

## License

Apache 2.0

---

> *Leonardo da Vinci and Claude Monet were both extraordinary observers of the natural world, and their artistic legacies resonate with the scientific mission of atmospheric model evaluation. Da Vinci, the Renaissance polymath, meticulously documented natural phenomena in his notebooks—from the mechanics of water flow to the blue haze of distant mountains, recognizing what we now call atmospheric perspective. His sfumato technique—from the Italian "to evaporate like smoke"—used subtle blending without harsh outlines to capture how the atmosphere softens and scatters light between observer and subject. Monet, the Impressionist master, devoted his career to capturing the ephemeral interplay of light and atmosphere. His serial paintings of haystacks, Rouen Cathedral, and the Thames recorded the same scenes under varying atmospheric conditions—fog, sunrise, midday sun—essentially conducting visual experiments on how the atmosphere transforms what we see. Both artists understood that the atmosphere is not empty space but an active medium that shapes our perception of the world. DAVINCI-MONET inherits this spirit of careful observation: just as da Vinci and Monet compared their perceptions against nature itself, this toolkit compares numerical model predictions against real-world observations, validating our mathematical representations of atmospheric chemistry against the truth that only measurement can provide.*
