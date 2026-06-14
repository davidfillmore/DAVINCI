# DAVINCI

**Data Analysis and Visual Intelligence for Climate**

A modern, type-safe Python toolkit for comparing climate and atmospheric composition datasets.

## Features

- **Unified Pairing Engine** - Single pairing system based on data geometry (point, track, profile, swath, grid)
- **Multiple Dataset Support** - CMAQ, WRF-Chem, UFS, CESM, and generic NetCDF
- **27 Statistical Metrics** - Bias, error, correlation, and agreement metrics with groupby support
- **Multiple Plot Types** - Time series, scatter, Taylor diagrams, spatial maps, 3D track, curtain, and more
- **Type-Safe Configuration** - Pydantic-validated YAML configs
- **Full Test Coverage** - 1000+ tests with synthetic data generation

### Supported Datasets

| Type | Reader | Description | Variables |
|------|--------|-------------|-----------|
| **Surface** | AirNow | EPA real-time air quality | O3, PM2.5, NO2, CO |
| | AQS | EPA Air Quality System | O3, PM2.5, NO2, SO2, CO |
| | AERONET | Aerosol Robotic Network | AOD, Angstrom exponent |
| | OpenAQ | Global air quality platform | O3, PM2.5, NO2, SO2, CO |
| **Column** | Pandora | Ground-based spectrometers | Tropospheric NO2 column |
| **Sonde** | Ozonesonde | Balloon profiles | O3 vertical profiles |
| **Aircraft** | ICARTT | NASA/NOAA flight campaigns | Multiple trace gases |
| **Satellite L2** | MODIS | Polar; Terra/Aqua | AOD |
| **Lightning** | LMA | Lightning Mapping Array | Flash density, source density |

**In development** (readers exist, not yet validated against real data):

| Type | Reader | Description | Status |
|------|--------|-------------|--------|
| Satellite L2 | TROPOMI | Sentinel-5P | Needs averaging kernel support |
| | TEMPO | Hourly N. America | Needs averaging kernel support |
| Satellite L3 | MOPITT | Terra CO | Needs averaging kernel support |
| | OMPS | Suomi-NPP Total O3 | Untested |
| | GOES | GOES-R/S AOD | Untested |

<details>
<summary><strong>Dataset Acronyms</strong></summary>

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
| LMA | Lightning Mapping Array |

</details>

<details>
<summary><strong>Dataset Acronyms</strong></summary>

| Acronym | Full Name |
|---------|-----------|
| CESM | Community Earth System Dataset |
| CAM-chem | Community Atmosphere Dataset with Chemistry |
| WRF-Chem | Weather Research and Forecasting dataset with Chemistry |
| CMAQ | Community Multiscale Air Quality dataset |
| UFS-AQM | Unified Forecast System - Air Quality Dataset |

</details>

## Quick Start

```bash
# Install from environment file
git clone https://github.com/NCAR/DAVINCI.git
cd DAVINCI
conda env create -f environment.yml
conda activate davinci

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

## AI Summary (Visual Intelligence)

Enable an optional final stage that asks Claude to read the run's statistics and
plots and write a structured markdown brief (`AI_summary.md`) into the output
directory. Requires the `[ai]` extra and an Anthropic API key.

```bash
pip install -e ".[ai]"
export ANTHROPIC_API_KEY=sk-ant-...
```

```yaml
summary:
  enabled: true
  dataset: claude-haiku-4-5          # cheapest vision dataset; bump to claude-sonnet-4-6
  plots: [scatter_o3, spatial_bias_o3]   # optional; omit to send up to max_images
  max_images: 8
  instructions: "Focus on coastal sites."   # optional steering
```

To use OpenRouter instead of the Anthropic API directly (e.g. with a key in a
file), set the provider and point at the key file:

```yaml
summary:
  enabled: true
  provider: openrouter
  api_key_file: OpenRouter.api          # gitignored; falls back to api_key_env
  dataset: anthropic/claude-haiku-4.5     # OpenRouter dataset id (default for this provider)
```

The stage is always non-fatal: with no key or no network it logs a warning and
is skipped, and the analysis run still succeeds.

## Examples

The `examples/` directory contains individual plot type examples using `davinci_monet.plots`:

| Plot Type | Script | Data Geometry |
|-----------|--------|---------------|
| Time Series | `plot_01_timeseries.py` | Point |
| Diurnal Cycle | `plot_02_diurnal.py` | Point |
| Scatter | `plot_03_scatter.py` | Point, Swath |
| Taylor Diagram | `plot_04_taylor.py` | Point |
| Box Plot | `plot_05_boxplot.py` | Point |
| Spatial Bias | `plot_06_spatial_bias.py` | Point |
| Spatial Overlay | `plot_07_spatial_overlay.py` | Point + Grid |
| Spatial Field (single-source) | `plot_08_spatial_field.py` | All geometries |
| Curtain | `plot_09_curtain.py` | Track |
| Scorecard | `plot_10_scorecard.py` | Point |
| 3D Track Map | `plot_13_track_map_3d.py` | Track |
| Satellite Swath | `plot_14_satellite_swath.py` | Swath |
| Satellite Gridded | `plot_15_satellite_gridded.py` | Grid |

Run all examples:
```bash
cd examples && python run_all_examples.py
```

Output is saved to `examples/output/plots/` as both PNG (300 DPI) and PDF.

## Analyses

The `analyses/` directory contains real-world dataset evaluation studies:

| Analysis | Dataset | Datasets | Period | Description |
|----------|-------|--------------|--------|-------------|
| [`ASIA-AQ`](analyses/asia-aq/) | CESM/CAM-chem | AirNow, AERONET, Pandora | Feb 2024 | NASA ASIA-AQ campaign evaluation |

See the [ASIA-AQ Analysis](../../wiki/ASIA-AQ-Analysis) wiki page for detailed documentation.

## Documentation

See the [Wiki](../../wiki) for full documentation:

- [Installation](../../wiki/Installation) - Setup and dependencies
- [Configuration](../../wiki/Configuration) - YAML configuration guide
- [CLI Geometry](../../wiki/CLI-Geometry) - Command-line interface
- [API Geometry](../../wiki/API-Geometry) - Python API documentation
- [Examples](../../wiki/Examples) - Detailed example walkthroughs

## Architecture

```
davinci_monet/
├── config/       # Pydantic schemas, YAML parsing
├── datasets/       # Dataset readers (CMAQ, WRF-Chem, UFS, CESM)
├── datasets/ # Dataset handlers by type
├── pairing/      # Unified pairing engine + strategies
├── plots/        # Plotting system with registry
├── stats/        # Statistics calculation
├── pipeline/     # Execution orchestration
├── io/           # File readers/writers
└── cli/          # Command-line interface
```

## Data Flow

```
Dataset Files ──► Dataset Reader ──► xr.Dataset ──┐
                                              ├──► Pairing Engine ──► Paired Dataset
Geometry Files ────► Geometry Reader ───► xr.Dataset ──┘         │
                                                       ▼
                                              Statistics + Plots
```

## Requirements

- Python 3.11+
- Core: xarray, numpy, pandas, matplotlib, cartopy
- I/O: netCDF4, monet, monetio
- Config: pydantic, pyyaml
- CLI: typer

## License

Apache 2.0

---

> *Leonardo da Vinci and Claude Monet both studied the natural world with unusual care, and that attention resonates with atmospheric dataset evaluation. Da Vinci documented natural phenomena in his notebooks, from water flow to the blue haze of distant mountains, recognizing what we now call atmospheric perspective. Monet devoted his career to the changing interplay of light and atmosphere. His serial paintings of haystacks, Rouen Cathedral, and the Thames recorded the same scenes under varying atmospheric conditions: fog, sunrise, midday sun. DAVINCI inherits this spirit of careful comparison: the toolkit places numerical datasets beside measurement datasets, bringing visual intelligence to climate-system analysis.*
