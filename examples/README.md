# DAVINCI Plot Examples

This directory contains examples demonstrating all 15 plot types available in DAVINCI, using synthetic data and the `davinci_monet.plots` module.

## Quick Start

```bash
# Run all examples
python run_all_examples.py

# Run a single example
python plot_01_timeseries.py
```

Output is saved to `output/plots/` as PNG (300 DPI) and PDF.

## Plot Types

| # | Script | Plot Type | Data Geometry | Description |
|---|--------|-----------|---------------|-------------|
| 1 | `plot_01_timeseries.py` | `timeseries` | Point | Time series with uncertainty bands |
| 2 | `plot_02_diurnal.py` | `diurnal` | Point | Mean diurnal cycle comparison |
| 3 | `plot_03_scatter.py` | `scatter` | Point, Swath | Dataset vs geometry scatter with regression |
| 4 | `plot_04_taylor.py` | `taylor` | Point | Taylor diagram (correlation, std, RMSE) |
| 5 | `plot_05_boxplot.py` | `boxplot` | Point | Distribution comparison |
| 6 | `plot_06_spatial_bias.py` | `spatial_bias` | Point | Geographic bias distribution |
| 7 | `plot_07_spatial_overlay.py` | `spatial_overlay` | Point + Grid | Dataset contours + geometry points |
| 8 | `plot_08_spatial_distribution.py` | `spatial_distribution` | Point | Geographic value distribution |
| 9 | `plot_09_curtain.py` | `curtain` | Track | Vertical cross-section (time x altitude) |
| 10 | `plot_10_scorecard.py` | `scorecard` | Point | Multi-metric performance heatmap |
| 11 | `plot_11_site_timeseries.py` | `site_timeseries` | Point | Multi-panel site-by-site comparison |
| 12 | `plot_12_flight_timeseries.py` | `flight_timeseries` | Track | Multi-panel flight-by-flight comparison |
| 13 | `plot_13_track_map_3d.py` | `track_map_3d` | Track | 3D flight trajectory visualization |
| 14 | `plot_14_satellite_swath.py` | `spatial_bias`, `spatial_distribution` | Swath | Satellite swath (TROPOMI-like) plots |
| 15 | `plot_15_satellite_gridded.py` | pcolormesh | Grid | L3 gridded satellite data visualization |

## Architecture

All examples use the same pattern:

```python
from davinci_monet.plots import plot_scatter  # or any plot type
from _helpers import create_paired_surface_data, save_figure

# Generate synthetic paired data
paired = create_paired_surface_data(n_sites=30, variables=["O3"])

# Create plot using davinci_monet.plots
fig = plot_scatter(paired, geometry_var="geometry_o3", dataset_var="dataset_o3", title="My Plot")

# Save output
save_figure(fig, "my_scatter")
```

### Helper Module (`_helpers.py`)

Provides functions to create paired dataset-dataset datasets:

| Function | Geometry | Dimensions |
|----------|----------|------------|
| `create_paired_surface_data()` | Point | `(time, site)` |
| `create_paired_track_data()` | Track | `(time,)` + lat/lon/alt coords |
| `create_paired_profile_data()` | Profile | `(time, level)` |
| `create_paired_swath_data()` | Swath | `(scanline, pixel)` |
| `create_paired_gridded_data()` | Grid | `(time, lat, lon)` |

## Using in Your Analysis

Import plotters directly:

```python
from davinci_monet.plots import (
    plot_scatter,
    plot_spatial_bias,
    plot_taylor,
    plot_timeseries,
    # ... etc
)

# With your paired data
fig = plot_scatter(my_paired_data, "geometry_pm25", "dataset_pm25")
fig.savefig("my_scatter.png")
```

Or use the registry:

```python
from davinci_monet.plots import get_plotter

plotter = get_plotter("scatter")
fig = plotter.plot(my_paired_data, "geometry_pm25", "dataset_pm25")
```

## Requirements

DAVINCI must be installed:

```bash
conda activate davinci
pip install -e ..
```
