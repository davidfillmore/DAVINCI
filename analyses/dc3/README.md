# DC3 Analysis

Dataset-only analysis for the NASA/NSF **Deep Convective Clouds and Chemistry (DC3)** field campaign (May-June 2012).

## Campaign Overview

DC3 studied the impact of deep midlatitude continental convective clouds on upper tropospheric composition and chemistry. Three research aircraft (NASA DC-8, NSF/NCAR GV, DLR Falcon) sampled thunderstorms from inflow to outflow across three U.S. regions: NE Colorado, N. Alabama, and Oklahoma/W. Texas.

**Geometry:** Barth et al. (2015), *Bull. Amer. Meteor. Soc.*, 96, 1281-1309. [doi:10.1175/BAMS-D-13-00290.1](https://doi.org/10.1175/BAMS-D-13-00290.1)

## Quick Start

### 1. Download Data

```bash
# Requires earthaccess: pip install earthaccess
# Requires Earthdata Login credentials (~/.netrc or interactive)
python scripts/download_dc3_aircraft.py
```

Downloads ICARTT merge files to `~/Data/DC3/aircraft/merge/`.

### 2. Run Analysis

```bash
# DC-8 geometry-only analysis (default)
python scripts/run_geometry_analysis.py dc3-geometry-dc8

# GV geometry-only analysis
python scripts/run_geometry_analysis.py dc3-geometry-gv

# Combined DC-8 + GV
python scripts/run_geometry_analysis.py dc3-geometry-all-aircraft
```

## Directory Structure

```
dc3/
├── configs/
│   ├── dc3-geometry-dc8.yaml            # DC-8 geometry-only
│   ├── dc3-geometry-gv.yaml             # GV geometry-only
│   └── dc3-geometry-all-aircraft.yaml   # Combined DC-8 + GV
├── scripts/
│   ├── download_dc3_aircraft.py    # Download ICARTT from NASA ASDC
│   └── run_geometry_analysis.py         # Run geometry-only pipeline
├── data/                            # Downloaded datasets
├── output/                          # Plots and statistics
└── logs/                            # Pipeline logs
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DC3_DATA` | `~/Data/DC3` | Root data directory |
| `DC3_ANALYSIS` | *(auto-set by run script)* | Analysis directory |

## Key Species

| Variable | DC-8 Instrument | GV Instrument |
|----------|-----------------|---------------|
| NO | NO_ESRL | NO_NOxyO3 |
| NO2 | NO2_TDLIF | NO2_NOxyO3 |
| O3 | O3_ESRL | O3_NOxyO3 |
| CO | CO_DACOM | CO_ACOMCO |
| NOy | NOy_ESRL | NOy_NOxyO3 |

## Plot Types

The geometry-only pipeline produces:
- **Flight track maps** -- 2D Cartopy maps with flight paths colored by variable value
- **Vertical profiles** -- Altitude vs. concentration (scatter or binned)
- **Time series** -- Variable along flight time, with optional altitude shading
- **Histograms** -- Distribution of dataset values with summary statistics

## Data Source

NASA ASDC: [asdc.larc.nasa.gov/project/DC3](https://asdc.larc.nasa.gov/project/DC3)

ICARTT merge files (10-second resolution):
- `dc3-mrg10-dc8_merge_*.ict` -- DC-8 aircraft
- `dc3-mrg10-gv_merge_*.ict` -- GV aircraft
