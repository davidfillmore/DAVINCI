# FIREX-AQ Analysis

Observation and model evaluation analysis for the NOAA/NASA **Fire Influence on Regional to Global Environments and Air Quality (FIREX-AQ)** field campaign (July--September 2019).

## Campaign Overview

FIREX-AQ was a multi-platform intensive study of North American fires that deployed
the NASA DC-8, NASA ER-2, NOAA Twin Otters, ground sites, and mobile laboratories
to investigate smoke from ignition through chemical transformation to downwind air
quality impacts. DC-8 operations comprised 23 science flights (15 from Boise, ID
targeting western wildfires; 8 from Salina, KS targeting southeastern
agricultural/prescribed fires) between 22 July and 5 September 2019.

**Reference:** Warneke, C. et al. (2023), *J. Geophys. Res. Atmos.*, 128, e2022JD037758. [doi:10.1029/2022JD037758](https://doi.org/10.1029/2022JD037758)

## Quick Start

### 1. Download Data

```bash
# Requires earthaccess: pip install earthaccess
# Requires Earthdata Login credentials (~/.netrc or interactive)
python scripts/download_firex_aircraft.py
```

Downloads DC-8 ICARTT merge files to `~/Data/FIREX-AQ/aircraft/merge/`.

### 2. Run Analysis

```bash
# DC-8 obs-only analysis (default)
python scripts/run_obs_analysis.py firex-aq-obs-dc8

# Or run directly via CLI
davinci-monet run analyses/firex-aq/configs/firex-aq-obs-dc8.example.yaml
```

## Directory Structure

```
firex-aq/
├── configs/
│   ├── firex-aq-obs-dc8.example.yaml      # DC-8 obs-only (env-var paths)
│   └── firex-aq-model-dc8.example.yaml    # Model + DC-8 evaluation template
├── scripts/
│   ├── download_firex_aircraft.py          # Download ICARTT from NASA ASDC
│   └── run_obs_analysis.py                # Run obs-only pipeline
├── data/                                   # Downloaded observations
├── output/                                 # Plots and statistics
└── logs/                                   # Pipeline logs
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FIREX_AQ_DATA` | `~/Data/FIREX-AQ` | Root data directory |
| `FIREX_AQ_ANALYSIS` | *(auto-set by run script)* | Analysis directory |

## Key Species

Core trace gas species from the DC-8 ICARTT merge files:

| Variable | Merge Column | Instrument | PI |
|----------|-------------|------------|-----|
| O3 | O3_CL | NOAA Chemiluminescence | Tom Ryerson |
| CO | CO_DACOM | DACOM | Glenn Diskin |
| NO | NO_CL | NOAA Chemiluminescence | Tom Ryerson |
| NO2 | NO2_CL | NOAA Chemiluminescence | Tom Ryerson |
| CH2O | CH2O_ISAF | ISAF | Tom Hanisco |
| CO2 | CO2_DACOM | DACOM | Glenn Diskin |

## Deployment Phases

| Phase | Dates | Base | Targets |
|-------|-------|------|---------|
| Western Wildfire | 22 Jul -- 19 Aug 2019 | Boise, ID | NW U.S. wildfires |
| Agricultural/Small Fire | 19 Aug -- 5 Sep 2019 | Salina, KS | SE U.S. agricultural burns |

## Plot Types

The obs-only pipeline produces:
- **Flight track maps** -- 2D Cartopy maps with flight paths colored by variable value
- **Vertical profiles** -- Altitude vs. concentration (scatter or binned)
- **Time series** -- Variable along flight time, with optional altitude shading
- **Histograms** -- Distribution of observed values with summary statistics

## Data Source

- **NASA ASDC:** [asdc.larc.nasa.gov/project/FIREX-AQ](https://asdc.larc.nasa.gov/project/FIREX-AQ)
- **DOI:** [10.5067/SUBORBITAL/FIREXAQ2019/DATA001](https://doi.org/10.5067/SUBORBITAL/FIREXAQ2019/DATA001)
- **Data format:** ICARTT (.ict) merge files
