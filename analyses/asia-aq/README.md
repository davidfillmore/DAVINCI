# ASIA-AQ Analysis

Model evaluation for the NASA ASIA-AQ (Airborne and Satellite Investigation of Asian Air Quality) campaign.

## Campaign Overview

ASIA-AQ was an international cooperative field study conducted from January-March 2024 to address local air quality challenges across eastern Asia. The campaign deployed multiple aircraft (including NASA's DC-8 and G-III) over four countries: South Korea, Philippines, Taiwan, and Thailand.

**Science Goals:**
- Satellite validation and interpretation (especially GEMS geostationary sensor)
- Emissions quantification and verification
- Model evaluation
- Aerosol and ozone chemistry

**Data Sources:**
- [NASA ESPO ASIA-AQ](https://espo.nasa.gov/asia-aq/)
- [NASA ASDC ASIA-AQ Data](https://asdc.larc.nasa.gov/project/ASIA-AQ)
- [CASEI Campaign Archive](https://impact.earthdata.nasa.gov/casei/campaign/ASIA-AQ)

## Model Data

**Model:** CESM/CAM with FC (full chemistry) configuration, nudged to meteorology

**Case:** `f.e3b06m.FCnudged.t6s.01x01.01`

**Resolution:** 0.1° x 0.1° (450 x 500 grid), 32 vertical levels

**Domain:** 0°-45°N, 90°-140°E (covers Southeast Asia, Korea, Taiwan, S. China, S. Japan)

**Period:** February 1-10, 2024 (hourly output)

**Location:** `~/Data/ASIA-AQ/CAM/`

**Key Variables:**
| Variable | Description | Unit Conversion |
|----------|-------------|-----------------|
| O3 | Ozone | × 1e9 (mol/mol → ppb) |
| NO2 | Nitrogen dioxide | × 1e9 (mol/mol → ppb) |
| CO | Carbon monoxide | × 1e9 (mol/mol → ppb) |
| PM25 | PM2.5 | × 1.2e9 (kg/kg → µg/m³) |
| AODVISdn | Aerosol optical depth (550 nm) | none |

## Observation Data

| Type | Source | Variables | Status |
|------|--------|-----------|--------|
| Surface | AirNow | PM2.5, O3, NO2, CO | ✓ Complete |
| Surface | AERONET | AOD (500 nm) | ✓ Complete |
| Aircraft | ICARTT from DC-8/G-III | Various | Pending |
| Satellite | TROPOMI NO2/CO | NO2, CO | Pending |

## Directory Structure

```
asia-aq/
├── README.md
├── configs/
│   └── asia-aq.yaml                # Pipeline configuration
├── scripts/
│   ├── download_airnow.py          # Download AirNow data
│   └── run_evaluation.py           # Run pipeline
├── data/                           # Observation data (NetCDF)
├── output/                         # Plots and statistics
├── logs/                           # Pipeline logs (timestamped)
└── misc/                           # Exploratory scripts
```

## Setup

Set the environment variables for data and analysis directories:

```bash
# Model and raw observation data (required)
export ASIA_AQ_DATA=~/Data/ASIA-AQ

# Analysis directory (set automatically by run_evaluation.py)
export ASIA_AQ_ANALYSIS=/path/to/analyses/asia-aq
```

If `ASIA_AQ_DATA` is not set, scripts default to `~/Data/ASIA-AQ`.
The `ASIA_AQ_ANALYSIS` variable is set automatically when running `run_evaluation.py`.

## Usage

**Download observations:**
```bash
python scripts/download_airnow.py
davinci-monet get aeronet -s 2024-02-01 -e 2024-02-03 -d data
```

**Run the evaluation pipeline:**
```bash
python scripts/run_evaluation.py
```

Or via CLI:
```bash
davinci-monet run configs/asia-aq.yaml
```

## Results

| Species | N | Mean Obs | Mean Model | R | NMB |
|---------|---|----------|------------|------|------|
| PM2.5 | 98 | 37.9 µg/m³ | 48.5 µg/m³ | 0.36 | +28% |
| O3 | 14 | 11.4 ppb | 3.1 ppb | 0.54 | -73% |
| NO2 | 5 | 21.4 ppb | 50.9 ppb | 0.72 | +138% |
| CO | 12 | 1375 ppb | 1182 ppb | -0.16 | -14% |
| AOD | 498 | 0.43 | 0.19 | 0.59 | -55% |

**Output files:**
- `output/statistics_summary.csv` - Evaluation metrics
- `output/*_scatter.png` - Model vs obs scatter plots
- `output/*_timeseries.png` - Time series with uncertainty bands (mean ± std)
- `output/*_spatial_bias.png` - Spatial bias maps with city labels
- `logs/pipeline_YYYYMMDD_HHMMSS.log` - Pipeline execution logs with timing
