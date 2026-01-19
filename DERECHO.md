# Derecho Environment Notes

Notes for running DAVINCI-MONET on NCAR's Derecho HPC system.

## System Info

- **Hostname**: `derecho6` (login nodes: derecho1-8)
- **Home**: `/glade/u/home/fillmore`
- **Work**: `/glade/work/fillmore`
- **Project**: `/glade/work/fillmore/DAVINCI-MONET`

## Conda Environment

```bash
conda activate davinci-monet
```

## ASIA-AQ Data Paths

### Model Data (CESM/CAM-chem)

**Location**: `/glade/campaign/acom/acom-weather/emmons/ASIAAQ_sims/`

**Regridded 0.1° output**:
```
/glade/campaign/acom/acom-weather/emmons/ASIAAQ_sims/f.e3b06m.FCnudged.t6s.GEMSne30x8.01/regridded_0.1deg/
```

**File pattern**: `f.e3b06m.FCnudged.t6s.01x01.01.cam.h2i.2024-02-DD-SSSSS.nc`
- DD = day (01-29)
- SSSSS = seconds of day (00000, 03600, 07200, ...)

### Observation Data

**Location**: `/glade/campaign/acom/acom-weather/emmons/ASIAAQ_obs/`

| Dataset | Path | Description |
|---------|------|-------------|
| AERONET | `ASIAAQ_obs/AERONET/` | AOD measurements |
| DC8 | `ASIAAQ_obs/DC8/` | DC-8 aircraft ICARTT files |
| Pandora | `ASIAAQ_obs/Pandora/` | NO2 column measurements |
| OPENAQ | `ASIAAQ_obs/OPENAQ/` | Surface air quality |
| MOPITT_L2 | `ASIAAQ_obs/MOPITT_L2/` | CO profiles |
| MOPITT_L3_daily | `ASIAAQ_obs/MOPITT_L3_daily/` | CO gridded daily |
| MODIS_L2 | `ASIAAQ_obs/MODIS_L2/` | Aerosol products |
| ISD_met | `ASIAAQ_obs/ISD_met/` | Meteorology |
| IMERG_precip | `ASIAAQ_obs/IMERG_precip/` | Precipitation |

## Directory Structure

```
/glade/work/fillmore/
├── DAVINCI-MONET/          # Code repository
│   └── analyses/asia-aq/
│       └── configs/        # YAML config files
└── ASIA-AQ/                # Project output (separate from repo)
    ├── output/             # Plots, statistics
    └── logs/               # Pipeline logs
```

## Environment Variables (optional)

The `asia-aq-derecho.yaml` config uses hardcoded paths, but for the original config:

```bash
export ASIA_AQ_DATA=/glade/campaign/acom/acom-weather/emmons
export ASIA_AQ_ANALYSIS=/glade/work/fillmore/ASIA-AQ
```

## Running Analyses (Headless Mode)

Derecho login/compute nodes have no display, so use headless mode (default):

```bash
cd /glade/work/fillmore/DAVINCI-MONET
davinci-monet run analyses/asia-aq/configs/asia-aq-derecho.yaml
```

Output goes to `/glade/work/fillmore/ASIA-AQ/output/`.

### Config Files

| Config | Description |
|--------|-------------|
| `asia-aq.yaml` | Original config (uses `${ASIA_AQ_DATA}` env var) |
| `asia-aq-derecho.yaml` | Derecho-specific with campaign storage paths |

### Available Data in Derecho Config

- **AERONET AOD** - L1.5 processed NetCDF
- **DC-8 Aircraft** - 10-second merge ICARTT files (O3, NO2, CO)

### Not Yet Available

- **AirNow** - Needs download via `scripts/download_airnow.py`
- **Pandora NO2 columns** - Raw txt files need preprocessing

To preview plots interactively (requires X11 forwarding):

```bash
davinci-monet run analyses/asia-aq/configs/asia-aq-derecho.yaml --show-plots
```

## File Transfer Notes

Tar archives created for transferring data:
- `ASIA-AQ.2024-02-DD.tar` - Daily model output bundles
- `Pandora.tar` - Pandora observations
- `DC8.tar` - DC-8 aircraft data
