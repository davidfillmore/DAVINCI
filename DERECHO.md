# Derecho Environment Notes

Notes for running DAVINCI on NCAR's Derecho HPC system.

## System Info

- **Hostname**: `derecho6` (login nodes: derecho1-8)
- **Home**: `/glade/u/home/fillmore`
- **Work**: `/glade/work/fillmore`
- **Project**: `/glade/work/fillmore/DAVINCI-MONET`
- **Branch**: `develop` (primary working branch)

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
/glade/work/fillmore/DAVINCI-MONET/
├── analyses/asia-aq/
│   ├── configs/            # YAML config files
│   ├── scripts/            # Download scripts (download_airnow.py)
│   ├── data/               # Downloaded observation data (local)
│   ├── output/             # Plots, statistics
│   └── logs/               # Pipeline logs
└── ...

/glade/derecho/scratch/fillmore/ASIA-AQ/
├── model/                  # 696 CESM NetCDF files (fast storage)
└── obs/                    # Observation data (AERONET, DC8, AirNow)
```

## Environment Variables (optional)

The `asia-aq-derecho.yaml` config uses hardcoded paths, but for the original config:

```bash
export ASIA_AQ_DATA=/glade/campaign/acom/acom-weather/emmons
export ASIA_AQ_ANALYSIS=/glade/work/fillmore/ASIA-AQ
```

## Running on Compute Nodes

**Do not run compute-intensive jobs on login nodes!** Use an interactive session or batch job.

### Interactive Session

```bash
qsub -I -A P19010000 -l select=1:ncpus=64:mpiprocs=64:ngpus=4 -l walltime=02:00:00 -q main
```

Once on a compute node:
```bash
conda activate davinci-monet
cd /glade/work/fillmore/DAVINCI-MONET
davinci-monet run analyses/asia-aq/configs/asia-aq-derecho.yaml
```

Output goes to `/glade/work/fillmore/DAVINCI-MONET/analyses/asia-aq/output/`.

### Batch Job

Create `run_asia_aq.pbs`:
```bash
#!/bin/bash
#PBS -N asia-aq
#PBS -A P19010000
#PBS -l select=1:ncpus=64:mpiprocs=64:ngpus=4
#PBS -l walltime=04:00:00
#PBS -q main
#PBS -j oe

conda activate davinci-monet
cd /glade/work/fillmore/DAVINCI-MONET
davinci-monet run analyses/asia-aq/configs/asia-aq-derecho.yaml
```

Submit with: `qsub run_asia_aq.pbs`

## Parallel Processing

The pipeline leverages parallelism at multiple levels:

1. **Xarray + Dask**: Lazy loading and parallel computation on chunked arrays
   - Dask 2024.2.1 available in environment
   - Automatically uses multiple cores for array operations

2. **Pipeline ParallelExecutor**: Concurrent model-observation pairing
   - Uses `concurrent.futures` ThreadPoolExecutor
   - Located in `davinci_monet/pipeline/parallel.py`

To explicitly configure Dask workers:
```python
from dask.distributed import Client
client = Client(n_workers=16, threads_per_worker=4)
```

### Config Files

| Config | Description |
|--------|-------------|
| `asia-aq-derecho.yaml` | Full config - all obs (AirNow, AERONET, DC8) |
| `asia-aq-airnow-derecho.yaml` | AirNow only (surface PM2.5, O3) |
| `asia-aq-aeronet-derecho.yaml` | AERONET only (AOD) |
| `asia-aq-dc8-derecho.yaml` | DC8 only (aircraft O3, NO2, CO) |
| `asia-aq-pandora-derecho.yaml` | Pandora only (NO2 column) |
| `asia-aq-gemini.yaml` | Mac testing config |

**Recommended workflow**: Run single-obs configs for faster iteration. Each loads the model
once and pairs quickly, avoiding the "democracy not monarchy" Dask problem where each pair
independently loads all 696 model files.

```bash
davinci-monet run analyses/asia-aq/configs/asia-aq-airnow-derecho.yaml
davinci-monet run analyses/asia-aq/configs/asia-aq-aeronet-derecho.yaml
davinci-monet run analyses/asia-aq/configs/asia-aq-dc8-derecho.yaml
davinci-monet run analyses/asia-aq/configs/asia-aq-pandora-derecho.yaml
```

### Available Data in Derecho Config

- **AirNow** - Surface PM2.5, O3 from 36 US Embassy/Consulate monitors (Bangkok, Beijing, etc.)
- **AERONET AOD** - L1.5 processed NetCDF
- **DC-8 Aircraft** - 10-second merge ICARTT files (O3, NO2, CO)
- **Pandora** - NO2 tropospheric column from 14 sites (preprocessed from L2 txt)

### Preprocessing Scripts

For Pandora, run these scripts first (already done, output on scratch):
```bash
python analyses/asia-aq/scripts/preprocess_pandora.py   # Pandora L2 → NetCDF
python analyses/asia-aq/scripts/compute_no2_column.py   # CESM 3D NO2 → column
```

### Not Yet Implemented

- **MOPITT** - CO profile evaluation
- **MODIS** - AOD comparison

To preview plots interactively (requires X11 forwarding):

```bash
davinci-monet run analyses/asia-aq/configs/asia-aq-derecho.yaml --show-plots
```

## Storage Performance

| Storage | Type | Characteristics |
|---------|------|-----------------|
| `/glade/campaign` | Tape-backed | Slow for I/O, optimized for archival |
| `/glade/work` | Parallel FS (GPFS) | Good throughput, high metadata latency |
| `/glade/derecho/scratch` | Parallel FS | **Fastest**, but temporary (purged after 60 days) |
| Mac SSD | Flash | Low latency, fast random I/O |

**Why Derecho feels slower than a Mac**:
- Parallel file systems have high latency for metadata operations
- Opening 696 hourly files = 696 metadata lookups (slow)
- Campaign storage adds tape-staging delays even when files are cached

**Optimization strategies**:
1. **Use scratch storage** - Copy data to `/glade/derecho/scratch` for active work
2. Narrow file glob for testing: `2024-02-0[1-3]-*.nc` (3 days instead of 29)
3. Pre-concatenate files into daily/weekly chunks
4. Use `xr.open_mfdataset(..., parallel=True)` with Dask

### Observed Performance

**Full month (Feb 2024) on scratch storage (2026-01-23, peak hours)**:

| Stage | Time | Data | Notes |
|-------|------|------|-------|
| load_models | 96s | 696 NetCDF files | Builds Dask task graph |
| load_observations | 165s | AERONET + DC8 (7 ICARTT) | ICARTT parsing dominates |
| pairing | 600s | 4 pairs | Each pair loads 696 files |
| statistics | 0.1s | 4 pairs | |
| plotting | 31s | 14 plots | |
| **Total** | **~15 min** | | |

**Output (verified 2026-01-23):**
- 14 plots: scatter, timeseries, spatial_bias, flight_timeseries, track_3d
- Statistics: N=8.5k-19k paired points, R=0.27-0.52

**Notes:**
- Scratch storage performance varies significantly with cluster load
- ICARTT (ASCII) parsing dominates obs load time, not I/O
- Pairing is the main bottleneck due to repeated Dask `.compute()` calls
- The `preload: true` feature (not yet implemented) would help significantly

### Scratch Storage Setup

Tar archives on scratch need extraction before use:

```bash
# Create directories
mkdir -p /glade/derecho/scratch/fillmore/ASIA-AQ/{model,obs,output,logs}

# Extract model data (full month - Feb 2024)
cd /glade/derecho/scratch/fillmore/ASIA-AQ/model

# Daily files (Feb 1-9)
for day in 01 02 03 04 05 06 07 08 09; do
  tar -xf /glade/derecho/scratch/fillmore/ASIA-AQ.2024-02-${day}.tar --strip-components=8
done

# Multi-day bundles (Feb 10-28)
tar -xf /glade/derecho/scratch/fillmore/ASIA-AQ.2024-02-10_19.tar --strip-components=8
tar -xf /glade/derecho/scratch/fillmore/ASIA-AQ.2024-02-20_28.tar --strip-components=8

# Feb 29
tar -xf /glade/derecho/scratch/fillmore/ASIA-AQ.2024-02-29.tar --strip-components=8

echo "Total files: $(ls *.nc | wc -l)"  # Should be 696

# Extract DC-8 aircraft data
cd /glade/derecho/scratch/fillmore/ASIA-AQ/obs
tar -xf /glade/derecho/scratch/fillmore/DC8.tar --strip-components=7

# Copy AERONET
cp /glade/work/fillmore/ASIA-AQ/model-subset/AERONET_L15_20240101_20240501.nc .
```

Then run:
```bash
davinci-monet run analyses/asia-aq/configs/asia-aq-derecho.yaml
```

## File Transfer Notes

Tar archives created for transferring data:
- `ASIA-AQ.2024-02-DD.tar` - Daily model output bundles
- `Pandora.tar` - Pandora observations
- `DC8.tar` - DC-8 aircraft data
