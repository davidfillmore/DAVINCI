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

Output goes to `/glade/work/fillmore/ASIA-AQ/output/`.

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
| `asia-aq.yaml` | Original config (uses `${ASIA_AQ_DATA}` env var) |
| `asia-aq-derecho.yaml` | Derecho-specific with campaign storage paths |
| `asia-aq-scratch.yaml` | **Fastest** - uses scratch storage (requires setup above) |
| `asia-aq-derecho-1day.yaml` | Single day test config |

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

**3-day test (72 model files)**:

| Stage | Campaign Storage | Scratch Storage | Speedup |
|-------|------------------|-----------------|---------|
| load_models | 190s | **6.8s** | **28x** |
| load_observations | 172s | 163s | ~1x |
| pairing (AERONET) | 10+ min | **2.3s** | **~260x** |

**Full month (696 model files) with all optimizations**:

| Stage | Before Optimization | After Optimization | Speedup |
|-------|---------------------|-------------------|---------|
| load_models | ~190s | **55s** | 3.5x |
| load_observations | 163s | **0.1s** | **1,630x** |
| pairing (AERONET) | 10+ min | **2.4s** | **~260x** |
| **Total pipeline** | ~175s (3-day) | **~8s** (3-day) | **22x** |

**Key optimizations applied**:
1. **Scratch storage** - Model loading 28x faster than campaign storage
2. **Dask parallel scheduler** - Pairing uses explicit threaded scheduler with 32 workers
3. **Time filtering at load** - Two-level filtering:
   - File-level: filters by YYYYMMDD in filename (skips out-of-range ICARTT files)
   - Data-level: filters by time dimension after loading (subsets large NetCDF files)
4. **Track strategy optimization** - Same Dask fix applied to aircraft pairing

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

Then run with scratch config:
```bash
davinci-monet run analyses/asia-aq/configs/asia-aq-scratch.yaml
```

## File Transfer Notes

Tar archives created for transferring data:
- `ASIA-AQ.2024-02-DD.tar` - Daily model output bundles
- `Pandora.tar` - Pandora observations
- `DC8.tar` - DC-8 aircraft data
