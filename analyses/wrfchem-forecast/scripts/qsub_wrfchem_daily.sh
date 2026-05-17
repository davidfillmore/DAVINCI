#!/bin/bash
# Submit the daily WRF-Chem forecast vs AirNow evaluation pipeline to PBS.
#
# Exports YYYY/MM/DD (yesterday by default; or an explicit YYYYMMDD as $1) into
# DAVINCI's ${VAR} expansion so the YAML config can resolve the dated forecast
# directory and AirNow file. No Python wrapper.
#
# Usage:
#   qsub_wrfchem_daily.sh             # yesterday
#   qsub_wrfchem_daily.sh 20250801    # explicit date (historical replay)
#
# Cron line (run 30 min after fetch_airnow):
#   30 09 * * * /glade/work/fillmore/DAVINCI-MONET/analyses/wrfchem-forecast/scripts/qsub_wrfchem_daily.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(dirname "${SCRIPT_DIR}")"
CONFIG="${ANALYSIS_DIR}/configs/wrfchem-forecast.example.yaml"

if [ "$#" -ge 1 ]; then
    fcst_date=$1
else
    fcst_date=$(date --date=yesterday '+%Y%m%d')
fi

export YYYY=${fcst_date:0:4}
export MM=${fcst_date:4:2}
export DD=${fcst_date:6:2}

# HDF5 thread safety: WRF-Chem mfdataset under parallel dask occasionally
# segfaults via libhdf5 file locks; disable file locking and pin Dask
# workers to be safe in batch runs.
export HDF5_USE_FILE_LOCKING=FALSE
export DASK_NUM_WORKERS=1

qsub \
    -l walltime=02:00:00 \
    -A P19010000 \
    -l select=1:ncpus=1 \
    -q casper \
    -v YYYY,MM,DD,HDF5_USE_FILE_LOCKING,DASK_NUM_WORKERS \
    -- davinci-monet run "${CONFIG}"
