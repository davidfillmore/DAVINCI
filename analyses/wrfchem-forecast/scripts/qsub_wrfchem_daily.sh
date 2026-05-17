#!/bin/bash
# Submit the daily WRF-Chem forecast vs AirNow + AERONET evaluation to PBS
# on Casper. Designed to be invoked from cron.hpc.ucar.edu (which is why
# the queue is fully qualified as casper@casper-pbs and the run is wrapped
# in run_pipeline.sh — the PBS compute node gets a sparse environment with
# no conda init).
#
# Usage:
#   qsub_wrfchem_daily.sh             # yesterday
#   qsub_wrfchem_daily.sh 20250801    # explicit date (historical replay)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    -q casper@casper-pbs \
    -v YYYY,MM,DD,HDF5_USE_FILE_LOCKING,DASK_NUM_WORKERS \
    -- "${SCRIPT_DIR}/run_pipeline.sh"
