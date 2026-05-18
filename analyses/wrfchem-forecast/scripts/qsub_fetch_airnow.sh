#!/bin/bash
# Submit the daily AirNow fetch to PBS on Casper. Designed to be invoked
# from a cron entry on cron.hpc.ucar.edu (where the queue must be fully
# qualified with the PBS server name).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

qsub \
    -l walltime=00:30:00 \
    -A P19010000 \
    -l select=1:ncpus=1 \
    -q casper@casper-pbs \
    -- "${SCRIPT_DIR}/fetch_airnow.sh"
