#!/bin/bash
# Submit the daily AirNow fetch to PBS on Casper.
#
# Cron line (10-min stagger before the WRF-Chem plot job):
#   20 08 * * * /glade/work/fillmore/DAVINCI-MONET/analyses/wrfchem-forecast/scripts/qsub_fetch_airnow.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

qsub \
    -l walltime=00:30:00 \
    -A P19010000 \
    -l select=1:ncpus=1 \
    -q casper \
    -- "${SCRIPT_DIR}/fetch_airnow.sh"
