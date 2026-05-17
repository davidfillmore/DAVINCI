# Daily WRF-Chem AQ Forecast Evaluation

Daily evaluation of the operational AQ_WATCH WRF-Chem forecast against AirNow
surface observations, replacing the legacy `melodies-scripts/wrfchem.yaml` +
`run_wrfchem.py` pipeline with a pure DAVINCI configuration.

## What it does

1. `fetch_airnow.sh` downloads the previous UTC day's AirNow observations into
   `/glade/work/fillmore/Data/AirNow/AirNow_YYYYMMDD.nc`.
2. `qsub_wrfchem_daily.sh` exports `YYYY/MM/DD` (yesterday by default), submits
   `davinci-monet run` to PBS on Casper. The pipeline pairs the dated WRF-Chem
   forecast with that day's AirNow + AERONET files and writes CONUS scatter,
   timeseries, and spatial bias plots for PM2.5 / OZONE / AOD to
   `/glade/campaign/acom/acom-da/fillmore/DAVINCI/WRF-Chem/YYYY/MM/DD/`.

No Python wrapper — date substitution is done entirely via DAVINCI's `${VAR}`
env-var expansion in the YAML config.

## Data sources

| Source | Path |
|---|---|
| WRF-Chem forecast | `/glade/campaign/acom/acom-da/shawnh/AQ_WATCH/YYYYMMDD/wrf/wrfout_d01_YYYY-MM-DD_*` |
| AirNow observations | `/glade/work/fillmore/Data/AirNow/AirNow_YYYYMMDD.nc` |
| AERONET observations | `/glade/work/fillmore/Data/AeroNet/AeroNet_YYYYMMDD.nc` |
| Plot output | `/glade/campaign/acom/acom-da/fillmore/DAVINCI/WRF-Chem/YYYY/MM/DD/` |

The WRF-Chem forecast is the operational AQ-WATCH run (CONUS d01, every
6 hours per init day, multi-day forecast horizon), owned by `shawnh` in
the acom-da group. Plots from this run are published at
<https://acom.ucar.edu/firex-aq/forecast.html>.

Note that the WRF-Chem files are named `wrfout_d01_YYYY-MM-DD_HH:MM:SS`
(no `_hourly_` infix, no `.nc` extension) — DAVINCI's WRF-Chem reader
opens them via netCDF4 regardless of extension.

## Files

| File | Purpose |
|---|---|
| `configs/wrfchem-forecast.example.yaml` | Pipeline config (env-var dates) |
| `scripts/fetch_airnow.sh` | Download yesterday's AirNow file |
| `scripts/qsub_fetch_airnow.sh` | PBS wrapper around `fetch_airnow.sh` |
| `scripts/qsub_wrfchem_daily.sh` | PBS wrapper around `davinci-monet run` |

## Usage

**Historical replay (for development/testing):**
```bash
analyses/wrfchem-forecast/scripts/qsub_wrfchem_daily.sh 20250801
```

**Yesterday's forecast (operational):**
```bash
analyses/wrfchem-forecast/scripts/qsub_wrfchem_daily.sh
```

**Manual run without qsub:**
```bash
export YYYY=2025 MM=08 DD=01
davinci-monet run analyses/wrfchem-forecast/configs/wrfchem-forecast.example.yaml
```

## Cron template

Two crontab entries replace the legacy `airnow.sh` + `wrfchem.yaml` chain:

```
20 08 * * * /glade/work/fillmore/DAVINCI-MONET/analyses/wrfchem-forecast/scripts/qsub_fetch_airnow.sh
30 09 * * * /glade/work/fillmore/DAVINCI-MONET/analyses/wrfchem-forecast/scripts/qsub_wrfchem_daily.sh
```

## Notes

- **Domain**: vweeks's system writes `d01` only (the legacy rkumar config used
  `d02`).
- **State-level subdomains**: the legacy config produced per-state panels for
  CA and CO. DAVINCI does not currently support `state_name` domains out of
  the box; only CONUS + EPA regions R1–R10 are plotted. Add state extents
  to `davinci_monet/plots/renderers/spatial/base.py:get_domain_extent` if
  per-state panels are needed.
- **Mechanism**: `mod_kwargs: {mech: racm_esrl_vcp}` is forwarded to monetio's
  WRF-Chem reader for proper variable resolution.
- **wrf-python compatibility**: monetio's WRF-Chem reader requires `wrf-python`,
  which in turn breaks against `netCDF4 >= 1.7` (raises
  `NotImplementedError: Dataset is not picklable` from wrf-python's internal
  `copy.copy()` on a `netCDF4.Dataset`). The DAVINCI WRF-Chem reader catches
  this and falls back to a plain xarray open with a loud warning. In the
  fallback path, raw WRF variables are returned without mech-aware decoding —
  notably `o3` stays in **ppmv** instead of being converted to ppb. The
  example config keeps `o3.unit_scale: 1.0` for the monetio path; if the
  fallback fires (the warning will tell you), switch it to `1.0e3` so the
  AirNow comparison is unit-consistent. To restore the monetio path,
  pin `netCDF4<1.7` or an older `wrf-python` build in `environment.yml`.
