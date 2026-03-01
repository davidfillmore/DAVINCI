# DC3 Field Campaign Support Plan

## Observation-First Approach for DC3 and ASIA-AQ

---

## Overview

Add support for the **DC3** field campaign (May-June 2012) and extend **ASIA-AQ** support in DAVINCI-MONET, starting with **observation-only pipelines**. No model runs are available initially, so the first phases focus on loading, visualizing, and characterizing observations. Model evaluation (MPAS-DAVINCI) is deferred to later phases when model output becomes available.

### Scope

| Component | Scope |
|-----------|-------|
| **Phase A** | Obs-only pipeline infrastructure (both campaigns) |
| **Phase B** | DC3 observation readers + analysis directory |
| **Phase C** | ASIA-AQ obs-only configs |
| **Phase D** | MPAS model reader + model-obs evaluation (deferred) |

### Design Principles

1. **Observations first** — understand the data before evaluating models against it
2. **Works for both campaigns** — obs-only infrastructure serves DC3 and ASIA-AQ equally
3. **Incremental** — each phase is independently useful; model evaluation layers on top later
4. **Reuse existing infrastructure** — ICARTT reader, pipeline stage-skipping, NCAR plot styling

---

## Campaign Reference

### DC3 Campaign

DC3 studied the impact of deep midlatitude continental convective clouds on upper tropospheric composition and chemistry (Barth et al., 2015).

- **Dates:** 15 May - 30 June 2012 (field phase)
- **Operations base:** Salina, Kansas
- **Ground regions:** NE Colorado (CSU), N. Alabama (UAH/NSSTC), Oklahoma/W. Texas (OU/NSSL)
- **PIs:** Mary C. Barth (NCAR), Christopher Cantrell (CU), William Brune (Penn State), Steven Rutledge (CSU), James Crawford (NASA), Heidi Huntrieser (DLR)
- **Sponsors:** NSF, NASA, NOAA, DLR
- **Full reference:** `docs/campaigns/DC3.md`

### Aircraft

| Aircraft | Operator | Role | Ceiling |
|----------|----------|------|---------|
| GV (HIAPER) | NCAR EOL | Outflow sampling (anvil, aged) | ~15.5 km |
| DC-8 | NASA | Inflow/boundary layer + DIAL | ~12 km |
| Falcon 20 | DLR | Fresh anvil outflow | ~12 km |

### Ground-Based Data

| Source | Type | Format |
|--------|------|--------|
| COLMA, NALMA, OKLMA | Lightning Mapping Arrays | NetCDF (NCAR EOL) |
| ARM SGP | Radiosondes (238 during DC3) | NetCDF (ARM archive) |

### Key Species (NOx-focused, matching DAVINCI Phase 3)

| Variable | GV Instrument | DC-8 Instrument | Falcon Instrument |
|----------|--------------|-----------------|-------------------|
| NO | NOxyO3 | NOAA NOyO3 | DLR suite |
| NO2 | NOxyO3 | TD-LIF | DLR suite |
| O3 | NOxyO3 | DIAL + NOAA NOyO3 | DLR suite |
| CO | ACOM CO | DACOM | DLR suite |
| NOy | NOxyO3 | NOAA NOyO3 | DLR suite |

### Data Access

| Source | URL | Format |
|--------|-----|--------|
| NASA ASDC | asdc.larc.nasa.gov/project/DC3 | ICARTT (aircraft) |
| NCAR EOL | data.eol.ucar.edu/project/DC3 | NetCDF (ground, GV) |
| ARM | adc.arm.gov | NetCDF (sondes) |

### Benchmark Case

**29-30 May 2012 Oklahoma Supercell** — most complete multi-aircraft coverage, OKLMA flash data, day-after outflow tracking, multiple published model studies (Pickering et al. 2024, Cummings et al. 2024).

---

## Phase A: Obs-Only Pipeline Infrastructure

**Goal:** Enable DAVINCI-MONET pipelines to run with observations only — no model data, no pairing, no model-obs statistics. Produces observation-only diagnostics: flight tracks, vertical profiles, time series, distributions.

**Applies to both DC3 and ASIA-AQ.**

### A1: Obs-Only Pipeline Mode

The existing pipeline already skips stages when validation fails (no `model` section → `LoadModelsStage` skipped). Extend this to provide a clean obs-only path:

1. **Add `create_obs_only_pipeline()` factory** in `pipeline/stages.py`
   ```python
   def create_obs_only_pipeline() -> list[BaseStage]:
       return [
           LoadObservationsStage(),
           ObsPlottingStage(),      # New: obs-only plots
           SaveResultsStage(),
       ]
   ```

2. **Auto-detect obs-only mode in `run_analysis()`**
   - If YAML config has no `model` section (or `model` is empty/absent), use obs-only pipeline
   - If `model` section present, use standard pipeline as before
   - Log clearly: `"Running in observation-only mode (no model data)"`

3. **Add `obs_only` flag to analysis config**
   ```yaml
   analysis:
     obs_only: true    # Optional explicit flag; auto-detected if model section absent
   ```

### A2: Obs-Only Plot Types

New plot types that operate on raw observation data (not paired datasets):

**New files in `davinci_monet/plots/renderers/`:**

| Plot Type | File | Description |
|-----------|------|-------------|
| Flight track map | `obs_track_map.py` | 2D map with flight path colored by species value |
| Vertical profile | `obs_vertical_profile.py` | Altitude vs. species concentration (scatter or binned) |
| Obs time series | `obs_timeseries.py` | Species value along flight time axis |
| Histogram | `obs_histogram.py` | Distribution of observed values with summary stats |

Each plot type:
- Takes an `xr.Dataset` (observation data) and a variable name
- Does NOT require `model_var` or paired data
- Registers via `@register_plotter("obs_track_map")` etc.
- Supports per-flight panels (via `flight` coordinate from ICARTT reader)
- Uses NCAR brand styling

**Obs track map (`obs_track_map.py`):**
- 2D Cartopy map with flight path as colored line/scatter
- Color = species concentration (or altitude)
- Configurable map extent, projection
- Optional: overlay multiple aircraft on same map (multi-obs)

**Vertical profile (`obs_vertical_profile.py`):**
- Y-axis: altitude (km), X-axis: species concentration
- Scatter plot of all data points, or binned median/mean with percentile shading
- Optional: overlay multiple aircraft (inflow vs outflow comparison)
- Altitude bins configurable (default 500m)

**Obs time series (`obs_timeseries.py`):**
- X-axis: time, Y-axis: species concentration
- Per-flight panels or all flights concatenated
- Optional altitude shading on secondary axis

**Histogram (`obs_histogram.py`):**
- Distribution of observed values
- Summary statistics annotation (N, mean, median, std, percentiles)
- Optional: separate by altitude band (BL vs free troposphere vs UT)

### A3: Obs-Only Plotting Stage

**New class in `pipeline/stages.py`:**

```python
class ObsPlottingStage(BaseStage):
    """Stage for generating observation-only plots."""

    def validate(self, context: PipelineContext) -> bool:
        return bool(context.observations) and "plots" in context.config

    def execute(self, context: PipelineContext) -> StageResult:
        # For each plot config entry, generate obs-only plots
        # using observation datasets directly (not paired data)
        ...
```

### A4: Obs-Only YAML Config Pattern

```yaml
analysis:
  start_time: "2012-05-18"
  end_time: "2012-06-22"
  output_dir: ${DC3_ANALYSIS}/output
  log_dir: ${DC3_ANALYSIS}/logs

# No model section — triggers obs-only mode

obs:
  gv:
    obs_type: icartt
    filename: ${DC3_ANALYSIS}/data/dc3-gv-merge*.ict
    variables:
      NO_NOxyO3:
        obs_min: 0
        obs_max: 50000
      O3_NOxyO3:
        obs_min: 0
        obs_max: 500

# No pairs section

plots:
  gv_no_track:
    type: obs_track_map
    obs: gv
    variable: NO_NOxyO3
    title: "GV NO along flight track"

  gv_no_profile:
    type: obs_vertical_profile
    obs: gv
    variable: NO_NOxyO3
    title: "GV NO vertical profile"
    altitude_bins: 500  # meters

  gv_no_timeseries:
    type: obs_timeseries
    obs: gv
    variable: NO_NOxyO3
    per_flight: true

  gv_no_histogram:
    type: obs_histogram
    obs: gv
    variable: NO_NOxyO3
    altitude_bands:
      BL: [0, 2000]
      FT: [2000, 8000]
      UT: [8000, 16000]

stats:
  # Obs-only stats: N, mean, median, std, percentiles per variable
  metrics: [N, mean, median, std, p25, p75]
```

### A5: Obs-Only Statistics

Extend the statistics stage (or add an `ObsStatisticsStage`) to compute observation-only summary stats when no paired data exists:

- N, mean, median, std, min, max, percentiles (5, 25, 75, 95)
- Per-variable, optionally per-flight and per-altitude-band
- Output to CSV in same format as model-obs statistics

### A6: Tests

- Test obs-only pipeline with synthetic ICARTT data (no model)
- Test each obs-only plot type produces a figure without error
- Test obs-only statistics output matches expected values
- Test auto-detection of obs-only mode from YAML without model section

---

## Phase B: DC3 Observation Readers and Analysis Directory

**Goal:** Load DC3 aircraft, LMA, and sonde observations. Create the `analyses/dc3/` directory with obs-only configs and download scripts.

### B1: LMA Observation Reader

**New files:**
- `davinci_monet/observations/lightning/__init__.py`
- `davinci_monet/observations/lightning/lma.py`

Lightning Mapping Array data from NCAR EOL:

- **Format:** NetCDF with flash locations (lat, lon, alt), times, types
- **Networks:** COLMA (Colorado), NALMA (Alabama), OKLMA (Oklahoma)
- **Resolution:** Individual flashes (millisecond timestamps)
- **Coverage:** ~100-350 km detection range per network

#### Tasks

1. **Create `lightning/` observation subpackage**

2. **Implement LMA reader**
   - Register as `@observation_registry.register('lma')`
   - Read NCAR EOL NetCDF flash files
   - Parse flash locations, times, types (IC/CG when available)
   - Aggregate to flash rates in configurable time bins (default 1 min)
   - Provide both raw flash data and aggregated rates

3. **Obs-only diagnostics for LMA**
   - Flash rate time series
   - Flash location map (scatter on Cartopy)
   - Flash altitude distribution (histogram)
   - IC/CG ratio time series (when classification available)

4. **Tests**
   - Synthetic LMA NetCDF with known flash locations and times
   - Verify flash rate aggregation
   - Verify coordinate parsing

### B2: ARM Sonde Reader

**New file:** `davinci_monet/observations/sonde/arm_sonde.py`

ARM SGP radiosondes:

- **Format:** NetCDF from ARM Data Discovery
- **Site:** Southern Great Plains Central Facility, Lamont, OK (36.61°N, 97.49°W)
- **Frequency:** 4/day during DC3 (00, 06, 12, 18 UTC), 238 total
- **Variables:** Temperature, relative humidity, wind speed/direction, pressure, altitude

#### Tasks

1. **Implement ARM sonde reader**
   - Register as `@observation_registry.register('arm_sonde')`
   - Read ARM NetCDF sonde files
   - Standardize to profile geometry: `(time, level)` with altitude coordinate
   - Map variable names to standard meteorological names

2. **Obs-only diagnostics for sondes**
   - Temperature/humidity profile plots (altitude vs T, RH)
   - Time-height cross sections
   - Stability indices (CAPE, CIN) if derivable

3. **Tests**
   - Synthetic ARM-like NetCDF with known profiles
   - Verify profile geometry and coordinate standardization

### B3: DC3 Analysis Directory

```
analyses/dc3/
├── README.md
├── configs/
│   ├── dc3-obs-gv.yaml              # GV obs-only
│   ├── dc3-obs-dc8.yaml             # DC-8 obs-only
│   ├── dc3-obs-falcon.yaml          # Falcon obs-only
│   ├── dc3-obs-all-aircraft.yaml    # Combined 3-aircraft obs-only
│   ├── dc3-obs-lma.yaml             # LMA flash data obs-only
│   └── dc3-obs-sondes.yaml          # ARM SGP sonde obs-only
├── scripts/
│   ├── download_dc3_aircraft.py     # Download ICARTT from NASA ASDC
│   ├── download_dc3_lma.py          # Download LMA from NCAR EOL
│   ├── download_dc3_sondes.py       # Download ARM SGP sondes
│   └── run_obs_analysis.py          # Obs-only pipeline execution
├── data/                             # Downloaded observations
├── output/                           # Plots and obs statistics
└── logs/                             # Pipeline logs
```

#### Tasks

1. Create directory structure
2. Write README.md with campaign overview, setup, usage
3. Write download scripts (ASDC, EOL, ARM)
4. Write obs-only YAML configs for each observation source
5. Write `run_obs_analysis.py` — runs obs-only pipeline, sets env vars, logs

#### Environment Variables

```bash
DC3_DATA=~/Data/DC3              # Raw observation data downloads
DC3_ANALYSIS=analyses/dc3        # Analysis directory (auto-set by run script)
```

#### ICARTT Variable Names (DC3-specific)

DC3 merge files use instrument-suffixed names:

| Variable | GV Name | DC-8 Name | Falcon Name |
|----------|---------|-----------|-------------|
| NO | NO_NOxyO3 | NO_ESRL | NO_DLR |
| NO2 | NO2_NOxyO3 | NO2_TDLIF | NO2_DLR |
| O3 | O3_NOxyO3 | O3_ESRL | O3_DLR |
| CO | CO_ACOMCO | CO_DACOM | CO_DLR |
| NOy | NOy_NOxyO3 | NOy_ESRL | NOy_DLR |

Configured per-obs in YAML — no code changes needed for variable mapping.

---

## Phase C: ASIA-AQ Obs-Only Configs

**Goal:** Add obs-only pipeline configs for existing ASIA-AQ observations. All readers already exist (AirNow, AERONET, Pandora, DC-8 ICARTT); this phase just creates the YAML configs.

### New Configs in `analyses/asia-aq/configs/`

| Config | Observations | Diagnostics |
|--------|-------------|-------------|
| `asia-aq-obs-airnow.yaml` | AirNow surface (PM2.5, O3, NO2, CO) | Time series, histograms, spatial maps |
| `asia-aq-obs-aeronet.yaml` | AERONET AOD | Time series, site maps |
| `asia-aq-obs-pandora.yaml` | Pandora NO2 columns | Site time series, histograms |
| `asia-aq-obs-dc8.yaml` | DC-8 aircraft (O3, NO2, CO) | Track maps, profiles, flight time series |
| `asia-aq-obs-all.yaml` | All observations combined | Full obs-only diagnostic suite |

### Tasks

1. Write obs-only YAML configs (no `model` or `pairs` sections)
2. Write `run_obs_analysis.py` script for ASIA-AQ obs-only mode
3. Verify existing readers work in obs-only pipeline

---

## Phase D: Model Evaluation (Deferred)

**Goal:** When MPAS-DAVINCI model runs become available, add model reader and model-obs evaluation configs.

### D1: MPAS Model Reader

**New file:** `davinci_monet/models/mpas.py`

MPAS output uses Voronoi mesh conventions:

| Aspect | MPAS Convention | Standard (CESM/WRF) |
|--------|----------------|---------------------|
| Dimensions | `(Time, nCells, nVertLevels)` | `(time, lev, lat, lon)` |
| Horizontal | `latCell`, `lonCell` (radians) | `lat`, `lon` (degrees) |
| Vertical | `zgrid` (geometric height, m) | hybrid sigma-pressure |
| Tracers | `scalars` array or individual vars | individual vars |
| Grid | Unstructured Voronoi | Regular lat/lon |

#### Tasks

1. **Create `mpas.py` model reader**
   - Register as `@model_registry.register('mpas')`
   - Read MPAS history/diagnostics stream NetCDF files
   - Convert `latCell`/`lonCell` from radians to degrees
   - Map MPAS variable names (`qNO` → `NO`, `qNO2` → `NO2`)
   - Unit conversions (kg/kg → ppb: `× 1e9 × M_air / M_species`)

2. **Handle unstructured mesh for pairing**
   - KDTree index on cell centers for nearest-neighbor queries
   - Vertical interpolation using `zgrid` midpoints

3. **Variable mapping (NOx-focused)**

   | MPAS Variable | Standard Name | Unit Conversion |
   |---------------|---------------|-----------------|
   | `qNO` | NO | × 1e9 × 28.97/30.01 (ppb) |
   | `qNO2` | NO2 | × 1e9 × 28.97/46.01 (ppb) |
   | `theta` | potential_temperature | K |
   | `w` | vertical_velocity | m/s |

4. **Register MPAS in config schema** (`mod_type: mpas`)

5. **Tests** — synthetic MPAS-like NetCDF

### D2: DC3 Model-Obs Evaluation Configs

When model runs are available, add paired configs:

```
analyses/dc3/configs/
├── dc3-eval-gv.yaml             # GV vs MPAS-DAVINCI
├── dc3-eval-dc8.yaml            # DC-8 vs MPAS-DAVINCI
├── dc3-eval-falcon.yaml         # Falcon vs MPAS-DAVINCI
├── dc3-eval-all-aircraft.yaml   # Combined evaluation
├── dc3-eval-lma.yaml            # LMA flash rates vs model source
└── dc3-eval-sondes.yaml         # ARM sondes vs model profiles
```

### D3: ASIA-AQ Model-Obs Configs for MPAS

Add MPAS model evaluation configs alongside existing CESM configs:

```
analyses/asia-aq/configs/
├── asia-aq-mpas.yaml            # MPAS-DAVINCI vs all ASIA-AQ obs
```

---

## Implementation Order

| Step | Phase | Description | Dependencies | New Files |
|------|-------|-------------|--------------|-----------|
| 1 | A1 | Obs-only pipeline mode | None | Edit `pipeline/stages.py`, `pipeline/runner.py` |
| 2 | A2 | Obs track map plot | Step 1 | `plots/renderers/obs_track_map.py` |
| 3 | A2 | Obs vertical profile plot | Step 1 | `plots/renderers/obs_vertical_profile.py` |
| 4 | A2 | Obs time series plot | Step 1 | `plots/renderers/obs_timeseries.py` |
| 5 | A2 | Obs histogram plot | Step 1 | `plots/renderers/obs_histogram.py` |
| 6 | A3 | Obs-only plotting stage | Steps 2-5 | Edit `pipeline/stages.py` |
| 7 | A5 | Obs-only statistics | Step 1 | Edit `pipeline/stages.py` or new stage |
| 8 | A6 | Obs-only pipeline tests | Steps 1-7 | `tests/pipeline/test_obs_only.py` |
| 9 | B1 | LMA observation reader | None | `observations/lightning/lma.py` |
| 10 | B1 | LMA reader tests | Step 9 | `tests/observations/test_lma.py` |
| 11 | B2 | ARM sonde reader | None | `observations/sonde/arm_sonde.py` |
| 12 | B2 | ARM sonde tests | Step 11 | `tests/observations/test_arm_sonde.py` |
| 13 | B3 | DC3 analysis directory + configs | Steps 1-7, 9, 11 | `analyses/dc3/` tree |
| 14 | B3 | DC3 download scripts | Step 13 | `analyses/dc3/scripts/download_*.py` |
| 15 | C | ASIA-AQ obs-only configs | Steps 1-7 | `analyses/asia-aq/configs/asia-aq-obs-*.yaml` |
| 16 | D1 | MPAS model reader | None | `models/mpas.py` (deferred) |
| 17 | D2 | DC3 model-obs configs | Step 16 | `analyses/dc3/configs/dc3-eval-*.yaml` (deferred) |
| 18 | D3 | ASIA-AQ MPAS configs | Step 16 | `analyses/asia-aq/configs/asia-aq-mpas.yaml` (deferred) |

**Parallelizable:** Steps 2-5 (obs plot types), Steps 9+11 (readers), Step 15 (ASIA-AQ configs).

**Phase A (Steps 1-8)** and **Phase B readers (Steps 9-12)** can proceed in parallel.

---

## Species Expansion Roadmap

As DAVINCI-MPAS adds chemistry phases, expand the DC3 observation variables:

| DAVINCI Phase | New Species | DC3 Instruments |
|---------------|-------------|-----------------|
| Phase 3 (current) | NO, NO2 | NOxyO3 (GV), NOAA NOyO3 (DC-8), DLR suite |
| Phase 5 (Chapman) | O3 (prognostic) | NOxyO3 (GV), DIAL (DC-8) |
| Phase 6 (Chapman+NOx) | PAN, HNO3 | PAN-CIMS (GV), GT-CIMS (DC-8) |
| Phase 7 (CO+CH4) | CO, CH4, HCHO | DACOM (DC-8), ACOM CO (GV), CAMS (GV) |
| Phase 9 (Trop O3) | VOCs, OH, HO2 | TOGA (GV), ATHOS (DC-8), WAS (DC-8) |

Each expansion requires only new YAML config entries — no new reader code.

---

## References

- Barth, M. C., et al. (2015). "The Deep Convective Clouds and Chemistry (DC3) Field Campaign." *Bull. Amer. Meteor. Soc.*, 96, 1281-1309. doi:10.1175/BAMS-D-13-00290.1
- Barth, M. C., et al. (2019). "Introduction to the DC3 2012 Studies." *J. Geophys. Res. Atmos.*, 124. doi:10.1029/2019JD030944
- Pickering, K. E., et al. (2024). "Lightning NOx in the 29-30 May 2012 DC3 Severe Storm." *J. Geophys. Res. Atmos.*, 129. doi:10.1029/2023JD039439
- Cummings, K. A., et al. (2024). "Evaluation of Lightning Flash Rate Parameterizations." *J. Geophys. Res. Atmos.*, 129. doi:10.1029/2023JD039492
- Nault, B. A., et al. (2017). "Lightning NOx Emissions: Reconciling Measured and Modeled Estimates." *Geophys. Res. Lett.*, 44, 9479-9488. doi:10.1002/2017GL074436

### Data Sources

- NASA ASDC: asdc.larc.nasa.gov/project/DC3
- NCAR EOL: data.eol.ucar.edu/project/DC3
- ARM Data Discovery: adc.arm.gov
- NASA CASEI: impact.earthdata.nasa.gov/casei/campaign/DC3
