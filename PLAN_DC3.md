# DC3 Field Campaign Support Plan

## Deep Convective Clouds and Chemistry (DC3) — Model Evaluation in DAVINCI-MONET

---

## Overview

Add support for evaluating **MPAS-DAVINCI** chemistry against observations from the **DC3** field campaign (May-June 2012). DC3 provides the most comprehensive observational dataset for lightning NOx production, convective transport, and post-convective ozone chemistry — the direct validation targets for DAVINCI-MPAS Phases 3-9.

### Scope

| Component | Scope |
|-----------|-------|
| Model | MPAS-DAVINCI (new model reader) |
| Aircraft | All three: NSF/NCAR GV, NASA DC-8, DLR Falcon |
| Ground | Lightning Mapping Arrays (COLMA, NALMA, OKLMA) |
| Sondes | ARM SGP radiosondes (238 during DC3) |
| Species | NOx-focused: NO, NO2, NOy, O3, CO |
| Benchmark | 29-30 May 2012 Oklahoma supercell (primary case) |

### Design Principles

1. **Mirror ASIA-AQ pattern** — same directory structure, script conventions, YAML config style
2. **Reuse existing infrastructure** — ICARTT reader, track pairing strategy, statistics engine
3. **Build for DAVINCI progression** — species list expands as DAVINCI chemistry adds phases
4. **Native MPAS support** — read Voronoi mesh output directly, no pre-processing step

---

## Campaign Reference

### Campaign Overview

DC3 was a multi-agency field experiment studying the impact of deep midlatitude continental convective clouds on upper tropospheric composition and chemistry (Barth et al., 2015).

- **Dates:** 15 May - 30 June 2012 (field phase)
- **Operations base:** Salina, Kansas
- **Ground regions:** NE Colorado (CSU), N. Alabama (UAH/NSSTC), Oklahoma/W. Texas (OU/NSSL)
- **PIs:** Mary C. Barth (NCAR), Christopher Cantrell (CU), William Brune (Penn State), Steven Rutledge (CSU), James Crawford (NASA), Heidi Huntrieser (DLR)
- **Sponsors:** NSF, NASA, NOAA, DLR

### Science Objectives

1. **Lightning NOx production** — NOx per flash, vertical distribution, storm-type dependence
2. **Convective transport** — boundary layer pollutant lofting efficiency
3. **Wet scavenging** — removal efficiencies for soluble gases (H2O2, CH3OOH, CH2O)
4. **Post-convective chemistry** — ozone production from lightning NOx in aged outflow

### Aircraft

| Aircraft | Operator | Role | Ceiling | Flights |
|----------|----------|------|---------|---------|
| GV (HIAPER) | NCAR EOL | Outflow sampling (anvil, aged) | ~15.5 km | Multiple |
| DC-8 | NASA | Inflow/boundary layer + DIAL | ~12 km | Multiple |
| Falcon 20 | DLR | Fresh anvil outflow | ~12 km | 13 (29 May - 14 Jun) |

### Key Instruments (NOx-focused subset)

**GV:**
| Instrument | Measurement | Notes |
|------------|-------------|-------|
| NOxyO3 | NO, NO2, NOy, O3 | 4-channel chemiluminescence, 1 s |
| ACOM CO | CO | VUV fluorescence |
| TOGA | 60+ VOC species | GC-MS (future phases) |
| Picarro | CO2, CH4 | Cavity ring-down |

**DC-8:**
| Instrument | Measurement | Notes |
|------------|-------------|-------|
| NOAA NOyO3 | NO, NO2, NOy, O3 | Chemiluminescence |
| TD-LIF | NO2, sum PNs, sum ANs, HNO3 | Thermal dissociation LIF |
| DACOM | CO, N2O, CH4 | Tunable diode laser |
| DIAL | O3, H2O profiles | UV/vis/IR lidar (remote) |

**Falcon:**
| Instrument | Measurement | Notes |
|------------|-------------|-------|
| DLR suite | CO, O3, SO2, CH4, NO, NOx, NOy | In-situ |

### Ground-Based Lightning Data

| Network | Location | Type |
|---------|----------|------|
| COLMA | NE Colorado | 15-station VHF, installed for DC3 |
| NALMA | N. Alabama | Pre-existing VHF near Huntsville |
| OKLMA | Oklahoma | Pre-existing at NSSL |

### Sounding Sites

- **ARM SGP** (Lamont, OK): 238 radiosondes, 4/day (00, 06, 12, 18 UTC)
- Additional mobile sounding teams in Colorado and Alabama

### Data Access

| Source | URL | Format |
|--------|-----|--------|
| NASA ASDC | asdc.larc.nasa.gov/project/DC3 | ICARTT (aircraft) |
| NCAR EOL | data.eol.ucar.edu/project/DC3 | NetCDF (ground, GV) |
| NASA ESPO | espo.nasa.gov/dc3 | Mission info |
| NASA CASEI | impact.earthdata.nasa.gov/casei/campaign/DC3 | Archive |

### Benchmark Case: 29-30 May 2012 Oklahoma Supercell

The most extensively studied DC3 storm:
- Complete multi-aircraft coverage (GV, DC-8, Falcon)
- OKLMA flash data with high spatial resolution
- Day-after outflow tracking (10-11 ppbv O3/day production)
- Published model studies: Pickering et al. (2024), Cummings et al. (2024)
- Primary candidate for MPAS-DAVINCI convection-allowing simulation

---

## Implementation Plan

### Phase 1: MPAS Model Reader

**New file:** `davinci_monet/models/mpas.py`

MPAS output uses Voronoi mesh conventions that differ from lat/lon gridded models:

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
   - Map MPAS variable names to standard names (`qNO` → `NO`, `qNO2` → `NO2`, etc.)
   - Handle unit conversions (kg/kg mixing ratio → ppb: `× 1e9 × M_air / M_species`)
   - Extract cell-center coordinates for pairing

2. **Handle unstructured mesh for pairing**
   - For track pairing (aircraft): nearest-cell lookup using `latCell`/`lonCell`
   - Build a KDTree index on cell centers for efficient nearest-neighbor queries
   - Vertical interpolation: use `zgrid` midpoints to interpolate to aircraft altitude

3. **Handle MPAS vertical coordinate**
   - `zgrid` has dimension `(nVertLevels+1, nCells)` — interface heights
   - Compute midpoint heights: `z_mid[k] = 0.5 * (zgrid[k] + zgrid[k+1])`
   - Surface extraction: last level (highest pressure) or `zgrid` nearest to terrain

4. **Variable mapping table (initial, NOx-focused)**

   | MPAS Variable | Standard Name | Unit Conversion |
   |---------------|---------------|-----------------|
   | `qNO` | NO | × 1e9 × 28.97/30.01 (ppb) |
   | `qNO2` | NO2 | × 1e9 × 28.97/46.01 (ppb) |
   | `qPassive` | passive_tracer | none |
   | `theta` | potential_temperature | K |
   | `w` | vertical_velocity | m/s |
   | `pressure_p + pressure_base` | pressure | Pa |

5. **Tests**
   - Synthetic MPAS-like NetCDF with known cell positions and tracer values
   - Verify radians → degrees conversion
   - Verify unit conversion round-trip
   - Verify nearest-cell lookup against known geometry
   - Verify vertical interpolation to target altitude

#### Verification

- Reader opens MPAS NetCDF files without error
- Cell coordinates are in degrees, within valid lat/lon range
- Tracer values have physically reasonable magnitudes after unit conversion
- KDTree nearest-cell matches expected cell for known query points

---

### Phase 2: LMA Observation Reader

**New file:** `davinci_monet/observations/lightning/lma.py`

Lightning Mapping Array data from NCAR EOL provides flash-level observations for validating DAVINCI's lightning NOx source parameterization.

#### LMA Data Characteristics

- **Source:** NCAR EOL archive (NetCDF)
- **Content:** Flash locations (lat, lon, alt), flash times, flash type (IC/CG when available)
- **Networks:** COLMA (Colorado), NALMA (Alabama), OKLMA (Oklahoma)
- **Temporal resolution:** Individual flashes (millisecond timestamps)
- **Spatial coverage:** ~100-350 km detection range per network

#### Tasks

1. **Create `lightning/` observation subpackage**
   - `davinci_monet/observations/lightning/__init__.py`
   - `davinci_monet/observations/lightning/lma.py`

2. **Implement LMA reader**
   - Register as `@observation_registry.register('lma')`
   - Read NCAR EOL NetCDF flash files
   - Parse flash locations, times, types
   - Aggregate to flash rates (flashes/min or flashes/5min) in configurable time bins
   - Optionally aggregate spatially (cell-level counts matching MPAS grid)

3. **Define validation metrics**
   - Flash rate time series: model `S_ltg` activation vs observed flash rate
   - Spatial distribution: model source region vs LMA flash locations
   - Flash altitude distribution: LMA flash channel altitudes vs model source vertical extent

4. **Pairing approach**
   - Use point geometry for flash locations
   - For rate comparison: bin both model source and LMA flashes in matching time/space windows
   - Radius of influence: configurable (default ~10 km to match storm scale)

5. **Tests**
   - Synthetic LMA NetCDF with known flash locations and times
   - Verify flash rate aggregation
   - Verify spatial binning

#### Verification

- Reader opens EOL NetCDF files without error
- Flash rates are non-negative, physically reasonable (~1-100+ flashes/min for active storms)
- Spatial distribution matches expected network coverage area

---

### Phase 3: ARM Sonde Reader

**New file:** `davinci_monet/observations/sonde/arm_sonde.py`

ARM SGP radiosondes provide thermodynamic profile validation for the model's meteorological state.

#### ARM Sonde Data Characteristics

- **Source:** ARM Data Discovery (NetCDF)
- **Site:** Southern Great Plains Central Facility, Lamont, OK (36.61°N, 97.49°W)
- **Frequency:** 4/day during DC3 (00, 06, 12, 18 UTC), 238 total
- **Variables:** Temperature, relative humidity, wind speed/direction, pressure, altitude

#### Tasks

1. **Implement ARM sonde reader**
   - Register as `@observation_registry.register('arm_sonde')`
   - Read ARM NetCDF sonde files
   - Standardize to profile geometry: `(time, level)` with altitude coordinate
   - Map variable names to standard meteorological names

2. **Pairing**
   - Use existing **profile pairing strategy**
   - Match model column at SGP location to sonde profiles
   - Temporal matching: nearest model time step to sonde launch time

3. **Tests**
   - Synthetic ARM-like NetCDF with known profiles
   - Verify profile geometry and coordinate standardization

#### Verification

- Reader opens ARM NetCDF files without error
- Profiles have physically reasonable temperature (200-310 K) and humidity
- Altitude coordinate is monotonically increasing

---

### Phase 4: Analysis Directory and Configs

**New directory:** `analyses/dc3/`

```
analyses/dc3/
├── README.md
├── configs/
│   ├── dc3-gv.yaml                 # GV outflow evaluation
│   ├── dc3-dc8.yaml                # DC-8 inflow evaluation
│   ├── dc3-falcon.yaml             # Falcon anvil evaluation
│   ├── dc3-all-aircraft.yaml       # Combined 3-aircraft evaluation
│   ├── dc3-lma.yaml                # Lightning flash rate evaluation
│   └── dc3-sondes.yaml             # ARM SGP sonde evaluation
├── scripts/
│   ├── download_dc3_aircraft.py    # Download ICARTT from NASA ASDC
│   ├── download_dc3_lma.py         # Download LMA from NCAR EOL
│   ├── download_dc3_sondes.py      # Download ARM SGP sondes
│   └── run_evaluation.py           # Pipeline execution
├── data/                            # Downloaded observations
├── output/                          # Plots and statistics
└── logs/                            # Pipeline logs
```

#### Tasks

1. **Create directory structure**

2. **Write README.md**
   - Campaign overview (adapted from DC3 reference above)
   - Setup instructions (environment variables, data paths)
   - Usage instructions (download, run, interpret)

3. **Write download scripts**
   - `download_dc3_aircraft.py`: Fetch GV, DC-8, Falcon ICARTT merge files from NASA ASDC
   - `download_dc3_lma.py`: Fetch LMA flash data from NCAR EOL
   - `download_dc3_sondes.py`: Fetch ARM SGP sondes from ARM archive

4. **Write YAML pipeline configs**

   **Aircraft config pattern (`dc3-gv.yaml` example):**
   ```yaml
   analysis:
     start_time: "2012-05-18"
     end_time: "2012-06-22"
     output_dir: ${DC3_ANALYSIS}/output
     log_dir: ${DC3_ANALYSIS}/logs

   model:
     mpas_davinci:
       mod_type: mpas
       files: ${DC3_DATA}/model/*.nc
       variables:
         NO:
           unit_scale: 965.7    # 1e9 * 28.97/30.01
         NO2:
           unit_scale: 629.5    # 1e9 * 28.97/46.01
         O3:
           unit_scale: 603.5    # 1e9 * 28.97/48.00
         CO:
           unit_scale: 1033.9   # 1e9 * 28.97/28.01

   obs:
     gv:
       obs_type: icartt
       filename: ${DC3_ANALYSIS}/data/dc3-gv-merge*.ict
       variables:
         NO_NOxyO3:
           obs_min: 0
           obs_max: 50000       # pptv
         NO2_NOxyO3:
           obs_min: 0
           obs_max: 50000
         O3_NOxyO3:
           obs_min: 0
           obs_max: 500         # ppbv
         CO_ACOMCO:
           obs_min: 0
           obs_max: 1000        # ppbv

   pairs:
     gv_no:
       model: mpas_davinci
       obs: gv
       variable:
         model_var: NO
         obs_var: NO_NOxyO3

   plots:
     gv_no_scatter:
       type: scatter
       pairs: [gv_no]
       title: "GV NO: MPAS-DAVINCI vs DC3"
       show_density: true

   stats:
     metrics: [N, MB, RMSE, R, NMB, NME, IOA]
   ```

5. **Write `run_evaluation.py`**
   - Set environment variables
   - Run pipeline for selected config
   - Display progress with tqdm
   - Log to timestamped file

6. **Environment variables**
   ```bash
   DC3_DATA=~/Data/DC3            # Model and raw observation data
   DC3_ANALYSIS=analyses/dc3      # Analysis directory (auto-set by run script)
   ```

#### ICARTT Variable Mapping (DC3-specific)

The ICARTT reader needs to handle DC3 variable naming conventions. DC3 merge files use instrument-suffixed names:

| Variable | GV Name | DC-8 Name | Falcon Name |
|----------|---------|-----------|-------------|
| NO | NO_NOxyO3 | NO_ESRL | NO_DLR |
| NO2 | NO2_NOxyO3 | NO2_TDLIF | NO2_DLR |
| O3 | O3_NOxyO3 | O3_ESRL | O3_DLR |
| CO | CO_ACOMCO | CO_DACOM | CO_DLR |
| NOy | NOy_NOxyO3 | NOy_ESRL | NOy_DLR |

These are configured per-pair in the YAML, so no code changes needed — just correct variable names in configs.

---

### Phase 5: MPAS Model Reader Integration with Config

Register the MPAS model type in the Pydantic config schema so YAML configs can specify `mod_type: mpas`.

#### Tasks

1. **Add `mpas` to model type enum in config schema**
2. **Add MPAS-specific config options** (if any beyond standard model config)
3. **Test end-to-end**: YAML config → config parser → MPAS reader → paired dataset

---

## Implementation Order

| Step | Description | Dependencies | New Files |
|------|-------------|--------------|-----------|
| 1 | MPAS model reader | None | `models/mpas.py` |
| 2 | MPAS reader tests | Step 1 | `tests/models/test_mpas.py` |
| 3 | LMA observation reader | None | `observations/lightning/__init__.py`, `observations/lightning/lma.py` |
| 4 | LMA reader tests | Step 3 | `tests/observations/test_lma.py` |
| 5 | ARM sonde reader | None | `observations/sonde/arm_sonde.py` |
| 6 | ARM sonde tests | Step 5 | `tests/observations/test_arm_sonde.py` |
| 7 | Register MPAS in config schema | Step 1 | Edit `config/schemas.py` |
| 8 | Analysis directory + README | None | `analyses/dc3/` tree |
| 9 | Download scripts | Step 8 | `analyses/dc3/scripts/download_*.py` |
| 10 | YAML pipeline configs | Steps 1, 3, 5, 7 | `analyses/dc3/configs/*.yaml` |
| 11 | Run evaluation script | Step 10 | `analyses/dc3/scripts/run_evaluation.py` |
| 12 | Campaign reference doc | None | `docs/campaigns/DC3.md` |

Steps 1, 3, 5, 8, 12 are independent and can be parallelized.

---

## Species Expansion Roadmap

As DAVINCI-MPAS adds chemistry phases, expand the DC3 evaluation species:

| DAVINCI Phase | New Species | DC3 Instruments |
|---------------|-------------|-----------------|
| Phase 3 (current) | NO, NO2 | NOxyO3 (GV), NOAA NOyO3 (DC-8), DLR suite |
| Phase 5 (Chapman) | O3 (prognostic) | NOxyO3 (GV), DIAL (DC-8) |
| Phase 6 (Chapman+NOx) | PAN, HNO3 | PAN-CIMS (GV), GT-CIMS (DC-8) |
| Phase 7 (CO+CH4) | CO, CH4, HCHO | DACOM (DC-8), ACOM CO (GV), CAMS (GV) |
| Phase 9 (Trop O3) | VOCs, OH, HO2 | TOGA (GV), ATHOS (DC-8), WAS (DC-8) |

Each expansion requires only new YAML config entries and variable mappings — no new reader code.

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
