# DAVINCI

**Data Analysis and Validation Infrastructure for Chemistry**

A modern, type-safe Python toolkit for evaluating atmospheric chemistry and air quality models against observations, based on MELODIES-MONET.

---

**Session Reminder**: Agentic AI coding is cognitively intense - you're managing a fast-moving collaborator while maintaining your own mental model. Remember to:
- Take breaks between major tasks (stretch, move around)
- Step away when stuck - insights often come when you're not at the keyboard
- You control the pace - slow down when needed

---

## Cross-Model Handoff Convention

This repo uses **cross-model code reviews and hand-offs** (e.g., Claude Opus writes implementation, Codex reviews, then back). Handoff files follow a consistent structure so any model can pick up context quickly.

### Handoff File Format

Use `REVIEW_<MODEL>.md` or `HANDOFF_<TOPIC>.md` in the repo root. Structure:

```markdown
## Context
Branch, task description, which files are involved

## Changes Made
What was done, with file paths and line references

## Decisions & Rationale
Why choices were made — prevents the next model from undoing or redoing work

## Open Questions / Concerns
Things the next model should investigate or address

## Suggested Next Steps
Specific actionable items
```

### Rules

- **One file per task/feature** — scoped context, not a running log
- **Always include Decisions & Rationale** — this is the highest-value section
- **Reference file paths and line numbers** — so the next model can verify without searching
- **Check for handoff files at session start** — look for `REVIEW_*.md` or `HANDOFF_*.md` in repo root
- **Git diff supplements the handoff** — the file gives intent, `git diff` gives the changes

---

## Git Workflow

- **NEVER auto commit or push**: Do NOT commit or push changes until the user explicitly confirms
- **Hold off on merge to main**: Do NOT merge to `main` until the user has verified the runs and explicitly asks for a merge
- **After merge, return to develop**: Always switch back to `develop` branch after merging to `main`

---

## ⚠️ CRITICAL: CESM Vertical Coordinate Convention

**This issue has been rediscovered 4+ times. READ THIS FIRST when working with CESM/CAM-chem data.**

CESM uses hybrid sigma-pressure coordinates where **pressure increases with level index**:
- `lev=0` → **Top of Atmosphere** (stratosphere, ~3 hPa)
- `lev=-1` (last index) → **Surface** (highest pressure, ~1000 hPa)

**Common symptom**: Model O3 values of 5000-10000 ppb (stratospheric) instead of 30-80 ppb (surface).

**The fix** (implemented in `_extract_surface()` in `base.py`):
```python
# Auto-detect if pressure increases with index
if vert_vals[-1] > vert_vals[0]:
    surface_idx = -1  # CESM convention: last level is surface
else:
    surface_idx = 0   # Other conventions: first level is surface
```

**If you see impossibly high trace gas values, check vertical level extraction FIRST.**

---

## Pre-Implementation Audit (REQUIRED)

**Before building any new component**, audit existing code for patterns it must conform to. This has caused repeated rework — new features were built in isolation, ignoring existing conventions, styles, renderers, CLI features, and naming patterns.

**Audit checklist** (search the codebase for each before writing code):

1. **Existing renderers/implementations** — Does a similar component already exist? (e.g., 3D flight track plotter already existed when 2D was introduced; should have extended, not reinvented)
2. **Style and theming** — Check `plots/style.py` for color conventions, fonts, palettes. New plotters must use the correct colors for their context (see Plot Styling section)
3. **CLI integration** — Check `cli/app.py` and `pipeline/runner.py` for flags like `--show-plots`. New pipeline stages must work with existing CLI features
4. **Config naming conventions** — Check `analyses/*/configs/` for naming patterns (e.g., `*-gemini.yaml` for machine-specific configs with absolute paths)
5. **Pipeline stage contracts** — Check what data keys existing stages return (e.g., `plots_generated`) and ensure new stages follow the same contracts
6. **Data conventions** — Check observation readers for coordinate aliases, unit conventions, and standardization patterns already in place

**The audit is not optional.** Every missed convention costs a debugging + fix cycle that's slower than reading the code upfront.

---



```bash
# Activate conda environment
conda activate davinci-monet

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy davinci_monet

# Format code
black davinci_monet && isort davinci_monet
```

## Conda Environment

**Name**: `davinci-monet`

**Create from environment.yml**:
```bash
conda env create -f environment.yml
conda activate davinci-monet
```

**Key packages** (inherited from melodies-monet):
- monet, monetio - atmospheric data I/O
- xarray, numpy, pandas - data structures
- matplotlib, cartopy - plotting
- netCDF4 - file I/O

**Additional I/O support**:
- pyhdf - HDF4/HDF-EOS for MODIS satellite data
- tqdm - Progress bars for pipeline execution

**Development tools**:
- pydantic - configuration validation
- mypy - static type checking
- pytest, pytest-cov - testing
- black, isort - formatting

## Related Repositories

**MELODIES-MONET location**: `/Users/fillmore/EarthSystem/MELODIES-MONET`

**Wiki repository**: `/Users/fillmore/EarthSystem/DAVINCI-MONET.wiki`

Key files to reference:
- `melodies_monet/driver.py` - Main logic (3,116 lines) - decompose this
- `melodies_monet/_cli.py` - CLI implementation (1,524 lines)
- `melodies_monet/plots/` - Plotting modules
- `melodies_monet/stats/` - Statistics modules
- `examples/yaml/` - 31 example YAML configs for backward compat testing

## Project Goals

- **Maintainability**: Small, focused modules (<500 lines each)
- **Type Safety**: Full type hints, mypy strict mode
- **Performance**: Parallel processing, lazy loading
- **Extensibility**: Plugin architecture for models/observations/plotters

## Key Design Principles

1. **Uniform Pairing Logic**: Strategy based on data geometry (point, track, profile, swath, grid) not data source

2. **xarray-Only Data Model**: All data as `xr.Dataset` throughout pairing/analysis. Pandas only for I/O adapters and stats output tables.

3. **Synthetic Data for Testing**: Generate test data programmatically - no external dataset dependencies

## Architecture Overview

```
davinci_monet/
├── core/           # Protocols, registry, base classes, exceptions
├── config/         # Pydantic schemas, YAML parsing
├── models/         # Model implementations (CMAQ, WRF-Chem, etc.)
├── observations/   # Observation handlers (surface, aircraft, satellite)
├── pairing/        # Unified pairing engine + strategies
│   └── strategies/ # point, track, profile, swath, grid
├── plots/          # Modular plotting system
│   └── renderers/  # Individual plot types
├── stats/          # Statistics calculation
├── pipeline/       # Execution orchestration
├── io/             # File readers/writers
├── cli/            # Command-line interface
├── logging/        # Structured logging
├── util/           # Shared utilities
└── tests/
    └── synthetic/  # Test data generators
```

## Implementation Status

**STATUS: COMPLETE** — 961 tests passing. CI via GitHub Actions (pytest, mypy, black/isort).

## Running Analyses

**ALL analysis scripts MUST use DAVINCI pipelines.** Do not write custom pairing/plotting scripts.

### Pipeline Stages

The standard pipeline executes these stages in order:
1. `load_models` - Load model data, apply unit conversions
2. `load_observations` - Load observation data, apply unit conversions
3. `pairing` - Pair model with observations using geometry-specific strategies
4. `statistics` - Calculate evaluation metrics (N, MB, RMSE, R, NMB, NME, IOA)
5. `plotting` - Generate scatter, spatial bias, time series plots
6. `save_results` - Write statistics to CSV

### Running a Pipeline

```python
from davinci_monet.pipeline.runner import run_analysis

result = run_analysis("path/to/config.yaml")
if result.success:
    print(f"Completed in {result.total_duration_seconds:.1f}s")
```

Or via CLI:
```bash
davinci-monet run path/to/config.yaml
```

### YAML Configuration Pattern

```yaml
analysis:
  start_time: "2024-02-01"
  end_time: "2024-02-03"
  output_dir: ${MY_ANALYSIS}/output  # Supports env var expansion
  log_dir: ${MY_ANALYSIS}/logs       # Pipeline logs with timestamps

model:
  my_model:
    mod_type: cesm_fv  # or cmaq, wrfchem, ufs, generic
    files: ${MY_DATA}/model/*.nc     # Env vars in paths
    radius_of_influence: 15000
    variables:
      PM25:
        unit_scale: 1.2e9  # kg/kg to µg/m³
      O3:
        unit_scale: 1.0e9  # mol/mol to ppb

obs:
  my_obs:
    obs_type: pt_sfc
    filename: ${MY_ANALYSIS}/data/observations.nc
    variables:
      pm25:
        obs_min: 0
        obs_max: 500

pairs:
  model_obs_pm25:
    model: my_model
    obs: my_obs
    variable:
      model_var: PM25
      obs_var: pm25

plots:
  pm25_scatter:
    type: scatter
    pairs: [model_obs_pm25]
    title: "PM2.5 Model vs Observations"

stats:
  metrics: [N, MB, RMSE, R, NMB, NME, IOA]
```

### Config Naming Convention

Machine-specific configs use `{campaign}-{variant}-{machine}.yaml` with **full absolute paths** (no env vars). Examples:
- `asia-aq-dc8-gemini.yaml` — ASIA-AQ DC-8 analysis on Gemini (Mac)
- `dc3-obs-dc8-gemini.yaml` — DC3 obs-only DC-8 on Gemini
- `asia-aq-airnow-derecho.yaml` — ASIA-AQ AirNow on Derecho (HPC)

Machine names: `gemini` (local Mac), `derecho` (NCAR HPC)

### Environment Variable Expansion

YAML config paths support `${VAR}` syntax for environment variables, but **machine-specific configs should use absolute paths instead**. Env var configs are for portable/template use only:
```bash
export MY_DATA=~/Data/campaign
export MY_ANALYSIS=/path/to/analysis
```

### Variable Naming Convention

Paired datasets use **prefix** format:
- `model_pm25` - Model values
- `obs_pm25` - Observation values

NOT suffix format (`pm25_model`, `pm25_obs`).

## Key Design Patterns

1. **Plugin Registry**: Components register via decorators
   ```python
   @model_registry.register('cmaq')
   class CMAQModel(BaseModel): ...
   ```

2. **Protocol-based Interfaces**: Python Protocols define contracts

3. **Pydantic Configuration**: Type-safe YAML parsing with validation

4. **Pipeline Architecture**: Composable stages replace monolithic methods

## Data Model (xarray-only)

```
Model:   xr.Dataset with dims (time, level, lat, lon)
Point:   xr.Dataset with dims (time, site) + lat/lon coords
Track:   xr.Dataset with dims (time,) + lat/lon/alt coords
Profile: xr.Dataset with dims (time, level) + lat/lon coords
Swath:   xr.Dataset with dims (time, scanline, pixel)
Grid:    xr.Dataset with dims (time, lat, lon)
Paired:  xr.Dataset with aligned model + obs variables
```

## Backward Compatibility

- Full compatibility with existing MELODIES-MONET YAML configuration files
- Continues using monet/monetio libraries for data I/O

## Working Example: ASIA-AQ Analysis

Reference implementation in `analyses/asia-aq/`:

```
analyses/asia-aq/
├── configs/
│   └── asia-aq.yaml                # Pipeline configuration
├── scripts/
│   ├── download_airnow.py          # Data download
│   └── run_evaluation.py           # Pipeline execution
├── data/                           # Observation data
├── output/                         # Plots and statistics
├── logs/                           # Pipeline logs
└── misc/                           # Exploratory scripts (not part of workflow)
```

**Environment variables**:
- `ASIA_AQ_DATA`: Model/observation data root (default: `~/Data/ASIA-AQ`)
- `ASIA_AQ_ANALYSIS`: Analysis directory (set automatically by `run_evaluation.py`)

**Run the analysis**:
```bash
cd analyses/asia-aq
export ASIA_AQ_DATA=~/Data/ASIA-AQ
python scripts/run_evaluation.py
```

Pipeline displays progress with tqdm and logs to `logs/pipeline_YYYYMMDD_HHMMSS.log`.

## Plot Styling (NCAR Branding)

The plotting system uses NSF NCAR brand colors and fonts. Enable via YAML config or Python:

### YAML Configuration (Recommended)

```yaml
analysis:
  start_time: "2024-02-01"
  end_time: "2024-02-03"
  output_dir: ${MY_ANALYSIS}/output
  style:
    theme: ncar                # Apply NCAR branding
    context: default           # or: presentation, publication
    use_seaborn: true          # Apply seaborn whitegrid theme
```

### Python API

```python
from davinci_monet.plots import apply_ncar_style, plot_timeseries

# Apply NCAR styling globally (call once at start of script)
apply_ncar_style()

# Create plots with consistent styling
fig = plot_timeseries(paired_data, "obs_o3", "model_o3")
```

**Key colors** (`davinci_monet.plots.style`):
- `NCAR_PRIMARY`: NCAR Blue (#0A5DDA) — brand color, used for **obs-only** plots
- `OBS_COLOR`: Gray (#58595B) — observations **in model-vs-obs paired** plots (for contrast)
- `MODEL_COLOR`: NCAR Blue (#0A5DDA) — model data in paired plots
- `NCAR_PALETTE`: 8-color palette for multiple datasets / per-flight coloring

**Context presets**:
- `default`: Standard sizes for general use
- `presentation`: Larger fonts for slides
- `publication`: Smaller fonts for journal figures

**Font**: Poppins (with Helvetica/Arial fallbacks)

## Common Gotchas

1. **Unit conversions**: Model variables often need `unit_scale` in config:
   - CESM mixing ratios (mol/mol) → ppb: `unit_scale: 1.0e9`
   - CESM PM mass (kg/kg) → µg/m³: `unit_scale: 1.2e9`

2. **AERONET wavelengths**: Use `aod_500nm` or `aod_440nm` for Asia (not `aod_551nm`)

3. **CESM vertical levels**: Surface is `lev=-1` (last index), NOT `lev=0`. See CRITICAL warning above. This has caused bugs 4+ times.

4. **Observation coordinates**: Must have `latitude`, `longitude` as coordinates or variables

5. **Time alignment**: Pipeline uses nearest-neighbor interpolation for model→obs times

6. **High-frequency observations**: Use `resample` to average sub-hourly data to match model resolution:
   ```yaml
   obs:
     pandora:
       obs_type: pt_sfc
       filename: /data/pandora/*.nc
       resample: "h"           # Average to hourly
       min_obs_count: 3        # Require ≥3 obs per hour (reject sparse hours)
       track_obs_count: true   # Add obs_count variable for diagnostics
   ```

7. **Scatter plot density**: For busy scatter plots with many points, enable density coloring:
   ```yaml
   plots:
     my_scatter:
       type: scatter
       show_density: true      # Color points by local density
   ```

8. **HDF5 thread safety segfaults**: If you see HDF5 errors mentioning "thread 1/thread 2" followed by a segmentation fault, this is an HDF5 thread safety issue. The segfault happens at the C level before Python can catch it, so retry logic won't help. Fix by disabling HDF5 file locking:
   ```bash
   HDF5_USE_FILE_LOCKING=FALSE davinci-monet run config.yaml
   ```
   If it persists, also limit Dask workers:
   ```bash
   DASK_NUM_WORKERS=1 HDF5_USE_FILE_LOCKING=FALSE davinci-monet run config.yaml
   ```
