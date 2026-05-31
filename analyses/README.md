# DAVINCI Analyses

Real-data model evaluation analyses using DAVINCI.

## Directory Structure

```
analyses/
├── README.md           # This file
└── asia-aq/            # ASIA-AQ campaign analysis
    ├── README.md       # Campaign overview and data sources
    ├── configs/        # YAML configuration files
    ├── scripts/        # Python analysis scripts
    └── output/         # Generated plots and statistics
```

## Analyses

| Analysis | Campaign | Model | Period | Status |
|----------|----------|-------|--------|--------|
| [asia-aq](asia-aq/) | NASA ASIA-AQ | CESM/CAM-chem | Feb 2024 | In Progress |

## Adding New Analyses

1. Create a new directory: `analyses/<name>/`
2. Add a README.md with campaign/project overview
3. Create subdirectories: `configs/`, `scripts/`, `output/`
4. Add exploration scripts to understand the data
5. Create YAML configs for model-observation pairing
6. Run analyses using CLI or Python scripts

## Running Analyses

```bash
# Activate environment
conda activate davinci

# Explore model data
python analyses/asia-aq/scripts/explore_model.py

# Test CESM reader
python analyses/asia-aq/scripts/test_cesm_reader.py

# Run full analysis (when observation data available)
davinci-monet run analyses/asia-aq/configs/cesm_surface.yaml
```

## Data Locations

Model data is typically stored externally and referenced by path in configs:
- ASIA-AQ CESM: `~/Data/ASIA-AQ/`

Observation data should be downloaded to `analyses/<name>/data/` or referenced externally.
