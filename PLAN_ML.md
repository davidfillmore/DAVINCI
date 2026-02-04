# PLAN: ML Prior/Posterior Atmospheric Representation for ASIA-AQ

Date: 2026-02-04
Owner: TBD

## Goal
Build a statistical representation of atmospheric constituents over ASIA-AQ (O3, NOx, CO, aerosol) and refine prior distributions using observations. The output should provide posterior mean fields **and** uncertainty (variance/credible intervals) in space and time.

## Summary Recommendation
Start with a **Bayesian hierarchical spatiotemporal model** per species, using a **simple standard-atmosphere vertical profile** as the prior (no CAM-chem/CESM). Use the existing DAVINCI-MONET pairing/geometry tools as the observation operator. Begin with **independent species** (per-species fields) and add cross-species coupling later once the pipeline is stable. Use **MAP inference for the MVP** and add VI later if needed.

## Scope and Outputs
- **Spatial domain**: ASIA-AQ region on a fixed grid (match model grid initially).
- **Temporal domain**: configurable window (daily or hourly fields).
- **Species**: O3, NOx (NO + NO2 or NO2-only depending on obs), CO, aerosol (AOD or PM2.5).
- **Vertical representation**: surface + 3D (lat/lon/time/lev).
- **Outputs**:
  - Posterior mean concentration fields.
  - Posterior uncertainty (variance/credible interval).
  - Bias-corrected fields relative to priors.
  - Diagnostics by observation type and platform.

## Data Inputs (Initial)
Use existing DAVINCI-MONET ingestion and pairing:
- **Prior profiles**: standard-atmosphere-style vertical profiles per species (simple climatology).
- **Observations**:
  - Surface: AirNow (PM2.5/O3), AERONET (AOD)
  - Aircraft: DC-8 (O3, NO2, CO)
  - Pandora (NO2 column)
  - Satellites deferred to later phase (TROPOMI, MODIS, MOPITT)

## Technical Approach

### 1) Baseline Prior Model
Define a latent field for each species over (x, y, t, z optional):
- **Prior mean**: standard-atmosphere-style vertical profile (per species), optionally modulated by latitude/season.
- **Prior uncertainty**:
  - Start with broad, altitude-dependent variance (larger aloft, smaller near surface).
  - Later: heteroskedastic error by altitude, season, or regime.

Representation options:
- **Low-rank basis** (recommended initial):
  - Use spatial basis functions (e.g., PCA/EOFs or radial basis) to reduce dimensionality.
  - Latent coefficients evolve in time (AR(1) or random walk).
- **Full-grid Gaussian** (simple but heavy):
  - Independent per grid cell with temporal smoothing.

### 2) Observation Model (Likelihood)
Define observation operator H that maps latent field → observed quantity:
- Use existing pairing logic (Point/Track/Swath/Grid) as H.
- Observation error model:
  - Start with Gaussian noise; variance per obs type.
  - Add representativeness error for scale mismatch (grid vs point).

### 3) Posterior Inference
Start with stable, computationally safe inference:
- **MAP (MVP)**: optimize latent coefficients via gradient methods.
- **Variational inference (later)**: approximate posterior uncertainty more fully.
- **Ensemble Kalman update** (optional for sequential windows).
- Avoid full MCMC initially (too slow at scale).

### 4) Multi-Species Strategy
Phase 1: independent species (separate priors/likelihoods). 
Phase 2: coupled covariance (e.g., shared latent factors or cross-species priors).

## System Design (Integration with DAVINCI-MONET)
- **Data pipeline**:
  - Use `load_observations` and pairing geometry to map the latent grid to obs locations.
  - Optionally use `load_models` only for grid metadata (no model priors required).
  - Store paired data to a training-ready format (Parquet/Zarr).
- **ML module**: new package `davinci_monet/ml/` with:
  - `data.py` (feature assembly)
  - `priors.py` (prior construction)
  - `likelihoods.py` (obs models)
  - `inference.py` (MAP/VI/EnKF)
  - `outputs.py` (posterior fields + uncertainty)
- **Config**: add `ml` section in YAML to control species, priors, windows, inference.

## Validation and Diagnostics
- Hold-out by platform: e.g., train with surface+aircraft, validate on satellite.
- Time-split: fit on early weeks, validate on later weeks.
- Metrics: RMSE, bias, correlation, CRPS (uncertainty quality), coverage probability.

## Phased Delivery Plan

### Phase 0: Scoping (1-2 days)
- Confirm species list, obs sources, spatial/temporal resolution.
- Choose representation (low-rank basis vs full grid).
- Confirm surface + 3D targets, and Mac-friendly defaults.

### Phase 1: Data Products (1-2 weeks)
- Generate standardized paired datasets for each species.
- Build a simple prior-profile library (per species, per altitude, optional seasonal variants).
- Save to a versioned training dataset (Zarr recommended).

### Phase 2: Baseline Prior + MAP Posterior (2-3 weeks)
- Implement prior from standard-atmosphere profiles.
- Implement MAP inference for a single species (O3) on a week window.
- Produce posterior maps and uncertainty estimates.
- Keep defaults Mac-safe (limited workers, chunking).

### Phase 3: Multi-Species Expansion (2-4 weeks)
- Add NO2, CO, aerosol.
- Add optional cross-species coupling.
- Extend from surface-only to 3D fields if not already in Phase 2.

### Phase 4: Operationalization (2-4 weeks)
- Add CLI command `davinci-monet ml-fit ...`.
- Add evaluation reports and regression tests.
- Document workflow in `ARCHITECTURE.md` and `PERFORMANCE.md`.
- Add satellite ingestion to a later phase (post-Phase 4) when stable.

## Open Questions
- Which standard profile source to use per species (fixed global vs lat/season dependent)?
- How to treat NOx (NO + NO2) given available observations?

## Minimal MVP
- O3 only, one-week window, surface field only.
- Prior from standard-atmosphere vertical profile, posterior via **MAP**.
- Validate with AirNow + DC-8 only.

---

If you confirm the open questions above, I can refine this into a concrete implementation plan (config schema, data formats, and first tasks).
