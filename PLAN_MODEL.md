# PLAN: Model Prior/Posterior Atmospheric Representation for ASIA-AQ

Date: 2026-02-04
Owner: TBD

## Goal
Build a statistical representation of atmospheric constituents over ASIA-AQ (O3, NOx, CO, aerosol) and refine prior distributions using observations. The output should provide posterior mean fields **and** uncertainty (variance/credible intervals) in space and time.

## Summary Recommendation
Start with **campaign-mean vertical profiles** per species computed from all flights, then adjust those profiles to match column and/or surface observations. Use **optimal interpolation** between surface sites to create a simple horizontal dependence. Then **enforce photochemical consistency** using a simple box-model chemistry mechanism with a photolysis rate model. Use the existing DAVINCI pairing/geometry tools as the observation operator. Begin with **independent species** and add cross-species coupling later once the pipeline is stable.

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
Use existing DAVINCI ingestion and pairing:
- **Prior profiles**: campaign-mean vertical profiles per species derived from aircraft data.
  - **Fallback**: AFGL standard profiles if aircraft coverage is sparse or missing for a species.
  - **Manual provision**: AFGL tables will be fetched manually and stored as CSVs in `data/afgl/`.
  - Expected layout: one CSV per reference atmosphere, plus a short `data/afgl/README.md` with provenance.
- **Observations**:
  - Surface: AirNow (PM2.5/O3), AERONET (AOD)
  - Aircraft: DC-8 (O3, NO2, CO)
  - Pandora (NO2 column)
  - Satellites deferred to later phase (TROPOMI, MODIS, MOPITT)

## Profile Source Choices (Priors)

Primary prior source:
- **Campaign-mean aircraft profiles** for each species (DC-8), binned by altitude/pressure.

Fallback sources:
- **AFGL constituent profiles (Anderson et al., 1986)** for trace gas VMR profiles (includes O3, NO, NO2, CO).
- **U.S. Standard Atmosphere 1976** for baseline pressure/temperature/density and minor constituent tables (useful for conversions).

Optional refinements by species:
- **O3**: MLS + sonde climatology (monthly, latitude-banded profiles).
- **Aerosol**: OPAC aerosol models with standard mixtures and exponential vertical profiles.

Notes:
- AFGL profiles are climatological and not photochemically consistent across species; they are intended as practical standard profiles.
- We will **not** auto-download these tables; they will be added manually and versioned in-repo.

References (for selection):
- Anderson et al., 1986, AFGL-TR-86-0110 (trace gas vertical profiles)
- NOAA/NASA/USAF, 1976, U.S. Standard Atmosphere (NASA TM-X-74335)
- McPeters & Labow, 2012, MLS + sonde ozone climatology (GSFC.JA.6143.2012)
- Hess, Koepke & Schult, 1998, OPAC aerosol model (BAMS 79:831–844)

## Technical Approach

### 1) Campaign-Mean Vertical Profiles (Prior)
- Compute mean vertical profiles from all aircraft data (DC-8) per species.
- Standardize to common vertical coordinate (pressure or altitude).
- Store as baseline profiles with uncertainty from flight-to-flight variability.

### 2) Adjust Profiles to Match Surface/Column Observations
- For each species, compute a scale factor or smooth correction so the profile integrates to match column observations (e.g., Pandora NO2).
- Apply surface constraints by anchoring the lowest-level value to surface observations (e.g., AirNow O3, PM2.5).
- Resolve conflicts by weighted least squares using instrument error estimates.

### 3) Horizontal Dependence via Optimal Interpolation
- Use surface sites as control points and perform optimal interpolation (OI) on the lowest model layer.
- Extrapolate OI-adjusted surface field to full vertical profile using the adjusted vertical shape.
- Use a simple covariance model (e.g., isotropic exponential) with tunable length scale.

### 4) Photochemical Consistency (Box Model)
- Run a simple box-model chemistry mechanism to adjust the multi-species profiles toward photochemical balance.
- Use a photolysis rate model to compute J-values (e.g., clear-sky approximations initially).
- Apply this step after profile adjustment and OI, as a consistency correction.

### 5) Multi-Species Strategy
- Phase 1: independent species (separate profiles and OI fields).
- Phase 2: optional cross-species coupling (shared spatial covariance or ratio constraints).

## System Design (Integration with DAVINCI)
- **Data pipeline**:
  - Use `load_observations` and pairing geometry to map the latent grid to obs locations.
  - Optionally use `load_models` only for grid metadata (no model priors required).
  - Store paired data to a training-ready format (Parquet/Zarr).
- **ML module**: new package `davinci_monet/ml/` with:
  - `data.py` (feature assembly)
  - `priors.py` (prior construction)
  - `likelihoods.py` (obs models)
  - `updates.py` (profile adjustment + optimal interpolation)
  - `outputs.py` (posterior fields + uncertainty)
- **Config**: add `ml` section in YAML to control species, priors, windows, inference.

## Validation and Diagnostics
- Hold-out by platform: fit with aircraft + surface, validate on columns or withheld sites.
- Time-split: fit on early weeks, validate on later weeks.
- Metrics: RMSE, bias, correlation, column closure error, uncertainty coverage (if applicable).

## Phased Delivery Plan

### Phase 0: Scoping (1-2 days)
- Confirm species list, obs sources, spatial/temporal resolution.
- Choose representation (low-rank basis vs full grid).
- Confirm surface + 3D targets, and Mac-friendly defaults.

### Phase 1: Data Products (1-2 weeks)
- Generate standardized paired datasets for each species.
- Build a simple prior-profile library (per species, per altitude, optional seasonal variants).
- Ingest AFGL tables into `data/afgl/` as CSVs (manual fetch).
- Save to a versioned training dataset (Zarr recommended).

### Phase 2: Baseline Prior + MVP Update (2-3 weeks)
- Implement campaign-mean profile construction.
- Implement column/surface adjustment for one species (O3) on a week window.
- Implement OI horizontal interpolation for surface sites.
- Produce updated fields and uncertainty estimates (if applicable).
- Keep defaults Mac-safe (limited workers, chunking).

### Phase 3: Multi-Species Expansion (2-4 weeks)
- Add NO2, CO, aerosol.
- Add optional cross-species coupling.
- Extend from surface-only to 3D fields if not already in Phase 2.

### Phase 4: Photochemical Consistency (2-4 weeks)
- Add a simple box-model mechanism and photolysis rate calculation.
- Apply chemical consistency corrections across the adjusted profiles.

### Phase 5: Operationalization (2-4 weeks)
- Add CLI command `davinci-monet ml-fit ...`.
- Add evaluation reports and regression tests.
- Document workflow in `ARCHITECTURE.md` and `PERFORMANCE.md`.
- Add satellite ingestion to a later phase (post-Phase 4) when stable.

## Open Questions
- Which vertical coordinate should be canonical (pressure vs altitude) for profile averaging?
- How to weight aircraft vs surface vs column constraints when adjusting profiles?
- What horizontal length scale should OI use per species and season?
- Which box-model mechanism and photolysis scheme to use for consistency?
- How to treat NOx (NO + NO2) given available observations?

## Minimal MVP
- O3 only, one-week window.
- Prior from campaign-mean DC-8 profile.
- Surface anchoring via AirNow; optional column check if available.
- OI horizontal interpolation across surface sites.

## Plan Notes
- **MVP (Minimum Viable Product)**: the smallest end-to-end version that runs reliably and yields useful outputs before adding complexity.

---

If you confirm the open questions above, I can refine this into a concrete implementation plan (config schema, data formats, and first tasks).
