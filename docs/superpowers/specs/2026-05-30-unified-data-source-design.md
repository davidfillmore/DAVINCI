# Unified Data-Source Architecture — Design

**Date:** 2026-05-30
**Status:** Design approved, pending spec review
**Branch target:** `develop` (phased; each phase keeps the 961-test suite green)

## Context

DAVINCI currently draws a hard line between **models** and **observations**, even
though both are simply data with a topology (point, track, profile, swath, grid).
The pairing layer is *already* geometry-driven (`DataGeometry` enum + per-geometry
strategies), and CLAUDE.md already states the principle "Uniform Pairing Logic:
Strategy based on data geometry … **not data source**." The implementation has
drifted from that principle: the model/obs split survives in config, core
protocols, registries, the pipeline, variable naming, and plot styling.

**Goal:** remove the artificial model/obs distinction. Both become a single
`DataSource` abstraction distinguished only by topology. A model/obs *role* may be
carried as optional metadata for labeling and styling, but it never drives pairing
logic. This enables **arbitrary binary pairs** — two models, two observations, or
one of each, in either direction.

### Decisions locked during brainstorming

1. **Config: clean break + migration tool.** Replace `model:`/`obs:` with a unified
   `sources:` block. A `migrate-config` CLI converts legacy configs. The
   MELODIES-MONET backward-compat principle in CLAUDE.md is relaxed and updated as
   part of this work.
2. **Pair direction: precedence default + explicit override.** Geometry precedence
   rules pick the reference/comparand automatically; a pair may override with an
   explicit `reference:`.
3. **Naming: source-label prefixes + role for styling.** Paired variables use the
   source's own label as prefix (`cam_o3`, `airnow_o3`). An optional `role:
   model|obs` tag drives only default plot colors/legend; pairing ignores it.
4. **Scope: full unification, phased & test-green.** Every layer unified, landed in
   ordered phases that each keep the suite green.
5. **Structure: Approach A — single `DataSource` abstraction.** Unify
   protocols/registries/pipeline/pairing/config; keep existing `models/` and
   `observations/` directories (readers re-parent onto `SourceReader`). No large
   file-move diff.
6. **Pair var-mapping syntax:** `sources: [a, b]` + `variables: {a: VAR, b: var}`.
7. **Same-geometry default direction:** when two same-geometry sources are paired
   with no explicit `reference:`, default to the first-listed source as reference
   and emit a warning.

## The model/obs split today (what we are removing)

| Layer | Model side | Obs side |
|-------|-----------|----------|
| Config | `model:` block, `ModelConfig`, `mod_type` | `obs:` block, `ObservationConfig`, `obs_type` |
| Core protocols | `ModelReader`, `ModelProcessor` | `ObservationReader`, `ObservationProcessor` |
| Registries | `model_registry` (+ stray `reader_registry`) | `observation_registry` |
| Pipeline stages | `LoadModelsStage` | `LoadObservationsStage` |
| Pipeline context | `get_model(label)` | `get_observation(label)` |
| Obs-only path | — | `create_obs_pipeline`, `ObsPlottingStage`, `ObsStatisticsStage` |
| Pairing | `pair(model, obs)`, `_get_model_coords` | `_get_obs_coords` |
| Plots | `MODEL_COLOR`, `model_` prefix | `OBS_COLOR`, `obs_` prefix |

The only substantive protocol difference: `ObservationReader` declares a `geometry`
property; `ModelReader` does not (it assumes `GRID`). Unification means **every
reader declares its output geometry**.

## Architecture

### 1. Core abstraction & protocols

- **`SourceReader`** protocol replaces `ModelReader` + `ObservationReader`:
  - `name: str` — reader type id (e.g. `cesm_fv`, `pt_sfc`, `modis_l2`)
  - `geometry: DataGeometry` — required on every reader; model readers return `GRID`
  - `open(file_paths, variables=None, time_range=None, **kwargs) -> xr.Dataset` —
    the merged superset signature (`time_range` now available to all readers)
  - `get_variable_mapping() -> Mapping[str, str]`
- **`SourceProcessor`** protocol replaces `ModelProcessor` + `ObservationProcessor`:
  - `process(dataset, **kwargs) -> xr.Dataset`
  - Unit conversion, CESM surface extraction / vertical-coordinate handling,
    resampling, `min_obs_count`, QA/QC all become processors in a single chain.
    Concrete processors may still be specific to a reader's needs; they just share
    one protocol.
- `DataGeometry` enum is unchanged.
- Every loaded dataset carries attrs: `geometry`, `role` (`"model"`/`"obs"`/absent),
  `source_label`.

### 2. Registry consolidation

- One **`source_registry`** keyed by a single `type` string replaces
  `model_registry`, `observation_registry`, and the unused `reader_registry`.
- All `@model_registry.register(...)` / observation registrations migrate to
  `@source_registry.register(...)`.
- **Open task (phase 2):** reader type ids must be globally unique across former
  model and obs types. Initial audit shows `generic` is registered only on the
  model side. Full audit happens in phase 2; namespace any collisions found.
- During phasing, `model_registry`/`observation_registry` become deprecated aliases
  pointing at `source_registry`, removed in phase 6.

### 3. Config schema (clean break + migration)

```yaml
sources:
  cam:
    type: cesm_fv
    role: model            # optional — styling/legend only
    files: ${DATA}/cam/*.nc
    radius_of_influence: 15000
    variables:
      O3: { unit_scale: 1.0e9 }
  airnow:
    type: pt_sfc
    role: obs              # optional
    filename: ${DATA}/airnow.nc
    variables:
      o3: { obs_min: 0, obs_max: 500 }

pairs:
  cam_vs_airnow_o3:
    sources: [cam, airnow]   # binary; order does NOT imply direction
    reference: airnow        # optional override; default by precedence
    variables:               # per-source variable name
      cam: O3
      airnow: o3
```

- **Pairs are binary.** Two models, two obs, or one of each all work. N-way
  comparison is explicitly out of scope (YAGNI).
- **Paired output variables** are named `<source_label>_<canonical_var>`, where the
  canonical var is the reference source's variable name — e.g. `cam_o3`, `airnow_o3`.
- New Pydantic schema: `SourceConfig` (replaces `ModelConfig` + `ObservationConfig`)
  and a revised `PairConfig` (`sources`, optional `reference`, `variables` map).
- **Migration tool:** `davinci-monet migrate-config old.yaml -o new.yaml`. Reads
  legacy `model:`/`obs:`/`pairs:`, emits the unified form: `role: model` for former
  model entries, `role: obs` for former obs entries, and `reference: <obs-label>`
  on each pair to preserve the old model→obs sampling direction.

### 4. Pipeline

- **`LoadSourcesStage`** replaces `LoadModelsStage` + `LoadObservationsStage`:
  iterates `sources`, looks up `source_registry[type]`, opens, runs the processor
  chain, tags the dataset with `role`/`source_label`/`geometry`, stores in
  `context.sources[label]`.
- `PipelineContext`: `sources` dict + `get_source(label)`; `get_model` /
  `get_observation` kept as deprecated shims during phases, removed in phase 6.
- **Delete the obs-only pipeline** (`create_obs_pipeline`, `ObsPlottingStage`,
  `ObsStatisticsStage`). Plotting/statistics stages learn to reference *either a
  pair or a single source*, so "obs-only" becomes "a pipeline with one source and no
  cross-source pairs." This is the meatiest consolidation and owns phase 5.

### 5. Pairing

- `PairingStrategy.pair(reference, comparand, ...)` — role-neutral signature. The
  `comparand` is resampled onto the `reference`'s geometry.
- **Strategy dispatch** becomes a table keyed by `(reference_geometry,
  comparand_geometry)`. Existing strategies seed the
  `(POINT|TRACK|PROFILE|SWATH, GRID)` and `(GRID, GRID)` cells, exactly matching
  today's behavior. New like-geometry combos (e.g. point↔point nearest-match) are
  added incrementally; unsupported combos raise a clear, explicit error.
- **Precedence rule (default direction):** irregular geometries
  (POINT/TRACK/PROFILE/SWATH) outrank GRID as the reference, so a GRID source is
  sampled onto them — identical to current model→obs behavior. When both sources
  share a geometry and no `reference:` is given, default to the first-listed source
  as reference and emit a warning.
- Helper renames: `_get_model_coords` → `_get_comparand_coords`, `_get_obs_coords` →
  `_get_reference_coords` (or a single `_get_coords(ds)`).

### 6. Plots & styling

- Renderers resolve datasets by **source label** instead of `model_`/`obs_`
  prefixes.
- **Color resolver:** `role == "obs"` → `OBS_COLOR` (gray), `role == "model"` →
  `MODEL_COLOR` (blue) — preserving today's convention. Same-role or role-less pairs
  cycle `NCAR_PALETTE` by source order.

## Phasing (each phase keeps the suite green)

1. **Core** — add `SourceReader` / `SourceProcessor` protocols + `source_registry`;
   make old protocol/registry names aliases; every reader declares `geometry`.
2. **Readers/registry** — migrate all registrations to `source_registry`; audit and
   resolve type-id collisions.
3. **Pipeline** — `LoadSourcesStage` + `context.sources`; add `get_model` /
   `get_observation` shims.
4. **Pairing** — role-neutral `pair()` signatures + `(ref, comp)` dispatch table +
   precedence/override logic.
5. **Plots/stats + obs-only consolidation** — source-label prefixes, role-based
   styling, fold the obs-only pipeline into the standard pipeline (plots/stats
   reference a pair *or* a single source).
6. **Config + migration** — new `sources:`/`pairs:` schema, `migrate-config` CLI,
   delete legacy schema and all shims/aliases, update CLAUDE.md principles.

Aliases and shims introduced in early phases let the suite stay green; they are all
removed in phase 6.

## Error handling

- Unsupported `(reference_geometry, comparand_geometry)` combination → clear error
  naming both geometries and listing supported combos.
- Same-geometry pair without explicit `reference:` → warning + default to first
  source (not an error).
- Unknown `type` in `sources:` → registry lookup error listing available types.
- Duplicate registration of a `type` id across former model/obs readers → caught in
  the phase-2 audit; resolved by namespacing.

## Testing

- Each phase migrates its own tests and must leave `pytest` fully green before the
  next phase begins.
- Integration tests continue to run through `PipelineRunner.run_from_config()` per
  the CLAUDE.md testing rules (no bypassing the pipeline).
- New coverage: arbitrary pairs (model↔model, obs↔obs), precedence-based direction
  selection, explicit `reference:` override, same-geometry default-with-warning,
  `migrate-config` round-trip (legacy config → unified config → equivalent results).

## Out of scope (YAGNI)

- N-way (>2 source) comparisons in a single pair.
- Reorganizing `models/`/`observations/` into a `sources/` package (Approach C).
- New pairing strategies beyond seeding the existing geometry combinations (new
  like-geometry combos added on demand, not speculatively).

## Open questions / follow-ups

- Exact extent of the phase-5 obs-only consolidation — confirm which existing
  obs-only plots/stats need a "single-source" code path versus being expressible as
  trivial pairs.
- Whether `radius_of_influence` and other former model-only knobs move onto
  `SourceConfig` generally or stay reference/comparand-specific.
