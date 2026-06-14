# Design: General Intermediate-Gridding Pairing Strategy — Phase 1 (2-D)

**Date:** 2026-06-14
**Status:** Approved (pending spec review)
**Scope:** Phase 1 of 2. Phase 2 (3-D altitude grid + per-source vertical conversion) is **out of scope** here and gets its own spec.

---

## Context

"Intermediate gridding" — binning sources onto a common uniform grid and pairing
cell-to-cell instead of point-to-point — exists today only for satellite swaths
(`pairing/strategies/swath_grid.py`, `pairing/grid_binning.py`). It is:

- **swath-only** — the engine auto-selects it solely for the SWATH geometry
  (`pairing/engine.py` strategy registry keyed by `DataGeometry`);
- **2-D / surface-only** — bins to a `(time, lon, lat)` grid; a source with a
  vertical dim is reduced to the surface;
- **asymmetric** — it bins one source (the swath) and *aligns* the other (select/
  interp), rather than treating both the same;
- **not opt-in** — there is no config way to request it for other geometries
  (no per-pair strategy override exists — `PairingConfig`/`pair_sources` have none).

We want intermediate gridding **available for all geometries**, treating both
sources symmetrically. Phase 1 delivers the general **2-D** strategy + the config
opt-in + engine routing. Phase 2 adds the **3-D altitude** grid.

This is post-x/y-rename: a pair is an ordered `(x, y)`; there is no model/obs
role. The strategy is symmetric — `x`/`y` only label the two outputs.

## Goal

A single geometry-agnostic `IntermediateGridStrategy` that, when a pair opts in
with `method: grid`, bins **both** sources onto a common uniform `(time, lon, lat)`
grid and pairs the cells. It replaces `SwathGridStrategy` (swath keeps defaulting
to it). Reuses/extends the numba binning core (`grid_binning.py`).

## Non-Goals (Phase 2)

- The `vertical:` config block, the 3-D `(time, lon, lat, alt)` grid, and the
  per-source vertical mapping (native `z` → geopotential→z → pressure→z → error).
- Any change to the direct point/track/profile/grid strategies' default behavior
  (they remain the `method: auto` default for their geometries).

---

## Decision 1 — One geometry-agnostic strategy (generalize, don't fork)

- New `pairing/strategies/intermediate_grid.py`, class `IntermediateGridStrategy`.
- It **replaces** `SwathGridStrategy`: the engine registers it for the SWATH
  geometry (preserving today's swath default — `swath.py` per-pixel stays as the
  deprecated fallback), AND it is the strategy used for any pair with `method: grid`.
- Reuses the numba core in `grid_binning.py` (`bin_swath_to_grid`, `normalize_grid`,
  `edges_from_centers`) — generalized in name/signature to bin a flattened point
  cloud from *either* source (rename to a source-agnostic `bin_points_to_grid` if
  clearer; behavior unchanged).

## Decision 2 — Symmetric binning of both sources

- Each source is flattened to `(time, lon, lat, value)` points:
  - irregular geometries (point/track/profile/swath) → their data points directly;
  - already-gridded sources → their cell centers as points.
- Both point clouds are scatter-binned onto the **same** uniform grid: per cell,
  accumulate sum + count, then mean. (Reuses `bin_*`/`normalize_grid`.)
- Cells are paired where **both** `x` and `y` are finite; cells below
  `min_sample_count` (per source) are masked to NaN.

## Decision 3 — Grid definition

A uniform grid from:
- `horizontal_res` (degrees) — required when `method: grid`.
- `extent` `[lon0, lon1, lat0, lat1]` — optional; default = bounding box covering
  both sources' coverage (union), computed from the data.
- `time_resolution` (pandas freq, default `"1D"`) — temporal bin width.
- `min_sample_count` (default `1`) — per-source minimum samples per cell.

Longitude is normalized to the cartopy-friendly convention consistent with the
existing binning (the current swath_grid builds 0–360 / edges_from_centers; keep
its convention, documented in the strategy).

## Decision 4 — Config schema (opt-in)

In `config/schema.py`, extend `SourcePairConfig`:

```python
class GridConfig(FlexibleSchema):
    """Intermediate-grid settings for a pair using ``method: grid`` (2-D, Phase 1)."""
    horizontal_res: float                     # degrees; required for method: grid
    extent: tuple[float, float, float, float] | None = None  # lon0,lon1,lat0,lat1
    time_resolution: str = "1D"
    min_sample_count: int = 1

class SourcePairConfig(FlexibleSchema):
    x: AxisRef
    y: AxisRef
    method: Literal["auto", "grid"] = "auto"
    grid: GridConfig | None = None
    # ... existing validators ...
```

Validation: `method == "grid"` requires a `grid:` block with `horizontal_res`;
`method == "auto"` with a `grid:` block is **rejected** (a config error — avoids a
silent no-op where grid settings are ignored). Example:

```yaml
pairs:
  aeronet_vs_cam_aod:
    x: {source: aeronet, variable: aod_500nm}
    y: {source: cam,     variable: AODVISdn}
    method: grid
    grid:
      horizontal_res: 0.5
      time_resolution: 1D
      min_sample_count: 1
```

## Decision 5 — Engine routing

- `pairing/engine.py`: when a pair specifies `method: grid`, use
  `IntermediateGridStrategy` directly, **bypassing** geometry-precedence
  auto-selection (`resolve_pair_direction` / `get_strategy_for`). Binning is
  symmetric, so no source is "sampled onto" the other; `x`/`y` from the pair
  config label the two binned outputs.
- The grid spec (`horizontal_res`/`extent`/`time_resolution`/`min_sample_count`)
  is threaded from the pair config through `pipeline/stages/pair.py` into the
  strategy call (the strategy already accepts `**kwargs`; pass an explicit grid
  options dict).
- `method: auto` (default) preserves all current behavior exactly.

## Decision 6 — Output contract

A paired `xr.Dataset` with dims `(time, lon, lat)` and:
- `x_<x_var>`, `y_<y_var>` — per-cell means, tagged `axis: "x"|"y"` + `source_label`
  (matching the standard paired-variable convention so downstream stats/plots work);
- `x_sample_count`, `y_sample_count` — int per-cell counts.

---

## Files

- **Create** `davinci_monet/pairing/strategies/intermediate_grid.py` — `IntermediateGridStrategy`.
- **Modify** `davinci_monet/pairing/grid_binning.py` — generalize the binning core to a source-agnostic point→grid binner (behavior-preserving; reused for both sources).
- **Remove/replace** `davinci_monet/pairing/strategies/swath_grid.py` — its logic folds into the general strategy; swath geometry registers the general strategy. (`swath.py` unchanged.)
- **Modify** `davinci_monet/config/schema.py` — `GridConfig` + `method`/`grid` on `SourcePairConfig` + validator.
- **Modify** `davinci_monet/pairing/engine.py` — route `method: grid`; register `IntermediateGridStrategy` for SWATH; accept the grid options.
- **Modify** `davinci_monet/pipeline/stages/pair.py` — thread `method`/`grid` from pair config to the engine.
- **Create** `davinci_monet/tests/test_intermediate_grid.py`.
- **Modify** existing swath_grid tests → point at the generalized strategy.

## Error handling

- Source missing resolvable lat/lon → clear `ValueError`/`PairingError`.
- `method: grid` without a `grid:` block (or `horizontal_res`) → config validation error at load.
- No spatial/temporal overlap on the grid → empty paired dataset + a logged warning (no crash).
- Cells `< min_sample_count` (per source) → NaN; pairing keeps only cells where both finite.

## Acceptance criteria

1. `IntermediateGridStrategy.pair_sources(x_data, y_data, grid_opts)` bins **both**
   sources onto a `(time, lon, lat)` grid; verified on synthetic cases:
   - **point + point**, **point + grid**, **track + grid** — output dims `(time, lon, lat)`;
   - per-cell `x_<var>`/`y_<var>` equal the hand-computed mean of the points binned
     into that cell; `x_sample_count`/`y_sample_count` equal the hand-computed counts;
   - cells with no samples → NaN; cells `< min_sample_count` → NaN; a cell where only
     one source has data is unpaired (the other is NaN).
2. Config: `method: grid` + `grid:` validates; `method: grid` without `grid:` is
   rejected; `method: auto` is unchanged behavior.
3. **Integration** through `PipelineRunner.run_from_config()` with two file-backed
   synthetic sources and a `method: grid` pair → a paired result, stats, and a plot
   are produced (per the repo's "integration goes through the pipeline" rule).
4. **Regression:** existing swath pairing still works through the generalized
   strategy (adapted swath_grid tests pass).
5. Full suite green, mypy clean, black/isort clean (davinci conda env).

## Risks

- **Behavior drift for swath** when folding `swath_grid` into the general strategy —
  mitigated by keeping/adapting the swath_grid tests as a regression gate.
- **Gridded-source-as-points** binning (flatten cell centers) can alias if the
  target `horizontal_res` is finer than the source grid — acceptable (user controls
  resolution); documented. A cell with no source points → NaN (correct).
- **Longitude convention** (0–360 vs −180–180) — keep the existing binning
  convention and normalize inputs consistently to avoid split cells at the dateline.
