# Design: Intermediate-Gridding Pairing — Phase 2 (3-D altitude grid)

**Date:** 2026-06-14
**Status:** Approved (pending spec review)
**Scope:** Phase 2 of 2. Builds on Phase 1 (`docs/superpowers/specs/2026-06-14-intermediate-grid-pairing-design.md`, shipped on `develop`).

---

## Context

Phase 1 delivered the general 2-D intermediate-gridding strategy
(`IntermediateGridStrategy`, `method: grid`): both sources are flattened to
`(time, lon, lat, value)` points and scatter-binned onto a common uniform
`(time, lon, lat)` grid, paired cell-to-cell. Phase 2 adds a **vertical axis**:
when a pair's `grid:` block carries a `vertical:` sub-block, the common grid
becomes `(time, lon, lat, alt)` and both sources are binned in 3-D using
**altitude** (metres) as the vertical coordinate.

This serves comparisons that are intrinsically 3-D — aircraft tracks (which carry
geometric altitude) and vertical profiles vs. a gridded field — by aggregating
both onto common altitude bins rather than point-to-column interpolation.

Symmetric and role-free (post-x/y rename): both sources are binned the same way;
`x`/`y` only label the two outputs.

## Goal

Extend `IntermediateGridStrategy`'s symmetric path so a `grid.vertical:` block
produces a `(time, lon, lat, alt)` paired dataset. Each source contributes a
**per-datum altitude** derived from what the dataset supplies (native altitude,
geopotential height, or pressure); a source that supplies none errors.

## Non-Goals

- Visualizing the 4-D paired output (curtain / 3-D plots already exist and are a
  separate concern; Phase 2 produces the paired data).
- The geopotential-vs-geometric-altitude correction — geopotential height is
  treated as altitude (the difference is <0.3% in the troposphere).
- Decoding hybrid sigma-pressure coefficients in the strategy — the **dataset
  must supply** a usable vertical coordinate; the strategy never guesses.
- Any change to the 2-D path or to `method: auto`.

---

## Decision 1 — `vertical:` block triggers 3-D

- Config: `grid.vertical` is an optional sub-block. Its **presence** makes the
  grid 3-D; its absence keeps the Phase 1 2-D (surface-reduced) path.
- `IntermediateGridStrategy._pair_symmetric` branches on whether a vertical grid
  spec was passed. In 3-D mode it does NOT reduce sources to the surface.

```yaml
pairs:
  dc8_vs_cam_o3:
    x: {source: dc8, variable: O3}
    y: {source: cam, variable: O3}
    method: grid
    grid:
      horizontal_res: 0.5
      vertical: { res: 500, units: m, extent: [0, 12000] }   # presence ⇒ 3-D
      time_resolution: 1D
      min_sample_count: 1
```

## Decision 2 — per-source altitude (`_source_altitude(ds, var, units)`)

Returns a per-datum altitude `DataArray` broadcast to `ds[var]`'s dims, in the
configured units. Resolution order (the dataset supplies; the strategy converts):

1. **Native altitude** — a coordinate or variable with **length units** (`m`,
   `km`, `ft`, …) whose name is in `{altitude, alt, height, geometric_height}`.
   Used directly (unit-converted). *Length units are required* so CESM's `z`
   (a hybrid model level, not length) is correctly NOT treated as altitude.
2. **Geopotential height** — a data variable named in
   `{Z3, zg, geopotential_height, geopotential_height_msl}` with length units →
   used as altitude (broadcast to the field's dims).
3. **Pressure** — a vertical coordinate with pressure units (`hPa`/`Pa`/`mb`)
   named in `{lev, level, plev, pressure, p}` → converted via
   `pressure_to_altitude` (US Standard Atmosphere; the inverse of the existing
   `altitude_to_pressure` in `track.py`).
4. **None** → `PairingError` naming the source and listing its vertical dims, e.g.
   *"Source 'cam' has no usable vertical coordinate for a 3-D altitude grid;
   supply geometric altitude (m), geopotential height (m), or pressure (hPa).
   Found dims: [...]"*.

Units are normalized through a `{m: 1.0, km: 1000.0, ft: 0.3048}` map (both the
source altitude and the config `res`/`extent`); internal binning is in metres,
and the output `alt` axis is expressed in the configured `units`.

## Decision 3 — 4-D binning

- `_flatten_to_points` (3-D mode) additionally produces `alt_flat` (per-datum
  altitude from `_source_altitude`), broadcast in the same dim order as the
  value/lat/lon/time arrays (consistent C-order — the Phase 1 `broadcast_like`
  pattern extended to the altitude array).
- A 4-D numba binner in `grid_binning.py` (extends `bin_swath_to_grid` to a
  `(time, lon, lat, alt)` accumulation; `normalize_grid` already works on any
  shape via its `count > 0` mask). Per cell: sum→mean + count.
- Vertical edges from `vertical.res` + `extent` (default = altitude bounding box
  over both sources) via Phase 1's `_span_edges` (always ≥1 full-width cell
  covering the data).

## Decision 4 — output contract

A paired `xr.Dataset` with dims `(time, lon, lat, alt)` and:
- `x_<x_var>`, `y_<y_var>` — per-cell means, tagged `axis: "x"|"y"` + `source_label`;
- `x_sample_count`, `y_sample_count` — int per-cell counts;
- `alt` coordinate in the configured `units`.

Wrapped in `PairedData.from_sources(..., geometry=DataGeometry.GRID)` exactly as
the Phase 1 grid path (so downstream stages treat it identically).

## Decision 5 — both sources must supply a vertical

When `grid.vertical` is set, **both** sources must yield a usable altitude (via
Decision 2). A surface-only source (e.g. point sites with no vertical) paired on
a vertical grid → `PairingError` (clear message). The 2-D path is unaffected.

---

## Files

- **Modify** `davinci_monet/pairing/strategies/intermediate_grid.py` — 3-D branch
  in `_pair_symmetric`; `_source_altitude`; extend `_flatten_to_points`,
  `_bin_one_source`, and grid construction to 4-D; vertical-edge helper (reuse `_span_edges`).
- **Modify** `davinci_monet/pairing/grid_binning.py` — add the 4-D
  `(time, lon, lat, alt)` binner.
- **Modify** `davinci_monet/pairing/strategies/track.py` (or a small shared util) —
  add `pressure_to_altitude` (inverse of `altitude_to_pressure`).
- **Modify** `davinci_monet/config/schema.py` — `VerticalGridConfig` + `GridConfig.vertical`.
- **Modify** `davinci_monet/tests/test_intermediate_grid.py` — 3-D unit + integration tests.

## Error handling

- Source with no usable vertical (native altitude / geopotential / pressure) →
  `PairingError` naming the source + what to supply + its vertical dims.
- Unknown `vertical.units` → config/strategy error listing supported units.
- `vertical:` present but a source is inherently surface-only → the above
  no-usable-vertical error.

## Acceptance criteria

1. `_source_altitude` returns the right altitude for: a native-`altitude` (m)
   source; a source with a `Z3` geopotential field; a pressure-level source
   (verify `pressure_to_altitude`: 1013.25 hPa ≈ 0 m, 500 hPa ≈ 5.5 km within a
   tolerance); and raises `PairingError` for a hybrid-`lev`-only source.
2. The symmetric 3-D path bins **both** sources onto `(time, lon, lat, alt)`;
   on a synthetic track (native altitude) + grid-with-`Z3` case, a known cell's
   `x_<var>`/`y_<var>` equal the hand-computed binned mean and the counts match;
   data at a given altitude lands in the intended `alt` bin.
3. `vertical.units: km` produces an `alt` axis in km with correct binning.
4. **Integration** through `PipelineRunner.run_from_config()` with `method: grid`
   + a `vertical:` block and two file-backed 3-D synthetic sources → a
   `(time, lon, lat, alt)` paired result stored in `context.paired`.
5. **Regression:** all Phase 1 tests pass unchanged (no `vertical:` ⇒ the 2-D
   path is byte-for-byte unaffected).
6. Full suite green, mypy clean, black/isort clean (davinci conda env).

## Risks

- **Flatten alignment in 4-D** (the Phase 1 make-or-break, now with altitude):
  `alt_flat` must align element-wise with value/lat/lon/time. Mitigated by the
  same `broadcast_like(da).transpose(*da.dims).flatten()` pattern + a unit test
  that encodes each datum's altitude and checks it lands in its own `alt` bin.
- **Pressure→altitude accuracy** — US Standard Atmosphere is approximate above
  the troposphere; acceptable as the documented generic fallback (datasets that
  need better should supply geopotential height).
- **Geopotential broadcast** — `Z3` is a full 3-D field `(time, lev, lat, lon)`;
  binning the data variable with `Z3` as its altitude requires they share dims
  (they do, for model output). Verified in the track+grid test.
