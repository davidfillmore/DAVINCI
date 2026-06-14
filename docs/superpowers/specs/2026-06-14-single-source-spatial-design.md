# Design: General Single-Source Spatial Plots (all geometry types)

**Date:** 2026-06-14
**Status:** Approved (autonomous — proceed)
**Scope:** Workstream #2 of 2 (follows the x/y pair rename).

---

## Context

There is no general way to put **one source's** field on a map. The closest, `spatial_distribution`,
requires a 2-series *pair* and uses a `show_var` flag to display one side — it is pair-vocabulary
bound and only handles point/grid. We want a true single-source spatial renderer that takes **one
source**, inspects that source's **shape** (point/track/profile/swath/grid), and renders the right
mark — no pairs, no x/y.

The pipeline already supports single-source plots: `_execute_single_source` (`pipeline/stages/plot.py:103`)
reads a plot spec's `source:`+`variable:`, tags `source_label`, builds a 1-element series, and calls
`plotter.render([series])`, auto-splitting on a `flight` coord and saving PNG+PDF. So a new registered
plotter `type` flows through automatically — no pipeline changes needed.

## Goal

A new `spatial` plotter (single source, shape-aware) registered like the others, that maps a source's
field for **all five geometries**, choosing the mark from the source's declared shape:

| Shape | dims (typical) | Mark |
|---|---|---|
| `point` | (time, site) + 1-D lat/lon on `site` | **scatter** (colored markers per site) |
| `track` | (time,) + 1-D lat/lon/alt coords | **scatter along trajectory** (+ faint connecting line) |
| `profile` | (time, level) + lat/lon | reduce vertical (surface) → then point/track scatter |
| `swath` | (time, scanline, pixel) + 2-D lat/lon | **pcolormesh** (curvilinear) |
| `grid` | (time, lat, lon[, lev]) | **pcolormesh** (surface slice if 3-D) |

## Non-Goals

- Pairwise/comparison spatial plots (already `spatial_bias`/`spatial_overlay`).
- 3-D track maps (already `flight_track`) — this is the 2-D map renderer.
- Vertical/curtain plots (already `vertical_profile`/`curtain`).
- Changing the single-source pipeline dispatch (it already works).

---

## Decision 1 — A new renderer `SpatialPlotter`, registered `spatial`

- New file `davinci_monet/plots/renderers/spatial/field.py`, class `SpatialPlotter(BaseSpatialPlotter)`,
  `@register_plotter("spatial")`.
- `render(series)` accepts **exactly 1 series** (raises `NotImplementedError` otherwise, matching the
  contract style of the other renderers). Extracts `ds = series[0].dataset`, `var = series[0].var_name`.
- Add `"spatial"` to `SPATIAL_PLOTS` in `registry.py` so it gets spatial handling (map sizing/domain).
- Config usage (flows through `_execute_single_source` unchanged):
  ```yaml
  plots:
    cam_o3_map:
      type: spatial
      source: cam
      variable: O3
      # optional: domain_type/domain_name, cmap, vmin/vmax, time_index, level_index
  ```

## Decision 2 — Shape dispatch (authoritative attr, geometric fallback)

Resolve the shape in this order:
1. `ds.attrs.get("geometry")` (lowercase `DataGeometry` name, set by readers via `set_geometry_attr`) —
   the authoritative source shape.
2. Fallback: `detect_spatial_geometry(lat_da, lon_da, field)` → `point`/`regular_grid`/`curvilinear_grid`,
   mapped to point/grid/swath.

Then:
- **point / track** → scatter (`draw_spatial_field` scatter path). Track additionally draws a faint
  value-neutral connecting line (`zorder` below markers) so the trajectory reads as a path; points are
  colored by value. Track is NOT time-averaged (time is the trajectory).
- **profile** → reduce the vertical dim to a single level (`surface_level_index`, configurable
  `level_index`), then dispatch as point/track on the remaining horizontal shape.
- **swath** → pcolormesh with 2-D lat/lon (curvilinear).
- **grid** → pcolormesh; if a vertical dim is present, slice the surface via `surface_level_index`
  (configurable `level_index`).

## Decision 3 — Time & vertical handling

- **Vertical:** any `lev`/`level`/`z`/`altitude` field dim is reduced via `surface_level_index` (default
  surface), overridable by `level_index: int`.
- **Time:** for point/grid/swath, **time-average by default** (`time_average: true`), matching
  `spatial_distribution`. For track, time is the path — never averaged. `time_index: int` selects a
  single time instead of averaging (grid/swath).

## Decision 4 — Reuse via an extracted shared helper (DRY)

Extract the scatter/pcolormesh dispatch currently private in
`SpatialDistributionPlotter._plot_data` into a module-level helper in `spatial/base.py`:

```python
def draw_spatial_field(ax, values, lats, lons, *, geometry, cmap, vmin, vmax,
                       marker_size, alpha, transform) -> mappable: ...
```

- `SpatialPlotter` uses it for point/track/swath/grid.
- Refactor `SpatialDistributionPlotter._plot_data` to delegate to it (behavior-preserving) so there is
  one field-drawing implementation. (Covered by the existing distribution tests.)
- Reuse `BaseSpatialPlotter.create_map_figure`/`add_map_features`/`add_colorbar`, `MapConfig`,
  `detect_spatial_geometry`, `surface_level_index`, `get_domain_extent` (domain_type/domain_name →
  extent), and the 0..360 → −180..180 longitude shift already in distribution/bias.

## Decision 5 — Labels / colorbar / title

- Colorbar label: `format_label_with_units(get_variable_label(ds, var, include_prefix=False), units)`.
- Title: config title, else `"{var_label} ({SOURCE})"` using `series[0].source_label`.
- Colormap: sequential NCAR cmap (`get_sequential_cmap()`) by default; `cmap:` overrides.

---

## Files

- **Create** `davinci_monet/plots/renderers/spatial/field.py` — `SpatialPlotter` (`spatial`).
- **Modify** `davinci_monet/plots/renderers/spatial/base.py` — add `draw_spatial_field()` helper.
- **Modify** `davinci_monet/plots/renderers/spatial/distribution.py` — delegate `_plot_data` to the helper.
- **Modify** `davinci_monet/plots/renderers/spatial/__init__.py` — export `SpatialPlotter`.
- **Modify** `davinci_monet/plots/registry.py` — add `"spatial"` to `SPATIAL_PLOTS`.
- **Modify** `davinci_monet/plots/renderers/__init__.py` (if it imports renderers for registration) — ensure `field` is imported so the decorator runs.
- **Create** `davinci_monet/tests/test_spatial_single_source.py` — unit + integration tests.
- **Create** an example config block (e.g. extend a tracked example) showing `type: spatial`.
- **Modify** `CLAUDE.md` — document the `spatial` single-source plot.

## Acceptance criteria

1. A `SpatialPlotter().render(build_series(ds, var))` for each of point/track/profile/swath/grid produces
   a figure with the **correct mark, verified programmatically** (per the geometry-aware-rendering
   memory): grid/swath → a `QuadMesh` artist (pcolormesh) on the GeoAxes; point/track/profile → a
   `PathCollection` (scatter); each has a colorbar. NO "scatter for everything" — assert artist types.
2. Grid with a vertical dim slices the **surface** (not TOA) by default (CESM convention via
   `surface_level_index`).
3. An **integration test** runs a synthetic single-source analysis through
   `PipelineRunner.run_from_config()` with `type: spatial` and asserts a PNG is written.
4. `draw_spatial_field` refactor keeps existing `spatial_distribution` tests green (behavior-preserving).
5. Full suite green, mypy clean, black/isort clean (davinci conda env).
6. A real run renders a `spatial` map for at least a grid and a point source; render mode confirmed
   programmatically (artist type), not by eye.

## Risks

- **Scatter-for-everything regression** (the recurring bug): mitigated by asserting artist types per shape.
- **Swath time/overlap**: multiple granules overlapping — default time-average/first-time; documented.
- **Profile on a map** is unusual (often a single sounding location); handled by vertical reduction →
  point/track, producing a small number of markers — acceptable.
