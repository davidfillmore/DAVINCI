# Renderer Audit — 2026-06-14

Audit of the 19 registered plot renderers (`davinci_monet/plots/renderers/**`) for tech
debt, loose ends, inconsistencies, and dead/stale plot types. Synthesized from a 7-agent
fan-out (5 family groups + contract/conventions + coverage/dead-code), with judgment
applied to separate real debt from intentional design.

> Note on staleness: every renderer file shows a 2026-06-14 git date because the recent
> repo-wide x/y rename touched them all. **Usage** (tracked configs / tests / examples),
> not git-date, is the staleness signal used below.

---

## Executive summary — the real themes

1. **Dead code:** `per_site_timeseries.py` (598 lines) is registered but referenced in **0 configs, 0 examples** (only its own tests). Flagged independently by 3 agents.
2. **Contract robustness gaps (real):** `flight_track.render()` and `lma_density.render()` **silently ignore `series[1:]`** (no `len(series)!=1` guard); `flight_track` is typed `list[Any]`. Other single-source renderers (histogram, vertical_profile, the new `spatial`) DO guard. This is a correctness gap, not cosmetic.
3. **Loose ends that drop real behavior:** `spatial_overlay` accepts a `map_config` but never calls `add_map_features` (map styling silently ignored); `flight_track.plot()` doesn't pass through `city_labels` (parity loss vs `track_map_3d`); `diurnal` accepts an unused `aggregate_dim`.
4. **Duplication / consolidation (the big structural debt):** four clusters repeat near-identical logic — timeseries family (~3000 L), spatial family (coord-resolution + time-average), statistical (x/y extraction + NaN-clean + label resolution), track (coord extraction).
5. **Overlap from recent work:** `spatial_distribution` (2-series + `show_var`) now overlaps the new single-source `spatial` (`field.py`); `distribution._plot_data` is a thin wrapper over the shared `draw_spatial_field`.
6. **Inconsistent shared-helper/style usage:** `scatter` hand-rolls colors while `taylor`/`boxplot` use `get_axis_color`; `bias` inlines meshgrid instead of `draw_spatial_field`; ad-hoc title/label logic; mixed `lat`/`latitude` conventions.
7. **Coverage gaps:** 8 renderers have **no tracked-config use** (boxplot, scorecard, site_timeseries, spatial_overlay, curtain, lma_density, vertical_profile, per_site_timeseries); `spatial_overlay` has **0 tests**; `flight_track` has minimal tests.
8. **Registry categories stale:** the new `spatial` type and the single-source plots aren't clearly categorized; `SPATIAL_PLOTS`/`SPECIALIZED_PLOTS` semantics undocumented.

**Intentional / not worth chasing** (agents over-flagged these): the `.plot()`/`.render()` split is deliberate backward-compat; `histogram.plot()`'s single-source signature + `# type: ignore[override]` is by design; the ~74 cartopy/mpl `# type: ignore` are mostly unavoidable upstream-stub gaps.

---

## Prioritized remediation backlog

### P1 — correctness & cheap, high-value fixes (small, safe)
- **`per_site_timeseries`:** remove it, OR fold into `site_timeseries` (they're both per-site panel timeseries; only the iteration differs). Decision needed (see Disposition).
- **Contract guards:** add `if len(series) != 1: raise NotImplementedError(...)` to `flight_track.render` (`flight_track.py:54`) and `lma_density.render` (`lma_density.py:30`); type `flight_track.render(series: list[PlotSeries])`. Add a regression test for each.
- **`spatial_overlay` map_config:** either call `self.add_map_features(ax)` in `overlay.plot()` (so `map_config` works) or drop the unused `map_config` param. (`overlay.py:128,364-400`)
- **`flight_track` city_labels:** forward `city_labels` (+ the city marker params) to `draw_track_3d`, restoring parity with `track_map_3d`. (`flight_track.py:64-249`)
- **`diurnal` dead param:** remove the unused `aggregate_dim` (`diurnal.py:94`).
- **`flight_track` docstring:** drop the stale "dataset-only" wording (`flight_track.py:1`).

### P2 — consolidation / DRY (bigger refactors, behavior-preserving)
- **Spatial helpers:** extract `_resolve_spatial_coords()` + `_maybe_time_average()` into `spatial/base.py`; have `bias`/`distribution`/`overlay` call them; make `bias` use `draw_spatial_field` like `distribution`/`field`. Inline/remove `distribution._plot_data` (thin wrapper). Est. bias 496→~350L, distribution 451→~300L.
- **`spatial_distribution` vs `spatial`:** decide — fold `distribution`'s single-side display into the new `spatial` (field) renderer (with a `show=both` side-by-side option) and deprecate `spatial_distribution`, OR keep both with a documented distinction. (It now has 1 config use; `spatial` is the cleaner single-source base.)
- **Statistical base helpers:** add `_extract_paired_series(series) -> (paired, x_var, y_var)`, `_clean_paired_values(x, y)`, and `_resolve_axis_label(...)` to the base; use across `scatter`/`taylor`/`boxplot`/`scorecard` (removes 3× duplicated x/y-extraction + NaN-filter + label boilerplate). Standardize `scatter` onto `get_axis_color`.
- **Timeseries family:** consolidate `site_timeseries` + `flight_timeseries` into a parameterized `MultiPanelTimeSeriesPlotter` (panel-grid layout is ~80% identical; altitude overlay optional). Extract the duplicated `_superscript()` helper to a shared util. Est. ~900 L reduction.
- **Track family:** extract `_resolve_track_coordinates()` shared by `flight_track`/`track_map_3d`; standardize param names on `_var` (not `_coord`); consider whether the two should merge (flight_track = 1-series, track_map_3d = 2-series of the same 3-D map).

### P3 — coverage, docs, policy
- **Disposition of zero-config renderers** (boxplot, scorecard, curtain, lma_density, spatial_overlay, site_timeseries, vertical_profile): per renderer — add a real tracked config/example, mark NICHE in docstring, or DEPRECATE. (See Disposition table.)
- **Tests:** add `spatial_overlay` tests (currently 0); expand `flight_track` (currently registration-only) and curvilinear/empty-grid cases for spatial.
- **Document the `render(series)` contract** (1→single, 2→x/y paired, N→overlay; which raise) in `plots/base.py` + a validation matrix; clarify registry category semantics and categorize the new `spatial` type.
- **Coordinate-naming convention** (`latitude`/`lat` fallback order) in CLAUDE.md; `vertical_profile` alt-coord fallback chain.
- **`type: ignore` policy:** consolidate cartopy/mpl suppressions behind small typed wrappers in the base; document the unavoidable ones.

---

## Per-renderer disposition

| Renderer | Lines | cfg/tst/ex | Disposition |
|---|---|---|---|
| scatter | 551 | 10/13/4 | KEEP (core) — standardize colors onto `get_axis_color` |
| timeseries | 835 | 8/9/5 | KEEP (core) — reference for 1/2/N; large but distinct |
| spatial_bias | 496 | 10/7/5 | KEEP (core) — use shared coord/time helpers + `draw_spatial_field` |
| spatial (field) | 292 | 1/4/7 | KEEP (new) — make it the single-source map base |
| diurnal | 389 | 2/2/3 | KEEP — drop unused `aggregate_dim` |
| histogram | 122 | 2/4/0 | KEEP — reference single-source; add an example |
| flight_track | 257 | 2/3/0 | KEEP — add series guard + `city_labels` + tests |
| vertical_profile | 122 | 2/1/0 | KEEP — reference single+N; add N-series test + alt fallback |
| taylor | 432 | 1/3/3 | KEEP — use the shared paired-series helper |
| flight_timeseries | 780 | 1/2/2 | REFACTOR → consolidate w/ site_timeseries |
| track_map_3d | 576 | 1/2/2 | REFACTOR → share track-coord helper w/ flight_track |
| spatial_distribution | 451 | 1/2/3 | DECIDE → fold into `spatial` or document vs it |
| site_timeseries | 446 | 0/1/2 | CONSOLIDATE w/ flight_timeseries; needs a config |
| boxplot | 453 | 0/3/2 | DISPOSITION — add config or mark NICHE |
| scorecard | 449 | 0/2/2 | DISPOSITION — clarify `render` vs `plot_from_dataframe` API; add config |
| spatial_overlay | 400 | 0/2/2 | FIX map_config + ADD tests; or mark NICHE |
| curtain | 476 | 0/2/2 | DISPOSITION — niche; clean contract or deprecate |
| lma_density | 317 | 0/1/0 | DISPOSITION — niche (DC3); add series guard or wrap |
| per_site_timeseries | 598 | 0/0/0 | **REMOVE or CONSOLIDATE** (orphan) |

---

## Notes
- Full per-finding detail (≈85 findings across 7 areas, with file:line) is in the workflow
  output; this doc is the prioritized synthesis.
- Each P1/P2 item is independently shippable behind the full test suite + mypy/black/isort.
- Not committed — this is a working audit artifact for triage.
