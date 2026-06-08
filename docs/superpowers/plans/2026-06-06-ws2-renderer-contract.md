# WS2 — Single `render(series)` Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps tracked with `- [ ]`. Each increment must leave `pytest`+`mypy`+`black`+`isort` green in the `davinci` env (`HDF5_USE_FILE_LOCKING=FALSE`). Do NOT commit unless the orchestrator/user approves (the orchestrator commits per verified increment).

**Goal:** Make `BasePlotter.render(series: list[PlotSeries])` the one rendering contract for ALL renderers, route the pipeline's paired path through it, add a central geometry guard, and dedupe inline stats — then the `plot()`-vs-`render()` fork can be deleted (the actual deletion of `plot()` bodies lands here; legacy `plot()` thin-wrappers are removed once nothing calls them).

**Program context:** WS2 of `docs/superpowers/specs/2026-06-06-davinci-remediation-program-design.md` (after WS1). Current state mapped 2026-06-06.

**Migration pattern (per renderer):** move the figure-building logic into `render(self, series, ax=None, **kwargs)` deriving arrays from `series` (1→single, 2→reference/comparand, N→overlay where meaningful); make the legacy `plot(paired_data, obs_var, model_var, **kwargs)` a thin wrapper `return self.render(build_series(paired_data, obs_var, model_var), ax=ax, **kwargs)`. Behavior-preserving: existing `.plot(...)` tests must stay green. Replace inline metric math with `davinci_monet.stats` helpers.

---

## Current state (2026-06-06 map)

- Override `render()` already: `timeseries`, `histogram`, `vertical_profile`, `flight_track`, `lma_density` (5/18).
- Default `BasePlotter.render()` (`plots/base.py:325`): 2→`self.plot(series[0].dataset, ref.var_name, comp.var_name)` (BUG: should use `ref.dataset` for the obs side), 1/N→`NotImplementedError`.
- Pipeline: single-source path already gates on `type(plotter).render is not BasePlotter.render` (`stages.py:2490`); **paired path always uses `plot()`** (`stages.py:2937`) + `plot_per_flight`/`plot_per_site` generators (`stages.py:2873/2908`).
- `scatter.py`: no geometry guard. `spatial/bias.py:185-272` and `spatial/distribution.py:184-199` duplicate the point/regular-grid/curvilinear classifier. `spatial/overlay.py` has a simpler ndim check.
- Inline stats: `scatter.py:407`, `taylor.py:105`, `scorecard.py:99`, `site_timeseries.py:227`, `flight_timeseries.py:340/593`, `per_site_timeseries.py:399`.
- Stale `registry.py:174-179` frozensets omit: `flight_timeseries, site_timeseries, flight_track, histogram, lma_density, track_map_3d, vertical_profile`.
- Tests: `tests/test_plots.py` (direct `.plot()`), `tests/unit/plots/test_unification_p*` (direct `.render()`), `tests/test_integration.py` + `tests/integration/` (pipeline). Smoke-only (no pixel baselines).

---

## Increments (each = one verified, committed checkpoint)

**Increment 1 — Central spatial geometry classifier (safe, high-value first).**
- [ ] Add `detect_spatial_geometry(lat_da, lon_da, field_da) -> Literal["point","regular_grid","curvilinear_grid"]` to `plots/renderers/spatial/base.py`; refactor `bias.py`/`distribution.py` to use it (dedupe; behavior byte-identical). Have `overlay.py` use it where applicable.
- [ ] Tests: classifier returns correct labels for point / regular-grid / curvilinear; existing spatial tests (`test_plots.py::TestSpatialPlotters`) stay green.
- NOTE (correction to the review): `ScatterPlotter` is a model-vs-obs VALUE scatter, not a map. Gridded value-scatter is valid and is exercised by `tests/integration/test_merra2_modis_aod_pipeline.py` (grid×grid). Do NOT add a "reject gridded" guard to ScatterPlotter — geometry-awareness belongs to the SPATIAL renderers. The memory `plot-geometry-aware-rendering` is about spatial/map rendering specifically.

**Increment 2 — Fix default `render()` + EASY migrations.**
- [ ] Fix `BasePlotter.render()` 2-series to use `ref.dataset`/`comp.dataset` (not `series[0].dataset`).
- [ ] Migrate EASY renderers to the contract (logic→`render`, `plot`→thin wrapper): `scatter`, `boxplot`, `diurnal`, `taylor`. Replace their inline stats with `davinci_monet.stats` helpers.
- [ ] Tests: direct `.plot()` tests green; add `.render([ref,comp])` parity tests.

**Increment 3 — MEDIUM migrations.**
- [ ] `scorecard` (preserve `plot_from_dataframe`/`plot_multi_metric` side-entries), `curtain` (carry `alt_var`), `spatial_bias`, `spatial_distribution` (keep geometry branch via the classifier).
- [ ] `site_timeseries`, `per_site_timeseries`, `flight_timeseries`: move single-figure logic into `render`; keep `plot_per_site`/`plot_per_flight` generators as-is for now (the split-by-entity path stays a generator; see Increment 5). Dedupe their inline stats.

**Increment 4 — Route the paired path through `render()`.**
- [ ] In `PlottingStage.execute` (paired path, `stages.py:2937`), add the same gate as the single-source path: if `type(plotter).render is not BasePlotter.render`, call `plotter.render(build_series(paired_data, obs_var, model_var), **opts)`, else `plot(...)`. Keep `plot_per_flight`/`plot_per_site` generator branches unchanged.
- [ ] Verify all pipeline integration tests (`tests/test_integration.py`, `tests/integration/`) stay green.

**Increment 5 — HARD migrations + fork removal.**
- [ ] `spatial_overlay`: define how the external `model_field` is carried (a kwarg on `render`, or a documented series convention); migrate.
- [ ] `track_map_3d`: migrate the single-figure path to `render`; keep `plot_per_flight` generator.
- [ ] Decide the per-flight/per-site generator contract (e.g. `render_split(series, by=...)` or keep generators as an explicit secondary API). Document the decision.
- [ ] Once every registered renderer overrides `render()` and the paired path routes through it, remove the now-dead `plot()`-vs-`render()` introspection branches and convert remaining `plot()` methods to thin wrappers (full `plot()` deletion is WS3).

**Increment 6 — Metadata + cleanup.**
- [ ] Refresh `registry.py` category frozensets to include all registered types.
- [ ] Confirm renderer-reported stats equal `stats/` values (no display/CSV divergence).

---

## Acceptance (WS2 Definition of Done)
- Every registered renderer overrides `render(series)`; `get_plotter(name).render(series)` works for valid series counts across the registry (no blanket `NotImplementedError`).
- Pipeline paired + single-source plotting both flow through `render()` (introspection fork removed or reduced to the generator secondary-API).
- One shared spatial geometry classifier; spatial renderers (bias/distribution/overlay) select pcolormesh vs scatter correctly for point/regular/curvilinear inputs.
- No renderer recomputes metrics inline; all use `davinci_monet.stats`.
- `registry.py` categories complete.
- Full suite + mypy + black/isort green; pipeline integration tests green; example runs visually unchanged for valid configs.
