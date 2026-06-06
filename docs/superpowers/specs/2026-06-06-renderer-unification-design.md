Confirmed the design's "TRUE duplicate" claim: `obs_flight_track` (flight_track_map.py:24-28) imports `_get_coastline_segments`/`_get_border_segments`/`_render_surface_map` from `track_map_3d` then reimplements the body. I now have full evidence. Writing the final design document.

# Renderer Unification: Design & Migration Plan

**Status:** Proposed
**Date:** 2026-06-06
**Scope:** Complete the model/obs unification at the renderer boundary — collapse the parallel `obs_*` renderer fork and the `obs_plotting`/`obs_statistics` stage fork into the unified, source-agnostic path that already reaches config, pairing, and paired-series labeling (commits R-1..R-5).
**Mode of authoring:** Read-only source audit. Every load-bearing claim below is cited to `file:line` as verified against `develop`.

> **Correction to prior drafts (verified).** Two premises in the earlier design JSON are **factually wrong** and are corrected here before any plan is built on them:
> 1. `results['obs_statistics']` is **not** "consumed by nothing." It is read by the AI summary payload at `davinci_monet/ai/payload.py:34` (`_STATS_STAGES = ("statistics", "obs_statistics")`) and iterated at `payload.py:54-66`. Likewise `obs_plotting` is read at `payload.py:35` (`_PLOT_STAGES`). The "orphan" justification is invalid; the real defect is narrower and is restated correctly in §1.
> 2. The plotter registry has **no alias mechanism today** (`register_plotter` at `plots/registry.py:33-56` and the underlying `Registry.register` at `core/registry.py:122` accept only `name` + `replace`). The backward-compat story depends on an alias layer that must be **built first**, not assumed.
>
> Other corrected figures: the suite is **1324 tests** (verified `pytest --collect-only`), not 961 (CLAUDE.md is stale). `core/base.py` is 796 lines and the read-path helpers live there at 374-448; the styling helpers live in `plots/base.py` at 763-881 (≈40 lines later than the JSON cited, same functions).

---

## 1. Context & Problem

The project unified models and observations into one `sources:` concept: both are data sources distinguished only by geometry, with `role: model|obs` as optional styling metadata. The unification landed in config parsing, the pairing engine, and the paired-series labeling helpers (`paired_variable_role`/`paired_variable_pair_role`/`paired_canonical_name`/`iter_paired_variable_pairs` at `core/base.py:374-448`; the styling family `get_role_color`/`get_series_label`/`resolve_source_variable`/`dataset_source_label` at `plots/base.py:781-881`; `get_color_for_role` at `style.py:294-319`).

It **stopped at the renderer boundary.** The result is a parallel obs-only fork:

- **Two incompatible abstract base classes.** `BasePlotter` (`plots/base.py:258`) declares `plot(self, paired_data, obs_var, model_var, ax=None, **kwargs)` (`base.py:289-318`). `ObsPlotter` (`plots/obs_base.py:40`) is an **independent ABC** (does not inherit `BasePlotter`) declaring `plot(self, obs_data, variable, ax=None, **kwargs)` (`obs_base.py:71-97`). `ObsPlotter` re-implements `create_figure`/`save`/`close` (`obs_base.py:99-172`) — pure duplication.

- **Five `obs_*` renderers** in `plots/renderers/obs/` registered under `obs_`-prefixed type names: `obs_timeseries` (`obs_timeseries.py:25`), `obs_histogram` (`obs_histogram.py:24`), `obs_vertical_profile` (`vertical_profile.py:24`), `obs_flight_track` (`flight_track_map.py:37`), `obs_lma_density` (`obs_lma_density.py:23`). `obs_flight_track` is a near-verbatim copy of the canonical `track_map_3d` — it even **imports** `_get_coastline_segments`/`_get_border_segments`/`_render_surface_map` from it (`flight_track_map.py:24-28`) and reimplements the 3D body.

- **Two stage forks** living simultaneously in the standard pipeline (`create_standard_pipeline` at `stages.py:3017-3034` lists **all four**): `StatisticsStage`/`PlottingStage` (`stages.py:2036`, `2222`) and `ObsStatisticsStage`/`ObsPlottingStage` (`stages.py:2894`, `2753`). They are gated mutually exclusive: paired stages run on `bool(context.paired)` (`stages.py:2044`, `2230`); obs stages run on `bool(context.observations) and not bool(context.paired)` (`stages.py:2762`, `2903`).

- **A vestigial `create_obs_pipeline`** (`stages.py:3037-3052`) that the runner never selects — `PipelineRunner.__init__` always calls `create_standard_pipeline()` (`runner.py:1360`); there is no obs-only swap. It remains exported in the public API (`pipeline/__init__.py:34,54`).

### The spaghetti bug as a symptom
The forks have **inverted defaults** for the same operation. Canonical `timeseries` auto-averages over non-time dims by default (`timeseries.py:140-145`), producing one mean line. The obs fork `obs_timeseries` defaults `aggregate=False` (`obs_timeseries.py:59`) and plots **raw one-line-per-site spaghetti** (`obs_timeseries.py:136-144`). The same dataset gets opposite treatment depending only on which class ran — the user-visible face of an abstraction that never unified.

### The N-source gap
Because the obs stages are gated `not bool(context.paired)` (`stages.py:2762`, `2903`), the moment **any** pair exists, **all** `obs_*` output for unpaired/extra sources is skipped. And on the read side, `iter_paired_variable_pairs` uses `setdefault` (`core/base.py:447`), which silently **drops a third same-canonical source**. A 2-models-vs-1-obs comparison is unreachable through either path.

---

## 2. Current-State Map

### 2.1 Renderer correspondence (canonical ↔ obs_*)

| obs_* renderer (file, base class) | Canonical counterpart | Relationship | Disposition |
|---|---|---|---|
| `obs_timeseries` (`renderers/obs/obs_timeseries.py`, `ObsPlotter`) | `timeseries` (`renderers/timeseries.py`) | Duplicate with **inverted aggregate default** | **MERGE** into `timeseries` |
| `obs_flight_track` (`renderers/obs/flight_track_map.py`, `ObsPlotter`) | `track_map_3d` (`renderers/track_map_3d.py`) | **True duplicate** — imports internals of `track_map_3d` (`flight_track_map.py:24-28`) then reimplements | **MERGE** into `track_map_3d` |
| `obs_histogram` (`renderers/obs/obs_histogram.py`, `ObsPlotter`) | **NONE** (no canonical histogram; only paired `boxplot` covers distributions) | No counterpart | **PROMOTE** to new canonical `histogram` |
| `obs_vertical_profile` (`renderers/obs/vertical_profile.py`, `VerticalProfilePlotter`) | **NONE exact** (`curtain` is 2-D time-altitude, paired-only) | No counterpart | **PROMOTE** to new canonical `vertical_profile` |
| `obs_lma_density` (`renderers/obs/obs_lma_density.py`, `ObsPlotter`) | **Partial** (`spatial_distribution` pcolormesh path) | Overlapping but distinct (hourly split, flight_tracks overlay) | **PROMOTE** to new canonical `lma_density` (rebased on `BaseSpatialPlotter` — see §2.4 caveat) |

**Canonical-only renderers with no obs_* counterpart** (correctly classified, no completeness gap): `scatter`, `taylor`, `boxplot`, `scorecard`, `diurnal`, `curtain`, `spatial_bias`, `spatial_overlay`, `flight_timeseries`, `per_site_timeseries`, `site_timeseries`, `spatial_distribution`.

### 2.2 The stage fork

| Concern | Paired stage | Obs stage |
|---|---|---|
| Statistics class | `StatisticsStage` (`stages.py:2036`), gate `bool(context.paired)` (`2044`) | `ObsStatisticsStage` (`stages.py:2894`), gate `bool(obs) and not bool(paired)` (`2903`) |
| Stats result key / shape | `results['statistics']` = `{pair_key: {var: comparison_metrics}}` (`2077`) | `results['obs_statistics']` = `{obs_label: {var: descriptive_metrics}}` (`2939`) |
| Stats vocab | N/MB/RMSE/R/NMB/NME/IOA/MO/MP via `StatisticsCalculator` (`2094-2218`) | N/mean/median/std/min/max/p10/p25/p75/p90 — inline numpy (`2918-2934`) |
| Plotting class | `PlottingStage` (`stages.py:2222`), gate `bool(paired)` (`2230`) | `ObsPlottingStage` (`stages.py:2753`), gate `bool(obs) and not bool(paired)` (`2762`) |
| Spec dispatch | iterates `plot_spec['pairs']`/`['data']` (`2272`); `continue` if `pair_key not in context.paired` (`2304`); **no single-source code path** | `if not plot_type.startswith("obs_"): continue` (`2807`); reads `plot_spec['obs']` + `['variable']` (`2810-2811`) |
| Output layout | subdir-per-source + PNG **and** PDF (`2525-2526`, `2604-2611`) | flat output_dir + **PNG only** (`2872`) |
| Splitting | **flag-gated**: `split_by_flight`/`split_by_site` (`2529-2531`) → `hasattr(plotter,'plot_per_flight'/'plot_per_site')` dispatch (`2533`, `2566`); filename = `{id}_{file_index}_{name}` **prefix** for slideshow grouping (`2549`); `min_points` passed **into** the generator (`2536`, `2544`) | **flag-less auto-split** on `flight` coord (`2829-2835`); filename = `{name}{suffix}{fig_suffix}` **suffix** (`2864`) |
| Multi-figure return | (not handled) | `list[(fig, suffix)]` consumed at `2862-2868` (today unique to `obs_lma_density`, return type at `obs_lma_density.py:43`) |

### 2.3 Stats-schema consumers (all hardcode the comparison vocabulary)

Three consumers read stats. Two read **only** `results['statistics']` and emit a **fixed comparison schema**; the third reads both keys generically.

1. **`SaveResultsStage` CSV** (`stages.py:2653`) — reads only `results.get("statistics")`; builds fixed columns Variable/N/Mean_Obs/Mean_Model/MB/RMSE/R/IOA/NMB_%/NME_% via `_get_metric(... "MO","MP","MB",...)` (`2669-2705`). Descriptive rows (no MO/MP/MB) would NaN-fill every column and **drop** mean/median/percentiles.
2. **`LogCollector.to_markdown`** (`runner.py:372-411`) — reads only `self.statistics`; hardcoded header `| Variable | N | Mean Obs | Mean Model | MB | RMSE | R |` (`381`); reads MO/MP/MB/RMSE/R (`394-398`). Descriptive rows render as N + all "-".
3. **AI summary payload** (`ai/payload.py:38-66`) — iterates **both** `statistics` and `obs_statistics` (`_STATS_STAGES`, line 34) and copies **all** non-`_` keys generically (`metrics = {k: v for k, v in var_stats.items() ...}`, line 65). This consumer is **schema-agnostic** and already surfaces descriptive stats.

**Net:** obs descriptive stats reach the AI summary but **never** reach the CSV or the run-log markdown. The correct framing is not "fix an orphan" but "make the CSV and markdown writers schema-aware so descriptive rows are not silently dropped when the stages merge."

### 2.4 Base-class divergence & styling/role plumbing status

- `ObsPlotter` is a sibling ABC, duplicating `create_figure`/`save`/`close` (`obs_base.py:99-172`). Unification deletes it and rebases its subclasses on `BasePlotter`, gaining `set_labels`/`add_legend`/`set_limits`/`apply_text_style`.

- **Styling already generalizes to N series — no change needed for N-width:** `get_color_for_role(role, index)` special-cases `obs`→`OBS_COLOR`, `model`→`MODEL_COLOR`, else `NCAR_PALETTE[index % len]` (`style.py:315-319`). `get_role_color` (`base.py:781-816`) reads the `role` attr and falls through to `get_color_for_role`; the `obs_`/`model_` prefix fallback (`base.py:806-811`) fires **only** when no `role` attr exists (harmless for tagged data). `get_series_label` keys off arbitrary `source_label` (`base.py:830-849`). `resolve_source_variable` resolves `<label>_<canonical>` for any label (`base.py:852-881`). **Verified:** `NCAR_PALETTE[0] == NCAR_COLORS["ncar_blue"] == NCAR_PRIMARY` (`style.py:60,72`).

- **What does NOT yet support single-source:** single-source datasets carry identity at the **dataset level** (`dataset_source_label`, `base.py:819-827`) with per-variable `role`/`source_label` **absent**, so obs renderers hardcode `NCAR_PRIMARY` (`obs_timeseries.py:94`). `tag_paired_roles` (`stages.py:330-343`) **only** acts on `model_`/`obs_`-prefixed names and `continue`s past everything else (`343`) — it is a **no-op** on bare single-source vars. A **new** tagging helper is required (§3.2).

- **Does role/source-label plumbing already support N series?** On the **styling/read-helper** side, yes. On the **read-grouping** side, no: `iter_paired_variable_pairs` collapses to one reference + one comparand via `setdefault` (`core/base.py:447`), dropping a third same-canonical source. `pair_role` is strictly `reference|comparand|None` (`paired_variable_pair_role`, `core/base.py:391-407`) — there is **no third pair position**. So N-overlay is reachable only by grouping (not pairing); the pairing engine itself stays binary (see §8 Q4).

---

## 3. Target Architecture

### 3.1 One series-list contract on one base class

Replace the two incompatible abstract `plot()` signatures with a single `render(series, ax, **kwargs)` on `BasePlotter`; `ObsPlotter` is deleted.

```python
@dataclass(frozen=True)
class PlotSeries:
    dataset: xr.Dataset
    var_name: str                  # actual name in dataset: cam_o3 / airnow_o3 / o3
    canonical: str                 # o3
    role: str | None               # free-form: 'obs' / 'model' / None / custom
    pair_role: str | None          # 'reference' | 'comparand' | None
    source_label: str | None       # source key (cam, airnow, dc8, ...)
    index: int                     # order in the plot — consumed by get_color_for_role
    split_dim: str | None = None   # intra-series fan-out coord (flight/site); see §3.3
```

```python
@abstractmethod
def render(
    self,
    series: list[PlotSeries],
    ax: matplotlib.axes.Axes | None = None,
    **kwargs: Any,
) -> Figure | list[tuple[str, Figure]]:
    ...
```

**Series-count semantics:**
- `len(series) == 1` → single line/field. Color resolved by the §3.2 rule (1-series ⇒ `NCAR_PRIMARY` brand blue).
- `len(series) == 2` with `pair_role` set → reference-vs-comparand (obs gray / model blue), preserving today's paired look.
- `len(series) >= 2` without a strict pair → overlay cycling `NCAR_PALETTE` by `index`.

**Universal multi-figure return.** The return type is `Figure | list[(label, Figure)]`, promoting today's `obs_lma_density` multi-figure contract (`obs_lma_density.py:43`) to the interface. This subsumes per-flight/per-site/hourly splitting, retiring the off-interface generator methods (see §3.4 and the per-renderer table). **Note the ordering decision (resolving the review):** the universal element is `(label, fig)` where **`label` is used as a filename PREFIX** — this preserves the paired stage's slideshow grouping (`{flight_id}_{file_index}_{plot_name}`, `stages.py:2549`). The obs LMA case (today a `suffix`) is normalized to a prefix label at the save site. Per-split filtering (`min_points`, `stages.py:2536`) moves **into `render()`** (each split decides whether to emit) so the stage save loop stays uniform.

**Min-series declarations** (post-migration registered names):
- **Descriptive** (`min_series = 1`): `timeseries`, `histogram` (new), `vertical_profile` (new), `track_map_3d`, `lma_density` (new), and the single-field spatial path `spatial_distribution` with `show_var ∈ {obs, model}`.
- **Comparative** (`min_series = 2`, `requires_pair_role = True`): `scatter`, `taylor`, `spatial_bias`, `spatial_overlay`, `scorecard`, `diurnal`, `boxplot`, `curtain`. The stage skips these for a <2-series target (loud warning recorded in `results`, not just a log line — see §8 Q-usability).

> **Correction (resolving review API-soundness #6):** `spatial_distribution` is **not** intrinsically comparative for `show_var ∈ {obs, model}` — it renders a single field (`distribution.py:65,210`). It is comparative only for `show_var="both"` (`distribution.py:203`). It is therefore `min_series=1` and is the **latent single-source spatial seam**: under the series-list contract it naturally becomes the 1/2/N spatial value-map renderer once unlocked from its `(obs_var, model_var)` signature. Remove it from the "intrinsically comparative" list (it was listed there erroneously in the JSON).

> **Correction:** the JSON's descriptive list named a non-existent `distribution` type. There is no registered `distribution`; only `spatial_distribution` exists. `histogram`/`vertical_profile`/`lma_density` are **new** names this plan creates.

### 3.2 Series model — derivation moves OUT of renderers, INTO one stage-level resolver

Add an N-capable sibling to the binary helper in `core/base.py`:

```python
def iter_canonical_variable_series(dataset) -> dict[str, list[PlotSeries]]:
    # group every data_var by canonical (paired_canonical_name, core/base.py:410),
    # reading role/pair_role/source_label; PRESERVE dataset.data_vars insertion order
    # within each canonical group.
```

`iter_paired_variable_pairs` (`core/base.py:429`) is rewritten as a **thin wrapper** taking `group[0]` reference + `group[0]` comparand from this grouping. **Insertion-order guarantee** (resolving review ordering #9): the grouping must preserve `dataset.data_vars` order within each canonical so the wrapper's `[0]/[0]` selection is byte-identical to today's `setdefault`-first (`core/base.py:447`). This is pinned by a regression test on a 3-source dataset (§6 UNIT-1).

**Single-source styling.** Add a **new** helper `tag_source_roles(ds, source_label, role)` invoked at load/select time, setting `role`/`source_label` per data_var **unconditionally** (no prefix gate — unlike `tag_paired_roles`, which is a no-op on bare names). Then the **color rule** (resolving review §8 Q2 in-design): a series gets `NCAR_PRIMARY` when `index == 0 and role in {None, "obs"}`; `role == "model"` single-source ⇒ `MODEL_COLOR` (also blue); `index >= 1` cycles the palette. This is expressed **explicitly by series count/role**, not by relying on the `palette[0] == NCAR_PRIMARY` coincidence (which holds at `style.py:72` but is brittle). Color is chosen by series count/role, never by which class ran — preserving `test_obs_source_labels.py:49` in unified terms.

### 3.3 Intra-series fan-out (resolving review API-soundness #1)

A single source with a `flight`/`site` coord can expand to **N lines from one PlotSeries** — this is a real existing behavior (`obs_timeseries.py:122-135` colors one line per flight). The series model handles this via the **optional `split_dim` field**: the stage detects a coloring coord and either (a) sets `split_dim` on a single PlotSeries (renderer fans it out internally, one source = one legend group), or (b) explodes the source into N PlotSeries by coord slice (each `role=None`, distinct `index`). **Decision:** use `split_dim` (option a) so legend/label semantics stay "one source, multiple flights" rather than "N sources." `build_series` and the `>=2` overlay branch are written to distinguish intra-source fan-out (`split_dim` set, shared source identity) from inter-source overlay (distinct `source_label`s).

### 3.4 Single plotting & statistics stage

Collapse to **one** `PlottingStage` and **one** `StatisticsStage`. Delete `ObsPlottingStage`/`ObsStatisticsStage`/`create_obs_pipeline`.

**Run gate** (replacing the mutually-exclusive predicates): one broad gate `bool(context.paired) or bool(context.observations) or bool(context.models)`. Per-spec resolution decides what to render — not the presence-of-pairs gate. This closes the N-source gap (§1).

**Per-spec dispatch (new core loop).** Each `plot_spec` resolves a TARGET:
- a `pair_key` → 2+ series via `iter_canonical_variable_series` on the paired dataset (caps at 1-or-2 for paired data; see §8 Q4);
- a single source label + `variable` → 1 series from the raw source, tagged via `tag_source_roles`;
- an explicit list of source labels → N series from raw sources.

The renderer is selected by the spec's `type` (`timeseries`/`histogram`/…), **not** by an `obs_` prefix. Eliminate the `plot_type.startswith("obs_")` skip (`stages.py:2807`) and the `plot_spec['variable']` vs `variable.{obs_var,model_var}` schema fork. Build the `PlotSeries` list, then call `plotter.render(series, **opts)`.

**Unified output contract:** subdir-per-source + PNG **and** PDF for all (today's paired behavior, `stages.py:2604-2611`); the `list[(label, fig)]` return is the universal save path, absorbing the per-flight loop (`stages.py:2533-2562`) and the obs auto-split (`stages.py:2835`) into **one** splitting mechanism (label = filename prefix; §3.1).

**Unified statistics.** One `StatisticsStage` iterates targets: a 2+series target emits **comparison** metrics (`StatisticsCalculator`, `stages.py:2094`); a 1-series target emits **descriptive** metrics (the inline numpy of `stages.py:2918-2934`). Both land in `results['statistics']` with a per-row `kind: comparison|descriptive` tag. The CSV/markdown/AI consumers are made `kind`-aware (§3.5); per the **Q3 decision** the CSV writer emits **two files** (`statistics_summary.csv` = comparison, byte-identical; `statistics_descriptive.csv` = descriptive).

### 3.5 Stats-consumer migration (mandatory, not optional)

All three consumers from §2.3 are explicit migration deliverables:
- **`SaveResultsStage`** (`stages.py:2664-2705`): branch on `kind` and write **two files** (Q3 decision). Comparison rows → `statistics_summary.csv`, **byte-identical** to today (no new column). Descriptive rows (N/mean/median/std/min/max/p10..p90) → a new `statistics_descriptive.csv`. A run with only single-source targets now writes `statistics_descriptive.csv` (closing the current obs-only no-CSV gap); a paired run's `statistics_summary.csv` is unchanged.
- **`LogCollector.to_markdown`** (`runner.py:372-411`): emit a comparison table and a descriptive table, discriminated by `kind`.
- **AI payload** (`ai/payload.py`): already schema-agnostic; update `_STATS_STAGES`/`_PLOT_STAGES` to read the single unified `statistics`/`plotting` keys once the dual-write is removed (§5 P4). Keep an integration assertion that descriptive stats still reach the payload.

---

## 4. Per-Renderer Migration Table

| Renderer | Action | Detail | Citations |
|---|---|---|---|
| `obs_timeseries` → `timeseries` | **MERGE** | `render(series)`: 1 → single line (`NCAR_PRIMARY`, role=None idx 0); 2 → obs gray/model blue (current `timeseries`); N → palette overlay. Adopt canonical **auto-aggregate** default (`timeseries.py:140-145`); spaghetti opt-in via existing `show_individual_sites` (`timeseries.py:72,122`). Preserve per-flight coloring via `split_dim` (§3.3) and altitude twin-axis as a render kwarg. Delete `obs_timeseries.py` after the alias is registered. | `obs_timeseries.py:59` (default flip); `timeseries.py:122-145` |
| `obs_flight_track` → `track_map_3d` | **MERGE (true duplicate)** | `track_map_3d.render` accepts 1..N series: 1 → single colored track (today's `obs_flight_track` body + its `vmin`/`vmax`, `get_sequential_cmap` default, `city_labels`); 2 with `show_var='bias'` → model−obs bias (today's `track_map_3d`). `show_var='bias'` declared per series-count (≥2 only; degrades to plain track at 1). Highest-value collapse — eliminates a near-verbatim copy. | imports `_get_*`/`_render_surface_map` from `track_map_3d` (`flight_track_map.py:24-28`) |
| `obs_histogram` → **new** `histogram` | **PROMOTE** | New canonical renderer on `BasePlotter` (gains `set_labels`/`add_legend`/`set_limits`/`apply_text_style`). 1 → single histogram; N → overlaid/side-by-side colored by role/source_label. Genuine capability gap — no paired histogram exists; `obs_histogram`+`obs_lma_density` are 26 of 79 config specs. | `obs_histogram.py:24` |
| `obs_vertical_profile` → **new** `vertical_profile` | **PROMOTE** | New canonical renderer on `BasePlotter`. 1 → obs profile; 2 → obs-vs-model **overlay** (the missing paired profile); N → multi-source overlay. Preserve scatter/binned modes + altitude-bin ±std fill. Aggregation collapses non-**altitude** dims, not non-time. `curtain` (2-D, paired) cannot absorb a 1-D profile. | `vertical_profile.py:24` |
| `obs_lma_density` → **new** `lma_density` | **PROMOTE** | New geometry-specific renderer. **Rebase on `BaseSpatialPlotter` is BLOCKED** by capability gaps (see §8 Q6): `MapConfig` exposes only `show_states/show_countries/show_coastlines` (`spatial/base.py:53-55`) — no counties, no hourly, no `YlOrRd`; `obs_lma_density` hand-rolls `LambertConformal` (`obs_lma_density.py:134`), admin_2 counties (`obs_lma_density.py:206-211`), and hourly aggregation (`obs_lma_density.py:78-79`). **Decision:** keep on plain `BasePlotter` as a documented exception for the initial migration; extending `MapConfig` (counties field + hourly) is a follow-up. `min_series=1`. Preserve the `list[(label, fig)]` hourly return and the `flight_tracks` overlay. | `obs_lma_density.py:43,78,134,206` |

**Canonical generators that the JSON claimed to "retire" but left undispositioned (resolving review completeness #5).** `plot_per_flight` exists on `scatter.py:243`, `flight_timeseries.py:398`, `track_map_3d.py:711`; `plot_per_site` on `per_site_timeseries.py:188`; plus the public `plot_per_site_timeseries` (`per_site_timeseries.py:483`, re-exported `plots/__init__.py`). The stage dispatch is coupled to them via `hasattr(plotter, "plot_per_flight"/"plot_per_site")` (`stages.py:2533,2566`). **Decision:** in P3 these four renderers also adopt the `list[(label, fig)]` return from `render()`, and the stage's `hasattr`-dispatch (`stages.py:2528-2616`) is rewired onto the universal return path. `plot_per_site_timeseries` is **kept as a thin back-compat shim** delegating to the new path (it is a public import). `taylor.plot_multiple` (`taylor.py:270`) is folded into the N-series `render` as the first-class N-model overlay.

---

## 5. Phased Migration Plan

Each phase keeps the **1324-test suite green** and is independently shippable. Phase ordering is corrected to eliminate the double-emission blocker and to build the alias layer before any deletion.

### P0 — Series abstraction + helper generalization + **alias layer** (no behavior change)
- Add `PlotSeries`; add `iter_canonical_variable_series` (insertion-order-preserving, `core/base.py:429`); rewrite `iter_paired_variable_pairs` as a wrapper over it (byte-identical).
- Add `build_series(data, var_args)` + a concrete `BasePlotter.plot(data, *var_args, **kwargs)` facade that resolves `(obs_var, model_var)` | `(variable)` | `[v1..vN]` into a series list and calls `self.render(series, ...)`. Add `render()` with a default impl delegating to existing `plot()` so nothing breaks. **mypy:** provide `@overload` stubs for the three facade arities so type safety is preserved under strict mode (resolving review missing-from-plan #4). **Positional-ax:** `build_series` special-cases a trailing `matplotlib.axes.Axes` in `var_args` so `plot(obs_data, 'o3', some_ax)` is not misread as a 2-series call (resolving review API-soundness facade #4); claim is "keyword-ax callers unchanged," not "verbatim."
- **Build the registry alias layer** (review blocker): extend `register_plotter` / `Registry` with `register_alias(old, new)` storing a redirect; `get_plotter`/`has_plotter` resolve aliases and emit a one-time `LegacyConfigWarning`. No `obs_*` aliases registered yet.
- **Risk: Very low** — pure addition; existing `plot()` paths untouched.

### P1 — Tag single-source variables + unified 1-series styling
- Add `tag_source_roles` (new helper; **not** `tag_paired_roles`, which is a no-op on bare names — `stages.py:343`); invoke at load/select. Implement the explicit color rule (`index==0 and role in {None,obs}` ⇒ `NCAR_PRIMARY`; `role==model` ⇒ `MODEL_COLOR`).
- Re-express `test_obs_source_labels.py:49` in unified terms (single-series ⇒ NCAR blue) but keep `obs_*` renderers running unchanged.
- **Risk: Low** — obs renderers still hardcode `NCAR_PRIMARY`, visuals unchanged; new attrs additive, consumed only once renderers migrate in P3.

### P2 — Unify the stages behind the broad gate (no double-emission)
- Merge `ObsPlottingStage`→`PlottingStage` and `ObsStatisticsStage`→`StatisticsStage` with per-spec/per-target dispatch and the broad `validate()` gate. Eliminate the `startswith("obs_")` skip (`stages.py:2807`) so canonical types render in obs-only runs.
- **Avoid double-emission (review blocker):** in the **same** phase, **remove `ObsStatisticsStage`/`ObsPlottingStage` from `create_standard_pipeline`** (`stages.py:3030-3031`) — do **not** defer to P4. The `obs_*` **renderer classes** stay registered (used via the merged stage) until P3; only the **stages** are dropped here. This prevents the merged stage and the legacy stages from both firing on `bool(observations)`.
- **Stats schema (Q3 — two CSVs):** the merged `StatisticsStage` tags each row `kind: comparison|descriptive` in `results['statistics']` (comparison rows byte-identical). `SaveResultsStage` (`stages.py:2664`) writes comparison rows to `statistics_summary.csv` (**unchanged**) and descriptive rows to a **new** `statistics_descriptive.csv` — no `Kind` column added to the existing file. `LogCollector` (`runner.py:381`) emits a comparison table and a descriptive table. The AI payload (`ai/payload.py:34`) continues reading both stats keys until P4.
- **Output layout (review minor):** unify on subdir + PNG/PDF; update `_assert_plots`/`_copy_artifacts` expectations (`test_integration.py:372,468`) in the same commit.
- **Splitting semantics (review major):** the unified stage **auto-splits on a detected `flight`/`site` coord** (preserving the 17 obs flight specs that set no flag, `stages.py:2829`) **and** honors explicit `split_by_flight`/`split_by_site` flags (preserving paired behavior). Decide §8 Q before merging.
- **Retarget fork-pinning stage-name tests here** (the phase that breaks them): `test_obs_pipeline.py:188` (`test_create_obs_pipeline`) and `:205` (`test_run_from_config_detects_obs_only`) → assert the unified `plotting`/`statistics` stage names handle a no-pairs context.
- Add the **keystone integration test** (§6 A/B/C/D).
- **Risk: Medium** — touches run-gate logic + output layout + a brand-new single-source dispatch branch (`PlottingStage` has no single-source path today, `stages.py:2272-2304`). De-risked by dual-writing stats and keeping `obs_*` renderers callable through the merged stage.

### P3 — Migrate renderers onto `render()` + collapse duplicates
- Per §4: merge `obs_timeseries`→`timeseries`, `obs_flight_track`→`track_map_3d`; promote `obs_histogram`→`histogram`, `obs_vertical_profile`→`vertical_profile`, `obs_lma_density`→`lma_density`. **Each migration commit registers the new canonical name AND its `obs_*` alias together** (so the alias target always exists — resolving review ordering #4). Each migrated renderer overrides `render(series)`.
- Convert `scatter`/`flight_timeseries`/`track_map_3d`/`per_site_timeseries` generators to the `list[(label,fig)]` return; rewire the stage off `hasattr`-dispatch (`stages.py:2533,2566`); keep `plot_per_site_timeseries` as a shim.
- **Timeseries default flip (riskiest, gated sub-step):** land the test rewrite FIRST — rewrite `test_phase5_plots.py:62` to assert one line by default and spaghetti only under `show_individual_sites=True`; add an equivalence test — THEN redirect. Resolve §8 Q1 (the visible 17-config change) **before** starting.
- **Risk: Medium-high** — behavioral merges. Ship renderer-by-renderer; alias + facade keep old configs/tests green. Note `test_phase5_plots.py:62` constructs `ObsTimeSeriesPlotter()` **directly**, so an alias cannot save it — its rewrite must land in the same commit as the timeseries merge.

### P4 — Migrate configs + retire the fork
- Update the **2 tracked** templates with `obs_*` type specs to canonical names: `dc3-obs-dc8.example.yaml` (7 specs) and `firex-aq-obs-dc8.example.yaml` (10 specs). The 9 gitignored `*-gemini.yaml` stay on aliases (opportunistic).
- Delete `ObsPlotter` (`obs_base.py`), the 5 `renderers/obs/*` modules, the `results['obs_statistics']` dual-write, the AI payload dual-key read (collapse `_STATS_STAGES`/`_PLOT_STAGES` to single keys, `ai/payload.py:34-35`), and the runner's `("plotting","obs_plotting")` merge (`runner.py:1751`).
- **`create_obs_pipeline`:** remove the function **and** its import + `__all__` entry in `pipeline/__init__.py:34,54` (resolving review completeness #6) — or keep it as a thin deprecated shim returning the unified pipeline. Update/delete `test_create_obs_pipeline` accordingly.
- Keep the `obs_*` string aliases **indefinitely** (cheap) so the 79 specs never hard-break.
- **Gate:** full suite + an obs-only run + a paired run + an N-source run.
- **Risk: Low-medium** — everything routes through the unified path; deletion is mechanical.

---

## 6. Test Strategy

**UNIT**
1. `iter_canonical_variable_series` returns 1/2/N `PlotSeries` grouped by canonical; the `iter_paired_variable_pairs` wrapper is **byte-identical** to today on a **3-source dataset** where the dropped third var would change `[0]` selection under reordering (regression guard for the read-path rewrite + the `setdefault` order, `core/base.py:447`).
2. Styling: keep `test_phase5_plots.py::TestGetColorForRole`; add a pin for single-series `role=None` ⇒ `NCAR_PRIMARY` and 3-series ⇒ three distinct palette colors. Add a `role='model'` single-source ⇒ `MODEL_COLOR` pin (§3.2 rule).
3. Render-facade: `plot(paired, obs_var, model_var)`, `plot(obs_data, variable)`, `plot(data, [v1..vN])`, and `plot(obs_data, var, ax_positional)` all produce the right series count (with the trailing-Axes special-case).
4. Alias layer: `get_plotter('obs_timeseries')` returns the unified class and warns once; `has_plotter('obs_timeseries')` stays `True` (preserves the 5 `test_obs_plots.py` registry-name asserts); every registered alias target resolves (registry-integrity test).

**INTEGRATION — through `PipelineRunner.run_from_config` only** (the sole path CLAUDE.md counts as integration):
- **(A) OBS-ONLY** on the unified `sources:` schema (`role: obs`, no `pairs`): produces timeseries/histogram/profile/track/lma plots via the **unified** renderer+stage; assert exact PNG count, single-source styling == `NCAR_PRIMARY`, source-label legend, **and that descriptive stats reach both the CSV and the AI payload**. Replaces the legacy-schema coverage of `test_integration.py:369` + the unit-level obs_* tests.
- **(B) PAIRED**: obs gray/model blue unchanged AND **no duplicate single-source plots double-emitted** (proves the fork boundary is gone, not bypassed — a gap no current test covers). Assert the paired CSV's existing columns are unchanged.
- **(C) N-SOURCE** (2 models + 1 obs, same canonical) via an **explicit list-of-source-labels** plot target reading **raw** sources (not a paired dataset — §8 Q4): assert 3 overlaid series. Proves the third source survives the read path.
- **(D) Unified `StatisticsStage`**: descriptive (N/mean/median/p10..p90) for a 1-source target AND comparison for a pair land in `results['statistics']` with `kind` tags and reach the CSV with correct **values** (not just keys). Replaces `TestObsStatisticsStage` (`test_obs_pipeline.py:140`).

**RETIRE/REPLACE fork pins**
- `test_phase5_plots.py:62` (`test_default_plots_one_line_per_site`) → invert to one line by default; pin spaghetti behind `show_individual_sites=True` (rewrite lands with the timeseries merge, P3).
- `test_obs_source_labels.py:49` → keep the assertion, drive it through the unified styling path (P1).
- `test_obs_plots.py` registry-name asserts (5) → kept passing via aliases through P3.
- `test_obs_pipeline.py:188,205` → retarget onto unified stage names (P2).
- Keep alias tests until P4, then drop with the fork.

---

## 7. Backward Compatibility

- **`obs_*` plot-type strings stay valid indefinitely as deprecated aliases**, mirroring the existing legacy `model:`/`obs:` auto-conversion that already emits `LegacyConfigWarning`. Register `obs_timeseries→timeseries`, `obs_flight_track→track_map_3d`, `obs_histogram→histogram`, `obs_vertical_profile→vertical_profile`, `obs_lma_density→lma_density` (P3, paired with each renderer's migration). So the **79 obs_* specs across 11 config files** keep running unchanged; the hard-update set shrinks to docs + the 2 committed templates.
- The single-source `plots: { ..., obs: <label>, variable: <var> }` spec shape stays accepted and is normalized into a 1-source target by the unified stage (no schema break).
- The facade `plot(paired, obs_var, model_var)` and `plot(obs_data, variable)` signatures both stay valid; the legacy `PairingEngine` API still emits `model_`/`obs_` names, which `get_role_color`/`canonical_variable_name` handle via the prefix fallback (`base.py:806-811`). Internal callers, tests, and user scripts are unaffected (keyword-ax callers; positional-ax handled by the trailing-Axes special-case).
- `migrate-config` continues producing valid (aliased) configs; legacy `model:`/`obs:` blocks auto-convert to `sources:` exactly as today.
- Stats output is split into **two CSVs** (Q3): `statistics_summary.csv` stays **byte-identical** (comparison only); descriptive single-source stats go to a new `statistics_descriptive.csv`. No existing CSV columns change, so downstream parsers of `statistics_summary.csv` are unaffected.

---

## 8. Decisions (resolved 2026-06-06)

All of these were settled before implementation:

1. **Timeseries default flip — YES, aggregate by default.** Single-source multi-site timeseries renders one mean line by default (matching canonical `timeseries`); per-site lines opt in via `show_individual_sites: true`, ±1σ via `show_uncertainty: true`. This is the intended fix for the spaghetti and visibly changes ~17 obs_timeseries specs. Rewrite `test_phase5_plots.py:62` to assert one line by default (lands with the P3 timeseries merge).
2. **Single-series color — adopt the explicit rule.** `index==0 and role in {None, obs} ⇒ NCAR_PRIMARY`; `role==model ⇒ MODEL_COLOR` (both are NCAR blue; gray `OBS_COLOR` stays paired-only). Holds the `test_obs_source_labels.py:49` pin.
3. **Stats schema — TWO SEPARATE CSVs.** Keep `statistics_summary.csv` byte-identical (comparison rows only, no new column); write a new `statistics_descriptive.csv` for single-source descriptive stats (N/mean/median/std/min/max/p10..p90). Zero risk to existing downstream parsers; also closes the current "obs-only run writes no stats CSV" gap. (Supersedes the draft's one-tagged-table/`Kind`-column proposal — see §3.4, §3.5, §5 P2, §7.)
4. **Scope — RENDERING ONLY.** Ship the N-capable renderer + collapsed stages; leave the pairing engine + `tag_paired_roles` (`stages.py:330-343`) binary (reference/comparand). N-overlay is reachable **only** via an explicit multi-source plot spec reading raw sources (integration test C), never from a paired dataset. N-way pairing is out of scope for now.
5. **Output layout — UNIFY.** All plots write subdir-per-source + PNG/PDF (today's paired behavior); obs-only output paths relocate accordingly. Update `_assert_plots`/`_copy_artifacts` expectations (`test_integration.py:372,468`) in the same commit (P2).
6. **LMA density base — DEFER.** Keep `lma_density` on plain `BasePlotter` as a documented exception; do not extend `MapConfig` (counties/hourly) now — follow-up.
7. **Splitting semantics — AUTO + EXPLICIT.** Unified stage auto-splits on a detected `flight`/`site` coord (preserves the flagless obs flight specs, `stages.py:2829`) and honors explicit `split_by_flight`/`split_by_site` flags (preserves paired behavior, `stages.py:2529`).

---

### Audit notes (file:line evidence cited inline above)
Key verifications: two abstract `plot()` signatures (`base.py:289-318`, `obs_base.py:71-97`); `ObsPlotter` is a standalone ABC duplicating figure helpers (`obs_base.py:40,99-172`); `obs_flight_track` imports `track_map_3d` internals (`flight_track_map.py:24-28`); both stage forks live in `create_standard_pipeline` (`stages.py:3025-3034`); mutually-exclusive gates (`stages.py:2044,2230,2762,2903`); `setdefault` drop (`core/base.py:447`); `obs_` skip (`stages.py:2807`); inverted aggregate defaults (`obs_timeseries.py:59` vs `timeseries.py:140-145`); `obs_statistics` IS read by the AI payload (`ai/payload.py:34,54-66`) — design JSON premise corrected; CSV/markdown hardcode comparison schema (`stages.py:2669-2705`, `runner.py:381`); no registry alias support (`registry.py:33-56`, `core/registry.py:122`); `create_obs_pipeline` vestigial but exported (`runner.py:1360`, `pipeline/__init__.py:34,54`); `MapConfig` lacks counties/hourly (`spatial/base.py:53-55`); `spatial_distribution` single-field for `show_var=obs|model` (`distribution.py:65,210`); 79 obs_* specs across 11 files (2 tracked templates: dc3-obs-dc8=7, firex-aq-obs-dc8=10); suite = 1324 tests (CLAUDE.md's 961 is stale).
