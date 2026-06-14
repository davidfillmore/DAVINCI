# Design: x/y Pair Rename — Kill the `geometry`/`dataset` Role Vocabulary

**Date:** 2026-06-14
**Status:** Approved (pending spec review)
**Scope:** Workstream #1 of 2. The single-source spatial plots feature is **out of scope** here and gets its own spec after this lands.

---

## Context

The codebase overloads the word **geometry** to mean two unrelated things:

1. **Shape** — a source's data geometry: point / track / profile / swath / grid. Every source has one. This is the *correct, fundamental* meaning and drives pairing strategy selection.
2. **Role** — `pair_axis: "geometry" | "dataset"`, a per-variable tag meaning "which side of a pair am I." This is a relabel of the old model/obs distinction. The obs side became `"geometry"` because the obs source supplies the sampling *locations* — i.e. the role got named after the thing it provides, which collides head-on with meaning #1.

`dataset` as a role word is equally bad: everything in the system is an `xr.Dataset`.

This spec removes the **role** concept entirely. A pair is simply an ordered **(x, y)**:

- **scatter** draws x on the horizontal axis, y on the vertical axis;
- **diff / spatial bias** is **`y − x`**;
- **time series** needs no x/y — it overlays N source series sharing one value axis;
- **single-source spatial** (next workstream) needs no x/y — it dispatches on *shape*.

## Goals

- Eliminate the `geometry`/`dataset` **role** vocabulary from code, attributes, config, tests, and docs.
- Replace it with positional **x / y** (plot axes) plus **`source_label`** (a source's name).
- Leave the *shape* meaning of "geometry" (point/track/profile/swath/grid) untouched.
- Produce **pixel-identical plots** — this is a behavior-preserving rename, not a redesign.

## Non-Goals

- Single-source spatial plots (separate spec).
- Any change to pairing *direction* / resampling math.
- Back-compat for the old config shape (explicit clean break — see below).

---

## Key Insight: the rename is behavior-preserving

In the current code:

- scatter puts geometry on x, dataset on y — `scatter.py:242` (`xlabel=geometry_label, ylabel=dataset_label`);
- spatial bias is already `dataset − geometry` — `bias.py:126` (`bias = dataset_data - geometry_data`).

So the equivalence is exact:

```
geometry  ≡  x          dataset  ≡  y
```

Mechanically renaming `geometry_* → x_*` and `dataset_* → y_*` reproduces every existing figure unchanged. The default colors map straight across (x = gray, y = blue — see Styling), so no plot's appearance changes.

---

## Decision 1 — Semantic model

- A **pair** is an ordered `(x, y)`. Order carries meaning (axis assignment + diff sign) and nothing else.
- **Pairing direction is unchanged and is NOT driven by x/y.** Which source gets resampled onto which is still decided by **shape precedence** (irregular geometries — point/track/profile/swath — outrank grid). x/y only controls plot-axis assignment and the sign of diffs. A user may set `x: cam, y: airnow` or `x: airnow, y: cam`; either way the pairing still resamples the gridded source onto the point locations. Only the plotted axes / bias sign flip.
- The paired dataset still contains two source-prefixed variables (`<source_label>_<var>`, e.g. `cam_o3`, `airnow_o3`); variable **names are unchanged**. Only their **attributes** change.

## Decision 2 — Attribute renames (on paired variables)

| Old attr | New attr | Values |
|---|---|---|
| `pair_axis` | `axis` | `"x"` / `"y"` (was `"geometry"` / `"dataset"`) |
| `dataset_label` | `source_label` | unchanged (e.g. `"cam"`, `"airnow"`) |

Renderers dispatch by `axis` (e.g. `x_series = next(s for s in series if s.axis == "x")`).

## Decision 3 — Identifier renames (Python)

Behavior-preserving, applied across all ~1,400 sites:

| Old | New |
|---|---|
| `geometry_var` | `x_var` |
| `dataset_var` | `y_var` |
| `geometry_data` | `x_data` |
| `dataset_data` | `y_data` |
| `geometry_series` | `x_series` |
| `dataset_series` | `y_series` |
| `geometry_color` | `x_color` |
| `dataset_color` | `y_color` |
| `PlotSeries.pair_axis` | `PlotSeries.axis` |
| `PlotSeries.dataset_label` | `PlotSeries.source_label` |

### Overloaded-name caveats (must NOT be renamed uniformly)

`dataset_label` and `geometry_label`/`dataset_label` appear with **two different meanings**. They must be disambiguated, not blindly swapped:

- **Source identity** — `PlotSeries.dataset_label`, the paired-variable attr `dataset_label` → **`source_label`** (the source's name).
- **Custom axis-label override** — `PlotConfig.geometry_label` / `PlotConfig.dataset_label` (a user-supplied string to label an axis) → **`PlotConfig.x_label` / `PlotConfig.y_label`**.

Helper functions named for the role (e.g. `get_dataset_color`, `get_series_label`) get renamed to axis/series-oriented names during the plan (e.g. `get_axis_color`); the plan enumerates each. The word "dataset" as a generic synonym for "an `xr.Dataset`" in prose/docstrings is left alone.

## Decision 4 — Styling

`StyleConfig` role fields → axis fields, preserving current appearance:

| Old | New | Color |
|---|---|---|
| `geometry_color` | `x_color` | gray (`#58595B`) |
| `dataset_color` | `y_color` | NCAR blue (`#0A5DDA`) |
| `geometry_linestyle` / `geometry_marker` | `x_linestyle` / `x_marker` | unchanged |
| `dataset_linestyle` / `dataset_marker` | `y_linestyle` / `y_marker` | unchanged |

## Decision 5 — Config schema (clean break)

**New pair shape** (nested, explicit):

```yaml
pairs:
  cam_vs_airnow_o3:
    x: {source: airnow, variable: o3}   # horizontal axis; the "− x" in diffs
    y: {source: cam,    variable: O3}    # vertical axis;  the "y −" in diffs
```

- Rewrite `SourcePairConfig`: fields `x: AxisRef` and `y: AxisRef`, where `AxisRef = {source: str, variable: str}` (a new small pydantic model). Both required.
- Remove `sources`, `geometry`, `variables` from the pair model.
- Remove the per-source `pair_axis` field from `SourceConfig` (axis is now per-pair, not per-source).
- The **old** pair shape (`sources: [a, b]` + `geometry:` + `variables:`) raises a clear `ValueError` at validation with a migration hint pointing at the `x:`/`y:` form. (Clean break — no silent acceptance.)
- The top-level `sources:` block (reader type, files, per-variable unit/limit config) is **unchanged**.
- Plot specs still reference pairs by name via `data:`/`pairs:` lists — **unaffected**.

## Decision 6 — Pairing engine

`PairingEngine` / strategies read `x` and `y` (source + variable) from the new pair config instead of `geometry`/`dataset` roles. They:

1. determine the common geometry by **shape precedence** (unchanged logic);
2. resample accordingly (unchanged math);
3. tag the two resulting paired variables with `axis: "x"` / `axis: "y"` and `source_label`.

## Decision 7 — Diff / bias labeling

`spatial_bias` computes `y_data − x_data`. The colorbar label uses the source names when available (e.g. `"Bias (cam − airnow)"`), falling back to `"Bias (y − x)"`. (Replaces the current `"Bias (Dataset - Geometry)"`.)

## Decision 8 — Config migration (tracked configs)

Migrate every **tracked** pair block to the nested form using the equivalence `geometry ≡ x`:

- `x.source` = the pair's `geometry:` label (or the first-listed source if `geometry:` was implicit); `x.variable = variables[x.source]`.
- `y.source` = the other source; `y.variable = variables[y.source]`.

Tracked configs to migrate: `analyses/**/configs/*.example.yaml`, `examples/**/*.yaml` (~31 files reference these terms; the migration touches the pair blocks). **Gitignored machine-specific configs** (`*-gemini.yaml`, `*-derecho.yaml`) cannot be committed and may not all be visible — they are flagged for the user to migrate locally using the same rule; the validation error message documents the transformation.

## Decision 9b — Source-level QC vocabulary (geometry ≡ obs misuse)

"geometry" is also misused at the **source level** to mean "observation / datapoint."
These are NOT x/y (no pair is involved) — they are per-source value/QC settings and
must be renamed to what they actually are. Folded into this pass (clean break):

| Old | New | Where |
|---|---|---|
| `geometry_min` / `geometry_max` | `valid_min` / `valid_max` | `VariableConfig` value bounds (~54/52 config uses) |
| `geometry_count` | `sample_count` | diagnostic variable emitted per bin |
| `min_geometry_count` | `min_sample_count` | `SourceConfig` min valid samples per bin |
| `track_geometry_count` | `track_sample_count` | `SourceConfig` flag to emit the diagnostic |
| `rem_geometry_nan` | `rem_nan` | `DataProcConfig` NaN filter |
| `rem_geometry_by_nan_pct` | `rem_by_nan_pct` | `DataProcConfig` NaN-percent filter |

The **shape** meaning of "geometry" (`DataGeometry`, `detect_spatial_geometry`,
`surface_level_index`, etc. — ~199 sites) is correct and left untouched.

## Decision 9 — Tests & docs

- Update the ~25 test files + synthetic data generators that set `pair_axis`/`dataset_label` attrs or build pairs / pair configs, to `axis`/`source_label` and the nested config shape. Integration tests run the migrated configs through `PipelineRunner.run_from_config()` (per the repo's testing rules).
- Rewrite the affected CLAUDE.md sections (Variable Naming Convention, Unified Data-Source Config, Plot Styling colors, Common Gotchas) to the x/y vocabulary. CLAUDE.md is the source of truth and must not describe the dead vocab.

---

## Affected surfaces (for the plan to expand)

- `davinci_monet/core/base.py` — `PlotSeries` dataclass
- `davinci_monet/core/protocols.py`
- `davinci_monet/config/schema.py` — `SourcePairConfig`, `SourceConfig`, new `AxisRef`
- `davinci_monet/pairing/engine.py`, `pairing/strategies/*`
- `davinci_monet/pipeline/stages/*` — esp. `plot.py` (`_resolve_pair_labels_and_vars`, `_resolve_paired_dataset_variable`, `_render_pair`, `_save_single`, `_execute_single_source`), `pair.py`, `stats.py`
- `davinci_monet/plots/base.py`, `plots/series.py`, `plots/labels.py`, `plots/plot_config.py`, `plots/registry.py`, `plots/style.py`
- `davinci_monet/plots/renderers/**` — all renderers + `spatial/*`
- `davinci_monet/stats/calculator.py`
- `davinci_monet/datasets/**` — readers that reference these terms (mostly docstrings/labels; verify each)
- `davinci_monet/tests/**`
- tracked YAML configs under `analyses/` and `examples/`
- `CLAUDE.md`

## Acceptance criteria

1. No `geometry`/`dataset` **role** vocabulary remains in code, attrs, config, or docs (the *shape* meaning of "geometry" is retained; generic "dataset == xr.Dataset" prose is retained). Verified by grep for the role identifiers/attrs/values returning only shape/prose uses.
2. New nested `x:`/`y:` pair config validates; old pair shape raises a clear migration error.
3. All tracked example configs migrated and load.
4. Full suite green, mypy clean, black/isort clean — run locally in the `davinci` conda env (`HDF5_USE_FILE_LOCKING=FALSE python -m pytest`).
5. **Output-preservation check:** a representative paired analysis (e.g. ASIA-AQ AirNow) produces byte-identical-or-visually-identical scatter / spatial-bias / timeseries plots before vs. after the rename. (Per the geometry-aware-rendering memory: verify programmatically, not by eye.)

## Risks

- **Overloaded `dataset_label`** (source identity vs custom label override) — mishandling silently mislabels axes. Mitigated by Decision 3's caveat and per-file review.
- **String-literal dispatch** — `"geometry"`/`"dataset"` are load-bearing in renderer dispatch and attr writes; a missed literal breaks series selection silently. Mitigated by the output-preservation check.
- **Scale** (~1,400 sites) — mitigated by mechanical, behavior-preserving mapping + full suite as the backstop.
