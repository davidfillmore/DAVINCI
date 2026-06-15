# Plot Labeling & Title System — Design Spec

**Date:** 2026-06-15
**Status:** Draft for review
**Scope:** A single, consistent, publication-quality title/label system applied across **all** DAVINCI plot renderers.

## Context & Problem

Plot text is composed ad-hoc inside each renderer, producing inconsistent and
non-publication-quality output. Concrete defects (observed in regenerated
asia-aq Pandora, and latent in other renderers):

- **Internal config keys leak into plots.** `scatter._source_display_name()`
  does `str(key).replace("_"," ").upper()`, so source key `cesm_no2_column`
  renders on an axis as `CESM NO2 COLUMN` (answering "why is COLUMN all-caps"
  and "what are internal var names doing in plots"). The timeseries legend shows
  the raw key `cesm_no2_column`.
- **Duplication.** That mangled source string is prepended to a variable label
  that already names the quantity → `CESM NO2 COLUMN Tropospheric NO2 Column`.
- **x/y leaks to the viewer.** `spatial/bias.py`, `curtain.py`, `track_map_3d.py`
  hardcode `Bias (y - x)` / `Bias (Y - X)`; viewers don't know x/y roles.
- **Units not superscripted / missing.** `format_units` has no `mol/m2` rule and
  no general power handling → `(mol/m2)`; some timeseries paths omit units.
- **Caption material in titles.** Titles carry `(Mean +/- Std)`, date ranges,
  `vs <source>` — context that belongs elsewhere.

Root cause: **no single owner of text composition.** Each renderer invents its
own strings, so consistency is impossible to maintain.

## Goals

1. One grammar for titles, subtitles, axis labels, legends, and colorbar/bias
   labels, owned by a single module and used by every renderer.
2. Publication-quality output: terse titles, SI units, no internal identifiers,
   no x/y exposure, no duplication.
3. Fully unit-testable without rendering (pure functions over identifiers/attrs).

## Non-Goals

- Changing fonts/sizes/colors/themes (the existing `style.py` system stays).
- Changing which plots exist or their numeric content.
- A figure-caption sidecar system (rejected; context stays on-figure — see
  Decisions).

## Decisions (locked with the user)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Where contextual info lives | **On-figure**: terse title + **date subtitle** + **in-axes stats box**; uncertainty in the legend. No sidecar caption. |
| D2 | Source-name placement | **Context-aware (no redundancy)**: source on an axis only where no legend carries it (scatter); timeseries/single-source carry source in legend/title; bias colorbar names both. |
| D3 | Unit format | **Negative-exponent SI**: `mol m⁻²`, `W m⁻²`, `µg m⁻³`, `m s⁻¹`; ratios `mol mol⁻¹`/`kg kg⁻¹`; drop only true dimensionless (`1`,`none`,``,`dimensionless`); leave `ppb`/`ppm`/`%`/`K` as-is. |
| D4 | Architecture | **Central module** `plots/labeling.py` (pure functions); every renderer calls it. |
| D5 | Source display names | **Auto-clean keys** (acronym map + chemical formatting; never ALL-CAPS). No new config field required. |
| D6 | Operation word in title | Only when it changes meaning: **bias → "… Bias"**; scatter/timeseries/histogram get **no** plot-type word. |

## § 1 — Module API: `davinci_monet/plots/labeling.py`

Pure functions. `labels.py` retains the lookup tables (`VARIABLE_DISPLAY_NAMES`,
`TITLE_FORMULA_REPLACEMENTS`, `UNIT_REPLACEMENTS`) and re-exports the new helpers
for backward-compatible imports.

```python
format_units(units: str | None) -> str
    # Negative-exponent SI (D3). "mol/m2"->"mol m$^{-2}$"; "W m-2"->"W m$^{-2}$";
    # "ug/m3"->"$\\mu$g m$^{-3}$"; ratios "mol/mol"->"mol mol$^{-1}$".
    # Returns "" for dimensionless markers.

source_display_name(source_label: str | None) -> str
    # Auto-clean key (D5): split on "_", apply chemical formatting (no2->NO$_2$),
    # apply acronym map, title-case the remainder; NEVER force ALL-CAPS.
    # "cesm_no2_column" -> "CESM NO$_2$ Column"; "airnow" -> "AirNow".

quantity_label(dataset, var_name) -> str
    # The quantity only (display_name/long_name/lookup), chem-formatted,
    # NO source, NO units. "NO$_2$ Column".

axis_label(quantity: str, units: str | None, source: str | None = None) -> str
    # Context-aware (D2) + de-dup. With source: "[Source ]Quantity (units)";
    # if quantity ⊆ source-display, the quantity is not repeated.

legend_label(source_label: str, uncertainty: str | None = None) -> str
    # "CESM" or "CESM (mean ± σ)" (σ/IQR/range per uncertainty_type).

bias_label(y_source: str, x_source: str, units: str | None) -> str
    # "Bias, <Ysrc> − <Xsrc> (units)" — never x/y. Shared-quantity factoring:
    # if both display names end with the same quantity (also in the title),
    # strip it -> "Bias, CESM − Pandora (mol m⁻²)".

title_text(quantity: str, operation: str | None = None) -> str
    # Terse (D6). "NO$_2$ Tropospheric Column"[+ " Bias"]. Never dates/stats.

subtitle_text(start, end) -> str
    # "2024-02-01 – 2024-02-29" (single date if start==end).
```

**Acronym map (seed; extend as sources are added):** CESM, CAM, WRF-Chem,
MERRA-2, GEOS-Chem, AirNow, AERONET, Pandora, CERES, MODIS, TROPOMI, TEMPO, UFS,
CMAQ. Keys are matched token-wise (`cesm` in `cesm_no2_column`).

## § 2 — Per-plot-type rules

| Plot | Title | x-axis | y-axis | Legend | Colorbar |
|------|-------|--------|--------|--------|----------|
| Scatter | Quantity | *Xsrc* Quantity (units) | *Ysrc* Quantity (units) | 1:1 & fit only | Point Density |
| Timeseries | Quantity | Time | Quantity (units) | *each src* (+ mean ± σ) | — |
| Single-source map | Quantity | — | — | — | Quantity (units) |
| Spatial bias | Quantity **Bias** | — | — | — | Bias, *Ysrc* − *Xsrc* (units) |
| Curtain | Quantity [Bias] | Time/dist | Altitude (units) | — | Quantity (units) *or* Bias, *Y*−*X* |
| Track-3D | Quantity [Bias] | Lon | Lat | — | Quantity (units) *or* Bias |
| Profile | Quantity | Quantity (units) | Altitude (units) | *src(s)* | — |
| Histogram | Quantity | Quantity (units) | Count / Density | *src(s)* | — |

System-added to every figure (not the title): date **subtitle**, in-axes **stats
box** where stats apply.

## § 3 — Integration, config handling, testing, verification

### Renderer migration (each gated independently)
- `scatter.py`: delete `_source_display_name`; x/y via `axis_label(quantity, units, source)`; title via `title_text`.
- `timeseries.py`: legend via `legend_label`; y-axis via `axis_label(quantity, units)` (units always applied — fixes "no units"); subtitle/title via helpers.
- `spatial/bias.py`: colorbar via `bias_label(y_key, x_key, units)`.
- `spatial/field.py`: colorbar via `axis_label(quantity, units)`; title via `title_text`.
- `curtain.py`, `track_map_3d.py`: replace `Bias (Y - X)` with `bias_label`; single-source colorbars via `axis_label`.
- `profile`, `histogram`, flight (dc3/firex) renderers: titles/axes via helpers.
- Pipeline plot stage: config `title:` honored as a **terse override** (chem-formatted, never appended with date/stats); subtitle from `analysis.start_time/end_time`; stats box unchanged.

### Config handling
- Explicit `x_label`/`y_label` remain full overrides (power users).
- `ylabel_plot`-style strings with **baked units** (e.g. `"NO$_2$ Column (mol/m2)"`) are no longer the default source of axis text; the built `axis_label` (quantity + formatted units) is used. Baked units in any honored label string are re-run through `format_units`.
- **Config cleanup (regen):** fix gitignored gemini/-local titles — drop `(Mean +/- Std)`, `vs <source>`, baked `(mol/m2)` — since the system now supplies subtitle/legend/units/auto-title.

### Testing
- **Unit** (`tests/unit/plots/test_labeling.py`): `format_units` table incl. `mol/m2`, `W m-2`, `ug/m3`, `mol/mol`, `1`; `source_display_name` incl. acronym map + chem; `axis_label` de-dup; `bias_label` no-x/y + shared-quantity factoring; `title_text` terse; `subtitle_text`.
- **Renderer** (extend `tests/unit/plots/test_renderers.py`): assert the rendered Axes `get_xlabel()/get_ylabel()/get_title()`, legend texts, and colorbar label match expected strings — programmatic, not visual. Negative assertions: no label contains `"x"`/`"y"` role tokens, a raw source key, `".upper()"`-style ALL-CAPS keys, or `"/m2"`.
- **Integration** (pipeline): run a config end-to-end; assert across all produced figures that no label violates the rules.

### Verification
See **§ 4 — Verification Stage**. It is the acceptance gate for this work:
a synthetic sample gallery covering every plot type, plus a re-gen of all real
analyses, all rendered to **PDF only**, synced to iCloud, then a multimodal
inspect-and-iterate loop until every figure passes the label checklist.

### Rollout order
1. `labeling.py` + unit tests (green).
2. Migrate renderers one at a time; gate (`pytest`/`mypy`/`black`/`isort`) after each.
3. Config-title cleanup.
4. Full gate → run the **§ 4 Verification Stage** (gallery + all analyses, multimodal inspect-and-iterate until clean).
5. Update CLAUDE.md (Plot Label & Title Conventions) to match.
6. Commit on develop → merge to main (per repo workflow) when the user approves.

## § 4 — Verification Stage (acceptance gate)

The labeling work is **not done** until every figure below passes a multimodal
visual inspection. PDFs only throughout (PNG emission suppressed or not copied).

### 4a. Synthetic sample gallery (every plot type)
Build a pipeline-driven gallery from the `tests/synthetic/` generators (no
external data) that emits **one figure per plot type**, so renderers not covered
by the real analyses (curtain, profile, histogram, multi-source timeseries) are
still exercised:

- scatter (pairwise) · timeseries (multi-source + uncertainty band) ·
  single-source spatial map (grid **and** point) · spatial bias ·
  curtain · track-3D · profile · histogram · flight timeseries/track.
- Each uses synthetic sources with deliberately awkward keys (e.g.
  `cesm_no2_column`, `airnow`) and ratio/area units (`mol/mol`, `mol/m2`,
  `W m-2`, `ug/m3`) to exercise source-name cleaning, de-dup, and unit SI
  formatting.
- Output `analyses/_gallery/output/` (local) → mirror to `Claude/_gallery/`.

### 4b. Re-gen all real analyses
asia-aq Pandora, ceres EBAF + SSF, firex DC8, dc3 DC8 — rendered **locally**,
**PDF only**, then mirrored to their iCloud per-analysis subdirs.

### 4c. Multimodal inspection (every PDF)
Read **each** produced PDF visually and check against the per-slot checklist:

- **Title**: terse; no dates, stats, `Mean ± Std`, or `vs <source>`.
- **Subtitle**: date range only (or absent).
- **Axis labels**: quantity + SI superscript units (D3); source present per the
  context-aware rule (D2); **no** internal source keys, **no** ALL-CAPS keys,
  **no** duplicated quantity.
- **Legend**: clean source display names; `(mean ± σ)` only where a band is drawn.
- **Colorbar / bias**: never `x`/`y`; reads `Bias, <Ysrc> − <Xsrc> (units)`.
- **Units**: superscripted everywhere and **present** on timeseries.

### 4d. Iterate
Any defect → fix code or config → re-render the affected figure(s) → re-inspect.
Repeat until the **entire** gallery and all analysis PDFs pass. Only then is the
stage complete; re-mirror the final PDFs and proceed to the gate/commit.

### 4e. Acceptance criteria
- Every gallery figure and every analysis PDF passes the 4c checklist on visual
  inspection.
- iCloud holds **PDFs only** (no PNGs) under `Claude/_gallery/` and each
  `Claude/<analysis>/...` subdir.
- Full local gate green (`pytest`, `mypy`, `black`/`isort`).

## Edge cases & resolved nuances
- `mol/mol`, `kg/kg`: formatted as ratios (`mol mol⁻¹`); only explicit dimensionless markers dropped.
- De-dup: `axis_label` repeats the quantity per scatter axis (expected); `bias_label` factors the shared quantity out (it's in the title) → short `Bias, CESM − Pandora`.
- With auto-clean (D5), a key like `cesm_no2_column` displays as `CESM NO₂ Column`, not bare `CESM`; the bias-label shared-quantity factoring is what yields the short `CESM − Pandora` form.
- Title override: a config `title:` is used verbatim (chem-formatted) as the terse line; the system never appends context to it.

## Post-implementation revisions (from multimodal review)

The §2 colorbar rules were refined after rendering every plot type and inspecting
the PDFs — the title and colorbar were doubling up on the quantity/"Bias":

- **Title owns the quantity; colorbar owns units/direction only.** Single-source
  maps (spatial field, curtain, track-3D, flight-track): **colorbar = units only**
  (`mol m⁻²`), title = `Quantity (Source)`. Bias plots: **colorbar = `<Ysrc> −
  <Xsrc> (units)`** (no leading "Bias,"), title = `Quantity Bias`. So each of
  "quantity"/"Bias" appears exactly once.
- **Species words normalise in titles too** — `format_plot_title` now maps
  `Ozone→O₃` (etc.), not just formula tokens, so auto-titles match the lookup.
- **`axis_label` de-dup keeps word order**: if the quantity is a substring of the
  source name, use the source name as-is (`CESM NO₂ Column`), never a reordered
  `CESM Column NO₂`; otherwise distinctive source tokens + quantity.
- **Reader/file units survive the load stage** — a no-op `unit_scale: 1.0` no
  longer strips the `units` attr (ICARTT `ppbv`, altitude `m` now reach plots).
- **ICARTT header units** are parsed and applied (`datasets/aircraft/icartt.py`).
- **Gallery is full-pipeline** (`analyses/_gallery/run_gallery.py`): synthetic
  source NetCDFs → configs → `run_analysis`, per CLAUDE.md's pipeline mandate.
