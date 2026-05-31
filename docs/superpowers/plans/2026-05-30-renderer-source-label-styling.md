# Renderer Source-Label Resolution + Role Styling — Implementation Plan

> Final piece of the model/obs unification (the "CFG-4 proper" renderer rewire).
> Sequenced as green sub-increments, each landed on `develop` with the full
> `davinci` conda-env suite passing (only the 2 pre-existing `test_stats.py`
> xarray-squeeze failures allowed).

## Context

Phases 1–6 (consolidation) unified core protocols, the registry, the pipeline
(`LoadSourcesStage` load-bearing), pairing (`pair_sources`/`(ref,comp)` dispatch),
the obs-only fold, the `sources:` config schema + `migrate-config`, and runtime
`sources:` support. The building blocks for this phase already exist and are
tested but **not yet consumed by the renderers**:

- `plots.get_color_for_role(role, index=0)` — role → color (obs gray, model blue,
  else palette cycle).
- `plots.resolve_source_variable(ds, canonical_var, source_label)` — resolve a
  variable by `<label>_<var>` with canonical fallback.
- `stages.tag_paired_roles(ds)` — paired variables carry a `role` attr.

What remains: the renderers still resolve series by hard-coded `model_`/`obs_`
prefixes and color via `style.obs_color`/`style.model_color`. There are
**~1166 `model_`/`obs_` references across `davinci_monet/plots/`**, plus the
plot test suite (`test_plots.py`, `test_obs_plots.py`) and the example/gemini
configs. This phase is the high-churn one and must be staged carefully.

## Goal

1. Paired output uses **source-label** variable names (`<label>_<var>`, e.g.
   `cam_o3`/`airnow_o3`) in addition to (then instead of) `model_`/`obs_`.
2. Renderers resolve series via `resolve_source_variable` and color via
   `get_color_for_role`, so two-model or two-obs pairs get distinct colors/legends.
3. Tests and example/gemini configs migrated; suite stays green throughout.

## Strategy: dual-naming bridge, then flip

Do **not** rename paired variables in one breaking step. Instead:

- **R-1** emits BOTH names: keep `model_<var>`/`obs_<var>` AND add aliases
  `<source_label>_<var>` (pointing at the same DataArray) on the paired dataset,
  with `role`/`source_label` attrs on each. Fully additive; existing renderers
  keep working off the legacy prefixes. Green.
- **R-2/R-3** migrate renderers to prefer source-label + role resolution, falling
  back to the legacy prefixes when a dataset only has those (older paired data).
- **R-5** drops the legacy `model_`/`obs_` aliases once nothing reads them, and
  migrates remaining tests/configs.

This keeps every step green and lets the ~1166 references move in reviewable
batches rather than one mega-diff.

---

## R-1: Dual-name paired output (additive)

**Files:** `davinci_monet/pipeline/stages.py` (PairingStage assembly /
`tag_paired_roles` area), `davinci_monet/pairing/engine.py` (where `obs_`/`model_`
prefixes are assigned), new test `tests/test_paired_source_labels.py`.

- Thread the `reference`/`comparand` source labels (from the pair config /
  `expand_sources_to_legacy` output) into the paired assembly.
- For each paired variable `model_<v>` add alias var `<comparand_label>_<v>`;
  for `obs_<v>` add `<reference_label>_<v>` — same data, `role` + `source_label`
  attrs set on all four.
- Test: a paired dataset exposes both `model_o3`/`obs_o3` and
  `<label>_o3` names with correct role/source_label attrs; values identical.

## R-2: Renderer variable resolution via source label

**Files:** `davinci_monet/plots/renderers/*.py` (timeseries, scatter, spatial_bias,
taylor, boxplot, …), `tests/test_plots.py`.

- Replace direct `obs_`/`model_` lookups with `resolve_source_variable(ds, var,
  label)`, falling back to the legacy prefix when no source-labelled var exists.
- Drive labels from the pair config (`pairs[*].sources` / `reference`); thread the
  pair spec into the plotting stage → renderer kwargs.
- Migrate `test_plots.py` assertions that hard-code `model_`/`obs_` var names.

## R-3: Role-based color + legend

**Files:** renderers + `tests/test_plots.py`, `tests/unit/plots/test_style.py`.

- Color each series with `get_color_for_role(ds[var].attrs.get("role"), index=i)`;
  same-role/role-less pairs cycle `NCAR_PALETTE` by source order.
- Legends use `source_label` instead of "Observed"/"Modeled".
- Verify obs-vs-model output is visually unchanged (obs gray, model blue); add
  coverage for a two-model pair getting two distinct palette colors.

## R-4: Source-label naming for the obs-only/single-source path

**Files:** `obs_*` renderers, `tests/test_obs_plots.py`.

- Obs-only plots already plot a single source; switch their labels/colors to
  `source_label` + role for consistency. Low risk (single series).

## R-5: Drop legacy prefixes + migrate fixtures (clean break tail)

**Files:** pairing assembly (stop emitting `model_`/`obs_`), all renderers, the
plot test suites, example configs (`examples/configs/*`, `analyses/*/configs/*`),
CLAUDE.md (variable-naming section).

- Remove the legacy `model_`/`obs_` aliases once R-2/R-3/R-4 read only source
  labels. Migrate remaining tests and example/gemini configs to `sources:` +
  `<label>_<var>` (use `migrate-config` for the YAML).
- This is the only intentionally-breaking step; gate it behind a green run of the
  full suite after R-2–R-4.

---

## Risks & notes

- **Blast radius:** ~1166 prefix references — move them per-renderer in R-2/R-3,
  not all at once. Each renderer + its tests in one commit.
- **Stat tables / CSV:** `StatisticsStage` output keys may also use `model_`/`obs_`
  prefixes; audit `stats/` before R-5 (out of scope for R-1).
- **Backward-compat:** keep the dual-naming bridge until R-5 so older paired
  datasets and any external consumers keep working.
- **Verification:** after each R-step, run the full suite in `davinci`
  (`HDF5_USE_FILE_LOCKING=FALSE python -m pytest`) and regenerate the ASIA-AQ
  obs plots to the iCloud folder to eyeball styling.

## Out of scope

- New plot types or new pairing geometries.
- The `model_registry`/`observation_registry`, `get_model`/`get_observation`, and
  `LoadModelsStage`/`LoadObservationsStage` shims — these are deprecated and can
  be deleted in a separate final cleanup once a full release cycle has passed.
