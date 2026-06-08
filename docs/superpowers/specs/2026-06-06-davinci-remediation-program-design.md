# DAVINCI Remediation & Migration-Completion — Program Design

**Date:** 2026-06-06 · **Branch:** `develop` · **Status:** approved (design) · **Author:** architecture review follow-up
**Source review:** `REVIEW.md` (repo root)
**Supersedes stance of:** `docs/superpowers/plans/2026-06-06-complete-source-unification.md` (keeps legacy forever → now to be deleted)
**References:** `docs/superpowers/specs/2026-06-06-renderer-unification-design.md`, `docs/superpowers/specs/2026-05-30-unified-data-source-design.md`

---

## 1. Context & motivation

`REVIEW.md` found DAVINCI has the right target architecture but is **caught mid-migration**: two unifications (model/obs → "data source"; per-renderer → `render(series)`) were landed *additively*. The new abstractions exist next to the old ones, both are live, and the result is **systematic forking** at every layer, accumulated **dead/vestigial code**, and **new "going-forward" paths that are silently incomplete** (the documented `sources:` path does *less* than the deprecated `model:`/`obs:` path without telling the user).

This program finishes the migration and addresses every issue in the review. The guiding thesis from the review: **the highest-leverage work is subtractive — finish each migration by deleting its legacy half, and make the new paths fail loudly when they cannot do the job.**

### Ratified decisions (see §9 decision log)
1. **Hard break now.** Delete the legacy side outright (legacy config schema, `ModelData`/`ObservationData`, legacy load/pair/plot paths, dual `pair`/`plot` contracts, `obs_`/`model_` vocabulary). No permanent compat shims. `migrate-config` is retained as a one-time converter; a legacy `model:`/`obs:` config passed to `run` must fail with a clear "run migrate-config" error, never silently. This is a **breaking release**, flagged prominently in the CHANGELOG (the project uses CalVer `26.03`, so the break is marked conspicuously rather than via a semver major bump).
2. **Master program plan + per-workstream sub-plans.** This spec is the program roadmap; each workstream gets its own implementation plan via `superpowers:writing-plans`, starting with WS1.

---

## 2. Goals & non-goals

**Goals**
- One of each: loader, data container, pairing path, plot contract, naming vocabulary, reader base.
- The unified `sources:`/`pairs:` config is the *only* runtime config; legacy is convertible but not runnable.
- No dead code, no ghost bytecode, no vestigial "looks-done" exports.
- No silent success-with-zero-output; consistent stage failure semantics; per-stage errors surfaced.
- HDF5-thread-safe behavior **by default**.
- Every core module < 500 lines or a documented exception.
- CLAUDE.md / MEMORY claims are all true (mypy, type-safe config, renderer unification, `_extract_surface` location, test count).

**Non-goals (YAGNI)**
- No new science, readers, plot types, or metrics. (Behavior-preserving refactor + bug/safety fixes only.)
- No change to scientific outputs except where the review identified a correctness bug.
- No dependency-stack change (stay on monet/monetio, pandas 1.x) — that is a separate future effort.
- No performance rewrite beyond the one identified hotspot (2-D nearest-neighbour → KD-tree) and HDF5 defaults.

---

## 3. Target end-state architecture

| Concern | Today (forked) | Target (single) |
|---|---|---|
| Config | `model:`/`obs:` **and** `sources:`/`pairs:` | `sources:`/`pairs:` only; `migrate-config` converts legacy offline |
| Container | `ModelData` + `ObservationData` + `SourceData` | one `SourceData` (geometry-tagged, role optional) |
| Loading | `LoadModelsStage` + `LoadObservationsStage` + `LoadSourcesStage` | `LoadSourcesStage` only |
| Reader base | per-reader copy/paste (6×) | `BaseReader` ABC; readers satisfy `SourceReader` (incl. `time_range`, set `attrs['geometry']`) |
| Pairing API | `pair(model, obs)` + `pair_sources(reference, comparand)` | `pair_sources` only (legacy adapter deleted) |
| Pairing execution | source-jobs + legacy inline ThreadPool + dead `parallel.py` | one orchestration, HDF5-safe by default |
| Plot contract | `plot(paired, obs_var, model_var)` + partial `render(series)` | `render(series)` only |
| Paired var names | `obs_`/`model_` intermediate, renamed to `<label>_<var>` | pairing emits `<label>_<var>` + `role`/`pair_role`/`source_label` attrs directly (no `obs_`/`model_` step) |
| Protocols | `ModelReader`/`ObservationReader` + `SourceReader` | `SourceReader`/`SourceProcessor` only |
| Registry | unified `source_registry` ✓ (already done) | unchanged |

---

## 4. Approach & sequencing

**Approach A — Complete → Delete → Clean up.** Make the unified path feature-complete and the renderer contract singular *before* deleting legacy; then dedup/harden/split once behavior is settled.

```
Phase 0  ▶ WS0  Repo hygiene & doc honesty                 (start now; no deps)
Phase 1  ▶ WS1  Unified sources path = feature-complete   ─┐ foundation
         ▶ WS2  Renderer unification = one contract        ─┘ (WS2 paired-routing needs WS1)
Phase 2  ▶ WS3  HARD BREAK: delete the legacy side          (keystone; needs WS1 + WS2)
Phase 3  ▶ WS4  Reader dedup (BaseReader)   ║  WS5  Correctness & safety   (after WS3)
Phase 4  ▶ WS6  Tests & CI hardening        ║  WS7  God-file split + typed events  (last)
```

Rationale: never delete a legacy path until the unified path provably replaces it (WS1/WS2 add the regression tests that make WS3 safe). WS7 (restructuring) is last because splitting files that WS3 will gut is wasted work. WS0 is independent and can ship immediately as its own PR.

---

## 5. Workstreams

Each workstream below is the scope for one implementation plan (`writing-plans`). Format: Goal · In scope · Out of scope · Key changes (with file anchors from the review) · Dependencies · Risk · Acceptance.

### WS0 — Repo hygiene & doc honesty  (size S, risk very low)
- **Goal:** Remove clutter and make stale docs true; zero behavioral change.
- **In scope:** delete ghost bytecode `davinci_monet/{addons,radiative,daemon}/` (560K, 0 `.py`, untracked, importable shadows); stop tracking `.coverage`/`.DS_Store` and gitignore them; delete unambiguously-dead exports with no entanglement — `pairing_registry` (never populated/read), `create_paired_dataset` (no prod caller; engine uses `_assemble_paired_dataset`), the `ConfigMigration` version chain (`detect_config_version`/`_migrate_*`/`validate_version_compatibility`, never called by parser), `TeeWriter`; fix purely-stale docs (CLAUDE.md "961 tests" → current count; `_extract_surface` "in base.py" → pairing strategy location; add `ai/` to the architecture overview).
- **Out of scope:** anything entangled with the legacy teardown (`ModelData`, `expand_sources_to_legacy`, `logging/`, `io/`) — those move in WS3/WS4/WS6.
- **Deps:** none. **Risk:** very low. **Acceptance:** `pytest`/`mypy`/`black`/`isort` green; `import davinci_monet.daemon` no longer resolves; grep shows the deleted exports gone from `__init__` and tests.

### WS1 — Unified `sources:` path is feature-complete  (size L, risk medium)
- **Goal:** The `sources:` path does everything the legacy path does, so deleting legacy in WS3 loses nothing. Basis: the existing `2026-06-06-complete-source-unification.md` (execute its feature-completion tasks; drop its "keep legacy forever" framing).
- **In scope:** typed `PairConfig` (unified `sources`/`reference`/`variables`) with validation; single source (`role: model|obs|none`) → descriptive stats + supported single-source plots keyed by **source label** not obs key; **pairing failures FAIL the `pairing` stage** (no COMPLETED-with-zero-pairs); apply `resample`/`min_obs_count`/`track_obs_count` in `LoadSourcesStage` (the silent regression, `stages.py:1340` vs legacy `stages.py:835`); make **MODIS-L2 reachable via `sources: type:`** (register a swath→grid binning source; today `MODISL2Reader` is reachable only via the legacy `sat_type` branch `stages.py:952`); route `spatial_overlay` field hydration and MODIS grid targets through `context.sources`; output/CSV naming uses source labels + reference/comparand.
- **Out of scope:** deleting legacy code (WS3); renderer migration (WS2); broad error-semantics (WS5) beyond the specific pairing-fail rule.
- **Deps:** WS0. **Risk:** medium (touches load/pair/stats/plot wiring). **Acceptance:** new regression tests (single-source stats/plot via `source:`; unsupported pair fails; grid-grid obs-obs pair; high-frequency source resamples under `sources:`; MODIS-L2 via `type:`) pass through `PipelineRunner.run_from_config`.

### WS2 — Renderer unification: `render(series)` is the only contract  (size L, risk medium)
- **Goal:** Retire the `plot()`-vs-`render()` fork; one geometry-aware contract.
- **In scope:** migrate the remaining renderers (~17: scatter, boxplot, diurnal, taylor, scorecard, curtain, spatial/{bias,distribution,overlay}, the timeseries family, track_map_3d, flight_track) to override `render(series)`; **route the paired pipeline path through `render`** and delete the introspection fork (`stages.py:2476`, `:2923`); add a **central geometry classifier** and make `scatter` (and point-only renderers) refuse/redirect gridded input (`plots/renderers/scatter.py` has no guard today); replace inline renderer metric computation with `stats/` functions (kills display/CSV divergence); refresh the stale `registry.py` category frozensets.
- **Out of scope:** deleting the `plot()` methods themselves (WS3, once paired path no longer calls them); the god-module split of `plots/base.py` (WS7).
- **Deps:** WS1 (paired routing needs unified paired naming). **Risk:** medium (visual output regressions). **Acceptance:** paired + single-source plots both flow through `render`; `get_plotter(name).render(series)` works for every registered name and series count; gridded-into-scatter is rejected with a clear error; image-smoke tests pass; renderer-reported stats equal `stats/` values.

### WS3 — Hard break: delete the legacy side  (size L, risk HIGH — keystone)
- **Goal:** Collapse every fork by deleting the legacy half.
- **In scope:** remove `ModelConfig`/`ObservationConfig` and the `model:`/`obs:` blocks from `MonetConfig`; add explicit legacy-config **detection → `ConfigurationError`** pointing to `migrate-config` (configs are `FlexibleModel`, so absent fields are silently ignored — must detect, not rely on validation); collapse `ModelData`+`ObservationData` → one `SourceData`/container; delete `LoadModelsStage`/`LoadObservationsStage` and fix `PipelineBuilder` to wire `LoadSourcesStage` (`runner.py:1997`); delete the legacy inline pairing path and the `pair(model,obs)` API + `BasePairingStrategy.pair`/protocol `pair`; make pairing **emit `<label>_<var>` + attrs directly** (delete the `obs_`/`model_` intermediate and the dual-resolution machinery in `core/base.py:374-625`, `_assemble_paired_dataset`); delete `plot(paired, obs_var, model_var)` once WS2 lands; remove `obs_`/`model_` fallbacks, `get_obs`/`get_model`, `model_label`/`obs_label`, `context.models`/`observations`, `get_model`/`get_observation`; delete `ModelReader`/`ObservationReader`/legacy processors from `protocols.py`; delete `expand_sources_to_legacy`; flag the breaking release in the CHANGELOG (CalVer — mark the break conspicuously, do not rely on a semver major).
- **Out of scope:** `BaseReader` extraction (WS4); concurrency/eval/error-semantics (WS5); file splitting (WS7). (Deletion only — restructuring follows.)
- **Deps:** WS1 **and** WS2 (and their regression tests). **Risk:** high — the big subtractive change. **Mitigation:** land behind a branch; rely on the WS1/WS2 test net; golden-output comparison of example runs before/after; do deletions in reviewable slices (config → container → stages → pairing → plot contract → vocabulary).
- **Acceptance:** grep confirms removal of `ModelData`/`ObservationData`/`LoadModelsStage`/`LoadObservationsStage`/`pair(model`/`get_model`/`obs_`+`model_` prefixes/`expand_sources_to_legacy`; legacy config → clear error; full suite green; example analyses produce equivalent outputs.

### WS4 — Reader-layer dedup  (size M, risk medium)
- **Goal:** One reader skeleton; finish the `SourceReader` contract.
- **In scope:** `BaseReader` ABC owning `open()`, the 3-attempt monetio→xarray fallback, variable selection, dim/coord standardization, and glob handling — subclasses supply only a monetio-module id, a rename table, and reader kwargs (removes the ~6× open/standardize and ~7× glob duplication across `models/` + `observations/`); readers accept `time_range` and set `attrs['geometry']` with **one encoding** (fix the int-`.value`/string/`.name.lower()` inconsistency); one variable-config implementation (the two — `models/base.py:172`, `stages.py:1431` — collapse once WS3 leaves one container) and one surface-extractor (delete the dead `ModelData.extract_surface`; keep the live strategy copy or hoist to a shared helper — resolves the 4×-rediscovered CESM duplication); decide `get_variable_mapping`/`*_VARIABLE_MAPPING`: wire into loading or delete as vestigial.
- **Out of scope:** new readers; reader behavior changes.
- **Deps:** WS3 (single container). **Risk:** medium (all readers touched). **Acceptance:** reader files shrink materially; one geometry encoding; `cesm.py` < 500; reader unit tests green; capability negotiation no longer needs `inspect.signature` (`stages.py:1388`).

### WS5 — Correctness & safety  (size M, risk medium)
- **Goal:** Stop silent degradation; make concurrency safe by default.
- **In scope:** one stage failure rule (no COMPLETED-with-all-failures; with the legacy pairing path removed in WS3, ensure the remaining path fails loudly — the old COMPLETED-on-all-fail behavior at `stages.py:1831` is gone); surface per-stage error lists in `PipelineResult` (today buried in `context.metadata`); stop swallowing stats exceptions to NaN silently (`calculator.py:429`,`:598` — log + count); make `validate()`-as-skip distinguish "nothing to do" from "misconfigured" (`runner.py:1922`); **HDF5-safe concurrency by default** (single Dask worker / locking off) instead of post-crash advice (`stages.py:1712`,`:1831`); collapse the three pairing orchestrations to one (adopt or delete `parallel.py`); replace `eval()` in `compute_derived_variable` (`models/base.py:434`) with an allowlisted evaluator; evaluate tightening config to `extra="forbid"` (decision at plan time).
- **Out of scope:** the pairing-fail rule already in WS1; the scatter guard already in WS2.
- **Deps:** WS3 (coordinate concurrency edits with the pairing-path deletion). **Risk:** medium. **Acceptance:** a run with every plot/pair failing reports `success=False` with errors in `PipelineResult`; default run is HDF5-safe without env vars; `eval` removed; one pairing executor.

### WS6 — Tests & CI hardening  (size M, risk low)
- **Goal:** Make the quality gates real and cover the final architecture.
- **In scope:** enable stricter mypy (incrementally: `disallow_untyped_defs`, etc.) and install the package in the typecheck job (today `--ignore-missing-imports` on an uninstalled package → cosmetic); raise the coverage floor above 67%; add a Python matrix (3.11/3.12); rename/relocate the 6 mislabeled `*Integration` classes (`test_pairing.py:834`, `test_plots.py:1415`, `test_stats.py:579`, `test_model_readers.py:740`, `test_cli.py:536`, `test_swath_grid_strategy.py:206`) and register an `integration` marker; finish the `unit/` mirror (move flat top-level tests; delete empty `unit/{pairing,stats}` stubs); add public `sample_obs_from` to `scenarios.py` (remove the 5× private-method coupling); add focused tests for `pairing/parallel`/`grid_binning`/`ProfileStrategy`/`SwathStrategy`/`util`; add `ai` deps to the `dev` extra (or guard `ai` tests).
- **Out of scope:** rewriting existing good tests.
- **Deps:** after WS3 (test the final shape; CI-job fixes may land earlier). **Risk:** low. **Acceptance:** stricter mypy green in CI against the installed package; coverage gate raised; matrix green; no `*Integration` class bypasses the pipeline.

### WS7 — God-file decomposition & typed events  (size L, risk low-to-correctness/high-churn — last)
- **Goal:** Bring core modules under the size budget and remove the stringly-typed display coupling.
- **In scope:** split `stages.py` (3203) → `stages/{load,pair,stats,plot,io,summary}.py`; extract `runner.py` (2107) UI → `pipeline/display.py` (ProgressFormatter) + `pipeline/reporting.py` (LogCollector); split `plots/base.py` (1106) → config/plotter+contract/labels+tables/series modules; split `config/schema.py` (925), `core/base.py` (898), and over-budget renderers; **replace the progress-string protocol with typed events** so stages emit structured events instead of prefix-matched strings the runner regex-parses back (`runner.py:1641-1710`) — removes the dispatcher and the stage↔display layering violation; consolidate the duplicated `_normalize_var_configs` (`stages.py:505,652,1297`), number/duration formatters, `_get_metric`, and `unwrap_to_dataset` helpers; relocate `compute_tropospheric_column` out of the reader layer; resolve the orphaned `logging/` (376 lines — adopt by wiring `configure_logging` into the runner/CLI bootstrap, or delete) and trim the dead ~600 lines of `io/` to the surface actually used (`write_dataset`).
- **Out of scope:** behavior changes (pure restructuring).
- **Deps:** WS3 (don't split code you're deleting); pairs with WS5/WS6. **Risk:** low to correctness, high churn. **Acceptance:** every targeted module < 500 lines (or documented exception); no progress strings parsed by regex; helper duplications gone; suite green.

---

## 6. Cross-cutting concerns

- **Versioning & release.** WS3 is a breaking change → write a prominent CHANGELOG BREAKING entry (project uses CalVer `26.03`; mark the break conspicuously, no silent acceptance), document the `migrate-config` path in CLAUDE.md/README. Legacy configs error with a migrate hint (never silently ignored).
- **Backward-compatibility boundary.** Retained: `migrate-config` (offline legacy→unified conversion via `migrate_to_sources`) and registry plot-type aliases (e.g. `obs_timeseries`→`timeseries`, harmless naming). Removed: every *runtime* legacy path, container, contract, and the `obs_`/`model_` variable vocabulary.
- **Testing strategy.** Per CLAUDE.md: integration tests run through `PipelineRunner.run_from_config`; present test design before writing; no green-checkmark shortcuts. WS1/WS2 add the regression net *before* WS3 deletes. Validate in the `davinci` conda env with `HDF5_USE_FILE_LOCKING=FALSE python -m pytest`.
- **Gates each WS must pass:** `pytest`, `mypy davinci_monet`, `black --check`, `isort --check` in `davinci`.
- **Plots output:** generated example plots go to the iCloud Claude folder per CLAUDE.md.
- **Commits/merges:** never auto-commit/push or merge to `main` without explicit user confirmation; return to `develop` after any approved merge.

## 7. Program Definition of Done

One loader, one container, one pairing path, one plot contract, one naming vocabulary; legacy config rejected with a migrate hint; no ghost/dead code; no silent success-with-zero-output; HDF5-safe by default; every core module < 500 lines (or documented exception); `pytest` + stricter `mypy` + `black`/`isort` green in the `davinci` env; all CLAUDE.md/MEMORY claims true.

## 8. Relationship to existing specs/plans

- `2026-06-06-complete-source-unification.md` → the **basis for WS1**; its feature-completion tasks are reused, its "keep legacy as permanent shim" framing is replaced by WS3 deletion.
- `2026-06-06-renderer-unification-design.md` → reference for **WS2**; this program completes what that design started (the review found it incomplete despite the "COMPLETE" MEMORY note).
- This program is the umbrella; each WS produces its own plan under `docs/superpowers/plans/`.

## 9. Decision log & deferred sub-decisions

**Decided (this session):**
- Legacy handling: **hard break now** (delete; no permanent shims). `migrate-config` retained.
- Deliverable: **master program plan + per-workstream sub-plans.**
- Sequencing: **Approach A** (complete → delete → clean up); WS7 last.

**Deferred to each workstream's plan:**
- `parallel.py`: adopt as the one executor vs delete (WS5).
- `get_variable_mapping`/`*_VARIABLE_MAPPING`: wire vs delete (WS4).
- Config strictness: `extra="forbid"` vs keep flexible (WS5).
- Exact mypy strictness ramp and coverage floor target (WS6).
- Whether to fold WS0 into the first PR or ship standalone (recommended standalone).
