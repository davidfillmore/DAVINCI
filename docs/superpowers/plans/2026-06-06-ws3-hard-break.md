# WS3 — Hard-Break: Delete the Legacy model/obs Half

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Each phase = a verified, committed checkpoint (revert point). `pytest`+`mypy`+`black`+`isort` green in the `davinci` env (`HDF5_USE_FILE_LOCKING=FALSE`) before each commit. This is a BREAKING release — flag in CHANGELOG; `migrate-config` retained; a legacy `model:`/`obs:` config must ERROR (not run).

**Program context:** WS3 of `docs/superpowers/specs/2026-06-06-davinci-remediation-program-design.md` (keystone). Prereqs WS1+WS2 are complete (unified path feature-complete; renderers on `render(series)`). Deletion order below is derived from a full dependency map (2026-06-06).

**MUST KEEP (used by unified path / CLI):** `resample_dataset` (module fn), `migrate_to_sources`, `detect_config_version`, `check_deprecated_fields`, `LegacyConfigWarning` (until Phase 6), `SourceReader`/`SourceProcessor`, `source_registry`, all reader CLASSES (their `open()` returns `xr.Dataset` and the unified loader uses them).

---

## Deletion order (each phase keeps the suite green)

**Phase 1 — No-dependency deletions (safe first):**
- Delete legacy protocols `ModelReader`/`ObservationReader`/`ModelProcessor`/`ObservationProcessor` (`core/protocols.py`) + `core/__init__` exports + `tests/unit/core/test_protocols.py` legacy cases.
- Delete `model_registry`/`observation_registry` aliases (`core/registry.py:311,314`) + `core/__init__` exports + alias tests in `test_registry.py`.
- Delete `expand_sources_to_legacy` (`config/migration.py`) + `config/__init__` export + `tests/unit/config/test_expand_sources.py`.
- Delete the dead `"obs_plotting"` stage-name check (`runner.py:~1804`).

**Phase 2 — Internal refactors that unblock deletion:**
- Pairing: move each of the 5 strategies' `pair()` body into `pair_sources()` (or make `pair_sources` the real impl); reduce `pair()` to a private/compat shim; delete `PairingEngine.pair()` legacy method; migrate `parallel.py` + `test_pairing.py` callers to `pair_sources`.
- Plots: migrate the module-level `plotter.plot()` convenience fns (boxplot/curtain/diurnal/flight_timeseries) and the `plot_per_flight` callers (scatter/track_map_3d) to `render()`; then drop `BasePlotter.plot()` abstract + `render()` shim body. (Renderer `plot()` thin-wrappers may remain until their tests migrate; full removal optional.)

**Phase 3 — Remove the legacy loader/container layer (largest):**
- Confirm reader CLASSES' `open()` return `xr.Dataset` (unified path uses them) — they STAY. Delete the legacy wrapping: `create_model_data`/`open_model`+per-model `open_*` convenience fns and `ModelData` (extract CESM `_extract_surface` + any still-needed transform into a shared util first); same for `create_observation_data`/`open_*` + `ObservationData` (keep module-level `resample_dataset`).
- Write/confirm a `sources:`-native MODIS-L2 path before deleting `_load_modis_l2`/`MODISL2Reader` (swath via SwathGridStrategy already works; port binned-file caching only if a config needs it — else drop).
- Delete `LoadModelsStage` + `LoadObservationsStage`; collapse `LoadSourcesStage` delegation block; fix `_has_data_to_process`/`iter_single_source_datasets` to use `context.sources`.

**Phase 4 — Pairing/plotting stage legacy branches:**
- Delete `PairingStage` inline model×obs loop + `_build_source_pair_jobs` legacy branch; delete `PlottingStage` legacy pair-spec handling; delete `PipelineBuilder.add_models/add_observations`.

**Phase 5 — Config schema:**
- Delete `SourcePairConfig` legacy fields (`model`/`obs`/`variable`) + legacy validator branch; delete `MonetConfig.model`/`obs` fields + `parse_models`/`parse_observations` + `validate_data_references` legacy branch + `get_model_obs_pairs`; delete `ModelConfig`/`ObservationConfig`.
- Set `MonetConfig` `extra="forbid"` (or a root validator) so a `model:`/`obs:` config ERRORS with a "run `davinci-monet migrate-config`" hint. Update `cli/commands/validate.py` to read `config.sources`.

**Phase 6 — Vocabulary cleanup:**
- Remove `context.models`/`context.observations` + `get_model`/`get_observation` from `PipelineContext` + fallbacks; remove `PairedData.get_obs`/`get_model`/`model_label`/`obs_label`/`model_variables`/`obs_variables` aliases.
- Make unified pairing emit `<label>_<var>` directly (adjust `_assemble_paired_dataset` + `tag_paired_roles`) and drop the `obs_`/`model_` prefix fallbacks in `paired_variable_role`/`paired_canonical_name`/`_resolve_role_var` — ONLY after confirming no untagged prefixed paired data is produced anywhere.

---

## Acceptance (WS3 DoD)
- No `ModelConfig`/`ObservationConfig`/`ModelData`/`ObservationData`/`LoadModelsStage`/`LoadObservationsStage`/legacy pairing path/`pair(model,obs)`/`expand_sources_to_legacy`/legacy protocols/registry aliases remain.
- A legacy `model:`/`obs:` config → `ConfigurationError` pointing to `migrate-config`; `migrate-config` still converts it.
- One container, one loader, one pairing path, one plot contract, source-label vocabulary.
- Full suite + mypy + black/isort green; example analyses produce equivalent outputs. CHANGELOG notes the breaking change.
