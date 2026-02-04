# DAVINCI-MONET Code Review (Codex)

**Summary**
Strong foundations: clear modular architecture (config, pipeline, pairing, plots, stats), disciplined typing, and a large test suite. The main risks are correctness gaps in pairing output construction, statistics integration, and a broken parallel pairing utility. Several pipeline behaviors diverge from documented config semantics, which can silently yield wrong results.

**High Severity Findings**
- Pairing can silently overwrite observation data when model and obs variable names match. All pairing strategies build a combined dataset by inserting obs vars then model vars into the same namespace, so identical names replace obs values. This then propagates into `create_paired_dataset`, yielding `obs_*` derived from model values. This is a correctness bug for mappings like `{"O3": "O3"}`. Affects `davinci_monet/pairing/strategies/point.py:258-303` and similar logic in `davinci_monet/pairing/strategies/profile.py`, `grid.py`, `swath.py`, `track.py`, plus `davinci_monet/pairing/engine.py:198-204`. Recommendation: keep obs/model datasets separate, or prefix model variables before merging, or pass distinct `obs_data` and `model_data` to `create_paired_dataset`.
- `ParallelPairingExecutor` is effectively broken. It passes `radius` and `time_tolerance` as kwargs to `PairingEngine.pair`, which does not accept them and will forward them to strategy calls, risking `TypeError` due to duplicate `time_tolerance` and ignoring `radius_of_influence`. See `davinci_monet/pipeline/parallel.py:230-235`. Either update to build a `PairingConfig` and pass via the `config=` parameter, or remove the utility if unused.
- Statistics output is not aligned with the statistics subsystem or config. `StatisticsStage` computes only mean bias, RMSE, correlation, and means, ignoring `stats.stat_list` and the richer metric set in `davinci_monet/stats/*`. `SaveResultsStage` then labels `NME_%` but computes it from RMSE, and emits `IOA` which is never computed. See `davinci_monet/pipeline/stages.py:918-990` and `davinci_monet/pipeline/stages.py:1305-1331`. This can materially misstate evaluation results.

**Medium Severity Findings**
- Time filtering widens the requested range by a full day, regardless of whether `end_time` includes a time-of-day. This can unintentionally include an extra 24 hours. See `davinci_monet/pipeline/stages.py:662-667`. Consider using inclusive slicing without offset, or only add a day when the input is date-only.
- Variable configuration options are partially ignored in the pipeline. `obs_min`, `obs_max`, `nan_value`, and `rename` are defined and implemented in `apply_variable_config`, but `LoadModelsStage` and `LoadObservationsStage` apply only `unit_scale`, `units`, and `display_name`. This can silently skip intended filtering and renaming. See `davinci_monet/models/base.py:171-213`, `davinci_monet/observations/base.py:167-210`, and `davinci_monet/pipeline/stages.py:334-355` and `davinci_monet/pipeline/stages.py:524-535`.
- `DataContainer.subset_bbox` assumes coordinate ordering is ascending. Many geospatial datasets store latitudes descending, which yields empty selections with `slice(lat_min, lat_max)`. See `davinci_monet/core/base.py:222-260`. Consider detecting coordinate order and swapping slice bounds accordingly.

**Low Severity Findings**
- Progress rendering sleeps 1 second for every parallel item completion, which can add minutes for large pair sets and negate parallelism. See `davinci_monet/pipeline/runner.py:933-939`. This should be optional or use a much smaller delay.
- Configuration docs and parsing disagree about pair naming and mapping direction. `ModelConfig.mapping` docstring states `{model_var: obs_var}` but examples and code use `{obs_var: model_var}`. `MonetConfig.validate_data_references` expects `obs_model` but examples use `model_obs`. See `davinci_monet/config/schema.py:253-255` and `davinci_monet/config/schema.py:670-713`.
- `LoadModelsStage` does not validate `files` presence; a missing path will fall through to `open_model` and raise a `TypeError` rather than a clear config error. See `davinci_monet/pipeline/stages.py:286-335` and `davinci_monet/models/generic.py:246-287`.

**Test Gaps**
- No test that pairing preserves obs values when obs and model variable names are identical. Add a unit test for each strategy or a shared test against `PairingEngine` with mapping like `{"O3": "O3"}`.
- No integration test that `stats.stat_list` or `StatisticsCalculator` metrics drive `StatisticsStage` output. Add a pipeline test that requests multiple metrics and verifies correct values and labels.
- No test validating the time range filter behavior when `end_time` includes a time-of-day.
- No test that pipeline respects `obs_min`, `obs_max`, `nan_value`, or `rename` from config.
- `ParallelPairingExecutor` has no coverage and would currently fail for `time_tolerance`.

**Recommendations**
1. Fix pairing namespace collisions by separating obs/model data or prefixing model variables before merging, then update `create_paired_dataset` usage.
2. Refactor `StatisticsStage` to use `StatisticsCalculator` and honor `stats.stat_list`; ensure `SaveResultsStage` uses correct definitions for NME and IOA.
3. Tighten time filtering semantics and add tests for inclusive end bounds with times.
4. Apply full variable configuration in load stages by calling `apply_variable_config` or mirroring its logic.
5. Either repair or remove `ParallelPairingExecutor` to avoid broken API surface.

