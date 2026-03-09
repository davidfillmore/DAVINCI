# PLAN: MODIS AOD Full Pipeline Integration

**Date**: 2026-03-09
**Branch**: `feature/modis`
**Status**: Revised after auditing the working implementation in `~/EarthSystem/MELODIES-MONET`

## Goal

Wire the working MODIS AOD smoke-test logic into the standard DAVINCI-MONET pipeline so it can be driven from YAML via `davinci-monet run`, while keeping the design close to the proven MELODIES-MONET workflow.

Secondary goals:
- Write gridded observation caches to NetCDF for reuse
- Preserve the new swath-grid primitive already added on this branch
- Keep the first production path simple enough to test end-to-end

## Current State In This Branch

Already implemented:
- `davinci_monet/pairing/grid_binning.py`
- `davinci_monet/pairing/strategies/swath_grid.py`
- Gridded spatial plotting support (`pcolormesh`) in the spatial renderers
- `analyses/modis-aod/scripts/smoke_test.py`, including Terra+Aqua combination and lon wrapping

Still missing for real pipeline support:
- A MODIS loader that matches the real `monetio.sat._modis_l2_mm` API
- Observation-stage dispatch that uses `sat_type` / observation readers instead of generic `xr.open_mfdataset()`
- Time-aware MODIS file subsetting before load
- A clean gridded observation contract for downstream pairing/stats/plots
- End-to-end tests for the actual pipeline path

## Reference Implementation: MELODIES-MONET

The working reference is in `~/EarthSystem/MELODIES-MONET`:

- `melodies_monet/driver.py:288-297`
  - For `sat_type == 'modis_l2'`, it first calls `subset_MODIS_l2(...)`
  - Then calls `mio.sat._modis_l2_mm.read_mfdataset(flst, variable_dict, ...)`
  - The loaded object is an `OrderedDict` of per-granule `xr.Dataset`s

- `melodies_monet/util/time_interval_subset.py:61-78`
  - `subset_MODIS_l2()` filters files by the timestamp encoded in `MOD04_L2` / `MYD04_L2` filenames
  - This is hourly filtering, not just day-level filtering

- `melodies_monet/driver.py:1071-1148`
  - `setup_obs_grid()` allocates `(time, lon, lat)` data/count arrays
  - `update_obs_gridded_data()` loops granules and bins each onto the target grid
  - `normalize_obs_gridded_data()` converts sums to means and creates the output dataset

- `examples/process_swath_data/control_modis_l2.yaml`
  - Uses `obs_type: 'sat_swath_clm'`
  - Uses `sat_type: 'modis_l2'`
  - Represents Terra and Aqua as separate observation labels
  - Uses normal `model.mapping` entries for pairing

### Takeaways From The Reference

1. MODIS support should be keyed off `sat_type: modis_l2`, not a brand-new geometry contract.
2. MODIS loading must not go through generic xarray file opening.
3. Time subsetting should happen before `read_mfdataset()` for performance.
4. Gridding is best treated as observation preprocessing.
5. The first DAVINCI implementation should stay close to the MELODIES config shape, then add merged Terra+Aqua behavior as a small extension.

## Recommended DAVINCI-MONET Design

### 1. Keep the YAML close to MELODIES-MONET

Do not introduce `obs_type: modis_l2` as a new primary contract.

Use:
- `obs_type: sat_swath_clm`
- `sat_type: modis_l2`

This matches:
- The existing schema field names
- The working MELODIES-MONET configuration
- The mental model that MODIS is a satellite subtype, not a new observation geometry

This also avoids inventing a second parallel config surface that the rest of the pipeline does not understand.

### 2. Add explicit observation-reader dispatch in `LoadObservationsStage`

Right now `LoadObservationsStage` mostly does generic file opening. That is the wrong seam for MODIS.

The stage should:
- Detect `sat_type`
- Route to a registered reader or dedicated satellite loader
- Only fall back to generic `xr.open_dataset()` / `xr.open_mfdataset()` when no specialized reader applies

Code sketch:

```python
def _load_observation_dataset(
    self,
    label: str,
    config: dict[str, Any],
    context: PipelineContext,
) -> xr.Dataset | None:
    obs_type = config.get("obs_type", "pt_sfc")
    sat_type = config.get("sat_type")

    if sat_type == "modis_l2":
        return self._load_modis_l2(label, config, context)

    if sat_type in observation_registry:
        reader_cls = observation_registry.get(sat_type)
        reader = reader_cls()
        files = self._resolve_files(config["filename"])
        return reader.open(files, variables=list(config.get("variables", {})))

    # Existing ICARTT / LMA / AERONET / generic xarray paths remain here.
```

### 3. Port the MELODIES file-subsetting helper

The current `_filter_files_by_date()` in DAVINCI only filters by `YYYYMMDD` in filenames. That is too coarse for MODIS and does not match the proven path.

Add a small MODIS-specific subset helper, probably near the reader:

```python
def subset_modis_l2_files(
    file_pattern: str,
    start_time: str,
    end_time: str,
) -> list[str]:
    """Filter MOD04/MYD04 files by hourly timestamps in the filename."""
```

Behavior should mirror MELODIES:
- Expand the glob first
- Build an hourly interval over the analysis window
- Match `M?D04_L2.AYYYYDDD.HH*.hdf`
- Return a sorted file list

This helper should be used before calling `monetio.sat._modis_l2_mm.read_mfdataset(...)`.

### 4. Rework the existing MODIS reader around the real `monetio` contract

The current `davinci_monet/observations/satellite/modis_l2_aod.py` assumes `read_mfdataset()` returns a single `xr.Dataset`. It does not.

The plan should be:
- Rework `modis_l2_aod.py`, or replace it with `modis_l2.py` plus a thin compatibility shim
- Accept `variable_dict`, not just a list of variable names
- Treat the `OrderedDict[str, xr.Dataset]` return shape as the source of truth
- Grid the granules before returning data to the pipeline
- Build the `monetio` `variable_dict` from DAVINCI variable config fields
  (`source_name`, `unit_scale`, `obs_min`, `obs_max`)

Code sketch:

```python
class MODISL2Reader:
    """Read MODIS L2 granules and grid them for pipeline use."""

    def read_granules(
        self,
        files: Sequence[str | Path],
        variable_dict: dict[str, dict[str, Any]],
        *,
        debug: bool = False,
    ) -> OrderedDict[str, xr.Dataset]:
        import monetio.sat._modis_l2_mm as modis_mod
        return modis_mod.read_mfdataset(list(files), variable_dict, debug=debug)

    def read_and_grid(
        self,
        files: Sequence[str | Path],
        variable_dict: dict[str, dict[str, Any]],
        lat_centers: np.ndarray,
        lon_centers: np.ndarray,
        start_time: str,
        end_time: str,
        time_resolution: str = "1D",
        min_obs_count: int = 1,
    ) -> xr.Dataset:
        granules = self.read_granules(files, variable_dict)
        return self._grid_granules(
            granules,
            lat_centers=lat_centers,
            lon_centers=lon_centers,
            start_time=start_time,
            end_time=end_time,
            time_resolution=time_resolution,
            min_obs_count=min_obs_count,
        )

def build_modis_variable_dict(
    variables: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for var_name, cfg in variables.items():
        source_name = cfg.get("source_name", var_name)
        result[source_name] = {
            "scale": cfg.get("unit_scale", 1.0),
            "minimum": cfg.get("obs_min"),
            "maximum": cfg.get("obs_max"),
        }
    return result
```

### 5. Grid during observation loading, not during pairing

This remains the right high-level decision, and it is also what the MELODIES implementation effectively does.

Why:
- MODIS enters as an `OrderedDict`, not a normal pipeline dataset
- Downstream pipeline stages already know how to work with gridded `xr.Dataset`s
- This keeps the pipeline contract clean: load observations -> pair -> stats -> plots

The gridding function should:
- Use the already-loaded model grid from `context.models[grid_source]`
- Build time edges from `analysis.start_time`, `analysis.end_time`, and `time_resolution`
- Use the new `grid_binning.py` helpers
- Track `obs_count`
- Return an `xr.Dataset` with dims `(time, lon, lat)`

Important detail:
- If the model longitude convention is `0..360`, shift MODIS longitudes from `-180..180` to `0..360`
- Preserve the smoke-test wraparound fix for points that land on the domain edge

Code sketch:

```python
def _grid_granules(...):
    time_edges, time_centers = build_time_grid(start_time, end_time, time_resolution)
    lat_edges = edges_from_centers(lat_centers)
    lon_edges = edges_from_centers(lon_centers)

    count_grid = np.zeros((ntime, nlon, nlat), dtype=np.int32)
    data_grid = np.zeros((ntime, nlon, nlat), dtype=np.float64)

    for granule_key, granule in granules.items():
        obs_timestamp = parse_modis_granule_key(granule_key)
        obs = granule[var_name].values.flatten().astype(np.float64)
        lat = granule["lat"].values.flatten().astype(np.float64)
        lon = granule["lon"].values.flatten().astype(np.float64)

        obs[(lat < -900) | (lon < -900)] = np.nan
        lon = normalize_lon_to_model(lon, lon_edges)
        time_flat = np.full(obs.size, obs_timestamp, dtype=np.float64)

        bin_swath_to_grid(
            time_edges, lon_edges, lat_edges,
            time_flat, lon, lat, obs,
            count_grid, data_grid,
        )

    normalize_grid(count_grid, data_grid)
```

### 6. Represent the post-gridded product as gridded observations

The config source remains `obs_type: sat_swath_clm`, but once gridding is complete the in-memory object should behave as a grid.

Recommended approach:
- Create the output dataset with `attrs["geometry"] = "grid"`
- Create the `ObservationData` wrapper with `obs_type="sat_grid_clm"` so its container geometry is also correct
- Preserve original metadata in attrs, e.g.:
  - `source_obs_type = "sat_swath_clm"`
  - `sat_type = "modis_l2"`
  - `grid_source = "cam6_base"`

This also implies a small fix in `ObservationData.geometry_from_obs_type()`:
- `sat_swath_clm` should map to `DataGeometry.SWATH`
- `sat_grid_clm` should map to `DataGeometry.GRID`

### 7. Use the normal `GridStrategy` after load-time gridding

This is the main design change from the earlier draft.

If the MODIS loader bins onto the target model grid, then pairing no longer needs a swath-specific algorithm. At that point the observation is a grid product.

Recommended pairing path:
- Loader returns gridded obs on the chosen target grid
- `PairingEngine` sees `geometry=GRID`
- Standard `GridStrategy` handles time alignment and model/obs assembly

Why this is preferable:
- It is closer to the MELODIES mental model
- It avoids a special `pairing_strategy` escape hatch in `PairingStage`
- It leaves `SwathGridStrategy` available for direct swath-pairing workflows and testing

Implication:
- `SwathGridStrategy` stays in the repo and remains useful
- It is not required for the first YAML-driven MODIS pipeline path

### 8. Cache the gridded output

This part of the original plan is still good.

Recommended config fields:
- `grid_source: cam6_base`
- `time_resolution: "1D"`
- `load_binned: true`
- `save_binned: true`
- `binned_file: /abs/path/modis_terra_binned.nc`

Behavior:
- If `load_binned` and the file exists, open the cached grid and skip HDF4 loading/binning
- Otherwise subset files, load granules, grid them, and optionally save the result

### 9. Start with separate Terra and Aqua obs entries

This is the biggest practical change from the previous plan.

The smoke test combines Terra+Aqua in one field, but the working MELODIES implementation uses one observation label per platform. That is the better first step for DAVINCI because it matches the existing config and pairing model.

Phase 1:
- `terra_modis` and `aqua_modis` are separate obs entries
- Each is gridded independently
- Each has its own cache file
- Each can be paired and plotted independently

Phase 2 (optional):
- Add a merge helper that combines two already-gridded products using `obs_count` weights
- Expose that as either:
  - a small utility script, or
  - a post-load synthetic observation product like `modis_combined`

Code sketch for weighted merge:

```python
def merge_gridded_obs(ds_a: xr.Dataset, ds_b: xr.Dataset, var: str) -> xr.Dataset:
    count_a = ds_a["obs_count"].fillna(0)
    count_b = ds_b["obs_count"].fillna(0)
    total = count_a + count_b

    merged = xr.where(
        total > 0,
        ((ds_a[var].fillna(0) * count_a) + (ds_b[var].fillna(0) * count_b)) / total,
        np.nan,
    )

    return xr.Dataset({var: merged, "obs_count": total}, coords=ds_a.coords)
```

## Proposed YAML: Phase 1

This shape is intentionally close to the working MELODIES config and closer to the current DAVINCI runtime contract.

```yaml
analysis:
  start_time: "2019-12-21"
  end_time: "2019-12-24"
  output_dir: /Users/fillmore/EarthSystem/DAVINCI-MONET/analyses/modis-aod/output
  log_dir: /Users/fillmore/EarthSystem/DAVINCI-MONET/analyses/modis-aod/logs

model:
  cam6_base:
    mod_type: generic
    files: /Users/fillmore/Data/CAM6/FCnudged_f09.mam.BaseMar27.2019_2021.001_AODVIS.nc
    mapping:
      terra_modis:
        AOD_550: AODVIS
      aqua_modis:
        AOD_550: AODVIS

  cam6_newdust:
    mod_type: generic
    files: /Users/fillmore/Data/CAM6/FCnudged_f09.mam.newdustMar282025.2019_2021.001_AODVIS.nc
    mapping:
      terra_modis:
        AOD_550: AODVIS
      aqua_modis:
        AOD_550: AODVIS

obs:
  terra_modis:
    obs_type: sat_swath_clm
    sat_type: modis_l2
    filename: /Users/fillmore/Data/MODIS/Terra/C61/2019/*/MOD04_L2.*.hdf
    grid_source: cam6_base
    time_resolution: "1D"
    load_binned: true
    save_binned: true
    binned_file: /Users/fillmore/EarthSystem/DAVINCI-MONET/analyses/modis-aod/data/terra_modis_binned.nc
    variables:
      AOD_550_Dark_Target_Deep_Blue_Combined:
        source_name: AOD_550_Dark_Target_Deep_Blue_Combined
        rename: AOD_550
        obs_min: 0.0
        obs_max: 10.0
        unit_scale: 0.001

  aqua_modis:
    obs_type: sat_swath_clm
    sat_type: modis_l2
    filename: /Users/fillmore/Data/MODIS/Aqua/C61/2019/*/MYD04_L2.*.hdf
    grid_source: cam6_base
    time_resolution: "1D"
    load_binned: true
    save_binned: true
    binned_file: /Users/fillmore/EarthSystem/DAVINCI-MONET/analyses/modis-aod/data/aqua_modis_binned.nc
    variables:
      AOD_550_Dark_Target_Deep_Blue_Combined:
        source_name: AOD_550_Dark_Target_Deep_Blue_Combined
        rename: AOD_550
        obs_min: 0.0
        obs_max: 10.0
        unit_scale: 0.001
```

## Optional Phase 2: Combined Terra+Aqua Product

Only add this after Phase 1 is working.

Possible approaches:
- Small analysis script that loads `terra_modis_binned.nc` and `aqua_modis_binned.nc` and writes `modis_combined_binned.nc`
- A pipeline-side merge helper that creates a synthetic observation label from multiple cached gridded obs inputs

This phase is where the smoke-test output should be matched exactly.

## Concrete Work Items

### A. Fix the observation geometry mapping

Files:
- `davinci_monet/observations/base.py`

Changes:
- Map `sat_swath_clm` -> `DataGeometry.SWATH`
- Map `sat_grid_clm` -> `DataGeometry.GRID`

### B. Add satellite-reader dispatch to the observation loading stage

Files:
- `davinci_monet/pipeline/stages.py`

Changes:
- Add a loader branch keyed by `sat_type`
- Route `sat_type == "modis_l2"` to a dedicated MODIS path

### C. Rework the MODIS reader to match the real API

Files:
- `davinci_monet/observations/satellite/modis_l2_aod.py` or `davinci_monet/observations/satellite/modis_l2.py`

Changes:
- Add `subset_modis_l2_files()`
- Use `variable_dict`
- Read an `OrderedDict`
- Grid to `(time, lon, lat)`
- Preserve `obs_count`

### D. Add binned-cache support

Files:
- Same MODIS reader module
- `davinci_monet/config/schema.py`

Changes:
- Add explicit schema fields for `grid_source`, `time_resolution`, `load_binned`, `save_binned`, `binned_file`
- Load cache if present
- Save cache after gridding

### E. Keep pairing on the normal grid path

Files:
- `davinci_monet/pairing/strategies/grid.py` if any small tweaks are needed
- Possibly `davinci_monet/pipeline/stages.py` if we need `regrid_to="obs"` explicitly

Changes:
- Prefer normal `GridStrategy`
- Do not add a `SWATH_GRID` pseudo-geometry unless a real blocker appears

### F. Add example configs and a thin run script

Files:
- `analyses/modis-aod/configs/modis-aod-cam6-gemini.yaml`
- `analyses/modis-aod/scripts/run_evaluation.py`

### G. Add end-to-end tests for the real pipeline seam

Files:
- New pipeline integration tests

Tests should cover:
- `subset_modis_l2_files()` filename filtering
- MODIS loader reading an `OrderedDict` from a monkeypatched `monetio` call
- Load stage producing a gridded dataset with `obs_count`
- Pairing stage using the normal grid path successfully
- Cache load/save behavior

## Implementation Order

| Step | Description | Status |
|------|-------------|--------|
| 1 | `grid_binning.py` and `SwathGridStrategy` primitive | DONE |
| 2 | Gridded spatial plotting support | DONE |
| 3 | Smoke-test reference workflow | DONE |
| 4 | Geometry mapping fix for `sat_swath_clm` / `sat_grid_clm` | TODO |
| 5 | Observation-stage `sat_type` dispatch | TODO |
| 6 | Real MODIS loader with file subsetting and gridding | TODO |
| 7 | Binned-cache support + schema fields | TODO |
| 8 | Example YAML and run script | TODO |
| 9 | Pipeline integration tests | TODO |
| 10 | Optional Terra+Aqua merged product | LATER |

## Testing and Validation

Required before calling MODIS support complete:
- Unit test for `subset_modis_l2_files()`
- Unit test for gridding one synthetic MODIS granule set onto a model grid
- Integration test for `LoadObservationsStage` with monkeypatched `monetio`
- Integration test for full `run_analysis(...)` on a synthetic config
- Real-data spot check against the smoke-test outputs for 2019-12-21 through 2019-12-23

Acceptance criteria:
- YAML-driven run succeeds without using the standalone smoke-test script
- Cached NetCDF output reloads correctly
- The gridded AOD field and `obs_count` are consistent with the smoke-test path
- Terra and Aqua each work independently
- Combined Terra+Aqua output, if implemented, matches the smoke-test logic

## Non-Goals For The First Pass

- Do not bypass the standard pipeline stages
- Do not rely on generic xarray HDF opening for MODIS HDF4
- Do not introduce a brand-new `obs_type: modis_l2` contract unless the existing `sat_type` design proves inadequate
- Do not make combined Terra+Aqua support a blocker for initial pipeline integration
