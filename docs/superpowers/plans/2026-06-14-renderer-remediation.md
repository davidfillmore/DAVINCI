# Renderer Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Act on the 2026-06-14 renderer audit — drop the multipanel timeseries renderers, fold `spatial_distribution` into the single-source `spatial` renderer, apply the P1 correctness/loose-end fixes, and the P2 DRY consolidations.

**Architecture:** Three phases. (A) **Drops/fold** — delete `site_timeseries`/`flight_timeseries`/`per_site_timeseries` and `spatial_distribution`, with all registry/export/test/config/example references, migrating the two folded usages to `spatial`. (B) **P1** — contract guards + loose-end fixes. (C) **P2** — extract shared helpers (spatial coords/time-average, paired-series extraction, track coords) and route renderers through them; behavior-preserving. The full test suite + mypy + black/isort + "all configs load" + "examples run" are the gate at every commit.

**Tech Stack:** Python 3.11/3.12, xarray, numpy, matplotlib/cartopy, pydantic, pytest, mypy, black, isort. `davinci` conda env.

**Audit:** `docs/audits/2026-06-14-renderer-audit.md`

---

## Conventions (every task)
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
```
**The gate** (run at the end of each task):
```bash
cd /Users/fillmore/EarthSystem/DAVINCI
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && mypy davinci_monet && black --check davinci_monet && isort --check-only davinci_monet
```
(If a bare pytest run hits an HDF5/numba teardown segfault, re-run with `DASK_NUM_WORKERS=1 HDF5_USE_FILE_LOCKING=FALSE` per CLAUDE.md gotcha #8.)
**Configs-load check** (Tasks 1, 2, 7):
```bash
python - <<'PY'
import glob
from davinci_monet.config.parser import load_config
bad=[]
for f in glob.glob("analyses/**/*.yaml", recursive=True)+glob.glob("examples/**/*.yaml", recursive=True):
    try: load_config(f)
    except Exception as e: bad.append((f, str(e).splitlines()[-1][:80]))
print("FAILS:", *bad, sep="\n") if bad else print("all tracked configs load")
PY
```
(Two configs fail to load for PRE-EXISTING unrelated reasons — `wrfchem-forecast.example.yaml` env-var date template, `cmaq_airnow.yaml` malformed line 15 — ignore those; only worry about NEW failures.)
**Examples-run check** (Tasks 1, 2): `HDF5_USE_FILE_LOCKING=FALSE python examples/run_all_examples.py 2>&1 | tail -5` must finish with no traceback.
Commit footer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. No push/merge.

---

## Task 1: Drop the 3 multipanel timeseries renderers

Remove `site_timeseries`, `flight_timeseries`, `per_site_timeseries` entirely — they render per-site/per-flight subplot grids and are being dropped.

**Files:**
- Delete: `davinci_monet/plots/renderers/{site_timeseries,flight_timeseries,per_site_timeseries}.py`
- Delete: `davinci_monet/tests/test_per_site_timeseries.py`
- Modify: `davinci_monet/plots/registry.py:150`, `davinci_monet/plots/renderers/__init__.py`, `davinci_monet/plots/__init__.py`, `davinci_monet/tests/test_plots.py`, `davinci_monet/tests/test_integration.py`, `davinci_monet/tests/unit/plots/test_unification_comparison_renderers.py`
- Modify: `analyses/firex-aq/configs/firex-aq-dataset-dc8.example.yaml` (remove the 2 `flight_timeseries` plot specs at ~lines 121, 128)
- Delete: `examples/plot_11_site_timeseries.py`, `examples/plot_12_flight_timeseries.py`; update `examples/README.md` + any cross-reference in `examples/plot_14_satellite_swath.py`.

- [ ] **Step 1: Delete the renderer files + dedicated test**
```bash
cd /Users/fillmore/EarthSystem/DAVINCI
git rm davinci_monet/plots/renderers/site_timeseries.py \
       davinci_monet/plots/renderers/flight_timeseries.py \
       davinci_monet/plots/renderers/per_site_timeseries.py \
       davinci_monet/tests/test_per_site_timeseries.py \
       examples/plot_11_site_timeseries.py \
       examples/plot_12_flight_timeseries.py
```

- [ ] **Step 2: Remove all references**
- `registry.py:150` — change `TEMPORAL_PLOTS = frozenset({"timeseries", "diurnal", "per_site_timeseries", "site_timeseries", "flight_timeseries"})` → `frozenset({"timeseries", "diurnal"})`.
- `plots/renderers/__init__.py` and `plots/__init__.py` — delete the `from ...flight_timeseries/per_site_timeseries/site_timeseries import ...` blocks and the matching `"plot_flight_timeseries"`/`"plot_per_site_timeseries"`/`"plot_site_timeseries"` `__all__` entries.
- `test_plots.py`, `test_integration.py`, `test_unification_comparison_renderers.py` — remove the test classes/functions/parametrize entries that reference the three dropped renderers (grep each for `site_timeseries|flight_timeseries|per_site_timeseries` and delete those tests/specs).
- `firex-aq-dataset-dc8.example.yaml` — delete the two `flight_timeseries` plot blocks (keys at ~121, ~128).
- `examples/README.md` / `plot_14_satellite_swath.py` — remove references to the deleted examples.
Verify nothing dangling:
```bash
grep -rnE "site_timeseries|flight_timeseries|per_site_timeseries" davinci_monet examples analyses | git ls-files --error-unmatch 2>/dev/null; \
grep -rnE "\bsite_timeseries\b|\bflight_timeseries\b|\bper_site_timeseries\b" $(git ls-files davinci_monet examples 'analyses/**/*.yaml') 2>/dev/null
```
Expected: empty (no tracked references remain).

- [ ] **Step 3: Gate (suite + configs-load + examples-run + mypy/format)**

Run the gate, the configs-load check, and the examples-run check (all from Conventions). Expected: green; "all tracked configs load"; run_all_examples finishes without traceback (it now skips the deleted examples).

- [ ] **Step 4: Commit**
```bash
black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "refactor: drop multipanel timeseries renderers (site/flight/per_site_timeseries)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Fold `spatial_distribution` into `spatial`

`spatial_distribution` (2-series + `show_var`) is superseded by the single-source `spatial` (`field.py`). Remove it; migrate its usages to `spatial`. The shared `draw_spatial_field` helper stays in `spatial/base.py` (still used by `field.py`).

**Files:**
- Delete: `davinci_monet/plots/renderers/spatial/distribution.py`
- Modify: `davinci_monet/plots/registry.py:153`, `davinci_monet/plots/renderers/spatial/__init__.py`, `davinci_monet/plots/renderers/__init__.py`, `davinci_monet/plots/__init__.py`, tests referencing `spatial_distribution`
- Migrate: `analyses/modis-aod/configs/modis-aod-cam6.example.yaml:117`, `examples/plot_08_spatial_distribution.py`

- [ ] **Step 1: Delete the renderer**
```bash
cd /Users/fillmore/EarthSystem/DAVINCI
git rm davinci_monet/plots/renderers/spatial/distribution.py
```

- [ ] **Step 2: Remove references**
- `registry.py:153` — `SPATIAL_PLOTS = frozenset({"spatial", "spatial_bias", "spatial_overlay", "spatial_distribution"})` → drop `"spatial_distribution"`.
- `spatial/__init__.py`, `plots/renderers/__init__.py`, `plots/__init__.py` — remove `SpatialDistributionPlotter`/`plot_spatial_distribution` imports + `__all__` entries + the `plots/__init__.py:31` docstring line.
- Tests — remove `spatial_distribution`/`SpatialDistributionPlotter` test cases (grep `davinci_monet/tests` and delete those).
Confirm `draw_spatial_field` is NOT removed (still imported by `field.py`):
```bash
grep -rn "draw_spatial_field" davinci_monet/plots/renderers/spatial
```

- [ ] **Step 3: Migrate the one config usage**

`modis-aod-cam6.example.yaml:117` has `type: spatial_distribution` referencing a pair. READ that plot block + the pair it names. Replace it with a single-source `spatial` plot using the pair's **x** source (the obs/reference side):
```yaml
# was: type: spatial_distribution, data: [<pair>], show_var: x   (or geometry)
<plot_name>:
  type: spatial
  source: <the pair's x source label>
  variable: <the pair's x variable>
  title: <keep the title>
```
(If the original `show_var` selected the y side, use the y source/variable instead. If `show_var: both`, pick the x side — the 2-panel mode is dropped.)

- [ ] **Step 4: Migrate the example**

`examples/plot_08_spatial_distribution.py` uses `SpatialDistributionPlotter`/paired data. Rewrite it to build a SINGLE-source dataset and render with `SpatialPlotter` (`type: spatial` / `from davinci_monet.plots.renderers.spatial.field import SpatialPlotter`), mirroring how the other single-source examples call `render(build_series(ds, var))`. Rename the file to `examples/plot_08_spatial_field.py` (and update `run_all_examples.py`/README references). Keep it minimal but runnable.

- [ ] **Step 5: Gate + commit**

Run the gate + configs-load + examples-run. Confirm no `spatial_distribution` references remain:
```bash
grep -rnE "spatial_distribution|SpatialDistributionPlotter" $(git ls-files davinci_monet examples 'analyses/**/*.yaml') 2>/dev/null || echo clean
```
```bash
black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "refactor: fold spatial_distribution into single-source spatial; migrate usages

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: P1 — contract guards + loose-end fixes

**Files:** `davinci_monet/plots/renderers/{flight_track,lma_density,diurnal}.py`, `davinci_monet/plots/renderers/spatial/overlay.py`, test file for the guards.

- [ ] **Step 1: Write failing guard tests** (append to `davinci_monet/tests/test_plots.py` or a small new `davinci_monet/tests/test_renderer_contracts.py`)
```python
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from davinci_monet.plots.base import build_series


def _track_ds():
    n = 10
    return xr.Dataset(
        {"O3": (["time"], np.arange(n, dtype=float))},
        coords={"time": pd.to_datetime(["2012-05-29"] * n),
                "latitude": ("time", np.linspace(34, 36, n)),
                "longitude": ("time", np.linspace(-98, -96, n)),
                "altitude": ("time", np.linspace(500, 8000, n), {"units": "m"})},
    )


def test_flight_track_requires_one_series():
    from davinci_monet.plots.renderers.flight_track import FlightTrackPlotter
    s = build_series(_track_ds(), "O3")
    with pytest.raises(NotImplementedError, match="1 series"):
        FlightTrackPlotter().render(s + s)


def test_lma_density_requires_one_series():
    from davinci_monet.plots.renderers.lma_density import LMADensityPlotter
    s = build_series(_track_ds(), "O3")
    with pytest.raises(NotImplementedError, match="1 series"):
        LMADensityPlotter().render(s + s)
```

- [ ] **Step 2: Run — expect FAIL** (no guards yet)

- [ ] **Step 3: Implement the fixes**
- `flight_track.py` `render` (~line 54): change the signature/body to guard and type:
```python
    def render(
        self, series: list[PlotSeries], ax: matplotlib.axes.Axes | None = None, **kwargs: Any
    ) -> Any:
        """Unified entry: render a single source's flight track."""
        if len(series) != 1:
            raise NotImplementedError(
                f"FlightTrackPlotter.render requires exactly 1 series; got {len(series)}."
            )
        s = series[0]
        return self.plot(s.dataset, s.var_name, **kwargs)
```
  Add `from davinci_monet.core.base import PlotSeries` under `TYPE_CHECKING`. Forward `city_labels` (+ `city_marker_size`/`city_marker_color`/`city_font_size`, defaults matching `track_map_3d`) through `plot()` into `draw_track_3d`. Update the module docstring line 1 to drop "dataset-only" (say "single-source geometry data").
- `lma_density.py` `render` (~line 30): add the same guard:
```python
        if len(series) != 1:
            raise NotImplementedError(
                f"LMADensityPlotter.render requires exactly 1 series; got {len(series)}."
            )
        s = series[0]
```
- `spatial/overlay.py` `plot` (~line 128, after fig/ax creation): add `self.add_map_features(ax)` so the `map_config` is actually applied (it is currently ignored).
- `diurnal.py`: remove the unused `aggregate_dim` parameter from the `plot()`/`render()` signatures and docstring (grep `aggregate_dim` in the file — it is accepted but never used).

- [ ] **Step 4: Run guard tests — expect PASS**, then gate + commit
```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && mypy davinci_monet
black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "fix(plots): P1 — flight_track/lma_density series guards, overlay map_config, flight_track city_labels, drop diurnal aggregate_dim

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: P2 — extract shared spatial helpers

Extract the duplicated coordinate-resolution + time-average logic into `spatial/base.py` and route `bias`/`overlay`/`field` through it; make `bias` use `draw_spatial_field`. Behavior-preserving.

**Files:** `davinci_monet/plots/renderers/spatial/base.py`, `bias.py`, `overlay.py`, `field.py`

- [ ] **Step 1: Add the helpers to `spatial/base.py`** (module-level functions, near `detect_spatial_geometry`):
```python
_LAT_CANDIDATES = ["latitude", "lat", "LAT", "Latitude"]
_LON_CANDIDATES = ["longitude", "lon", "LON", "Longitude"]


def resolve_spatial_coords(ds, lat_var="latitude", lon_var="longitude"):
    """Resolve (lat_name, lon_name, lat_values, lon_values) from a dataset, with
    0..360 -> -180..180 longitude normalization. Raises ValueError if absent."""
    import numpy as np

    lat_name = next((c for c in [lat_var, *_LAT_CANDIDATES] if c in ds.coords or c in ds), None)
    lon_name = next((c for c in [lon_var, *_LON_CANDIDATES] if c in ds.coords or c in ds), None)
    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Could not find latitude/longitude coordinates. Available: {list(ds.coords)}"
        )
    lats = ds[lat_name].values
    lons = ds[lon_name].values
    if lons.ndim >= 1 and np.any(lons > 180):
        lons = np.where(lons > 180, lons - 360, lons)
    return lat_name, lon_name, lats, lons


def maybe_time_average(data, time_average=True, time_dim="time"):
    """Mean over the time dim when present and requested; else return data."""
    if time_average and time_dim in getattr(data, "dims", ()):
        return data.mean(dim=time_dim)
    return data
```

- [ ] **Step 2: Route `bias.py`, `overlay.py`, `field.py` through them**

Replace each renderer's inline lat/lon-candidate resolution + 0..360 shift with `resolve_spatial_coords(...)`, and inline `if time_average and "time" in ...: .mean("time")` with `maybe_time_average(...)`. In `bias.py`, replace the hand-rolled scatter/pcolormesh + meshgrid/broadcast block with a call to `draw_spatial_field(ax, bias.values, lats, lons, plot_type=..., cmap=cmap, vmin=vmin, vmax=vmax, marker_size=ms, alpha=alpha)` (the `plot_type` resolved from `detect_spatial_geometry` as `field.py`/`distribution`-removed did). Keep the symmetric-cbar / TwoSlopeNorm handling. Do NOT change numeric output — the full suite's bias tests + the `test_xy_contract`/spatial tests are the gate.

- [ ] **Step 3: Gate** — run the gate. The existing spatial tests (bias/overlay/field) must stay green (behavior-preserving). If a test asserts an artist type or value, confirm it still holds.

- [ ] **Step 4: Commit**
```bash
black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "refactor(plots): extract shared spatial coord/time-average helpers; bias uses draw_spatial_field

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: P2 — shared paired-series helpers (statistical renderers)

Extract the repeated x/y-series extraction + NaN cleaning + axis-label resolution into `plots/base.py` and route `scatter`/`taylor`/`boxplot`/`scorecard` through them. Behavior-preserving.

**Files:** `davinci_monet/plots/base.py`, `scatter.py`, `taylor.py`, `boxplot.py`, `scorecard.py`

- [ ] **Step 1: Add helpers to `plots/base.py`** (on `BasePlotter` or module-level):
```python
def extract_xy_series(series):
    """From a 2-element series list, return (paired_dataset, x_var, y_var) selecting
    by the 'x'/'y' axis attr (falling back to positional order)."""
    if len(series) != 2:
        raise NotImplementedError(f"requires exactly 2 series; got {len(series)}.")
    x_series = next((s for s in series if s.axis == "x"), series[0])
    y_series = next((s for s in series if s.axis == "y"), series[1])
    return x_series.dataset, x_series.var_name, y_series.var_name


def clean_xy(x, y):
    """Flatten and drop non-finite pairs; return (x_clean, y_clean)."""
    import numpy as np

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]
```

- [ ] **Step 2: Route the four renderers through them**

In `scatter.py`, `taylor.py`, `boxplot.py`, `scorecard.py` `render()`: replace the duplicated `x_series = next(...); y_series = next(...); paired_data = x_series.dataset; x_var = ...; y_var = ...` block with `paired_data, x_var, y_var = extract_xy_series(series)`, and replace the inline `mask = np.isfinite(x) & np.isfinite(y)` blocks with `x, y = clean_xy(x_raw, y_raw)`. In `scatter.py`, replace the inline color logic with `get_axis_color(...)` like `taylor`/`boxplot` use. Do NOT change visual/numeric output — the suite is the gate.

- [ ] **Step 3: Gate + commit**
```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && mypy davinci_monet
black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "refactor(plots): shared extract_xy_series/clean_xy helpers; scatter uses get_axis_color

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: P2 — shared track-coordinate helper

Extract the duplicated lat/lon/alt extraction shared by `flight_track` and `track_map_3d` into `_track3d.py`; standardize the param names on `_var`. Behavior-preserving.

**Files:** `davinci_monet/plots/renderers/_track3d.py`, `flight_track.py`, `track_map_3d.py`

- [ ] **Step 1: Add the helper to `_track3d.py`**
```python
def resolve_track_coords(dataset, variable, lat_var="latitude", lon_var="longitude",
                         alt_var="altitude", alt_scale=1.0):
    """Return finite (lons, lats, alts, values) flat arrays for a track variable."""
    import numpy as np

    lats = dataset[lat_var].values
    lons = dataset[lon_var].values
    alts = dataset[alt_var].values * alt_scale
    vals = dataset[variable].values
    valid = np.isfinite(vals) & np.isfinite(lats) & np.isfinite(lons) & np.isfinite(alts)
    return lons[valid], lats[valid], alts[valid], vals[valid]
```

- [ ] **Step 2: Route both renderers through it**

In `flight_track.py` and `track_map_3d.py`, replace the inline coordinate extraction + `np.isfinite(...)` masking (flight_track ~167-188, track_map_3d ~179-221) with `lons, lats, alts, values = resolve_track_coords(...)`. Standardize `track_map_3d` to use `np.isfinite` (already does via the helper). Keep `flight_track`'s `_coord`→ rename to `_var` params (already standardized in the helper call). Behavior-preserving — the map tests are the gate.

- [ ] **Step 3: Gate + commit**
```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && mypy davinci_monet
black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "refactor(plots): shared resolve_track_coords for flight_track/track_map_3d

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Registry/docs cleanup, gitignored gemini configs, final verification

**Files:** `davinci_monet/plots/registry.py`, `CLAUDE.md`, gitignored `*-gemini.yaml` (local only)

- [ ] **Step 1: Registry + docs**
- Confirm `registry.py` `TEMPORAL_PLOTS`/`SPATIAL_PLOTS` no longer list the dropped types and that `"spatial"` is categorized under `SPATIAL_PLOTS` (it is). Update the `get_plot_category`/category docstrings if they mention dropped types.
- `CLAUDE.md` — remove any mention of the dropped plot types; note `spatial` is the single-source map (replaces `spatial_distribution`).

- [ ] **Step 2: Migrate the gitignored gemini configs LOCALLY (not committed)**

These are gitignored machine configs the user runs; dropping the renderers would break them. For each, remove the dropped plot specs and fold `spatial_distribution`→`spatial` (same rule as Task 2 Step 3). Enumerate + fix:
```bash
find analyses -name '*-gemini.yaml' | xargs grep -lnE "type:\s*(site_timeseries|flight_timeseries|per_site_timeseries|spatial_distribution)\b" 2>/dev/null
```
(Known: `modis-aod-cam6-gemini.yaml`, `asia-aq-pandora-gemini.yaml`, `asia-aq-gemini.yaml`, `asia-aq-dc8-gemini.yaml`.) Edit each in place; verify it loads with the configs-load check pattern. Do NOT `git add` them (they're gitignored). Report the list to the user.

- [ ] **Step 3: Final gate**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
cd /Users/fillmore/EarthSystem/DAVINCI
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q
mypy davinci_monet
black --check davinci_monet && isort --check-only davinci_monet
```
Expected: all green. Also confirm the registry/`@register_plotter` count dropped by 4 and no dangling references:
```bash
grep -rnE "site_timeseries|flight_timeseries|per_site_timeseries|spatial_distribution|SpatialDistributionPlotter" $(git ls-files davinci_monet 'analyses/**/*.yaml' examples) 2>/dev/null || echo "clean"
```

- [ ] **Step 4: Commit + report**
```bash
git add davinci_monet/plots/registry.py CLAUDE.md docs/ && git commit -m "chore(plots): registry/docs cleanup after renderer remediation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
Report the suite/mypy results, the list of removed renderers/types, and the gitignored gemini configs migrated. Do not push/merge.

---

## Self-Review notes
- **Coverage:** drops → Tasks 1–2; fold → Task 2; P1 (flight_track/lma_density guards, overlay map_config, city_labels, diurnal) → Task 3; P2 spatial/statistical/track helpers → Tasks 4/5/6; registry/docs/gemini → Task 7.
- **Green-at-each-commit:** each drop task removes the renderer + its tests + config specs + examples in one commit, so the suite/examples never reference a deleted symbol mid-task.
- **Behavior-preserving P2:** Tasks 4–6 are extract-method refactors; the full suite (incl. the spatial/scatter/map renderer tests) is the regression gate — they must NOT change numeric/visual output.
- **The two pre-existing config-load failures** (`wrfchem-forecast`, `cmaq_airnow`) are unrelated and expected; don't chase them.
- **Out of scope (deferred dispositions):** whether to keep/deprecate the surviving niche renderers (boxplot, scorecard, curtain, lma_density, spatial_overlay) — left for a follow-up decision.
