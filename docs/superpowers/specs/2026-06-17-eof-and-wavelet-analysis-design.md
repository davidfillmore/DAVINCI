# EOF & Wavelet Analysis — Design Spec

**Date:** 2026-06-17
**Status:** Approved design, pre-implementation (revised after adversarial spec review)
**Scope:** Two new analysis features delivered on one shared "derived-analysis" layer:
1. **EOF** (Empirical Orthogonal Function) decomposition of any 2-D or 3-D gridded field.
2. **Wavelet** time-series analysis (Torrence & Compo continuous wavelet transform), including of EOF principal-component coefficients.

> **Note on code references.** This spec cites symbol names and file paths (not line numbers, which drift). All cited symbols were verified against the tree on 2026-06-17. `xeofs` and `pycwt` are **not yet installed** in the `davinci` env — their exact API signatures must be re-verified against the pinned installed version before coding (see §8, §11).

---

## 1. Motivation & Goals

DAVINCI today computes only **scalar** statistics (paired `x, y → float`, `stats/metrics.py`). There is no abstraction for an analysis that consumes a field and **emits a derived dataset** (modes, principal components, a time–period power spectrum). EOF and wavelet are the first two such analyses and motivate a small, reusable **derived-analysis layer**.

**Goals**

- Decompose any 2-D `(time, lat, lon)` or 3-D `(time, lev, lat, lon)` gridded source into EOF spatial modes + principal-component (PC) time series + explained variance.
- Run a climate-standard wavelet analysis (Morlet CWT, AR(1) red-noise significance, cone of influence, global spectrum) on any 1-D series — a station/point series, an area-mean of a gridded field, or **an EOF PC**.
- Make derived outputs **first-class sources** so the existing single-source plot path operates on them with no special-casing, and so "wavelet of an EOF coefficient" is just "wavelet of a source variable."
- Deliver four visualizations: EOF pattern maps, EOF scree (explained variance), EOF PC time series (reused renderer), and a wavelet scalogram with global-spectrum side panel.

**Non-goals (YAGNI)** — see §10.

---

## 2. Architecture & Data Flow

### 2.1 Derived-analysis layer

A new package `davinci_monet/analysis/` introduces a registry-backed base class, mirroring the statistics registry pattern (`core/registry.py`, `@statistic_registry.register(...)`):

```python
# davinci_monet/analysis/base.py
from abc import ABC, abstractmethod
import xarray as xr
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import analysis_registry  # new registry

class DerivedAnalysis(ABC):
    """An analysis that consumes ONE source dataset and emits a derived dataset."""
    name: str = "base"
    long_name: str = "Base Derived Analysis"
    output_geometry: DataGeometry          # geometry of the principal output field

    @abstractmethod
    def analyze(self, data: xr.Dataset, spec: "AnalysisSpec") -> xr.Dataset:
        """Return a derived xr.Dataset. `data` is the fully-built input dataset
        (raw source OR an already-built derived source). `spec` is the validated
        Pydantic params for this entry."""
        ...
```

- `analysis_registry: Registry[type] = Registry("analysis")` is added to `core/registry.py` alongside `source_registry`/`plotter_registry`/`statistic_registry`.
- Each concrete analysis (`EOFAnalysis`, `WaveletAnalysis`) registers via `@analysis_registry.register("eof"|"wavelet")` and declares its `output_geometry`.

### 2.2 Pseudo-source model

`SourceData` is defined in `pipeline/stages/base.py` (imported/produced by `LoadSourcesStage`). Its required fields are `data: xr.Dataset`, `label: str`, `source_type: str`, `geometry: DataGeometry`; optional `variables: dict = {}`, `config: dict = {}`. The `AnalysesStage` wraps each `analyze()` result into a `SourceData` and inserts it into `context.sources` under the analysis's config key, with **exact** field values:

- `data` = the derived dataset (with `data.attrs['geometry']` set from `output_geometry.name.lower()`, exactly as `LoadSourcesStage` does, **plus** `data.attrs['derived'] = True`).
- `label` = the analysis config key (e.g. `cam_O3_eof`).
- `source_type` = the analysis type string (`"eof"` / `"wavelet"`).
- `variables` = `{}`.
- `config` = the `AnalysisSpec.model_dump()` (provenance; also marks the source derived).

The `derived` marker (attr + `source_type` in the analysis set) is what the pairing/stats guard keys off (§10).

**Consequence:** the **single-source plot path** (`source:` + `variable:`, `plots/contracts.py` `SINGLE_SOURCE_PLOTS`, dispatched in `pipeline/stages/plot.py`) renders derived sources unchanged.

**Mixed-shape derived sources.** An EOF derived source holds variables of *differing* dimensionality — `mode(mode, lev, lat, lon)`, `pc(time, mode)`, `explained_variance(mode)`. This is valid xarray. The source-level `geometry` attr reflects the **principal field** (GRID for EOF). Renderers and the wavelet reducer select behavior by the **chosen variable's dims**, not by the source-level geometry. Each derived variable also carries a `kind` attr (`"mode" | "pc" | "scalar"` for EOF; `"power" | "global" | "coi"` for wavelet) for unambiguous downstream selection.

### 2.3 New geometry kind

`core/protocols.py` `DataGeometry` (currently POINT/TRACK/PROFILE/SWATH/GRID via `auto()`) gains **`SPECTRUM = auto()`** for the wavelet output `(time, period)`. EOF output uses the existing **GRID** geometry. Neither derived geometry participates in pairing (§10).

### 2.4 Pipeline placement & dependency ordering

A new `AnalysesStage` (`pipeline/stages/analyses.py`) is inserted in `create_standard_pipeline()` (`pipeline/stages/factory.py`) **after `LoadSourcesStage` and before `PairingStage`**:

```
LoadSourcesStage → AnalysesStage → PairingStage → StatisticsStage → PlottingStage → SaveResultsStage → SummaryStage
```

Because one analysis may consume another's output (wavelet of an EOF PC), the stage:

1. Reads `context.analyses_config()` (ordered dict of `key → AnalysisSpec`).
2. Builds a dependency DAG: each spec's `source:` resolves to either a raw `context.sources` key **or** another analysis key.
3. **Topologically sorts**; raises a clear config error on an unknown reference or a cycle (mirrors the parse-time validator in §3.4, re-checked here against actually-loaded sources).
4. Executes analyses in order, inserting each result into `context.sources` before the next runs.
5. Writes a per-analysis summary (output kind, shape, n_modes / n_scales, timing) to `context.results["analyses"]`.

`AnalysesStage.validate()` returns `True` only when `analyses:` is non-empty, so existing configs are unaffected.

---

## 3. Configuration Schema

### 3.1 `analyses:` block

`config/schema.py` gains a top-level field on `MonetConfig`:

```python
analyses: dict[str, AnalysisSpec] = Field(default_factory=dict)
```

`PipelineContext` gains a typed accessor `analyses_config() -> dict[str, AnalysisSpec]`.

> **Naming footgun (explicit):** `MonetConfig` already has a **singular** `analysis: AnalysisConfig` (the `analysis:` block — start/end time, output_dir) with accessor `analysis_config()`. The new **plural** `analyses` / `analyses_config()` is distinct. Keep both; implementers must not conflate them.

### 3.2 Discriminated union (a new pattern for this repo)

`AnalysisSpec` is a Pydantic v2 discriminated union on the `Literal` `type` field. (This is a **new** pattern here — existing plot `type:` validation uses a plain `@field_validator` against the plotter registry, not a discriminated union. The `Literal` discriminator already constrains `type` to the known analyses, so **no** separate registry field-validator is added; the dependency DAG check is a model-validator, §3.4.)

```python
class PointReduce(StrictSchema):
    point: tuple[float, float]          # (lat, lon)

class _AnalysisBase(StrictSchema):
    source: str                         # raw source key OR another analysis key
    variable: str

class EOFSpec(_AnalysisBase):
    type: Literal["eof"]
    n_modes: int = 10
    standardize: bool = False           # False → covariance EOF; True → correlation EOF
    remove_seasonal_cycle: bool = False # in addition to always-on time-mean removal
    rotation: Literal["none", "varimax"] = "none"
    level: int | None = None            # analysis-time: restrict 3-D field to one level (→ 2-D EOF)

class WaveletSpec(_AnalysisBase):
    type: Literal["wavelet"]
    mode: int | None = None             # select PC mode N when `variable` has a `mode` dim
    reduce: Literal["area_mean"] | PointReduce | None = "area_mean"
    omega0: float = 6.0                 # Morlet wavenumber
    significance_level: float = 0.95
    dj: float = 0.25                    # sub-octaves per octave (pycwt)
    s0: float | None = None             # smallest scale; default 2*dt
    j: int | None = None                # config field → pycwt `J`; None ⇒ pass J=-1 (pycwt auto)

AnalysisSpec = Annotated[EOFSpec | WaveletSpec, Field(discriminator="type")]
```

### 3.3 Reduction semantics (wavelet) — precedence

After selecting `variable` and (if it has a `mode` dim) `mode: N`:

- **If the series is already 1-D in time** (e.g. an EOF `pc` with `mode` selected, or a point/track variable): **`reduce` is ignored** regardless of its value. An explicit non-`null` `reduce` on an already-1-D series is a **config error** (caught at validation).
- **If a spatial dim remains**: apply `reduce` — `area_mean` (area-weighted cos-lat mean → `(time,)`) or `{point: [lat, lon]}` (nearest grid cell). `area_mean` is the default and is applied only in this branch.

This resolves the default-vs-PC-example tension: the `pc1_wavelet` example (§3.5) selects `mode: 1`, yielding a 1-D series, so the defaulted `area_mean` is a no-op by the rule above.

### 3.4 Dependency validation

A `MonetConfig` model-validator builds the analysis dependency graph from `source:` references and rejects unknown references and cycles at parse time (re-verified against loaded sources in `AnalysesStage`, §2.4).

### 3.5 Example

```yaml
sources:
  cam:
    type: cesm_fv
    files: ${DATA}/cam/*.nc
    variables: { O3: { unit_scale: 1.0e9 } }

analyses:
  cam_O3_eof:
    type: eof
    source: cam
    variable: O3
    n_modes: 6
  pc1_wavelet:
    type: wavelet
    source: cam_O3_eof     # a derived source
    variable: pc
    mode: 1                # → 1-D series; reduce is a no-op
  areamean_wavelet:
    type: wavelet
    source: cam
    variable: O3
    reduce: area_mean      # spatial dims remain → area-weighted mean

plots:
  eof_maps:
    type: eof_pattern
    source: cam_O3_eof
    variable: mode
    display_level: -1       # plot-time level slice for 3-D modes (surface)
  eof_variance:
    type: eof_scree
    source: cam_O3_eof
    variable: explained_variance
  pc_series:
    type: timeseries        # reused renderer
    source: cam_O3_eof
    variable: pc
    mode: 1                 # plot-time mode selector (see §6.3)
  pc1_scalogram:
    type: wavelet_scalogram
    source: pc1_wavelet
    variable: power
```

---

## 4. EOF Analysis

`davinci_monet/analysis/eof.py`, `@analysis_registry.register("eof")`, `output_geometry = DataGeometry.GRID`.

### 4.1 Preprocessing (fixed order)

1. Select `variable`; if 3-D and `level:` (analysis-time) is set, slice that level (→ 2-D path).
2. **Remove time-mean** → anomalies (always).
3. If `remove_seasonal_cycle`: subtract the per-calendar-month climatology of the anomalies.
4. If `standardize`: divide each spatial point's (post-step-3) anomaly by its temporal std → correlation EOF; else covariance EOF.

### 4.2 Weighting

- 2-D covariance EOF: weight = `sqrt(cos(lat))` over `(lat, lon)`.
- 3-D **full state vector** covariance EOF: weight = `sqrt(cos(lat) · Δp_norm)` over `(lev, lat, lon)`, where `Δp_norm` is the normalized layer pressure thickness (mass weighting). **Δp source:** from CESM hybrid sigma-pressure edges `hyai`/`hybi` + surface pressure `PS` (a reference `P0`-based mean PS if per-time PS is absent), i.e. `Δp_k = (hyai_{k+1}-hyai_k)·P0 + (hybi_{k+1}-hybi_k)·PS`; or directly from `ilev` pressure edges when present. **Fallback:** if no usable vertical edge/PS info exists, use equal layer weight with a logged warning. Level orientation follows the existing `surface_level_index` / CESM convention (CLAUDE.md vertical-coordinate warning).
- **standardize + mass weighting are mutually exclusive in 3-D.** Per-point standardization already equalizes per-cell variance, making mass weighting moot/double-counting. Rule: when `standardize: true`, weighting is **cos-lat (area) only**; mass weighting applies **only** to covariance EOF (`standardize: false`). A 3-D `standardize: true` config logs a warning that vertical mass weighting is disabled.
- Weights are applied via xeofs's **fit-time** `weights=` argument with `use_coslat=False` (the single weight field already includes cos-lat — do **not** also set `use_coslat=True`, which would double-weight).

### 4.3 Decomposition (xeofs)

- Use `xeofs.single.EOF` (and `xeofs.single.EOFRotator` when `rotation: varimax`) — the v2+ import path; **pin a xeofs version** (§8) and re-verify class/method names before coding.
- Stack the spatial dims so a 3-D `(lev, lat, lon)` field collapses to one sample(time)×feature(space) problem → **coupled** modes spanning the vertical.
- Compute `n_modes` modes and the explained-variance ratio.

### 4.4 Scaling split (modes vs PCs)

xeofs `scores()`/`components()` are not unit-variance by default. We renormalize to a fixed, internally consistent convention:

- **PC** `pc_k` = score scaled to **unit temporal variance** (dimensionless).
- **Displayed mode** `mode_k` = `component_k · sqrt(λ_k) · (1/weight)` so that `Σ_k mode_k ⊗ pc_k` reconstructs the **de-weighted anomaly field in physical units**. (De-weighting divides out the §4.2 weight so maps are physical, not weighted-space.)
- Verify this split against xeofs's `normalized=` semantics during implementation.

### 4.5 Sign convention (mode-intrinsic, deterministic)

Each mode's sign is fixed by a **mode-intrinsic** rule: flip mode *k* so that the spatial loading with the **largest absolute value is positive** (ties broken by flattened index). This is robust for dipole/antisymmetric modes where a domain-mean projection is near zero and numerically unstable. (The earlier area-mean-correlation rule is dropped.)

### 4.6 Explained-variance error (North's rule — hand-computed)

xeofs does **not** provide North's rule. Compute it explicitly: `δλ_k = λ_k · sqrt(2 / N*)`, where `N*` is the **effective** number of independent samples (raw `len(time)` divided by the integral decorrelation time estimated from the series' lag-1 autocorrelation — autocorrelated geophysical fields inflate raw N). Document the autocorrelation caveat. North's rule governs **eigenvalue separation**, so the error bars belong on the scree/variance spectrum (§6.2).

**Rotation caveat:** North's rule and descending-variance ordering apply to the **unrotated** case only. When `rotation: varimax`: modes are reordered and are not eigenvectors, so (a) `explained_variance` is **recomputed post-rotation**, (b) **no North error bars** are emitted (scree shows rotated variance without error), and (c) the §4.5 sign rule is applied **after** rotation.

### 4.7 Output dataset

`xr.Dataset` (geometry GRID), variables:

| variable | dims | kind | units / notes |
|---|---|---|---|
| `mode` | `(mode, lev, lat, lon)` or `(mode, lat, lon)` | `mode` | physical anomaly units = source variable's `units`; `long_name` set |
| `pc` | `(time, mode)` | `pc` | dimensionless (unit variance); units `"1"` (omitted in labels) |
| `explained_variance` | `(mode)` | `scalar` | fraction 0–1; attr `percent` for display |
| `explained_variance_error` | `(mode)` | `scalar` | North δλ (unrotated only; absent when rotated) |

`mode` coordinate is 1-indexed for human-facing labels.

---

## 5. Wavelet Analysis

`davinci_monet/analysis/wavelet.py`, `@analysis_registry.register("wavelet")`, `output_geometry = DataGeometry.SPECTRUM`.

### 5.1 Series extraction & preprocessing (fixed order)

1. Select `variable`; if it has a `mode` dim, select `mode: N`.
2. Apply `reduce` per §3.3 (only if a spatial dim remains) → `(time,)`.
3. **Regularize** the time axis to regular spacing. **Log the chosen `dt` and the fraction of timestamps synthesized by regularization; warn (or reject) when interpolation exceeds a threshold** — heavy gap-filling biases both the AR(1) estimate and the spectrum, and can alias.
4. **Linear detrend.**
5. **Estimate AR(1) `alpha` = `pycwt.wavelet.ar1(detrended_series)`** — on the detrended, *pre-normalization* series (interpolation/normalization change autocorrelation and would bias the red-noise background).
6. **Normalize to unit variance** for the CWT. Record `dt`, original `std`, `mean` as attrs.

### 5.2 Transform & significance (pycwt — pinned calls)

- `wave, scales, freqs, coi, _, _ = pycwt.cwt(signal, dt, dj, s0, J, wavelet=pycwt.Morlet(omega0))`. Default `s0 = 2*dt`; `J = -1` when config `j is None` (pycwt auto), else `J = j`.
- `power = |wave|²`. `period = 1/freqs`; **`freqs` already includes the Morlet Fourier factor** (≈1.033 for `omega0=6`) — do **not** hardcode the factor anywhere downstream. Changing `omega0` changes the time/frequency-resolution tradeoff and COI width; everything follows from pycwt automatically.
- **Local significance** (per-scale, time-invariant): `signif, _ = pycwt.significance(1.0, dt, scales, 0, alpha, significance_level=significance_level, wavelet=mother)` (var=1.0 because the series is unit-variance; `sigma_test=0`). Broadcast to `(time, period)`; `power_significance = power / signif` (values > 1 significant).
- **Global significance**: `glbl_signif, _ = pycwt.significance(var, dt, scales, 1, alpha, significance_level=significance_level, dof=(N - scales), wavelet=mother)` — `sigma_test=1`, `var` = variance of the (normalized) series (=1.0), and the **mandatory per-scale `dof` vector** `N - scales` (time-average DOF correction). `global_power` = time-mean of `power`.
- **Cone of influence:** `coi(time)` gives the maximum reliable **period** at each time. Power at periods **greater than** `coi(time)` is edge-contaminated and must be masked/hatched.

### 5.3 Period units

Derive a unit string from the time-coordinate spacing: map the pandas freq / `np.timedelta64` resolution of `dt` to `"days"`/`"hours"`/etc. For non-datetime or irregular axes, fall back to index units (`"steps"`) and log it. The `period` coordinate carries this `units`.

### 5.4 Output dataset

`xr.Dataset` (geometry SPECTRUM), variables:

| variable | dims | kind | notes |
|---|---|---|---|
| `power` | `(time, period)` | `power` | wavelet power (normalized variance) |
| `power_significance` | `(time, period)` | `power` | from the **local** (`sigma_test=0`) curve; `>1` ⇒ significant |
| `coi` | `(time)` | `coi` | max reliable period at each time |
| `global_power` | `(period)` | `global` | time-mean spectrum |
| `global_significance` | `(period)` | `global` | from the **global** (`sigma_test=1`, dof) curve |

`period` carries `units` (§5.3).

---

## 6. Plots & Rendering

All new renderers register via `@register_plotter`, are added to `SINGLE_SOURCE_PLOTS` **and** an appropriate category set in `plots/contracts.py` so `ALL_PLOT_TYPES` (= union of category sets, which does **not** include `SINGLE_SOURCE_PLOTS`) contains them: `eof_pattern → SPATIAL_PLOTS`; `eof_scree`, `wavelet_scalogram → SPECIALIZED_PLOTS`. They are exported from `plots/renderers/__init__.py` and use `plots/labeling.py` for all label/title text. (`calculate_symmetric_limits` lives in `plots/labels.py`, not `labeling.py`.)

### 6.1 `eof_pattern` (new)

- Extends `BaseSpatialPlotter` (`plots/renderers/spatial/base.py`); reuses `draw_spatial_field()` / cartopy setup.
- One **signed** map per mode (or a faceted figure): diverging cmap + `matplotlib.colors.TwoSlopeNorm(vcenter=0)` with symmetric limits (`labels.calculate_symmetric_limits`).
- 3-D `mode(mode, lev, lat, lon)` defaults to a **surface-level slice** (`surface_level_index`, CESM convention) with a plot-spec `display_level:` override (distinct from the analysis-time `EOFSpec.level`). `display_level` is added to the single-source plot spec schema.
- Title via `title_text(quantity, operation="EOF Mode N")` (dynamic mode number — confirm `title_text(operation=...)` accepts the composed string); explained-variance % in the subtitle/stats box. Render mark verified programmatically (QuadMesh for grid).

### 6.2 `eof_scree` (new, small)

- Plain `BasePlotter`: explained variance (%) vs mode index as bar/stem; **North-rule error bars only when unrotated** (§4.6); horizontal noise-floor reference. Title `title_text(quantity, operation="EOF Explained Variance")`.

### 6.3 PC time series (reused, with a plot-time mode selector)

- The existing `timeseries` renderer (`plots/renderers/timeseries.py`) **averages over all non-time dims by default** — so plotting `pc(time, mode)` directly would average all PCs into one meaningless line. Fix: the single-source plot dispatch gains an optional **`mode:`** selector that does `pc.sel(mode=N)` (default `N=1` when a `mode` dim is present and unselected) **before** `build_series`, yielding a clean `(time,)` series. The renderer itself is unchanged. A test asserts the rendered series equals the selected PC, not a cross-mode mean.
- **Pre-implementation audit item:** confirm the timeseries renderer selects series by variable dims/axis attrs and tolerates a GRID-geometry source (it averages non-time dims; with `mode` pre-selected the series is already 1-D).

### 6.4 `wavelet_scalogram` (new)

- Plain `BasePlotter`, structurally modeled on `plots/renderers/curtain.py`: main panel `pcolormesh(time, period, power)` with a **log period** y-axis; sequential cmap; **COI** masked/hatched (periods above `coi(time)`); **significance contour** at `power_significance == 1`.
- A right-side **global-spectrum panel** (`global_power` vs period, sharing the period axis) with its significance curve — classic Torrence & Compo layout.
- Title `title_text(quantity, operation="Wavelet Power")`; period axis labeled with `format_units` on the period units (§5.3).

### 6.5 Colormaps & labeling

- EOF patterns (signed) → diverging (`get_bias_cmap()` / `RdBu_r` family) centered at zero.
- Scree, scalogram power, global spectrum (unsigned) → sequential (`get_sequential_cmap()` / viridis family).
- New operation words (`"EOF Mode N"`, `"EOF Explained Variance"`, `"Wavelet Power"`) are exercised by `tests/unit/plots/test_labeling.py` / `test_labels_rendered.py` so no ad-hoc strings slip in. PC labels omit units (dimensionless); mode labels use the source variable's units; scree uses `%`; power is dimensionless ("Power").

---

## 7. Module Layout

**New files**

- `davinci_monet/analysis/{__init__,base,eof,wavelet,reductions}.py` — `reductions.py` holds `area_mean`, `point`, and the shared time-axis regularize/detrend/ar1/normalize helpers.
- `davinci_monet/pipeline/stages/analyses.py` — `AnalysesStage`.
- `davinci_monet/plots/renderers/{eof_pattern,eof_scree,wavelet_scalogram}.py` (flat layout — there is no `renderers/temporal/`).

**Touched files**

- `core/registry.py` — add `analysis_registry`.
- `core/protocols.py` — add `DataGeometry.SPECTRUM`.
- `config/schema.py` — `PointReduce`, `AnalysisSpec` union (`EOFSpec`, `WaveletSpec`), `MonetConfig.analyses`, dependency-DAG validator; add `display_level` + `mode` to the single-source plot spec.
- `pipeline/stages/base.py` — `analyses_config()` accessor.
- `pipeline/stages/factory.py` — insert `AnalysesStage`; `pipeline/stages/__init__.py` — export it.
- `pipeline/stages/plot.py` — apply the single-source `mode:` selection before `build_series`.
- `pairing` + `stats` stages — guard against derived sources (§10).
- `plots/contracts.py` — add new types to `SINGLE_SOURCE_PLOTS` and category sets; `plots/renderers/__init__.py` — export.
- `environment.yml`, `pyproject.toml` — add pinned `xeofs`, `pycwt`.

Each module stays under the project's ~500-line target.

---

## 8. Dependencies

- **`xeofs`** (v2+) — xarray-native EOF: fit-time `weights=`, N-dimensional fields, `EOFRotator` (varimax), dask. Public classes at `xeofs.single.EOF` / `xeofs.single.EOFRotator`. Pulls scikit-learn + dask. **North's rule is NOT provided** (hand-computed, §4.6).
- **`pycwt`** — Torrence & Compo CWT: `pycwt.cwt`, `pycwt.significance`, `pycwt.Morlet`, `pycwt.wavelet.ar1`. Lightweight (numpy/scipy), lightly maintained.

Neither is currently installed in the `davinci` env. **Pin minimum versions** in `environment.yml`/`pyproject.toml` and **verify the exact signatures (§4.3, §5.2) against the installed version** as an implementation gate. Each dependency is introduced by the plan that first needs it (§11), not up front.

---

## 9. Testing Strategy

All integration tests run **through `PipelineRunner.run_from_config()`** — the same path as `davinci-monet run config.yaml` (project Testing Rules). Render marks are verified **programmatically**, never by eye.

**Synthetic generators (known structure)**

- EOF: a gridded field from two **orthogonal planted spatial patterns** × **known PC series** + controlled noise. Assert recovered pattern correlation, variance ordering, PC correlation, deterministic sign (§4.5), and (unrotated) that `explained_variance_error` matches `λ·sqrt(2/N*)`. A 3-D variant asserts coupled vertical-structure recovery and the mass-weight vs equal-weight fallback path.
- Wavelet: a series with an **injected period** + red noise. Assert the global-spectrum **peak at the injected period** exceeds 95% significance, COI present, period units correct, and that `alpha` is estimated pre-normalization.

**Integration (pipeline)**

- A config with `analyses:` (`eof`, then `wavelet` of its PC) asserts: derived sources registered in `context.sources`; **dependency order** resolved (wavelet after EOF); all four plot types produced; output files written; `eof_pattern`/`wavelet_scalogram` produce `QuadMesh`; PC time series equals the selected PC (not a cross-mode mean).
- A guard test: referencing a derived source in `pairs:` raises the clear "derived sources are not pairable" error (§10).

**Unit**

- EOF: weighting (single field, no double cos-lat), Δp computation + fallback, mode orthogonality, variance sums ≤ 1, North δλ + effective-N, sign determinism (incl. dipole), 3-D stacking, standardize-vs-mass-weight exclusion warning.
- Wavelet: reduction helpers + precedence branches (§3.3), AR(1) ordering, both significance calls (local var=1/sigma=0; global dof=N−scales/sigma=1), COI semantics, period-unit derivation.
- Config: `AnalysisSpec` discrimination, `PointReduce`, dependency-cycle/unknown-ref rejection, 1-D-series + explicit reduce → error.
- Labeling: terse titles + correct units/operation words for the three renderers.

**Gates**: `pytest`, `mypy davinci_monet`, `black`/`isort` — locally in the `davinci` env with `HDF5_USE_FILE_LOCKING=FALSE` (GitHub Actions disabled; local gates are source of truth).

---

## 10. Out of Scope (YAGNI)

- **Per-cell wavelet fields** `(time, period, lat, lon)` — heavy, rarely the goal.
- **Cross-wavelet / wavelet coherence** between two series — possible future analysis on the same layer.
- **Rotations beyond varimax**.
- **Hovmöller** and other new spatiotemporal plots.
- **EOF significance beyond North's rule** (Monte-Carlo / bootstrap).
- **Pairing or stats on derived sources is forbidden this release.** The `PairingStage` and `StatisticsStage` raise a clear "derived sources are not pairable/comparable" error when a `pairs:`/`stats:` entry references a `derived` source (keyed off the `derived` marker, §2.2). This removes the fragile untested path that mixed-dimensionality `(time, mode)` PCs under a single GRID geometry would otherwise expose.

---

## 11. Build Sequence — three plans behind a shared foundation

The work is too large and too sequenced for one implementation plan. Decompose into a shared foundation plus two independently shippable feature plans (each gets its own `writing-plans` plan):

- **Plan A — Derived-analysis foundation (no new deps).** `analysis_registry`; `DerivedAnalysis` base; `DataGeometry.SPECTRUM`; `analyses:` schema + `PointReduce` + discriminated union + dependency-DAG validator + `analyses_config()`; `AnalysesStage` (topo-order + pseudo-source construction per §2.2); pairing/stats derived-source guard (§10); single-source `mode:` + `display_level:` plot-spec fields and dispatch hook. Prove end-to-end through the pipeline with a **trivial pass-through analysis** + its integration test.
- **Plan B — EOF (adds + gates `xeofs`).** `EOFAnalysis` (preprocessing order, weighting incl. Δp + fallback, scaling split, sign rule, North error, rotation caveats, outputs); `eof_pattern` + `eof_scree` renderers; PC time-series reuse with the `mode:` selector; synthetic generator + unit + integration tests. Depends on A.
- **Plan C — Wavelet (adds + gates `pycwt`).** `reductions.py`; `WaveletAnalysis` (extraction/preprocessing order, pinned pycwt calls, COI, global spectrum, period units, outputs); `wavelet_scalogram` renderer; synthetic generator + unit + integration tests; verify wavelet-of-EOF-PC end-to-end. Depends on A (and on B only for the wavelet-of-PC integration test).
- **Docs (folded into B/C):** CLAUDE.md `analyses:` documentation; synthetic gallery entries for the new plot types; an example config.
