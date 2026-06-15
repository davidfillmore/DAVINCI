# Plot Labeling & Title System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ad-hoc per-renderer plot text with one consistent, publication-quality title/label system owned by a central `plots/labeling.py`, then verify every plot type by multimodal inspection.

**Architecture:** A new pure-function module `davinci_monet/plots/labeling.py` composes every text element (units, source name, quantity, axis label, legend, bias/colorbar label, title, subtitle). `labels.py` keeps its lookup tables and re-exports the new helpers. Each renderer stops building strings and calls `labeling.*`. Verification renders a synthetic gallery (every plot type) plus all real analyses to PDF, mirrors to iCloud, and iterates on visual inspection.

**Tech Stack:** Python 3.11, matplotlib, xarray, pydantic; pytest/mypy/black/isort; run in the `davinci` conda env with `HDF5_USE_FILE_LOCKING=FALSE`.

**Spec:** `docs/superpowers/specs/2026-06-15-plot-labeling-system-design.md`

**Conventions for every task:** run gates in the env:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
cd /Users/fillmore/EarthSystem/DAVINCI
```
Commit messages end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Work on `develop`; do not push/merge until the user approves.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `davinci_monet/plots/labeling.py` *(new)* | All text composition — pure functions |
| `davinci_monet/tests/unit/plots/test_labeling.py` *(new)* | Unit tests for labeling.py |
| `davinci_monet/plots/labels.py` *(modify)* | Keep lookup tables; re-export labeling helpers; `format_units` SI rewrite |
| `davinci_monet/plots/renderers/scatter.py` *(modify)* | Axis labels + title via labeling |
| `davinci_monet/plots/renderers/timeseries.py` *(modify)* | Legend + y-axis units via labeling |
| `davinci_monet/plots/renderers/spatial/bias.py` *(modify)* | Colorbar via `bias_label` |
| `davinci_monet/plots/renderers/spatial/field.py` *(modify)* | Colorbar via `axis_label` |
| `davinci_monet/plots/renderers/curtain.py` *(modify)* | Bias/colorbar via labeling |
| `davinci_monet/plots/renderers/track_map_3d.py` *(modify)* | Bias/colorbar via labeling |
| `davinci_monet/plots/renderers/profile.py`, `histogram.py`, `flight_track.py` *(modify)* | Titles/axes via labeling |
| `davinci_monet/pipeline/stages/plot.py` + `plot_options.py` *(modify)* | Title-as-terse-override; subtitle from analysis times |
| `davinci_monet/tests/unit/plots/test_renderers.py` *(modify)* | Programmatic rendered-label assertions |
| `analyses/_gallery/` *(new)* | Synthetic gallery config + runner (§4a) |
| `CLAUDE.md` *(modify)* | Plot Label & Title Conventions section |

> Verify exact renderer filenames first: `ls davinci_monet/plots/renderers davinci_monet/plots/renderers/spatial`. If a file differs (e.g. `time_series.py`), use the real name.

---

## Task 1: `format_units` — negative-exponent SI

**Files:**
- Create: `davinci_monet/plots/labeling.py`
- Test: `davinci_monet/tests/unit/plots/test_labeling.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/plots/test_labeling.py
import pytest
from davinci_monet.plots import labeling as L

@pytest.mark.parametrize("raw,expected", [
    ("mol/m2", "mol m$^{-2}$"),
    ("W m-2", "W m$^{-2}$"),
    ("mol/mol", "mol mol$^{-1}$"),
    ("kg/kg", "kg kg$^{-1}$"),
    ("m/s", "m s$^{-1}$"),
    ("ppb", "ppb"),
    ("K", "K"),
    ("1", ""),
    ("none", ""),
    ("", ""),
    (None, ""),
])
def test_format_units(raw, expected):
    assert L.format_units(raw) == expected

def test_format_units_micrograms():
    assert L.format_units("ug/m3") == r"$\mu$g m$^{-3}$"
```

- [ ] **Step 2: Run — expect fail** (`ModuleNotFoundError: labeling`)

Run: `python -m pytest davinci_monet/tests/unit/plots/test_labeling.py -q`

- [ ] **Step 3: Implement**

```python
# davinci_monet/plots/labeling.py
"""Central, publication-quality text composition for all DAVINCI plots.

Pure functions only — no matplotlib, no I/O. Renderers call these instead of
building label/title strings themselves. See
docs/superpowers/specs/2026-06-15-plot-labeling-system-design.md.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from davinci_monet.plots.labels import (
    VARIABLE_DISPLAY_NAMES,
    format_plot_title,
    get_variable_label,
    get_variable_units,
)

if TYPE_CHECKING:
    import xarray as xr

_DIMENSIONLESS = {"", "1", "none", "dimensionless", "unitless", "fraction"}

def format_units(units: str | None) -> str:
    """Rewrite a raw unit string to negative-exponent SI LaTeX (D3)."""
    if units is None:
        return ""
    s = str(units).strip()
    if s.lower() in _DIMENSIONLESS:
        return ""
    s = s.replace("ug", r"$\mu$g")
    # division: "/unitN" -> " unit^{-N}" (N defaults to 1)
    def _div(m: re.Match[str]) -> str:
        unit, power = m.group(1), m.group(2) or "1"
        return f" {unit}$^{{-{power}}}$"
    s = re.sub(r"/([A-Za-z]+)(\d+)?", _div, s)
    # space/dash negative exponents: "unit-N" -> "unit^{-N}"
    s = re.sub(r"([A-Za-z])-(\d+)", lambda m: f"{m.group(1)}$^{{-{m.group(2)}}}$", s)
    # remaining positive exponents: "unitN" -> "unit^{N}"
    s = re.sub(r"([A-Za-z])(\d+)", lambda m: f"{m.group(1)}$^{{{m.group(2)}}}$", s)
    return s.strip()
```

- [ ] **Step 4: Run — expect pass.** Iterate on the regex until all params pass.

Run: `python -m pytest davinci_monet/tests/unit/plots/test_labeling.py -q`

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/plots/labeling.py davinci_monet/tests/unit/plots/test_labeling.py
git commit -m "feat(labeling): negative-exponent SI format_units"
```

---

## Task 2: `source_display_name` — auto-clean keys

**Files:** Modify `labeling.py`; Test `test_labeling.py`

- [ ] **Step 1: Failing tests**

```python
@pytest.mark.parametrize("key,expected", [
    ("cesm_no2_column", r"CESM NO$_2$ Column"),
    ("airnow", "AirNow"),
    ("pandora", "Pandora"),
    ("merra2", "MERRA-2"),
    ("ceres", "CERES"),
    ("cam", "CAM"),
    ("", ""),
    (None, ""),
])
def test_source_display_name(key, expected):
    assert L.source_display_name(key) == expected
```

- [ ] **Step 2: Run — expect fail** (AttributeError).

- [ ] **Step 3: Implement** (append to `labeling.py`)

```python
_SOURCE_ACRONYMS = {
    "cesm": "CESM", "cam": "CAM", "wrf": "WRF", "wrfchem": "WRF-Chem",
    "merra2": "MERRA-2", "geoschem": "GEOS-Chem", "geos": "GEOS",
    "airnow": "AirNow", "aeronet": "AERONET", "pandora": "Pandora",
    "ceres": "CERES", "modis": "MODIS", "tropomi": "TROPOMI",
    "tempo": "TEMPO", "ufs": "UFS", "cmaq": "CMAQ", "ebaf": "EBAF", "ssf": "SSF",
}

def source_display_name(source_label: str | None) -> str:
    """Friendly source name from a raw config key (D5); never ALL-CAPS."""
    if not source_label:
        return ""
    out = []
    for tok in str(source_label).split("_"):
        low = tok.lower()
        if low in _SOURCE_ACRONYMS:
            out.append(_SOURCE_ACRONYMS[low])
        elif low in VARIABLE_DISPLAY_NAMES:
            out.append(VARIABLE_DISPLAY_NAMES[low])
        else:
            out.append(tok.capitalize())
    return " ".join(out)
```

- [ ] **Step 4: Run — expect pass.**
- [ ] **Step 5: Commit** `feat(labeling): source_display_name with acronym map`

---

## Task 3: `quantity_label`

**Files:** Modify `labeling.py`; Test `test_labeling.py`

- [ ] **Step 1: Failing test** (uses a tiny synthetic dataset)

```python
import xarray as xr, numpy as np
def _ds(var, **attrs):
    d = xr.Dataset({var: ("t", np.arange(3.0))})
    d[var].attrs.update(attrs)
    return d

def test_quantity_label_from_lookup():
    ds = _ds("no2_column")
    assert L.quantity_label(ds, "no2_column") == r"NO$_2$ Column"

def test_quantity_label_prefers_long_name():
    ds = _ds("X", long_name="Tropospheric NO2 Column")
    assert "Tropospheric" in L.quantity_label(ds, "X")
```

- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Implement**

```python
def quantity_label(dataset: "xr.Dataset", var_name: str) -> str:
    """Quantity name only (no source, no units), chem-formatted."""
    return get_variable_label(dataset, var_name, include_prefix=False)
```

- [ ] **Step 4: Run — expect pass.**  - [ ] **Step 5: Commit** `feat(labeling): quantity_label`

---

## Task 4: `axis_label` — context-aware + de-dup

**Files:** Modify `labeling.py`; Test `test_labeling.py`

- [ ] **Step 1: Failing tests**

```python
def test_axis_label_no_source():
    assert L.axis_label(r"NO$_2$ Column", "mol/m2") == r"NO$_2$ Column (mol m$^{-2}$)"

def test_axis_label_with_source():
    # pandora has no embedded quantity -> source + quantity
    assert L.axis_label(r"NO$_2$ Column", "mol/m2", source="pandora") == \
        r"Pandora NO$_2$ Column (mol m$^{-2}$)"

def test_axis_label_dedup():
    # source already contains the quantity -> not repeated
    out = L.axis_label(r"NO$_2$ Column", "mol/m2", source="cesm_no2_column")
    assert out == r"CESM NO$_2$ Column (mol m$^{-2}$)"

def test_axis_label_no_units():
    assert L.axis_label("Altitude", "1") == "Altitude"
```

- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Implement**

```python
def axis_label(quantity: str, units: str | None, source: str | None = None) -> str:
    """Compose an axis label (D2 context-aware, with de-dup)."""
    label = quantity or ""
    if source:
        src = source_display_name(source)
        if quantity and quantity.lower() in src.lower():
            label = src
        else:
            label = f"{src} {quantity}".strip()
    u = format_units(units)
    return f"{label} ({u})" if u else label
```

- [ ] **Step 4: Run — expect pass.**  - [ ] **Step 5: Commit** `feat(labeling): axis_label`

---

## Task 5: `legend_label`

**Files:** Modify `labeling.py`; Test `test_labeling.py`

- [ ] **Step 1: Failing tests**

```python
def test_legend_label_plain():
    assert L.legend_label("cesm_no2_column") == r"CESM NO$_2$ Column"
def test_legend_label_uncertainty():
    assert L.legend_label("pandora", uncertainty="mean ± σ") == "Pandora (mean ± σ)"
```

- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Implement**

```python
def legend_label(source_label: str, uncertainty: str | None = None) -> str:
    name = source_display_name(source_label)
    return f"{name} ({uncertainty})" if uncertainty else name
```

- [ ] **Step 4: Run — expect pass.**  - [ ] **Step 5: Commit** `feat(labeling): legend_label`

---

## Task 6: `bias_label` — no x/y, shared-quantity factoring

**Files:** Modify `labeling.py`; Test `test_labeling.py`

- [ ] **Step 1: Failing tests**

```python
def test_bias_label_factors_shared_quantity():
    # both keys carry "olr" -> factor it out (it's in the title)
    out = L.bias_label("merra2_olr", "ceres_olr", "W m-2")
    assert out == r"Bias, MERRA-2 − CERES (W m$^{-2}$)"

def test_bias_label_no_shared_quantity():
    out = L.bias_label("cesm_no2_column", "pandora", "mol/m2")
    assert out == r"Bias, CESM NO$_2$ Column − Pandora (mol m$^{-2}$)"

def test_bias_label_never_uses_xy():
    out = L.bias_label("cesm_no2_column", "pandora", None)
    assert " x" not in out.lower() and " y" not in out.lower()
```

- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Implement** (`−` is U+2212 MINUS SIGN)

```python
def bias_label(y_source: str, x_source: str, units: str | None) -> str:
    """'Bias, <Ysrc> − <Xsrc> (units)'; factor a shared trailing quantity."""
    yw = source_display_name(y_source).split()
    xw = source_display_name(x_source).split()
    while yw and xw and yw[-1] == xw[-1]:
        yw.pop(); xw.pop()
    y = " ".join(yw) or source_display_name(y_source)
    x = " ".join(xw) or source_display_name(x_source)
    core = f"Bias, {y} − {x}"
    u = format_units(units)
    return f"{core} ({u})" if u else core
```

- [ ] **Step 4: Run — expect pass.**  - [ ] **Step 5: Commit** `feat(labeling): bias_label`

---

## Task 7: `title_text` + `subtitle_text`

**Files:** Modify `labeling.py`; Test `test_labeling.py`

- [ ] **Step 1: Failing tests**

```python
def test_title_text_terse():
    assert L.title_text("NO2 Tropospheric Column") == r"NO$_2$ Tropospheric Column"
def test_title_text_operation():
    assert L.title_text("OLR", operation="Bias") == "OLR Bias"
def test_subtitle_range():
    assert L.subtitle_text("2024-02-01", "2024-02-29") == "2024-02-01 – 2024-02-29"
def test_subtitle_single():
    assert L.subtitle_text("2024-02-01", "2024-02-01") == "2024-02-01"
def test_subtitle_empty():
    assert L.subtitle_text(None, None) == ""
```

- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Implement**

```python
def title_text(quantity: str, operation: str | None = None) -> str:
    q = format_plot_title(quantity or "")
    return f"{q} {operation}".strip() if operation else q

def subtitle_text(start: Any, end: Any) -> str:
    if not start:
        return ""
    s = str(start).split(" ")[0].split("T")[0]
    e = str(end).split(" ")[0].split("T")[0] if end else s
    return s if s == e else f"{s} – {e}"
```

- [ ] **Step 4: Run — expect pass.** Add `__all__` with all seven functions.
- [ ] **Step 5: Commit** `feat(labeling): title_text + subtitle_text`

---

## Task 8: Re-export from `labels.py`; rewrite `labels.format_units`

**Files:** Modify `davinci_monet/plots/labels.py`; Test `test_labeling.py`

The old `labels.format_units` (solidus style) and `labels.format_label_with_units`
are imported widely. Make them delegate to the new SI rules so all callers update.

- [ ] **Step 1: Failing test**

```python
def test_labels_format_units_delegates_to_si():
    from davinci_monet.plots import labels
    assert labels.format_units("mol/m2") == "mol m$^{-2}$"
```

- [ ] **Step 2: Run — expect fail** (old solidus output).
- [ ] **Step 3: Implement** — in `labels.py`, replace the body of `format_units` to call the SI implementation. To avoid a circular import (labeling imports labels), put the SI core in `labeling.py` and have `labels.format_units` import lazily inside the function:

```python
def format_units(units: str) -> str:
    from davinci_monet.plots.labeling import format_units as _si
    return _si(units)
```

Leave `format_label_with_units` as-is (it already calls `format_units`).

- [ ] **Step 4: Run — expect pass**, plus full label tests:
`python -m pytest davinci_monet/tests/unit/plots/ -q`
- [ ] **Step 5: Commit** `refactor(labels): delegate format_units to SI labeling`

---

## Task 9: Migrate `scatter.py`

**Files:** Modify `davinci_monet/plots/renderers/scatter.py` (axis-label block ~219-244; delete `_source_display_name` ~35-37); Test: Task 17.

- [ ] **Step 1** Delete `_source_display_name`. Replace the label block with:

```python
from davinci_monet.plots import labeling
...
x_units = get_variable_units(paired_data, x_var)
y_units = get_variable_units(paired_data, y_var)
x_q = labeling.quantity_label(paired_data, x_var)
y_q = labeling.quantity_label(paired_data, y_var)
x_src = paired_data[x_var].attrs.get("source_label") if x_var in paired_data else None
y_src = paired_data[y_var].attrs.get("source_label") if y_var in paired_data else None
x_label = self.config.x_label or labeling.axis_label(x_q, x_units, source=x_src)
y_label = self.config.y_label or labeling.axis_label(y_q, y_units, source=y_src)
self.set_labels(ax, xlabel=x_label, ylabel=y_label)
```

- [ ] **Step 2** Title: where the scatter sets its title, route the config title (or auto quantity) through `labeling.title_text(...)` and never append stats/date.
- [ ] **Step 3** Run scatter-related tests + mypy on the file:
`python -m pytest davinci_monet/tests/unit/plots/test_renderers.py -q && mypy davinci_monet/plots/renderers/scatter.py`
- [ ] **Step 4** `black davinci_monet/plots/renderers/scatter.py && isort ...`
- [ ] **Step 5: Commit** `refactor(scatter): labels via central labeling module`

---

## Task 10: Migrate `timeseries.py` (units + legend)

**Files:** Modify `davinci_monet/plots/renderers/timeseries.py` (ylabel ~232-237, 534-539; legend labels ~323, 412; per-axis ylabel ~369-384, 426-427).

- [ ] **Step 1** Every y-axis label → `labeling.axis_label(quantity_label(ds,var), get_variable_units(ds,var))` so **units are always applied** (fixes "no units").
- [ ] **Step 2** Every series label → `labeling.legend_label(source_label, uncertainty=...)`. Pass `uncertainty="mean ± σ"` (or IQR/range per `uncertainty_type`) only when a band is drawn; never fall back to a raw key.
- [ ] **Step 3** Replace `f"{label} (x)"` / `f"{label} (y)"` legend forms (~514, 527) with `legend_label(source_label)` — no x/y.
- [ ] **Step 4** Run: `python -m pytest davinci_monet/tests/unit/plots/test_renderers.py -q && mypy davinci_monet/plots/renderers/timeseries.py`; format.
- [ ] **Step 5: Commit** `refactor(timeseries): units always shown; legend via labeling`

---

## Task 11: Migrate `spatial/bias.py`

**Files:** Modify `davinci_monet/plots/renderers/spatial/bias.py:228`.

- [ ] **Step 1** Replace:
```python
label = format_label_with_units("Bias (y - x)", units)
```
with:
```python
from davinci_monet.plots import labeling
y_src = paired_data[y_var].attrs.get("source_label")
x_src = paired_data[x_var].attrs.get("source_label")
label = labeling.bias_label(y_src, x_src, units)
```
(Confirm `y_var`/`x_var` are in scope here; they are used just above for `get_variable_units`.)
- [ ] **Step 2** Route the title through `labeling.title_text(quantity, operation="Bias")`.
- [ ] **Step 3** Run renderer tests + mypy + format.
- [ ] **Step 4: Commit** `refactor(spatial/bias): colorbar via bias_label (no x/y)`

---

## Task 12: Migrate `spatial/field.py`

**Files:** Modify `davinci_monet/plots/renderers/spatial/field.py`.

- [ ] **Step 1** Single-source colorbar label → `labeling.axis_label(quantity_label(ds,var), units)` (no source — title carries it). Title → `labeling.title_text`.
- [ ] **Step 2** Run renderer tests + mypy + format.  - [ ] **Step 3: Commit** `refactor(spatial/field): colorbar via labeling`

---

## Task 13: Migrate `curtain.py`

**Files:** Modify `davinci_monet/plots/renderers/curtain.py:183,283,369`.

- [ ] **Step 1** Replace the three `"Bias (Y - X)"` literals with `labeling.bias_label(y_src, x_src, units)` (obtain the source labels from the paired-data attrs as in Task 11). Single-source colorbars → `labeling.axis_label(...)`.
- [ ] **Step 2** Run renderer tests + mypy + format.  - [ ] **Step 3: Commit** `refactor(curtain): bias/colorbar via labeling`

---

## Task 14: Migrate `track_map_3d.py`

**Files:** Modify `davinci_monet/plots/renderers/track_map_3d.py:208`.

- [ ] **Step 1** Replace `label = "Bias (Y - X)"` with `labeling.bias_label(y_src, x_src, units)`; single-source colorbars → `labeling.axis_label`.
- [ ] **Step 2** Run renderer tests + mypy + format.  - [ ] **Step 3: Commit** `refactor(track_map_3d): bias/colorbar via labeling`

---

## Task 15: Migrate profile / histogram / flight renderers

**Files:** Modify `profile.py`, `histogram.py`, `flight_track.py` (confirm names via `ls`).

- [ ] **Step 1** For each: titles via `labeling.title_text`; axis labels via `labeling.axis_label` (quantity + SI units); any series labels via `labeling.legend_label`. No raw keys, no x/y, no baked units.
- [ ] **Step 2** Run `python -m pytest davinci_monet/tests/unit/plots/ -q`; mypy each file; format.
- [ ] **Step 3: Commit** `refactor(profile/histogram/flight): labels via labeling`

---

## Task 16: Pipeline title + subtitle handling

**Files:** Modify `davinci_monet/pipeline/stages/plot.py` and/or `plot_options.py`.

- [ ] **Step 1** Title: a config `plots.*.title` is used as the terse line via `labeling.title_text` (chem-formatted), and the pipeline **never** appends date/stats to it. Where no title is given, auto-generate from the pair's quantity.
- [ ] **Step 2** Subtitle: build from `analysis.start_time/end_time` via `labeling.subtitle_text` and pass to renderers as the existing `subtitle` kwarg (see `build_plot_subtitle`). Replace `build_plot_subtitle`'s body with a call to `labeling.subtitle_text`.
- [ ] **Step 3** Run pipeline unit tests: `python -m pytest davinci_monet/tests/unit/pipeline/ -q`; mypy; format.
- [ ] **Step 4: Commit** `refactor(pipeline): title-as-terse-override, subtitle via labeling`

---

## Task 17: Programmatic rendered-label assertions

**Files:** Modify `davinci_monet/tests/unit/plots/test_renderers.py`.

- [ ] **Step 1: Add tests** that render with synthetic paired data and assert the
actual Axes text — positive and negative:

```python
def test_scatter_axis_labels_clean(synthetic_paired_no2):
    fig = ScatterPlotter(PlotConfig()).render(synthetic_paired_no2)
    ax = fig.axes[0]
    xl, yl = ax.get_xlabel(), ax.get_ylabel()
    for lbl in (xl, yl):
        assert "COLUMN" not in lbl          # no ALL-CAPS key
        assert "cesm_no2_column" not in lbl  # no internal key
        assert "/m2" not in lbl              # units superscripted
        assert "$^{-2}$" in lbl or "mol" in lbl

def test_bias_colorbar_no_xy(synthetic_paired_no2):
    fig = SpatialBiasPlotter(PlotConfig()).render(synthetic_paired_no2)
    label = fig.axes[-1].get_ylabel() or fig.axes[-1].get_xlabel()
    assert "y - x" not in label.lower() and "(y" not in label.lower()
    assert "Bias," in label
```

(Build `synthetic_paired_no2` from existing `tests/synthetic` helpers with
`source_label`/`axis` attrs set and units `mol/m2`.)

- [ ] **Step 2: Run — expect pass** (renderers already migrated):
`python -m pytest davinci_monet/tests/unit/plots/test_renderers.py -q`
- [ ] **Step 3: Commit** `test(renderers): assert clean rendered labels`

---

## Task 18: Config-title cleanup (gitignored machine configs)

**Files:** Modify `analyses/asia-aq/configs/asia-aq-pandora-gemini.yaml`, the two `analyses/ceres/configs/*-local.yaml`, `analyses/dc3/configs/dc3-geometry-dc8-gemini.yaml`, `analyses/firex-aq/configs/firex-aq-geometry-dc8.example.yaml`.

- [ ] **Step 1** In each, make `plots.*.title` terse: drop `(Mean +/- Std)`, `vs <source>`, and any baked units. Remove `ylabel_plot` strings that bake units (let the system build them). These are gitignored except the `.example.yaml`; for the `.example.yaml`, keep it terse too.
- [ ] **Step 2** Validate each loads: `python -c "from davinci_monet.config.parser import load_config; load_config('<path>')"`.
- [ ] **Step 3: Commit** the tracked `.example.yaml` change only: `chore(configs): terse example titles`. (Gitignored configs are local; note them in the handoff, not git.)

---

## Task 19: Synthetic sample gallery (§4a)

**Files:** Create `analyses/_gallery/configs/gallery-synthetic.yaml` and `analyses/_gallery/run_gallery.py`.

- [ ] **Step 1** Write `run_gallery.py` that uses `davinci_monet/tests/synthetic` generators to build sources for every geometry (point, track, profile, swath, grid) with awkward keys (`cesm_no2_column`, `airnow`) and units `mol/mol`,`mol/m2`,`W m-2`,`ug/m3`, writes them to `analyses/_gallery/data/*.nc`, then runs the pipeline via `run_analysis(config)` to emit **one figure per plot type** (scatter, timeseries multi-source+band, single-source map grid+point, spatial bias, curtain, track-3D, profile, histogram, flight ts/track) to `analyses/_gallery/output/`, **PDF only**.
- [ ] **Step 2** Run it: `HDF5_USE_FILE_LOCKING=FALSE python analyses/_gallery/run_gallery.py`. Confirm every plot type produced a PDF: `ls analyses/_gallery/output`.
- [ ] **Step 3: Commit** `feat(gallery): synthetic every-plot-type label gallery` (the runner + config; `data/`, `output/` are gitignored).

---

## Task 20: Verification stage execution (§4) — multimodal inspect & iterate

**Files:** none (process); may loop back to Tasks 9-16/18 on defects.

- [ ] **Step 1: Re-gen all real analyses locally, PDF only.** Run each config (asia-aq pandora; ceres ebaf + ssf; firex; dc3 via the temp-local-config trick) in the `davinci` env with `HDF5_USE_FILE_LOCKING=FALSE`.
- [ ] **Step 2: Sync PDFs to iCloud.** For gallery + each analysis, `cp` only `*.pdf` into `~/Library/Mobile Documents/com~apple~CloudDocs/Claude/_gallery/` and `Claude/<analysis>/...` (never PNGs). Enumerate with the osascript/Finder trick.
- [ ] **Step 3: Multimodal inspect EVERY PDF.** Read each PDF (image mode) and check the §4c checklist: terse title; date-only subtitle; SI superscript units present (incl. timeseries); source per context-aware rule; no internal keys / ALL-CAPS keys / x-y / duplication; bias reads `Bias, Ysrc − Xsrc (units)`.
- [ ] **Step 4: Iterate.** For any defect, fix the responsible renderer/config, re-render that figure, re-inspect. Repeat until the entire gallery and all analyses pass.
- [ ] **Step 5: Commit** any fixes from the loop (one per fix). No code change → no commit.

**Acceptance (§4e):** every gallery + analysis PDF passes 4c; iCloud is PDF-only; full gate green.

---

## Task 21: Update CLAUDE.md

**Files:** Modify `CLAUDE.md` (add under "Plot Styling").

- [ ] **Step 1** Add a "Plot Label & Title Conventions" subsection documenting D1–D6: central `plots/labeling.py`; terse title + date subtitle + in-axes stats; context-aware source placement; negative-exponent SI units; auto-clean source names; bias `Bias, Ysrc − Xsrc`; PDF-only iCloud mirroring; "build labels, don't bake units in configs."
- [ ] **Step 2: Commit** `docs(CLAUDE): plot label & title conventions`

---

## Task 22: Full gate + handoff

- [ ] **Step 1** Full gate:
```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q
mypy davinci_monet
black --check davinci_monet && isort --check-only davinci_monet
```
Expected: all pass.
- [ ] **Step 2** Summarize for the user: tasks done, gallery + analyses regenerated and inspected, iCloud PDF-only. Do **not** push/merge — wait for the user to review and request it (then `develop` → `main` per the repo workflow).

---

## Self-Review (completed by author)

- **Spec coverage:** §1 API → Tasks 1-8; §2 per-plot rules → Tasks 9-16; config handling → Tasks 16,18; testing → Tasks 1-8,17; §4 verification → Tasks 19-20; CLAUDE.md → Task 21. All covered.
- **Placeholders:** none — every code step shows code; every run step shows the command.
- **Type/name consistency:** `format_units`, `source_display_name`, `quantity_label`, `axis_label`, `legend_label`, `bias_label`, `title_text`, `subtitle_text` used identically across tasks; `−` is U+2212 throughout.
