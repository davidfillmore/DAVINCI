# MERRA2 vs MODIS AOD Global Evaluation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a global, multi-year monthly evaluation of MERRA2 reanalysis AOD against MODIS Terra/Aqua L3 AOD, by bootstrapping a minimal `modis_viirs` catalog-driven L3 reader and a new `analyses/merra2-aod/` analysis that runs entirely through `PipelineRunner`.

**Architecture:** A small catalog subsystem (`observations/satellite/catalog/`) provides product metadata (MOD08_M3 / MYD08_M3) as data. A new `MODISVIIRSReader` (registered `modis_viirs`) reads L3 HDF4 grids, resolves variables via the catalog (SDS → display name), parses the month from the filename, and returns a GRID `xr.Dataset`. MERRA2 AOD loads through the existing `generic` reader. The existing `GridStrategy` regrids and pairs the two gridded sources; existing stats/plots/CSV stages finish the run.

**Tech Stack:** Python 3.11, xarray, pydantic v2, netCDF4 (reads HDF4), pytest. Run everything in the `davinci` conda env with `HDF5_USE_FILE_LOCKING=FALSE`.

**Spec:** `docs/superpowers/specs/2026-06-01-merra2-modis-aod-evaluation-design.md`
**Related design:** `docs/superpowers/specs/2026-06-01-modis-viirs-catalog-readers-design.md` (this plan implements a vertical slice of it).

---

## Conventions for every task

- Activate env once per shell session:
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
  ```
- All test/pytest commands are prefixed with `HDF5_USE_FILE_LOCKING=FALSE`.
- Do **not** commit or push unless the user has explicitly approved (repo rule overrides any "commit" step — treat the commit steps below as "stage + prepare commit; ask before pushing"). Commit message footer:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```

---

## File structure

**Create:**
- `davinci_monet/observations/satellite/catalog/__init__.py` — package, re-exports `get_catalog`, `ProductEntry`.
- `davinci_monet/observations/satellite/catalog/schema.py` — pydantic `VariableEntry`, `ProductEntry`.
- `davinci_monet/observations/satellite/catalog/registry.py` — load YAML, `Catalog.resolve(product)`, `get_catalog()`.
- `davinci_monet/observations/satellite/catalog/data/modis_viirs_atmosphere.yaml` — MOD08_M3, MYD08_M3 entries.
- `davinci_monet/observations/satellite/modis_viirs.py` — `MODISVIIRSReader` (`@source_registry.register("modis_viirs")`).
- `davinci_monet/tests/unit/observations/satellite/test_catalog.py` — schema + registry tests.
- `davinci_monet/tests/unit/observations/satellite/test_modis_viirs_reader.py` — reader tests.
- `davinci_monet/tests/integration/test_merra2_modis_aod_pipeline.py` — pipeline integration test.
- `analyses/merra2-aod/configs/merra2-modis-aod.example.yaml` — portable template config.
- `analyses/merra2-aod/scripts/run_evaluation.py` — thin `run_analysis` wrapper.
- `analyses/merra2-aod/.gitignore` — ignore `output/`, `logs/`, `data/`.

**Modify:**
- `davinci_monet/observations/satellite/__init__.py` — import `modis_viirs` and catalog so registration fires.

---

### Task 1: Catalog schema

**Files:**
- Create: `davinci_monet/observations/satellite/catalog/__init__.py`
- Create: `davinci_monet/observations/satellite/catalog/schema.py`
- Test: `davinci_monet/tests/unit/observations/satellite/test_catalog.py`

- [ ] **Step 1: Write the failing test**

```python
# davinci_monet/tests/unit/observations/satellite/test_catalog.py
import pytest
from davinci_monet.observations.satellite.catalog.schema import ProductEntry, VariableEntry


def test_variable_entry_minimal():
    v = VariableEntry(display_name="aod_550nm", sds_name="Aerosol_Optical_Depth_Land_Ocean_Mean_Mean")
    assert v.display_name == "aod_550nm"
    assert v.sds_name.startswith("Aerosol_Optical_Depth")
    assert v.units == "1"  # default


def test_product_entry_resolves_variable_by_display_and_sds():
    p = ProductEntry(
        product_id="MOD08_M3",
        instrument="MODIS",
        platform="Terra",
        daac="LAADS",
        collection="061",
        level="L3",
        geometry="GRID",
        file_format="HDF4",
        time_parse="A%Y%j",
        dim_aliases={"XDim": "lon", "YDim": "lat"},
        variables=[VariableEntry(display_name="aod_550nm", sds_name="Aerosol_Optical_Depth_Land_Ocean_Mean_Mean")],
    )
    assert p.variable_by_display("aod_550nm").sds_name.startswith("Aerosol_Optical_Depth")
    assert p.variable_by_sds("Aerosol_Optical_Depth_Land_Ocean_Mean_Mean").display_name == "aod_550nm"
    assert p.variable_by_display("nope") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/observations/satellite/test_catalog.py -v`
Expected: FAIL with `ModuleNotFoundError: ...catalog.schema`.

- [ ] **Step 3: Write minimal implementation**

```python
# davinci_monet/observations/satellite/catalog/schema.py
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VariableEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str
    sds_name: str
    units: str = "1"
    wavelength_nm: float | None = None
    long_name: str | None = None


class ProductEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str
    instrument: str
    platform: str
    daac: str
    collection: str
    level: str
    geometry: str
    file_format: str
    time_parse: str  # strptime pattern applied to the filename "A%Y%j" token
    dim_aliases: dict[str, str] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)
    variables: list[VariableEntry] = Field(default_factory=list)

    def variable_by_display(self, name: str) -> VariableEntry | None:
        return next((v for v in self.variables if v.display_name == name), None)

    def variable_by_sds(self, name: str) -> VariableEntry | None:
        return next((v for v in self.variables if v.sds_name == name), None)
```

```python
# davinci_monet/observations/satellite/catalog/__init__.py
from davinci_monet.observations.satellite.catalog.schema import ProductEntry, VariableEntry

__all__ = ["ProductEntry", "VariableEntry"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/observations/satellite/test_catalog.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/observations/satellite/catalog/__init__.py \
        davinci_monet/observations/satellite/catalog/schema.py \
        davinci_monet/tests/unit/observations/satellite/test_catalog.py
git commit -m "feat(satellite): add MODIS/VIIRS catalog product schema"
```

---

### Task 2: Catalog data + registry

**Files:**
- Create: `davinci_monet/observations/satellite/catalog/data/modis_viirs_atmosphere.yaml`
- Create: `davinci_monet/observations/satellite/catalog/registry.py`
- Modify: `davinci_monet/observations/satellite/catalog/__init__.py`
- Test: `davinci_monet/tests/unit/observations/satellite/test_catalog.py` (extend)

- [ ] **Step 1: Write the catalog data file** (not a test step — concrete data the registry loads)

```yaml
# davinci_monet/observations/satellite/catalog/data/modis_viirs_atmosphere.yaml
products:
  - product_id: MOD08_M3
    instrument: MODIS
    platform: Terra
    daac: LAADS
    collection: "061"
    level: L3
    geometry: GRID
    file_format: HDF4
    time_parse: "A%Y%j"
    dim_aliases: { XDim: lon, YDim: lat, "XDim:mod08": lon, "YDim:mod08": lat }
    variables:
      - display_name: aod_550nm
        sds_name: Aerosol_Optical_Depth_Land_Ocean_Mean_Mean
        units: "1"
        wavelength_nm: 550
        long_name: "AOD at 0.55um (land+ocean), monthly mean of daily mean"
  - product_id: MYD08_M3
    instrument: MODIS
    platform: Aqua
    daac: LAADS
    collection: "061"
    level: L3
    geometry: GRID
    file_format: HDF4
    time_parse: "A%Y%j"
    dim_aliases: { XDim: lon, YDim: lat, "XDim:mod08": lon, "YDim:mod08": lat }
    variables:
      - display_name: aod_550nm
        sds_name: Aerosol_Optical_Depth_Land_Ocean_Mean_Mean
        units: "1"
        wavelength_nm: 550
        long_name: "AOD at 0.55um (land+ocean), monthly mean of daily mean"
```

- [ ] **Step 2: Write the failing test**

```python
# append to test_catalog.py
from davinci_monet.observations.satellite.catalog.registry import get_catalog, UnknownProductError


def test_catalog_resolves_known_products():
    cat = get_catalog()
    terra = cat.resolve("MOD08_M3")
    aqua = cat.resolve("MYD08_M3")
    assert terra.platform == "Terra"
    assert aqua.platform == "Aqua"
    assert terra.variable_by_display("aod_550nm").wavelength_nm == 550


def test_catalog_unknown_product_suggests_matches():
    cat = get_catalog()
    with pytest.raises(UnknownProductError) as exc:
        cat.resolve("MOD08_X3")
    assert "MOD08_M3" in str(exc.value)  # close-match suggestion
```

- [ ] **Step 3: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/observations/satellite/test_catalog.py -v`
Expected: FAIL with `ImportError: ...registry`.

- [ ] **Step 4: Write minimal implementation**

```python
# davinci_monet/observations/satellite/catalog/registry.py
from __future__ import annotations

import difflib
from functools import lru_cache
from pathlib import Path

import yaml

from davinci_monet.observations.satellite.catalog.schema import ProductEntry

_DATA_DIR = Path(__file__).parent / "data"


class UnknownProductError(KeyError):
    """Raised when a product id/alias is not in the catalog."""


class Catalog:
    def __init__(self, products: list[ProductEntry]) -> None:
        self._by_key: dict[str, ProductEntry] = {}
        for p in products:
            self._by_key[p.product_id] = p
            for alias in p.aliases:
                self._by_key[alias] = p

    def resolve(self, product: str) -> ProductEntry:
        if product in self._by_key:
            return self._by_key[product]
        close = difflib.get_close_matches(product, list(self._by_key), n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise UnknownProductError(f"Unknown MODIS/VIIRS product '{product}'.{hint}")

    def product_ids(self) -> list[str]:
        return sorted(self._by_key)


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    products: list[ProductEntry] = []
    for yaml_path in sorted(_DATA_DIR.glob("*.yaml")):
        raw = yaml.safe_load(yaml_path.read_text()) or {}
        for entry in raw.get("products", []):
            products.append(ProductEntry(**entry))
    return Catalog(products)
```

Update `__init__.py`:

```python
# davinci_monet/observations/satellite/catalog/__init__.py
from davinci_monet.observations.satellite.catalog.registry import (
    Catalog,
    UnknownProductError,
    get_catalog,
)
from davinci_monet.observations.satellite.catalog.schema import ProductEntry, VariableEntry

__all__ = ["Catalog", "UnknownProductError", "get_catalog", "ProductEntry", "VariableEntry"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/observations/satellite/test_catalog.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add davinci_monet/observations/satellite/catalog/
git add davinci_monet/tests/unit/observations/satellite/test_catalog.py
git commit -m "feat(satellite): add MODIS/VIIRS catalog registry + atmosphere products"
```

---

### Task 3: MODISVIIRSReader (L3 grid path)

**Files:**
- Create: `davinci_monet/observations/satellite/modis_viirs.py`
- Test: `davinci_monet/tests/unit/observations/satellite/test_modis_viirs_reader.py`

**Interface contract** (from `pipeline/stages.py:1252-1319`):
`reader.open(file_paths, variables=<list|None>, *, product=<str>, level=None, time_range=None, **kwargs)` → returns `xr.Dataset` with a `time` coord; the reader sets `self.geometry = DataGeometry.GRID`. `variables` are catalog **display names** (or `None`/`["*"]` for all cataloged).

> **Test fixture note:** Generating real HDF4 in tests is impractical. The reader opens via `xr.open_dataset(engine="netcdf4")`, which also reads `.nc`. The unit test writes a `.nc` fixture mirroring the MOD08 layout (dims `YDim:mod08`/`XDim:mod08`, 1-D `XDim`/`YDim` coordinate variables, an AOD data var named with the SDS name). xarray applies CF scale/fill on read, so the fixture stores already-physical values and the test asserts rename/time/geometry behavior. The HDF4-specific decode path is covered by the real-data smoke test (Task 8).

- [ ] **Step 1: Write the failing test**

```python
# davinci_monet/tests/unit/observations/satellite/test_modis_viirs_reader.py
import numpy as np
import xarray as xr
import pytest

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.observations.satellite.modis_viirs import MODISVIIRSReader


def _write_mod08_like(path, fname):
    # 4x8 grid mirroring MOD08_M3 dim/coord layout
    lat = np.linspace(89.5, -89.5, 4)
    lon = np.linspace(-179.5, 179.5, 8)
    aod = np.random.default_rng(0).uniform(0.0, 1.0, size=(4, 8)).astype("float32")
    ds = xr.Dataset(
        {"Aerosol_Optical_Depth_Land_Ocean_Mean_Mean": (("YDim:mod08", "XDim:mod08"), aod)},
        coords={"YDim": ("YDim:mod08", lat), "XDim": ("XDim:mod08", lon)},
    )
    fpath = path / fname
    ds.to_netcdf(fpath)
    return str(fpath)


def test_reader_returns_grid_with_time_and_display_name(tmp_path):
    # 2024-032 = day-of-year 32 = 2024-02-01
    f = _write_mod08_like(tmp_path, "MOD08_M3.A2024032.061.0000.nc")
    reader = MODISVIIRSReader()
    ds = reader.open([f], variables=["aod_550nm"], product="MOD08_M3")

    assert reader.geometry == DataGeometry.GRID
    assert "aod_550nm" in ds.data_vars                       # SDS renamed to display name
    assert {"lat", "lon"}.issubset(set(ds.coords))           # XDim/YDim → lon/lat
    assert "time" in ds.coords
    assert str(ds["time"].values[0])[:7] == "2024-02"        # filename month parsed
    assert ds["aod_550nm"].attrs.get("wavelength_nm") == 550


def test_reader_concatenates_months_sorted(tmp_path):
    f2 = _write_mod08_like(tmp_path, "MOD08_M3.A2024060.061.0000.nc")  # ~Mar
    f1 = _write_mod08_like(tmp_path, "MOD08_M3.A2024032.061.0000.nc")  # Feb
    reader = MODISVIIRSReader()
    ds = reader.open([f2, f1], variables=["aod_550nm"], product="MOD08_M3")
    months = [str(t)[:7] for t in ds["time"].values]
    assert months == ["2024-02", "2024-03"]                  # sorted ascending


def test_reader_unknown_product_raises(tmp_path):
    f = _write_mod08_like(tmp_path, "MOD08_M3.A2024032.061.0000.nc")
    reader = MODISVIIRSReader()
    with pytest.raises(Exception):
        reader.open([f], variables=["aod_550nm"], product="NOPE_M3")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/observations/satellite/test_modis_viirs_reader.py -v`
Expected: FAIL with `ModuleNotFoundError: ...modis_viirs`.

- [ ] **Step 3: Write minimal implementation**

```python
# davinci_monet/observations/satellite/modis_viirs.py
"""Catalog-driven MODIS/VIIRS reader (L3 grid slice).

Initial vertical slice of the modis_viirs catalog reader design: Level-3
regular-grid atmosphere products (MOD08_M3 / MYD08_M3). See
docs/superpowers/specs/2026-06-01-modis-viirs-catalog-readers-design.md.
"""
from __future__ import annotations

import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from davinci_monet.core.exceptions import DataNotFoundError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.observations.satellite.catalog import ProductEntry, get_catalog

_DATE_TOKEN = re.compile(r"\.A(\d{7})\.")  # ".A2024032." → 2024032


@source_registry.register("modis_viirs")
class MODISVIIRSReader:
    """Catalog-driven reader for MODIS/VIIRS products (L3 grid path)."""

    def __init__(self) -> None:
        self.geometry: DataGeometry = DataGeometry.GRID

    @property
    def name(self) -> str:
        return "modis_viirs"

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        *,
        product: str | None = None,
        level: str | None = None,
        time_range: tuple[Any, Any] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        if not product:
            raise ValueError("modis_viirs source requires a 'product' (e.g. MOD08_M3).")
        entry = get_catalog().resolve(product)  # raises UnknownProductError if absent
        if level and level.upper() != entry.level.upper():
            raise ValueError(
                f"Configured level '{level}' != catalog level '{entry.level}' for {product}."
            )
        if entry.geometry.upper() != "GRID":
            raise NotImplementedError(
                f"modis_viirs slice supports GRID products only; {product} is {entry.geometry}."
            )

        files = [Path(f) for f in file_paths]
        missing = [f for f in files if not f.exists()]
        if not files:
            raise DataNotFoundError("No MODIS/VIIRS files provided")
        if missing:
            raise DataNotFoundError(f"MODIS/VIIRS files not found: {missing}")

        per_file = [self._open_one(f, variables, entry) for f in files]
        per_file = [d for d in per_file if d is not None]
        if not per_file:
            raise DataNotFoundError("No valid MODIS/VIIRS data found")

        ds = xr.concat(per_file, dim="time").sortby("time")
        self.geometry = DataGeometry.GRID
        ds.attrs["geometry"] = DataGeometry.GRID.value
        ds.attrs.update(
            product_id=entry.product_id,
            instrument=entry.instrument,
            platform=entry.platform,
            level=entry.level,
            daac=entry.daac,
            collection=entry.collection,
        )
        return ds

    def _open_one(
        self, fpath: Path, variables: Sequence[str] | None, entry: ProductEntry
    ) -> xr.Dataset | None:
        try:
            raw = xr.open_dataset(str(fpath), engine="netcdf4", mask_and_scale=True)
        except Exception as e:  # pragma: no cover - exercised via smoke test
            warnings.warn(f"Failed to open {fpath}: {e}", UserWarning)
            return None

        # Resolve which catalog variables to keep (display names → SDS names).
        wanted = list(variables) if variables else None
        if wanted in (None, ["*"]):
            selected = entry.variables
        else:
            selected = []
            for name in wanted:
                v = entry.variable_by_display(name) or entry.variable_by_sds(name)
                if v is None:
                    warnings.warn(f"{entry.product_id}: variable '{name}' not in catalog", UserWarning)
                    continue
                selected.append(v)

        keep = {v.sds_name: v for v in selected if v.sds_name in raw.data_vars}
        if not keep:
            warnings.warn(f"{fpath.name}: none of the requested SDS present", UserWarning)
            return None
        ds = raw[list(keep)]

        # Rename SDS → display name and attach variable metadata.
        ds = ds.rename({sds: v.display_name for sds, v in keep.items()})
        for v in keep.values():
            attrs = ds[v.display_name].attrs
            attrs["units"] = v.units
            if v.wavelength_nm is not None:
                attrs["wavelength_nm"] = v.wavelength_nm
            if v.long_name:
                attrs["long_name"] = v.long_name

        # Standardize grid coords: dim_aliases maps file dim/coord names → lon/lat.
        ds = self._standardize_grid(ds, entry)

        # Parse the month from the filename and assign a time coordinate.
        ds = ds.expand_dims(time=[self._parse_time(fpath.name, entry)])
        return ds

    @staticmethod
    def _standardize_grid(ds: xr.Dataset, entry: ProductEntry) -> xr.Dataset:
        renames = {k: v for k, v in entry.dim_aliases.items() if k in ds.dims or k in ds.coords}
        if renames:
            ds = ds.rename(renames)
        # Promote lon/lat to coordinates if they are bare 1-D variables.
        for axis in ("lon", "lat"):
            if axis in ds.variables and axis not in ds.coords:
                ds = ds.set_coords(axis)
        return ds

    @staticmethod
    def _parse_time(filename: str, entry: ProductEntry) -> np.datetime64:
        m = _DATE_TOKEN.search(filename)
        if not m:
            raise ValueError(f"Cannot parse date token from filename: {filename}")
        dt = datetime.strptime("A" + m.group(1), entry.time_parse)
        # Monthly products: snap to first-of-month for clean monthly alignment.
        return np.datetime64(pd.Timestamp(dt).to_period("M").to_timestamp(), "ns")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/observations/satellite/test_modis_viirs_reader.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/observations/satellite/modis_viirs.py \
        davinci_monet/tests/unit/observations/satellite/test_modis_viirs_reader.py
git commit -m "feat(satellite): add modis_viirs catalog reader (L3 grid path)"
```

---

### Task 4: Register the reader on package import

**Files:**
- Modify: `davinci_monet/observations/satellite/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_modis_viirs_reader.py
def test_reader_is_registered():
    import davinci_monet.observations  # triggers package import / registration
    from davinci_monet.core.registry import source_registry
    assert "modis_viirs" in source_registry.list()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/observations/satellite/test_modis_viirs_reader.py::test_reader_is_registered -v`
Expected: FAIL — `modis_viirs` not in registry (module never imported).

- [ ] **Step 3: Add the import**

In `davinci_monet/observations/satellite/__init__.py`, alongside the other reader imports (e.g. near `from davinci_monet.observations.satellite.goes_l3_aod import ...`), add:

```python
from davinci_monet.observations.satellite import modis_viirs  # noqa: F401  (registers "modis_viirs")
from davinci_monet.observations.satellite.modis_viirs import MODISVIIRSReader
```

And add `"MODISVIIRSReader"` to the module's `__all__` list.

- [ ] **Step 4: Run test to verify it passes**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/observations/satellite/test_modis_viirs_reader.py::test_reader_is_registered -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/observations/satellite/__init__.py
git commit -m "feat(satellite): register modis_viirs reader on package import"
```

---

### Task 5: Verify MERRA2 loads as GRID through the generic reader

This is a guard test — confirm the existing `generic` reader exposes a 2-D `(time, lat, lon)` field as GRID with no `lev`, so no MERRA2-specific reader is needed.

**Files:**
- Test: `davinci_monet/tests/unit/models/test_generic_grid_geometry.py` (Create)

- [ ] **Step 1: Write the test**

```python
# davinci_monet/tests/unit/models/test_generic_grid_geometry.py
import numpy as np
import xarray as xr

from davinci_monet.core.registry import source_registry
import davinci_monet.models  # noqa: F401  (registers "generic")


def test_generic_reader_tags_2d_field_as_grid(tmp_path):
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(-180, 175, 8)
    t = np.array(["2024-02-01"], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {"TOTEXTTAU": (("time", "lat", "lon"), np.random.default_rng(0).uniform(0, 1, (1, 5, 8)))},
        coords={"time": t, "lat": lat, "lon": lon},
    )
    f = tmp_path / "merra2_like.nc"
    ds.to_netcdf(f)

    reader = source_registry.get("generic")()
    out = reader.open([str(f)], variables=["TOTEXTTAU"])
    assert "TOTEXTTAU" in out
    assert "lev" not in out.dims
    geom = getattr(reader, "geometry")
    assert str(getattr(geom, "value", geom)).upper() == "GRID"
```

- [ ] **Step 2: Run the test**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/unit/models/test_generic_grid_geometry.py -v`
Expected: PASS. **If it fails** (generic reader does not set `geometry`/tags it non-GRID for 2-D fields), STOP and use `superpowers:systematic-debugging`: inspect `davinci_monet/models/generic.py` to see how `geometry` is determined, and decide with the user whether to (a) fix generic's geometry inference for 2-D fields or (b) add a thin `merra2` registration. Do not silently work around it.

- [ ] **Step 3: Commit**

```bash
git add davinci_monet/tests/unit/models/test_generic_grid_geometry.py
git commit -m "test(models): assert generic reader tags 2-D field as GRID (MERRA2 AOD)"
```

---

### Task 6: Analysis directory scaffolding

**Files:**
- Create: `analyses/merra2-aod/configs/merra2-modis-aod.example.yaml`
- Create: `analyses/merra2-aod/scripts/run_evaluation.py`
- Create: `analyses/merra2-aod/.gitignore`

- [ ] **Step 1: Write the config**

```yaml
# analyses/merra2-aod/configs/merra2-modis-aod.example.yaml
# MERRA2 vs MODIS Terra/Aqua AOD — global monthly evaluation.
#
# Required environment variables:
#   MERRA2_DATA          — dir containing MERRA2_tavgM/aer_Nx/*.nc4
#   MODIS_DATA           — dir containing MOD08_M3/ and MYD08_M3/ (*.hdf)
#   MERRA2_AOD_ANALYSIS  — analysis dir (parent of output/ and logs/)
#
# Usage:
#   export MERRA2_DATA=/Volumes/Io/MERRA2_tavgM
#   export MODIS_DATA=/Volumes/Io
#   export MERRA2_AOD_ANALYSIS=$(pwd)
#   HDF5_USE_FILE_LOCKING=FALSE davinci-monet run analyses/merra2-aod/configs/merra2-modis-aod.example.yaml

analysis:
  # Default to a short window for a fast first run; widen to sweep the full
  # archive (Terra >=2000-02, Aqua >=2002-07, MERRA2 -> 2026-03).
  start_time: "2003-01-01"
  end_time: "2003-12-31"
  output_dir: ${MERRA2_AOD_ANALYSIS}/output
  log_dir: ${MERRA2_AOD_ANALYSIS}/logs
  style:
    theme: ncar
    context: default

sources:
  merra2:
    type: generic
    role: model
    files: ${MERRA2_DATA}/aer_Nx/*.nc4
    variables:
      TOTEXTTAU:
        units: "1"
        ylabel_plot: "AOD (550 nm)"
        vmin_plot: 0
        vmax_plot: 1
        vdiff_plot: 0.3
  modis_terra:
    type: modis_viirs
    role: obs
    product: MOD08_M3
    files: ${MODIS_DATA}/MOD08_M3/*.hdf
    variables:
      aod_550nm:
        units: "1"
  modis_aqua:
    type: modis_viirs
    role: obs
    product: MYD08_M3
    files: ${MODIS_DATA}/MYD08_M3/*.hdf
    variables:
      aod_550nm:
        units: "1"

pairs:
  merra2_vs_terra:
    sources: [merra2, modis_terra]
    reference: modis_terra
    variables: { merra2: TOTEXTTAU, modis_terra: aod_550nm }
  merra2_vs_aqua:
    sources: [merra2, modis_aqua]
    reference: modis_aqua
    variables: { merra2: TOTEXTTAU, modis_aqua: aod_550nm }

plots:
  terra_aod_spatial_bias:
    type: spatial_bias
    pairs: [merra2_vs_terra]
    title: "AOD Bias: MERRA2 - MODIS Terra"
  aqua_aod_spatial_bias:
    type: spatial_bias
    pairs: [merra2_vs_aqua]
    title: "AOD Bias: MERRA2 - MODIS Aqua"
  terra_aod_scatter:
    type: scatter
    pairs: [merra2_vs_terra]
    show_density: true
    title: "AOD: MERRA2 vs MODIS Terra"
  aqua_aod_scatter:
    type: scatter
    pairs: [merra2_vs_aqua]
    show_density: true
    title: "AOD: MERRA2 vs MODIS Aqua"
  aod_timeseries:
    type: timeseries
    pairs: [merra2_vs_terra, merra2_vs_aqua]
    title: "Global-Mean Monthly AOD"
    aggregate_dim: [lat, lon]

stats:
  output_table: true
  output_table_kwargs:
    precision: 3
  metrics: [N, MO, MP, MB, RMSE, R, NMB, NME, IOA]
```

- [ ] **Step 2: Write the run script**

```python
# analyses/merra2-aod/scripts/run_evaluation.py
"""Run the MERRA2 vs MODIS AOD evaluation through the DAVINCI pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

from davinci_monet.pipeline.runner import run_analysis

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "merra2-modis-aod.example.yaml"


def main() -> int:
    config = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    result = run_analysis(str(config))
    if result.success:
        print(f"Completed in {result.total_duration_seconds:.1f}s")
        return 0
    print("Analysis failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Write the .gitignore**

```gitignore
# analyses/merra2-aod/.gitignore
output/
logs/
data/
*-gemini.yaml
*-derecho.yaml
```

- [ ] **Step 4: Validate the config parses**

Run:
```bash
HDF5_USE_FILE_LOCKING=FALSE python -c "from davinci_monet.config.parser import load_config; load_config('analyses/merra2-aod/configs/merra2-modis-aod.example.yaml'); print('config OK')"
```
Expected: `config OK` (no validation error). **If `load_config` has a different name**, find the loader with `grep -rn "def load_config\|def parse_config\|def load" davinci_monet/config/parser.py` and use it.

- [ ] **Step 5: Commit**

```bash
git add analyses/merra2-aod/
git commit -m "feat(analyses): add merra2-aod global AOD evaluation scaffolding"
```

---

### Task 7: Pipeline integration test (grid-to-grid, real pipeline path)

Per repo rule, integration tests run through `PipelineRunner.run_from_config()`. This test exercises the **full path** with two synthetic GRID sources, asserting paired output, stats CSV, and generated plots — which also validates that `spatial_bias`/`scatter`/`timeseries` renderers handle GRID-paired data (the renderer-audit open item).

**Files:**
- Test: `davinci_monet/tests/integration/test_merra2_modis_aod_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# davinci_monet/tests/integration/test_merra2_modis_aod_pipeline.py
import numpy as np
import xarray as xr
import yaml
import pytest

from davinci_monet.pipeline.runner import PipelineRunner


def _grid(varname, nt=3, ny=6, nx=8, seed=0):
    rng = np.random.default_rng(seed)
    t = np.array(["2003-01-01", "2003-02-01", "2003-03-01"], dtype="datetime64[ns]")[:nt]
    lat = np.linspace(-87.5, 87.5, ny)
    lon = np.linspace(-175, 175, nx)
    data = rng.uniform(0.05, 0.8, size=(nt, ny, nx))
    return xr.Dataset({varname: (("time", "lat", "lon"), data)},
                      coords={"time": t, "lat": lat, "lon": lon})


def test_merra2_modis_aod_pipeline(tmp_path):
    # Two MERRA2-like and MODIS-like generic GRID sources (both readable by `generic`).
    merra2 = _grid("TOTEXTTAU", seed=1)
    modis = _grid("aod_550nm", seed=2)
    (tmp_path / "merra2").mkdir(); (tmp_path / "modis").mkdir()
    merra2.to_netcdf(tmp_path / "merra2" / "merra2.nc")
    modis.to_netcdf(tmp_path / "modis" / "modis.nc")

    out = tmp_path / "output"; logs = tmp_path / "logs"
    config = {
        "analysis": {"start_time": "2003-01-01", "end_time": "2003-03-31",
                     "output_dir": str(out), "log_dir": str(logs)},
        "sources": {
            "merra2": {"type": "generic", "role": "model",
                       "files": str(tmp_path / "merra2" / "*.nc"),
                       "variables": {"TOTEXTTAU": {"units": "1"}}},
            "modis_terra": {"type": "generic", "role": "obs",
                            "files": str(tmp_path / "modis" / "*.nc"),
                            "variables": {"aod_550nm": {"units": "1"}}},
        },
        "pairs": {"merra2_vs_terra": {"sources": ["merra2", "modis_terra"],
                                      "reference": "modis_terra",
                                      "variables": {"merra2": "TOTEXTTAU", "modis_terra": "aod_550nm"}}},
        "plots": {
            "bias": {"type": "spatial_bias", "pairs": ["merra2_vs_terra"], "title": "AOD bias"},
            "sc": {"type": "scatter", "pairs": ["merra2_vs_terra"], "title": "AOD scatter"},
            "ts": {"type": "timeseries", "pairs": ["merra2_vs_terra"], "title": "AOD ts",
                   "aggregate_dim": ["lat", "lon"]},
        },
        "stats": {"output_table": True, "metrics": ["N", "MB", "RMSE", "R"]},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    runner = PipelineRunner()
    result = runner.run_from_config(str(cfg_path))

    assert result.success, getattr(result, "error", "pipeline failed")
    plots = list(out.rglob("*.png"))
    assert len(plots) >= 3, f"expected >=3 plots, got {[p.name for p in plots]}"
    assert list(out.rglob("*.csv")), "expected a stats CSV"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/integration/test_merra2_modis_aod_pipeline.py -v`
Expected: FAIL initially. Likely failure modes and how to handle:
- `run_from_config` signature/name differs → `grep -rn "def run_from_config\|def run_analysis" davinci_monet/pipeline/runner.py` and adapt.
- Grid-to-grid pairing or a renderer errors on GRID-paired data → this is the renderer-audit item; use `superpowers:systematic-debugging` and extend the renderer (do not bypass the pipeline to force a pass — repo testing rule #3).

- [ ] **Step 3: Make it pass**

Resolve whatever the real pipeline surfaces (API name, renderer GRID support). Keep the test on the real `run_from_config` path.

- [ ] **Step 4: Run the full suite + type/format checks**

```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest
mypy davinci_monet
black --check davinci_monet && isort --check davinci_monet
```
Expected: all green. Fix `black`/`isort` by running them without `--check` if needed.

- [ ] **Step 5: Commit**

```bash
git add davinci_monet/tests/integration/test_merra2_modis_aod_pipeline.py davinci_monet
git commit -m "test(integration): MERRA2 vs MODIS AOD grid pipeline through PipelineRunner"
```

---

### Task 8: Real-data smoke test (skipped unless Io env vars set)

**Files:**
- Test: `davinci_monet/tests/integration/test_merra2_modis_aod_pipeline.py` (extend)

- [ ] **Step 1: Add the smoke test**

```python
# append to test_merra2_modis_aod_pipeline.py
import os
from pathlib import Path

@pytest.mark.skipif(
    not (os.environ.get("MERRA2_DATA") and os.environ.get("MODIS_DATA")),
    reason="set MERRA2_DATA and MODIS_DATA to run the real-data smoke test",
)
def test_real_data_one_month(tmp_path):
    """Read one real MOD08_M3 HDF4 file + MERRA2 month through the readers."""
    from davinci_monet.observations.satellite.modis_viirs import MODISVIIRSReader
    modis_dir = Path(os.environ["MODIS_DATA"]) / "MOD08_M3"
    files = sorted(modis_dir.glob("*.hdf"))[:1]
    assert files, f"no MOD08_M3 files under {modis_dir}"
    ds = MODISVIIRSReader().open([str(files[0])], variables=["aod_550nm"], product="MOD08_M3")
    assert "aod_550nm" in ds and "time" in ds.coords
    assert {"lat", "lon"}.issubset(ds.coords)
    assert float(ds["aod_550nm"].max()) < 10.0  # physical AOD, scale applied
```

- [ ] **Step 2: Run it locally with Io mounted**

```bash
export MERRA2_DATA=/Volumes/Io/MERRA2_tavgM
export MODIS_DATA=/Volumes/Io
HDF5_USE_FILE_LOCKING=FALSE python -m pytest \
  davinci_monet/tests/integration/test_merra2_modis_aod_pipeline.py::test_real_data_one_month -v
```
Expected: PASS (confirms the HDF4 decode path + scale/fill on real data). If the AOD comes back unscaled (max ~1000s), fix scale handling in the reader (`mask_and_scale` / explicit `scale_factor`) and re-run.

- [ ] **Step 3: Commit**

```bash
git add davinci_monet/tests/integration/test_merra2_modis_aod_pipeline.py
git commit -m "test(integration): real-data MOD08 smoke test (env-gated)"
```

---

### Task 9: End-to-end run on Io data + deliver plots

Not a TDD task — the real run the user wants, with output sent to the iCloud Claude folder per repo convention.

- [ ] **Step 1: Run a short window first**

```bash
export MERRA2_DATA=/Volumes/Io/MERRA2_tavgM
export MODIS_DATA=/Volumes/Io
export MERRA2_AOD_ANALYSIS="$(pwd)/analyses/merra2-aod"
HDF5_USE_FILE_LOCKING=FALSE davinci-monet run analyses/merra2-aod/configs/merra2-modis-aod.example.yaml
```
Expected: pipeline completes; plots + stats CSV land in `analyses/merra2-aod/output/`.

- [ ] **Step 2: Sanity-check the numbers**

Open the stats CSV; AOD means should be O(0.1–0.5) globally, R between MERRA2 and MODIS positive. If MERRA2 AOD is wildly off, re-check `TOTEXTTAU` selection and that no spurious `unit_scale` was applied.

- [ ] **Step 3: Copy plots to the iCloud Claude folder**

```bash
cp analyses/merra2-aod/output/**/*.png "$HOME/Library/Mobile Documents/com~apple~CloudDocs/Claude/" 2>/dev/null || \
  find analyses/merra2-aod/output -name '*.png' -exec cp {} "$HOME/Library/Mobile Documents/com~apple~CloudDocs/Claude/" \;
```

- [ ] **Step 4: Report to the user** — summarize stats per pair (Terra, Aqua), note the window used, and ask whether to widen to the full archive before any merge.

---

## Self-review notes (author)

- **Spec coverage:** catalog slice (Tasks 1–4) ↔ spec "Component 1"; MERRA2 generic (Task 5) ↔ "Component 2"; analysis dir/config (Task 6) ↔ "Component 3" + config shape; integration + renderer audit (Task 7) ↔ "Open items" + "Testing"; smoke test (Task 8) ↔ "Testing"; run + iCloud delivery (Task 9) ↔ repo output convention. Plots/stats sets match the spec.
- **Display-vs-SDS naming:** config `variables:` use display name `aod_550nm`; reader resolves via catalog and renames SDS→display; `_apply_variable_config` then matches on `aod_550nm`. Consistent across Tasks 3, 6, 7.
- **Geometry contract:** reader sets `self.geometry` (Task 3) because the unified loader reads `getattr(reader, "geometry")` (`stages.py:1309`); Task 5 guards the MERRA2 side.
- **Known unknowns deferred to runtime, not papered over:** generic-GRID tagging (Task 5 guard), renderer GRID-paired support (Task 7), HDF4 scale on real data (Task 8) — each has an explicit debug/stop instruction rather than a workaround.
