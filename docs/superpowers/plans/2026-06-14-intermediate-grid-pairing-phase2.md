# Intermediate-Gridding Pairing — Phase 2 (3-D altitude) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 3-D altitude axis to `IntermediateGridStrategy`'s symmetric path: a `grid.vertical:` block makes the common grid `(time, lon, lat, alt)`, binning both sources by altitude derived from what the dataset supplies (native altitude → geopotential → pressure, else error).

**Architecture:** Extend Phase 1's symmetric bin-both path. A new `_source_altitude` derives a per-datum altitude (in the configured units) per source; the flatten produces a 5th array (`alt`); a new 4-D numba binner bins `(time, lon, lat, alt)`; `pressure_to_altitude` (inverse of the existing barometric) backs the pressure fallback. Triggered by a `vertical:` sub-block; the 2-D path (no `vertical:`) is untouched.

**Tech Stack:** Python 3.11/3.12, xarray, numpy, numba, pydantic, pytest, mypy, black, isort. `davinci` conda env.

**Spec:** `docs/superpowers/specs/2026-06-14-intermediate-grid-pairing-phase2-design.md`

---

## Conventions (every task)
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci   # prefix test runs
```
Gate after each task:
```bash
cd /Users/fillmore/EarthSystem/DAVINCI
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && mypy davinci_monet && black --check davinci_monet && isort --check-only davinci_monet
```
Commit footer (required): `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. No push/merge — local commits on `develop`.

---

## Task 1: `pressure_to_altitude` (inverse barometric)

**Files:**
- Modify: `davinci_monet/pairing/strategies/track.py` (next to `altitude_to_pressure`, ~line 57)
- Test: `davinci_monet/tests/test_intermediate_grid.py`

- [ ] **Step 1: Write the failing test**
```python
def test_pressure_to_altitude_standard_atmosphere():
    from davinci_monet.pairing.strategies.track import pressure_to_altitude
    import numpy as np
    p = np.array([1013.25, 500.0, 700.0])
    z = pressure_to_altitude(p)
    assert z[0] == pytest.approx(0.0, abs=1.0)          # sea level
    assert z[1] == pytest.approx(5572.0, abs=50.0)       # ~500 hPa
    assert z[2] == pytest.approx(3012.0, abs=50.0)       # ~700 hPa
    # round-trips with the existing forward conversion
    from davinci_monet.pairing.strategies.track import altitude_to_pressure
    assert altitude_to_pressure(z)[1] == pytest.approx(500.0, rel=1e-3)
```

- [ ] **Step 2: Run — expect FAIL** (`pressure_to_altitude` not defined)
```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_intermediate_grid.py::test_pressure_to_altitude_standard_atmosphere -q
```

- [ ] **Step 3: Implement** — add to `track.py` directly below `altitude_to_pressure`:
```python
def pressure_to_altitude(
    pressure_hpa: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[Any]]:
    """Convert pressure (hPa) to altitude (metres) — inverse of altitude_to_pressure.

    Inverts the US Standard Atmosphere barometric formula (troposphere,
    valid to ~11 km): h = (T0/L) * (1 - (P/P0)**(1/exponent)).
    """
    P0 = 1013.25
    L = 0.0065
    T0 = 288.15
    g = 9.80665
    M = 0.0289644
    R = 8.31447
    exponent = g * M / (R * L)  # ≈ 5.2559
    return (T0 / L) * (1.0 - (pressure_hpa / P0) ** (1.0 / exponent))
```

- [ ] **Step 4: Run — expect PASS**, then gate + commit
```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_intermediate_grid.py::test_pressure_to_altitude_standard_atmosphere -q
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && mypy davinci_monet
black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "feat: pressure_to_altitude (US Std Atm inverse barometric)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Config — `VerticalGridConfig` + `GridConfig.vertical`

**Files:**
- Modify: `davinci_monet/config/schema.py` (the `GridConfig` added in Phase 1)
- Test: `davinci_monet/tests/test_config_grid_method.py`

- [ ] **Step 1: Write failing tests**
```python
def test_grid_vertical_block_parses():
    from davinci_monet.config.schema import SourcePairConfig
    p = SourcePairConfig(
        x={"source": "a", "variable": "v"}, y={"source": "b", "variable": "V"},
        method="grid",
        grid={"horizontal_res": 0.5, "vertical": {"res": 500, "units": "m", "extent": [0, 12000]}},
    )
    assert p.grid is not None and p.grid.vertical is not None
    assert p.grid.vertical.res == 500.0 and p.grid.vertical.units == "m"
    assert p.grid.vertical.extent == (0.0, 12000.0)

def test_grid_vertical_defaults_units_m_and_optional():
    from davinci_monet.config.schema import SourcePairConfig
    p = SourcePairConfig(
        x={"source": "a", "variable": "v"}, y={"source": "b", "variable": "V"},
        method="grid", grid={"horizontal_res": 0.5},
    )
    assert p.grid is not None and p.grid.vertical is None  # 2-D when omitted
    p2 = SourcePairConfig(
        x={"source": "a", "variable": "v"}, y={"source": "b", "variable": "V"},
        method="grid", grid={"horizontal_res": 0.5, "vertical": {"res": 1.0}},
    )
    assert p2.grid.vertical.units == "m"
```

- [ ] **Step 2: Run — expect FAIL**
```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_config_grid_method.py -q
```

- [ ] **Step 3: Implement** — in `schema.py`, add `VerticalGridConfig` ABOVE `GridConfig`, add the field + parser to `GridConfig`:
```python
class VerticalGridConfig(FlexibleSchema):
    """Vertical (altitude) settings for a 3-D intermediate grid (Phase 2)."""

    res: float
    units: str = "m"
    extent: tuple[float, float] | None = None
```
In `GridConfig`, add after `min_sample_count`:
```python
    vertical: VerticalGridConfig | None = None

    @field_validator("vertical", mode="before")
    @classmethod
    def _parse_vertical(cls, v: Any) -> Any:
        return VerticalGridConfig(**v) if isinstance(v, dict) else v
```
(Ensure `field_validator`/`Any` imported — they are.)

- [ ] **Step 4: Run — expect PASS**, gate + commit
```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_config_grid_method.py -q
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && mypy davinci_monet
black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "feat: GridConfig.vertical (VerticalGridConfig) for 3-D intermediate gridding

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: 4-D numba binner

**Files:**
- Modify: `davinci_monet/pairing/grid_binning.py`
- Test: `davinci_monet/tests/test_grid_binning_4d.py` (new)

- [ ] **Step 1: Write failing test**
```python
import numpy as np
from davinci_monet.pairing.grid_binning import bin_points_to_grid_4d, normalize_grid


def test_bin_points_to_grid_4d_accumulates_by_cell():
    # 2 time x 2 lon x 2 lat x 2 alt grid, edges 0..2 each
    edges = np.array([0.0, 1.0, 2.0])
    nt = nx = ny = nz = 2
    count = np.zeros((nt, nx, ny, nz), dtype=np.int32)
    acc = np.zeros((nt, nx, ny, nz), dtype=np.float64)
    # two points in cell (0,0,0,0), one in (1,1,1,1)
    t = np.array([0.5, 0.5, 1.5]); x = np.array([0.5, 0.5, 1.5])
    y = np.array([0.5, 0.5, 1.5]); z = np.array([0.5, 0.5, 1.5])
    d = np.array([2.0, 4.0, 9.0])
    bin_points_to_grid_4d(edges, edges, edges, edges, t, x, y, z, d, count, acc)
    normalize_grid(count, acc)
    assert count[0, 0, 0, 0] == 2 and acc[0, 0, 0, 0] == 3.0  # mean(2,4)
    assert count[1, 1, 1, 1] == 1 and acc[1, 1, 1, 1] == 9.0
    assert np.isnan(acc[0, 1, 0, 1])  # empty cell
```

- [ ] **Step 2: Run — expect FAIL**
```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_grid_binning_4d.py -q
```

- [ ] **Step 3: Implement** — append to `grid_binning.py`:
```python
@numba.jit(nopython=True)
def bin_points_to_grid_4d(
    time_edges: np.ndarray,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    alt_edges: np.ndarray,
    time_values: np.ndarray,
    lon_values: np.ndarray,
    lat_values: np.ndarray,
    alt_values: np.ndarray,
    data_values: np.ndarray,
    count_grid: np.ndarray,
    data_grid: np.ndarray,
) -> None:
    """Accumulate points into (time, lon, lat, alt) grid cells (sum + count, in-place)."""
    dt = time_edges[1] - time_edges[0]
    dx = lon_edges[1] - lon_edges[0]
    dy = lat_edges[1] - lat_edges[0]
    dz = alt_edges[1] - alt_edges[0]
    nt, nx, ny, nz = data_grid.shape
    for i in range(len(data_values)):
        if (
            not math.isnan(data_values[i])
            and not math.isnan(time_values[i])
            and not math.isnan(lon_values[i])
            and not math.isnan(lat_values[i])
            and not math.isnan(alt_values[i])
            and time_values[i] >= time_edges[0]
            and time_values[i] <= time_edges[-1]
            and lon_values[i] >= lon_edges[0]
            and lon_values[i] <= lon_edges[-1]
            and lat_values[i] >= lat_edges[0]
            and lat_values[i] <= lat_edges[-1]
            and alt_values[i] >= alt_edges[0]
            and alt_values[i] <= alt_edges[-1]
        ):
            it = int((time_values[i] - time_edges[0]) / dt)
            ix = int((lon_values[i] - lon_edges[0]) / dx)
            iy = int((lat_values[i] - lat_edges[0]) / dy)
            iz = int((alt_values[i] - alt_edges[0]) / dz)
            if it < 0:
                it = 0
            elif it >= nt:
                it = nt - 1
            if ix < 0:
                ix = 0
            elif ix >= nx:
                ix = nx - 1
            if iy < 0:
                iy = 0
            elif iy >= ny:
                iy = ny - 1
            if iz < 0:
                iz = 0
            elif iz >= nz:
                iz = nz - 1
            count_grid[it, ix, iy, iz] += 1
            data_grid[it, ix, iy, iz] += data_values[i]
```

- [ ] **Step 4: Run — expect PASS**, gate + commit
```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_grid_binning_4d.py -q
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && mypy davinci_monet
black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "feat: 4-D numba binner bin_points_to_grid_4d (time, lon, lat, alt)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `_source_altitude` — derive per-datum altitude

**Files:**
- Modify: `davinci_monet/pairing/strategies/intermediate_grid.py`
- Test: `davinci_monet/tests/test_intermediate_grid.py`

- [ ] **Step 1: Write failing tests**
```python
def _track_alt_ds(alts_m, var="O3"):
    import numpy as np, pandas as pd, xarray as xr
    n = len(alts_m)
    return xr.Dataset(
        {var: (["time"], np.arange(n, dtype=float))},
        coords={
            "time": pd.to_datetime(["2012-05-29"] * n),
            "latitude": ("time", np.full(n, 35.0)),
            "longitude": ("time", np.full(n, -97.0)),
            "altitude": ("time", np.asarray(alts_m, float), {"units": "m"}),
        },
    )


def test_source_altitude_native_meters():
    from davinci_monet.pairing.strategies.intermediate_grid import IntermediateGridStrategy
    ds = _track_alt_ds([500.0, 8000.0])
    alt = IntermediateGridStrategy()._source_altitude(ds, "O3", "m")
    assert list(alt.dims) == ["time"]
    assert float(alt.values[0]) == pytest.approx(500.0)
    assert float(alt.values[1]) == pytest.approx(8000.0)


def test_source_altitude_native_km_units_conversion():
    from davinci_monet.pairing.strategies.intermediate_grid import IntermediateGridStrategy
    ds = _track_alt_ds([500.0, 8000.0])  # source in metres
    alt = IntermediateGridStrategy()._source_altitude(ds, "O3", "km")  # request km
    assert float(alt.values[1]) == pytest.approx(8.0)


def test_source_altitude_pressure_fallback():
    import numpy as np, pandas as pd, xarray as xr
    from davinci_monet.pairing.strategies.intermediate_grid import IntermediateGridStrategy
    ds = xr.Dataset(
        {"O3": (["time", "lev"], np.zeros((1, 2)))},
        coords={
            "time": pd.to_datetime(["2012-05-29"]),
            "lev": ("lev", np.array([1013.25, 500.0]), {"units": "hPa"}),
            "latitude": ("time", [35.0]), "longitude": ("time", [-97.0]),
        },
    )
    alt = IntermediateGridStrategy()._source_altitude(ds, "O3", "m")
    # broadcast to (time, lev): lev 1013->~0 m, 500->~5572 m
    vals = alt.transpose("time", "lev").values[0]
    assert vals[0] == pytest.approx(0.0, abs=1.0)
    assert vals[1] == pytest.approx(5572.0, abs=50.0)


def test_source_altitude_errors_without_vertical():
    import numpy as np, pandas as pd, xarray as xr
    from davinci_monet.core.exceptions import PairingError
    from davinci_monet.pairing.strategies.intermediate_grid import IntermediateGridStrategy
    # hybrid 'z' level, no length units, no geopotential, no pressure
    ds = xr.Dataset(
        {"O3": (["time", "z"], np.zeros((1, 2)))},
        coords={"time": pd.to_datetime(["2012-05-29"]),
                "z": ("z", np.array([0.5, 0.9])),  # hybrid, unitless
                "latitude": ("time", [35.0]), "longitude": ("time", [-97.0])},
    )
    with pytest.raises(PairingError, match="vertical"):
        IntermediateGridStrategy()._source_altitude(ds, "O3", "m")
```

- [ ] **Step 2: Run — expect FAIL**
```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_intermediate_grid.py -k source_altitude -q
```

- [ ] **Step 3: Implement** — in `intermediate_grid.py`, add module-level constants near the top and the method on the class. Add the import `from davinci_monet.pairing.strategies.track import pressure_to_altitude` and `from davinci_monet.core.exceptions import PairingError` (PairingError is already imported).
```python
_UNIT_TO_M = {"m": 1.0, "meter": 1.0, "meters": 1.0, "km": 1000.0, "ft": 0.3048, "feet": 0.3048}
_ALT_NAMES = {"altitude", "alt", "height", "geometric_height"}
_GEOPOT_NAMES = {"z3", "zg", "geopotential_height", "geopotential_height_msl"}
_PRESSURE_NAMES = {"lev", "level", "plev", "pressure", "p"}
_PRESSURE_HPA_UNITS = {"hpa", "mb", "millibar", "hectopascal"}
_PRESSURE_PA_UNITS = {"pa", "pascal"}
```
```python
    def _source_altitude(self, ds: xr.Dataset, var: str, units: str) -> xr.DataArray:
        """Per-datum altitude (broadcast to ``ds[var]``'s dims) in ``units``.

        Order: native geometric altitude (length units) -> geopotential height
        (length units) -> pressure (US Std Atm). Errors if the dataset supplies
        none — it is the dataset's responsibility to carry a usable vertical.
        """
        from davinci_monet.pairing.strategies.track import pressure_to_altitude

        da = ds[var]
        tu = units.lower()
        if tu not in _UNIT_TO_M:
            raise PairingError(
                f"Unsupported vertical units '{units}'; use one of {sorted(_UNIT_TO_M)}"
            )
        tfac = _UNIT_TO_M[tu]

        # 1. native geometric altitude — a coord/var with length units
        for name in list(da.coords) + [v for v in ds.variables if v != var]:
            if str(name).lower() in _ALT_NAMES:
                cand = ds[name]
                su = str(cand.attrs.get("units", "")).lower()
                if su in _UNIT_TO_M:
                    return (cand * (_UNIT_TO_M[su] / tfac)).broadcast_like(da)

        # 2. geopotential height — a data variable with length units
        for name in ds.data_vars:
            if str(name).lower() in _GEOPOT_NAMES:
                cand = ds[name]
                su = str(cand.attrs.get("units", "m")).lower()
                if su in _UNIT_TO_M:
                    return (cand * (_UNIT_TO_M[su] / tfac)).broadcast_like(da)

        # 3. pressure vertical coordinate -> altitude (US Std Atm)
        for name in list(da.dims) + list(da.coords):
            if str(name).lower() in _PRESSURE_NAMES and name in ds.coords:
                cand = ds[name]
                su = str(cand.attrs.get("units", "")).lower()
                if su in _PRESSURE_HPA_UNITS or su in _PRESSURE_PA_UNITS:
                    p_hpa = cand if su in _PRESSURE_HPA_UNITS else cand / 100.0
                    alt_m = pressure_to_altitude(p_hpa)
                    return (alt_m * (1.0 / tfac)).broadcast_like(da)

        raise PairingError(
            f"Source variable '{var}' has no usable vertical coordinate for a 3-D "
            f"altitude grid; supply geometric altitude (m), geopotential height (m), "
            f"or pressure (hPa). Found dims: {list(da.dims)}"
        )
```

- [ ] **Step 4: Run — expect PASS**, gate + commit
```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_intermediate_grid.py -k source_altitude -q
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && mypy davinci_monet
black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "feat: _source_altitude — derive per-datum altitude (z/geopotential/pressure)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: 3-D symmetric path (`vertical:` branch)

**Files:**
- Modify: `davinci_monet/pairing/strategies/intermediate_grid.py` (`_pair_symmetric`, add `_pair_symmetric_3d`, `_flatten_to_points_3d`, `_uniform_vertical_grid`; the `horizontal_res` dispatch and engine routing pass `vertical`)
- Modify: `davinci_monet/pairing/engine.py` (grid routing passes `vertical`)
- Test: `davinci_monet/tests/test_intermediate_grid.py`

- [ ] **Step 1: Write the failing 3-D strategy test**
```python
def test_symmetric_3d_bins_by_altitude():
    import numpy as np, pandas as pd, xarray as xr
    from davinci_monet.pairing.strategies.intermediate_grid import IntermediateGridStrategy
    # x: a track with native altitude (m); two points at ~500 m, one at ~6000 m
    x = xr.Dataset(
        {"O3": (["time"], np.array([10.0, 30.0, 99.0]))},
        coords={"time": pd.to_datetime(["2012-05-29"] * 3),
                "latitude": ("time", [35.1, 35.2, 35.3]),
                "longitude": ("time", [-97.1, -97.2, -97.3]),
                "altitude": ("time", np.array([400.0, 700.0, 6000.0]), {"units": "m"})},
    )
    # y: a 3-D grid with geopotential height Z3 (m)
    lev = np.array([1, 2]); lat = np.array([35.0]); lon = np.array([-97.0])
    y = xr.Dataset(
        {"O3": (["time", "lev", "lat", "lon"], np.full((1, 2, 1, 1), 50.0)),
         "Z3": (["time", "lev", "lat", "lon"], np.array([[[[500.0]], [[6000.0]]]]), {"units": "m"})},
        coords={"time": pd.to_datetime(["2012-05-29"]), "lev": lev,
                "lat": lat, "lon": lon, "latitude": ("lat", lat), "longitude": ("lon", lon)},
    )
    paired = IntermediateGridStrategy().pair_sources(
        x_data=x, y_data=y, x_var="O3", y_var="O3", x_source="dc8", y_source="cam",
        horizontal_res=1.0, time_resolution="1D", min_sample_count=1,
        vertical={"res": 1000.0, "units": "m", "extent": [0.0, 7000.0]},
    )
    assert list(paired["x_O3"].dims) == ["time", "lon", "lat", "alt"]
    assert "alt" in paired.coords and "x_sample_count" in paired and "y_sample_count" in paired
    # the two ~500 m x points share the 0-1000 m alt bin -> mean(10,30)=20
    low = paired["x_O3"].sel(alt=500.0, method="nearest").max().item()
    assert low == pytest.approx(20.0)
    # the 6000 m x point sits in a higher alt bin, away from the 500 m bin
    assert int(paired["x_sample_count"].sel(alt=500.0, method="nearest").max().item()) == 2
```

- [ ] **Step 2: Run — expect FAIL** (no `vertical` handling)

- [ ] **Step 3: Implement** — in `intermediate_grid.py`:

(a) Extend the dispatch at the top of `pair_sources` to forward `vertical`:
```python
        if kwargs.get("horizontal_res") is not None:
            return self._pair_symmetric(
                x_data, x_data_var=kwargs.get("x_var"),
                y_data=y_data, y_data_var=kwargs.get("y_var"),
                x_source=kwargs.get("x_source"), y_source=kwargs.get("y_source"),
                horizontal_res=float(kwargs["horizontal_res"]),
                extent=kwargs.get("extent"),
                time_resolution=kwargs.get("time_resolution", "1D"),
                min_sample_count=int(kwargs.get("min_sample_count", 1)),
                vertical=kwargs.get("vertical"),
            )
```
(b) Add a `vertical=None` keyword to `_pair_symmetric` and branch to 3-D before the 2-D body:
```python
        if vertical is not None:
            return self._pair_symmetric_3d(
                x_data, x_data_var=x_data_var, y_data=y_data, y_data_var=y_data_var,
                x_source=x_source, y_source=y_source,
                horizontal_res=horizontal_res, extent=extent,
                time_resolution=time_resolution, min_sample_count=min_sample_count,
                vertical=vertical,
            )
```
(c) Add the 3-D method + helpers:
```python
    def _pair_symmetric_3d(
        self, x_data, *, x_data_var, y_data, y_data_var, x_source, y_source,
        horizontal_res, extent, time_resolution, min_sample_count, vertical,
    ):
        from davinci_monet.pairing.grid_binning import bin_points_to_grid_4d, normalize_grid

        x_var = x_data_var or str(list(x_data.data_vars)[0])
        y_var = y_data_var or str(list(y_data.data_vars)[0])
        units = str(vertical.get("units", "m")) if isinstance(vertical, dict) else "m"
        vres = float(vertical["res"]) if isinstance(vertical, dict) else float(vertical.res)
        vextent = vertical.get("extent") if isinstance(vertical, dict) else getattr(vertical, "extent", None)

        x_alt = self._source_altitude(x_data, x_var, units)
        y_alt = self._source_altitude(y_data, y_var, units)

        lon_centers, lat_centers, lon_edges, lat_edges = self._uniform_horizontal_grid(
            [x_data, y_data], horizontal_res, extent
        )
        time_centers, time_edges, time_coords = self._uniform_time_grid(
            [x_data, y_data], time_resolution
        )
        alt_centers, alt_edges = self._uniform_vertical_grid([x_alt, y_alt], vres, vextent)

        shape = (len(time_centers), len(lon_centers), len(lat_centers), len(alt_centers))
        xg, xc = self._bin_one_source_3d(
            x_data, x_var, x_alt, time_edges, lon_edges, lat_edges, alt_edges, shape, min_sample_count
        )
        yg, yc = self._bin_one_source_3d(
            y_data, y_var, y_alt, time_edges, lon_edges, lat_edges, alt_edges, shape, min_sample_count
        )
        dims = ["time", "lon", "lat", "alt"]
        paired = xr.Dataset(
            {
                f"x_{x_var}": (dims, xg.astype(np.float32)),
                f"y_{y_var}": (dims, yg.astype(np.float32)),
                "x_sample_count": (dims, xc),
                "y_sample_count": (dims, yc),
            },
            coords={"time": time_coords, "lon": lon_centers, "lat": lat_centers, "alt": alt_centers},
        )
        paired["alt"].attrs["units"] = units
        paired[f"x_{x_var}"].attrs.update({"axis": "x", "source_label": x_source or ""})
        paired[f"y_{y_var}"].attrs.update({"axis": "y", "source_label": y_source or ""})
        paired.attrs.update({"created_by": "davinci_monet", "paired": True})
        return paired

    def _bin_one_source_3d(
        self, ds, var, alt, time_edges, lon_edges, lat_edges, alt_edges, shape, min_sample_count
    ):
        from davinci_monet.pairing.grid_binning import bin_points_to_grid_4d, normalize_grid

        time_flat, lon_flat, lat_flat, data_flat = self._flatten_to_points(ds, var)
        da = ds[var]
        alt_flat = alt.broadcast_like(da).transpose(*da.dims).values.astype(np.float64).flatten()
        if lon_edges[0] >= 0 and np.any(lon_flat < 0):
            lon_flat = np.where(lon_flat < 0, lon_flat + 360.0, lon_flat)
        count = np.zeros(shape, dtype=np.int32)
        acc = np.zeros(shape, dtype=np.float64)
        bin_points_to_grid_4d(
            time_edges, lon_edges, lat_edges, alt_edges,
            time_flat, lon_flat, lat_flat, alt_flat, data_flat, count, acc,
        )
        normalize_grid(count, acc)
        if min_sample_count > 1:
            acc[count < min_sample_count] = np.nan
        return acc, count

    def _uniform_vertical_grid(self, alt_arrays, res, extent):
        if extent is not None:
            z0, z1 = float(extent[0]), float(extent[1])
        else:
            mins = [float(np.nanmin(a.values)) for a in alt_arrays]
            maxs = [float(np.nanmax(a.values)) for a in alt_arrays]
            z0, z1 = min(mins), max(maxs)
        edges = self._span_edges(z0, z1, res)
        centers = (edges[:-1] + edges[1:]) / 2.0
        return centers, edges
```
Notes: `_flatten_to_points` and `_span_edges` are the Phase 1 helpers (reused). `alt_flat` uses the SAME `broadcast_like(da).transpose(*da.dims).flatten()` pattern as the lat/lon flatten so it aligns element-wise.

(d) In `davinci_monet/pairing/engine.py`, the `method == "grid"` routing must pass `vertical`. Find the grid block (added in Phase 1) and add `vertical=kwargs.get("vertical")` to the `IntermediateGridStrategy().pair_sources(...)` call's kwargs.

- [ ] **Step 4: Run — expect PASS**, gate + commit
```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest davinci_monet/tests/test_intermediate_grid.py -q
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && mypy davinci_monet
black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "feat: 3-D symmetric intermediate gridding (time, lon, lat, alt) via vertical: block

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Integration test through the pipeline (`vertical:` block)

**Files:**
- Test: `davinci_monet/tests/test_intermediate_grid.py`

- [ ] **Step 1: Add the failing integration test**
```python
@pytest.mark.integration
def test_method_grid_3d_runs_through_pipeline(tmp_path):
    import numpy as np, pandas as pd, xarray as xr
    from davinci_monet.pipeline.runner import PipelineRunner
    # two file-backed sources, each with a native altitude coordinate (m)
    def alt_ds(seed):
        rng = np.random.default_rng(seed)
        n = 20
        return xr.Dataset(
            {"O3": (["time"], rng.uniform(20, 80, n))},
            coords={"time": pd.to_datetime(["2012-05-29"] * n),
                    "latitude": ("time", rng.uniform(34, 36, n)),
                    "longitude": ("time", rng.uniform(-98, -96, n)),
                    "altitude": ("time", rng.uniform(0, 10000, n), {"units": "m"})},
        )
    xp, yp = tmp_path / "x.nc", tmp_path / "y.nc"
    alt_ds(1).to_netcdf(xp); alt_ds(2).to_netcdf(yp)
    config = {
        "analysis": {"output_dir": str(tmp_path / "out")},
        "sources": {
            "obs": {"type": "generic", "files": str(xp), "variables": {"O3": {"units": "ppb"}}},
            "mod": {"type": "generic", "files": str(yp), "variables": {"O3": {"units": "ppb"}}},
        },
        "pairs": {
            "obs_vs_mod": {
                "x": {"source": "obs", "variable": "O3"},
                "y": {"source": "mod", "variable": "O3"},
                "method": "grid",
                "grid": {"horizontal_res": 1.0, "time_resolution": "1D",
                         "vertical": {"res": 1000.0, "units": "m"}},
            }
        },
    }
    result = PipelineRunner(show_progress=False).run_from_config(config)
    assert result.success, getattr(result, "error", None)
    ctx = result.context
    assert ctx is not None and "obs_vs_mod" in ctx.paired
    paired = ctx.paired["obs_vs_mod"]
    data = paired.data if hasattr(paired, "data") else paired
    assert "alt" in data.coords and list(data["x_O3"].dims) == ["time", "lon", "lat", "alt"]
```

- [ ] **Step 2: Run — expect FAIL or surface a wiring gap.** If it fails, the `vertical:` dict must reach the strategy (Phase 1 `_strategy_options` flattens `grid:` → `vertical` becomes a top-level kwarg dict; the engine grid-routing must forward it — Task 5(d)). Fix wiring, not the test.

- [ ] **Step 3: Make it pass**, gate + commit
```bash
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q && mypy davinci_monet
black davinci_monet >/dev/null && isort davinci_monet >/dev/null
git add -A && git commit -m "test: 3-D method: grid runs end-to-end through the pipeline

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Docs + final verification

**Files:**
- Modify: `CLAUDE.md` (extend the `method: grid` block with the `vertical:` option)

- [ ] **Step 1: Document the `vertical:` option in CLAUDE.md** — extend the Phase 1 `method: grid` example:
```yaml
    grid:
      horizontal_res: 0.5
      vertical: { res: 500, units: m }   # presence -> 3-D (time, lon, lat, alt) grid
```
with one sentence: "Add a `grid.vertical:` block for a 3-D altitude grid; each source must supply a usable vertical (geometric altitude, geopotential height, or pressure) or the strategy errors."

- [ ] **Step 2: Final gate**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate davinci
cd /Users/fillmore/EarthSystem/DAVINCI
HDF5_USE_FILE_LOCKING=FALSE python -m pytest -q
mypy davinci_monet
black --check davinci_monet && isort --check-only davinci_monet
```
Expected: all green, mypy clean, format clean.

- [ ] **Step 3: Commit + report**
```bash
git add CLAUDE.md docs/superpowers && git commit -m "docs: document grid.vertical 3-D intermediate gridding (Phase 2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
Report the suite/mypy results. Do not push/merge.

---

## Self-Review notes

- **Spec coverage:** Decision 1 (`vertical:` triggers 3-D) → Task 5; Decision 2 (`_source_altitude` order + error) → Task 4; Decision 3 (4-D binning, flatten alignment) → Tasks 3 + 5; Decision 4 (output `(time,lon,lat,alt)` tagged) → Task 5; Decision 5 (both sources must supply a vertical) → Task 4's error covers a surface-only source; `pressure_to_altitude` → Task 1; config → Task 2; integration → Task 6; regression (2-D unchanged) → the full suite gate each task.
- **Flatten alignment (the make-or-break):** Task 5's `_bin_one_source_3d` builds `alt_flat` with the IDENTICAL `broadcast_like(da).transpose(*da.dims).flatten()` pattern as the Phase 1 lat/lon flatten, so the i-th `alt_flat` aligns with the i-th value. The Task 5 test (two ~500 m points share a bin; a 6000 m point does not) checks this end-to-end. An adversarial reviewer should still encode per-datum altitude and confirm each lands in its own bin.
- **Type consistency:** `_source_altitude(ds, var, units) -> xr.DataArray`; `bin_points_to_grid_4d` arg order matches Task 5's call; `_uniform_vertical_grid` mirrors `_uniform_horizontal_grid`'s return shape `(centers, edges)`.
- **Vertical as dict vs model:** the strategy receives `vertical` as a plain dict (config is re-dumped to dicts in the pipeline — see Phase 1's `_strategy_options`); `_pair_symmetric_3d` handles both dict and model defensively.
