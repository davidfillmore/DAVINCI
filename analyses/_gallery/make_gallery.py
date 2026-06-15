"""Label gallery: one PDF per plot type rendered from synthetic data.

Exercises the DAVINCI labeling system across every renderer:
  scatter, timeseries (multi-source + uncertainty band), spatial (grid),
  spatial (point), spatial_bias, curtain, vertical_profile, histogram,
  flight_track.

Usage::

    HDF5_USE_FILE_LOCKING=FALSE python analyses/_gallery/make_gallery.py

Output: analyses/_gallery/output/*.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must come before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Resolve project root so the script runs from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from davinci_monet.plots import (
    CurtainPlotter,
    FlightTrackPlotter,
    HistogramPlotter,
    PlotConfig,
    ScatterPlotter,
    SpatialBiasPlotter,
    SpatialPlotter,
    TimeSeriesPlotter,
    VerticalProfilePlotter,
    apply_ncar_style,
    build_series,
)

# ---------------------------------------------------------------------------
# Constants: awkward source key exercises source-name cleaning + de-dup.
# _AWKWARD_SOURCE is used ONLY where the quantity is actually NO2 column,
# so the source name legitimately embeds the species word.
# ---------------------------------------------------------------------------
_AWKWARD_SOURCE = "cesm_no2_column"
_TROPOMI_SOURCE = "tropomi"

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic dataset builders
# ---------------------------------------------------------------------------


def _time_axis(n: int = 48, freq: str = "1h") -> pd.DatetimeIndex:
    return pd.date_range("2024-02-01", periods=n, freq=freq)


def _make_paired_point(n_time: int = 48, n_site: int = 25) -> xr.Dataset:
    """Paired point dataset: (time, site) dims with source-label/axis attrs.

    Variable: NO2 column [mol/m2].  x = TROPOMI (obs), y = CESM (model).
    Uses the awkward source key on the y-axis to exercise source-name
    cleaning and SI unit formatting.  The quantity IS an NO2 column, so the
    key legitimately embeds the species word.
    """
    rng = np.random.default_rng(42)
    times = _time_axis(n_time)
    lons = rng.uniform(-120, -70, n_site)
    lats = rng.uniform(25, 50, n_site)

    # x: tropomi (obs), y: cesm (model)
    x_vals = rng.uniform(1.0e-5, 8.0e-5, (n_time, n_site))
    noise = rng.normal(0, 0.5e-5, (n_time, n_site))
    y_vals = x_vals * 1.1 + noise

    ds = xr.Dataset(
        {
            f"{_TROPOMI_SOURCE}_NO2": (
                ["time", "site"],
                x_vals,
                {
                    "units": "mol/m2",
                    "long_name": "TROPOMI NO2 column",
                    "axis": "x",
                    "source_label": _TROPOMI_SOURCE,
                },
            ),
            f"{_AWKWARD_SOURCE}_NO2": (
                ["time", "site"],
                y_vals,
                {
                    "units": "mol/m2",
                    "long_name": "CESM NO2 column",
                    "axis": "y",
                    "source_label": _AWKWARD_SOURCE,
                },
            ),
        },
        coords={
            "time": times,
            "site": np.arange(n_site),
            "latitude": ("site", lats, {"units": "degrees_north"}),
            "longitude": ("site", lons, {"units": "degrees_east"}),
        },
    )
    ds.attrs["geometry"] = "point"
    return ds


def _make_multi_source_timeseries(n_time: int = 48, n_site: int = 10) -> list[xr.Dataset]:
    """Three independent single-source time datasets for the overlay path.

    Each carries source_label so legend_label() is exercised.
    Units: ppbv (O3) to exercise SI formatting.
    """
    rng = np.random.default_rng(7)
    times = _time_axis(n_time)

    def _src_ds(mean: float, source: str) -> xr.Dataset:
        vals = rng.normal(mean, mean * 0.15, (n_time, n_site)).clip(min=0)
        ds = xr.Dataset(
            {
                "O3": (
                    ["time", "site"],
                    vals,
                    {"units": "ppbv", "long_name": "Ozone", "source_label": source},
                )
            },
            coords={"time": times, "site": np.arange(n_site)},
        )
        ds.attrs["source_label"] = source
        return ds

    # Three sources with slightly different means: model A, model B, obs
    return [
        _src_ds(40.0, "cam"),
        _src_ds(44.0, "wrf"),
        _src_ds(37.0, "airnow"),
    ]


def _make_track(n_pts: int = 300) -> xr.Dataset:
    """Synthetic aircraft track: (time,) with lat/lon/altitude coords.

    Variable: O3 [ppbv], source = cam.  Used by histogram and flight_track.
    """
    rng = np.random.default_rng(55)
    times = _time_axis(n_pts, freq="1min")[:n_pts]

    t = np.linspace(0, 4 * np.pi, n_pts)
    lons = -95.0 + 15.0 * np.sin(t)
    lats = 37.0 + 10.0 * np.cos(t)
    alt = 500.0 + 7500.0 * (np.sin(t * 0.5) + 1) / 2

    o3 = (30.0 + 20.0 * np.sin(t) + rng.normal(0, 3, n_pts)).clip(min=0)

    ds = xr.Dataset(
        {
            "O3": (
                ["time"],
                o3,
                {"units": "ppbv", "long_name": "Ozone", "source_label": "cam"},
            )
        },
        coords={
            "time": times,
            "latitude": ("time", lats, {"units": "degrees_north"}),
            "longitude": ("time", lons, {"units": "degrees_east"}),
            "altitude": ("time", alt, {"units": "m", "long_name": "Altitude ASL"}),
        },
    )
    ds.attrs.update({"geometry": "track", "source_label": "cam"})
    return ds


_CURTAIN_OBS = "dc8"  # aircraft in-situ obs
_CURTAIN_MOD = "cam"  # model sampled along track


def _make_curtain_paired(n_pts: int = 200) -> xr.Dataset:
    """1-D paired track dataset for CurtainPlotter.

    Both x and y are along (time,) with altitude as a coordinate.
    Variable: O3 [ppbv].  x = DC-8 in-situ (obs), y = CAM (model).
    show_var='y' or 'x' avoids needing a bias to be non-trivial.
    """
    rng = np.random.default_rng(77)
    times = _time_axis(n_pts, freq="1min")[:n_pts]

    t = np.linspace(0, 2 * np.pi, n_pts)
    alt = 500.0 + 8000.0 * (0.5 + 0.5 * np.sin(t))
    o3_obs = (35.0 + 15.0 * np.sin(t) + rng.normal(0, 2, n_pts)).clip(min=0)
    o3_model = o3_obs * 1.08 + rng.normal(0, 1.5, n_pts)

    ds = xr.Dataset(
        {
            f"{_CURTAIN_OBS}_O3": (
                ["time"],
                o3_obs,
                {
                    "units": "ppbv",
                    "long_name": "Ozone",
                    "axis": "x",
                    "source_label": _CURTAIN_OBS,
                },
            ),
            f"{_CURTAIN_MOD}_O3": (
                ["time"],
                o3_model,
                {
                    "units": "ppbv",
                    "long_name": "Ozone",
                    "axis": "y",
                    "source_label": _CURTAIN_MOD,
                },
            ),
        },
        coords={
            "time": times,
            "altitude": ("time", alt, {"units": "m", "long_name": "Altitude ASL"}),
            "latitude": ("time", np.linspace(30, 45, n_pts), {"units": "degrees_north"}),
            "longitude": ("time", np.linspace(-110, -80, n_pts), {"units": "degrees_east"}),
        },
    )
    ds.attrs["geometry"] = "track"
    return ds


def _make_gridded(n_lon: int = 36, n_lat: int = 18, n_time: int = 5) -> xr.Dataset:
    """Gridded (lat, lon) single-source dataset for SpatialPlotter.

    Variable: NO2 column [mol/m2]; geometry='grid'.
    """
    rng = np.random.default_rng(22)
    lons = np.linspace(-130, -60, n_lon)
    lats = np.linspace(20, 55, n_lat)
    times = _time_axis(n_time, freq="1D")

    # Simple field with latitudinal gradient
    lon2d, lat2d = np.meshgrid(lons, lats)
    base = 2.0e-5 + 1.5e-5 * np.cos(np.radians(lat2d))
    field = np.stack([base + rng.normal(0, 1e-6, base.shape) for _ in range(n_time)], axis=0)
    field = field.clip(min=0)

    ds = xr.Dataset(
        {
            "NO2": (
                ["time", "lat", "lon"],
                field,
                {
                    "units": "mol/m2",
                    "long_name": "NO2 total column",
                    "source_label": _AWKWARD_SOURCE,
                },
            )
        },
        coords={
            "time": times,
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
        },
    )
    ds.attrs.update({"geometry": "grid", "source_label": _AWKWARD_SOURCE})
    return ds


def _make_profile(n_lev: int = 80) -> xr.Dataset:
    """Vertical profile dataset: 1-D (level,) with altitude coord.

    Uses a single mean profile (averaged from multiple soundings) so
    VerticalProfilePlotter._plot_binned can ravel both the variable and
    altitude coord to the same length.

    Variable: O3 [ppbv] with realistic vertical structure.
    """
    rng = np.random.default_rng(33)
    # Pressure levels (surface -> TOA)
    press = np.logspace(np.log10(1013.0), np.log10(10.0), n_lev)
    alt_m = 7640.0 * np.log(1013.0 / press)  # rough scale height in metres

    # O3 profile: low near surface, peak ~25 km (stratosphere)
    baseline = 30.0 + 100.0 * np.exp(-((alt_m / 1000.0 - 25.0) ** 2) / 50.0)
    o3_vals = (baseline + rng.normal(0, 2, n_lev)).clip(min=0)

    ds = xr.Dataset(
        {
            "O3": (
                ["level"],
                o3_vals,
                {"units": "ppbv", "long_name": "Ozone", "source_label": "cam"},
            )
        },
        coords={
            "level": ("level", press, {"units": "hPa"}),
            "altitude": ("level", alt_m, {"units": "m", "long_name": "Altitude ASL"}),
        },
    )
    ds.attrs.update({"geometry": "profile", "source_label": "cam"})
    return ds


_BIAS_OBS = "ceres"  # x-axis: CERES OLR (obs)
_BIAS_MOD = "merra2"  # y-axis: MERRA-2 OLR (model)


def _make_radiation_gridded(n_lon: int = 30, n_lat: int = 15, n_time: int = 3) -> xr.Dataset:
    """Paired gridded dataset for SpatialBiasPlotter.

    Variable: OLR [W m-2].  x = CERES (obs), y = MERRA-2 (model).
    Exercises SI unit formatting.  Colorbar reads "MERRA-2 − CERES (W m$^{-2}$)".
    """
    rng = np.random.default_rng(66)
    lons = np.linspace(-130, -60, n_lon)
    lats = np.linspace(20, 55, n_lat)
    times = _time_axis(n_time, freq="1D")

    lon2d, lat2d = np.meshgrid(lons, lats)
    # OLR increases toward the tropics (warmer surface)
    base = 220.0 + 80.0 * np.cos(np.radians(lat2d))

    def _field(noise_scale: float) -> np.ndarray:
        return np.stack(
            [base + rng.normal(0, noise_scale, base.shape) for _ in range(n_time)], axis=0
        ).clip(min=0)

    x_vals = _field(8.0)
    y_vals = x_vals + _field(15.0) * 0.12

    ds = xr.Dataset(
        {
            f"{_BIAS_OBS}_OLR": (
                ["time", "lat", "lon"],
                x_vals,
                {
                    "units": "W m-2",
                    "long_name": "CERES OLR",
                    "axis": "x",
                    "source_label": _BIAS_OBS,
                },
            ),
            f"{_BIAS_MOD}_OLR": (
                ["time", "lat", "lon"],
                y_vals,
                {
                    "units": "W m-2",
                    "long_name": "MERRA-2 OLR",
                    "axis": "y",
                    "source_label": _BIAS_MOD,
                },
            ),
        },
        coords={
            "time": times,
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
            "latitude": (["lat", "lon"], np.broadcast_to(lat2d, (n_lat, n_lon)).copy()),
            "longitude": (["lat", "lon"], np.broadcast_to(lon2d, (n_lat, n_lon)).copy()),
        },
    )
    ds.attrs["geometry"] = "grid"
    return ds


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _save(fig: "matplotlib.figure.Figure", name: str) -> Path:
    path = OUTPUT_DIR / f"{name}.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(OUTPUT_DIR.parent.parent)}")
    return path


# ---------------------------------------------------------------------------
# One function per plot type
# ---------------------------------------------------------------------------


def make_scatter() -> Path:
    """Scatter: paired point data, NO2 column mol/m2.

    x = TROPOMI (obs), y = CESM (model).  Axes read "TROPOMI NO2 Column" /
    "CESM NO2 Column" — the awkward source key is legitimate here because the
    quantity IS an NO2 column, so source-name de-dup works correctly.
    """
    import matplotlib.figure

    ds = _make_paired_point()
    x_var = f"{_TROPOMI_SOURCE}_NO2"
    y_var = f"{_AWKWARD_SOURCE}_NO2"
    plotter = ScatterPlotter(PlotConfig(title="NO2 Column"))
    series = build_series(ds, x_var, y_var)
    result = plotter.render(series, show_density=True, show_regression=True)
    # ScatterPlotter.render may return Figure or list of (label, Figure) tuples
    # (the latter only when split_by_flight=True, which is not used here).
    fig = result if isinstance(result, matplotlib.figure.Figure) else result[0][1]
    return _save(fig, "scatter")


def make_timeseries_multi_uncertainty() -> Path:
    """Timeseries: multi-source overlay (N=3) for a single site, plus
    a second figure variant with 1-source + uncertainty band.
    """
    src_datasets = _make_multi_source_timeseries(n_site=10)

    # --- multi-source overlay (N=3) ---
    # Build one PlotSeries per source dataset; each carries source_label.
    from davinci_monet.core.base import PlotSeries
    from davinci_monet.plots.labels import canonical_variable_name

    plotter = TimeSeriesPlotter(PlotConfig(title="O3"))
    series = []
    for i, ds in enumerate(src_datasets):
        series.append(
            PlotSeries(
                dataset=ds,
                var_name="O3",
                canonical=canonical_variable_name(ds, "O3"),
                axis=None,
                source_label=ds.attrs.get("source_label"),
                index=i,
            )
        )
    fig = plotter.render(series)
    _save(fig, "timeseries_multi_overlay")

    # --- single-source with uncertainty band ---
    plotter2 = TimeSeriesPlotter(PlotConfig(title="O3"))
    s0 = series[0]
    fig2 = plotter2.render([s0], show_uncertainty=True, uncertainty_type="std")
    return _save(fig2, "timeseries_uncertainty_band")


def make_spatial_grid() -> Path:
    """Single-source spatial map: gridded NO2 column mol/m2 (pcolormesh path).

    No explicit PlotConfig title — the renderer auto-titles as
    "NO$_2$ Column (CESM)", exercising the source-name de-dup logic.
    """
    ds = _make_gridded()
    plotter = SpatialPlotter(PlotConfig())
    # Gridded dataset has 'lat'/'lon' dims, not 'latitude'/'longitude' coords.
    # SpatialPlotter resolves lat/lon via _LAT_CANDIDATES/_LON_CANDIDATES.
    series = build_series(ds, "NO2")
    series[0] = series[0].__class__(
        dataset=ds,
        var_name="NO2",
        canonical=series[0].canonical,
        axis=series[0].axis,
        source_label=_AWKWARD_SOURCE,
        index=0,
    )
    fig = plotter.render(series, lat_var="lat", lon_var="lon", time_average=True)
    return _save(fig, "spatial_grid")


def make_spatial_point() -> Path:
    """Single-source spatial map: surface O3 point sites (scatter path).

    Source = AirNow surface network.  Auto-title exercises "O$_3$ (AirNow)".
    """
    from davinci_monet.core.base import PlotSeries
    from davinci_monet.plots.labels import canonical_variable_name

    rng = np.random.default_rng(99)
    n_time, n_site = 24, 30
    times = _time_axis(n_time)
    lons = rng.uniform(-120, -70, n_site)
    lats = rng.uniform(25, 50, n_site)
    o3_vals = rng.normal(45.0, 12.0, (n_time, n_site)).clip(min=0)

    single_ds = xr.Dataset(
        {
            "O3": (
                ["time", "site"],
                o3_vals,
                {
                    "units": "ppbv",
                    "long_name": "Ozone",
                    "source_label": "airnow",
                },
            )
        },
        coords={
            "time": times,
            "site": np.arange(n_site),
            "latitude": ("site", lats, {"units": "degrees_north"}),
            "longitude": ("site", lons, {"units": "degrees_east"}),
        },
    )
    single_ds.attrs.update({"geometry": "point", "source_label": "airnow"})

    plotter = SpatialPlotter(PlotConfig())
    series = [
        PlotSeries(
            dataset=single_ds,
            var_name="O3",
            canonical=canonical_variable_name(single_ds, "O3"),
            axis=None,
            source_label="airnow",
            index=0,
        )
    ]
    fig = plotter.render(series, time_average=True)
    return _save(fig, "spatial_point")


def make_spatial_bias() -> Path:
    """Spatial bias map: paired gridded OLR, W m-2.

    y = MERRA-2 (model), x = CERES (obs).
    Title "OLR Bias"; colorbar reads "MERRA-2 − CERES (W m$^{-2}$)".
    """
    ds = _make_radiation_gridded()
    x_var = f"{_BIAS_OBS}_OLR"
    y_var = f"{_BIAS_MOD}_OLR"
    plotter = SpatialBiasPlotter(PlotConfig(title="OLR Bias"))
    series = build_series(ds, x_var, y_var)
    fig = plotter.render(series, lat_var="lat", lon_var="lon", time_average=True)
    return _save(fig, "spatial_bias")


def make_curtain() -> Path:
    """Curtain: 1-D flight track, O3 ppbv, show_var='y' (CAM model values)."""
    ds = _make_curtain_paired()
    x_var = f"{_CURTAIN_OBS}_O3"
    y_var = f"{_CURTAIN_MOD}_O3"
    plotter = CurtainPlotter(PlotConfig(title="O3"))
    series = build_series(ds, x_var, y_var)
    fig = plotter.render(series, alt_var="altitude", show_var="y")
    return _save(fig, "curtain")


def make_vertical_profile() -> Path:
    """Vertical profile: CAM O3 ppbv vs altitude, binned."""
    ds = _make_profile()
    plotter = VerticalProfilePlotter(PlotConfig(title="O3"))
    from davinci_monet.core.base import PlotSeries
    from davinci_monet.plots.labels import canonical_variable_name

    series = [
        PlotSeries(
            dataset=ds,
            var_name="O3",
            canonical=canonical_variable_name(ds, "O3"),
            axis=None,
            source_label="cam",
            index=0,
        )
    ]
    fig = plotter.render(series, mode="binned", alt_coord="altitude")
    return _save(fig, "vertical_profile")


def make_histogram() -> Path:
    """Histogram: CAM O3 ppbv distribution from track data."""
    ds = _make_track()
    plotter = HistogramPlotter(PlotConfig(title="O3"))
    from davinci_monet.core.base import PlotSeries
    from davinci_monet.plots.labels import canonical_variable_name

    series = [
        PlotSeries(
            dataset=ds,
            var_name="O3",
            canonical=canonical_variable_name(ds, "O3"),
            axis=None,
            source_label="cam",
            index=0,
        )
    ]
    fig = plotter.render(series, n_bins=25, show_stats=True)
    return _save(fig, "histogram")


def make_flight_track() -> Path:
    """Flight track: 3-D colored by CAM O3 ppbv."""
    ds = _make_track()
    plotter = FlightTrackPlotter(PlotConfig(title="O3"))
    from davinci_monet.core.base import PlotSeries
    from davinci_monet.plots.labels import canonical_variable_name

    series = [
        PlotSeries(
            dataset=ds,
            var_name="O3",
            canonical=canonical_variable_name(ds, "O3"),
            axis=None,
            source_label="cam",
            index=0,
        )
    ]
    fig = plotter.render(series)
    return _save(fig, "flight_track")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_MAKERS = [
    ("scatter", make_scatter),
    ("timeseries (multi-source overlay + uncertainty band)", make_timeseries_multi_uncertainty),
    ("spatial map — grid (pcolormesh)", make_spatial_grid),
    ("spatial map — point (scatter)", make_spatial_point),
    ("spatial bias", make_spatial_bias),
    ("curtain", make_curtain),
    ("vertical profile", make_vertical_profile),
    ("histogram", make_histogram),
    ("flight track", make_flight_track),
]


def _verify_labels() -> list[str]:
    """Derive expected label strings via the labeling API and assert consistency.

    Returns a list of failure messages; empty means all assertions passed.
    """
    from davinci_monet.plots import labeling

    failures: list[str] = []

    # ---- multi-overlay legend entries -----------------------------------
    multi_sources = ["cam", "wrf", "airnow"]
    legend_entries = [labeling.legend_label(s) for s in multi_sources]
    print("\n[label verify] multi-overlay legend entries:")
    for s, e in zip(multi_sources, legend_entries):
        print(f"  {s!r} → {e!r}")
    expected_multi = {"CAM", "WRF", "AirNow"}
    actual_multi = set(legend_entries)
    if actual_multi != expected_multi:
        failures.append(f"multi-overlay legend: expected {expected_multi}, got {actual_multi}")
    for entry in legend_entries:
        if "Column" in entry or "NO2" in entry or "column" in entry:
            failures.append(f"multi-overlay legend entry {entry!r} contains NO2/Column")

    # ---- spatial bias colorbar ------------------------------------------
    bias_cb = labeling.bias_label("merra2", "ceres", "W m-2")
    print(f"\n[label verify] bias colorbar: {bias_cb!r}")
    if "MERRA-2" not in bias_cb or "CERES" not in bias_cb:
        failures.append(f"bias colorbar missing MERRA-2 or CERES: {bias_cb!r}")
    if "−" not in bias_cb:
        failures.append(f"bias colorbar missing minus sign: {bias_cb!r}")
    if "Bias" in bias_cb:
        failures.append(f"bias colorbar should NOT contain 'Bias': {bias_cb!r}")
    if "NO2" in bias_cb or "Column" in bias_cb:
        failures.append(f"bias colorbar should NOT contain NO2/Column: {bias_cb!r}")

    # ---- O3 quantity label ----------------------------------------------
    # Verify that "Ozone" long_name normalises to O$_3$ (LaTeX formula)
    ds_o3 = _make_track()
    o3_qty = labeling.quantity_label(ds_o3, "O3")
    print(f"\n[label verify] O3 quantity label: {o3_qty!r}")
    if "O$_3$" not in o3_qty and "O3" not in o3_qty:
        failures.append(f"O3 quantity label unexpected: {o3_qty!r}")
    if "Ozone" in o3_qty:
        failures.append(f"O3 quantity label should render formula not 'Ozone': {o3_qty!r}")

    # ---- scatter axis labels (NO2 column) -------------------------------
    ds_no2 = _make_paired_point()
    x_label = labeling.axis_label(
        labeling.quantity_label(ds_no2, f"{_TROPOMI_SOURCE}_NO2"),
        "mol/m2",
        source=_TROPOMI_SOURCE,
    )
    y_label = labeling.axis_label(
        labeling.quantity_label(ds_no2, f"{_AWKWARD_SOURCE}_NO2"),
        "mol/m2",
        source=_AWKWARD_SOURCE,
    )
    print(f"\n[label verify] scatter x-axis: {x_label!r}")
    print(f"[label verify] scatter y-axis: {y_label!r}")
    if "TROPOMI" not in x_label:
        failures.append(f"scatter x-axis should contain TROPOMI: {x_label!r}")
    if "CESM" not in y_label:
        failures.append(f"scatter y-axis should contain CESM: {y_label!r}")

    if not failures:
        print("\n[label verify] ALL assertions passed.")
    else:
        print(f"\n[label verify] {len(failures)} ASSERTION(S) FAILED:")
        for f in failures:
            print(f"  - {f}")

    return failures


def main() -> int:
    apply_ncar_style()
    print(f"\nLabel gallery → {OUTPUT_DIR}\n")
    failed: list[str] = []
    for label, maker in _MAKERS:
        print(f"[{label}]")
        try:
            maker()
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}")
            failed.append(label)
    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for f in failed:
            print(f"  - {f}")
        return 1
    pdfs = sorted(OUTPUT_DIR.glob("*.pdf"))
    print(f"\nAll {len(pdfs)} PDFs written to {OUTPUT_DIR}:")
    for p in pdfs:
        print(f"  {p.name}")

    label_failures = _verify_labels()
    if label_failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
