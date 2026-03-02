# LMA Flash Density Map Renderer — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an `obs_lma_density` plot renderer that produces hourly cartopy maps of LMA flash extent density, with optional flight track overlays, integrated into the DAVINCI-MONET pipeline.

**Architecture:** New obs renderer inheriting from `ObsPlotter`, using cartopy directly for map rendering. The renderer aggregates 5-minute LMA grid data into hourly totals and produces one PNG per active hour. A small extension to `ObsPlottingStage` enables multi-figure output from a single plotter call.

**Tech Stack:** matplotlib, cartopy, xarray, numpy. NCAR style system for branding.

---

### Task 1: Test fixture — synthetic LMA grid data

**Files:**
- Modify: `davinci_monet/tests/test_obs_plots.py` (add fixture near line 24)

**Step 1: Write the fixture**

Add a `grid_lma_data` fixture that creates a small synthetic LMA-like dataset matching the structure the LMA reader returns: dims `(time, latitude, longitude)`, 12 five-minute time steps (one hour), with a hotspot of flash activity.

```python
@pytest.fixture
def grid_lma_data() -> xr.Dataset:
    """Create synthetic LMA gridded flash density dataset (one hour)."""
    np.random.seed(42)
    n_times = 12  # 5-min steps in one hour
    n_lat = 20
    n_lon = 25

    time = pd.date_range("2012-05-29 22:00", periods=n_times, freq="5min")
    lats = np.linspace(33.5, 37.0, n_lat)
    lons = np.linspace(-101.0, -96.0, n_lon)

    # Create a hotspot in the center
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    hotspot = np.exp(
        -((lat_grid - 35.2) ** 2 + (lon_grid - 98.5) ** 2) / 0.5
    )
    # Vary intensity over time (ramp up then down)
    time_profile = np.sin(np.linspace(0, np.pi, n_times))
    flash_extent = np.zeros((n_times, n_lat, n_lon))
    for t in range(n_times):
        flash_extent[t] = hotspot * time_profile[t] * 10 + np.random.poisson(0.5, (n_lat, n_lon))

    ds = xr.Dataset(
        {
            "flash_extent": (
                ["time", "latitude", "longitude"],
                flash_extent,
                {"units": "flashes/grid cell", "long_name": "Flash Extent Density"},
            ),
        },
        coords={
            "time": time,
            "latitude": lats,
            "longitude": lons,
        },
        attrs={"geometry": "grid", "lma_network_id": "oklma"},
    )
    return ds
```

**Step 2: Verify fixture works**

Run: `pytest davinci_monet/tests/test_obs_plots.py -v -k "grid_lma" --collect-only`
Expected: fixture is collected (no import errors)

**Step 3: Commit**

```
feat: add synthetic LMA grid fixture for density map tests
```

---

### Task 2: Test fixture — multi-hour LMA data

**Files:**
- Modify: `davinci_monet/tests/test_obs_plots.py`

**Step 1: Write the multi-hour fixture**

```python
@pytest.fixture
def grid_lma_data_multihour() -> xr.Dataset:
    """Create synthetic LMA data spanning 3 hours with varying activity."""
    np.random.seed(42)
    n_lat = 20
    n_lon = 25

    lats = np.linspace(33.5, 37.0, n_lat)
    lons = np.linspace(-101.0, -96.0, n_lon)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    hotspot = np.exp(
        -((lat_grid - 35.2) ** 2 + (lon_grid - 98.5) ** 2) / 0.5
    )

    # 3 hours × 12 steps = 36 time steps
    time = pd.date_range("2012-05-29 22:00", periods=36, freq="5min")
    hourly_scale = [5.0, 10.0, 3.0]  # hour 2 is peak

    flash_extent = np.zeros((36, n_lat, n_lon))
    for t in range(36):
        hour_idx = t // 12
        flash_extent[t] = (
            hotspot * hourly_scale[hour_idx]
            + np.random.poisson(0.5, (n_lat, n_lon))
        )

    ds = xr.Dataset(
        {
            "flash_extent": (
                ["time", "latitude", "longitude"],
                flash_extent,
                {"units": "flashes/grid cell", "long_name": "Flash Extent Density"},
            ),
        },
        coords={
            "time": time,
            "latitude": lats,
            "longitude": lons,
        },
        attrs={"geometry": "grid", "lma_network_id": "oklma"},
    )
    return ds
```

**Step 2: Commit**

```
feat: add multi-hour LMA grid fixture for hourly aggregation tests
```

---

### Task 3: Core renderer — single-hour density map

**Files:**
- Create: `davinci_monet/plots/renderers/obs/obs_lma_density.py`
- Modify: `davinci_monet/plots/renderers/obs/__init__.py`
- Test: `davinci_monet/tests/test_obs_plots.py`

**Step 1: Write the failing test**

```python
class TestObsLMADensityPlotter:
    """Tests for the ObsLMADensityPlotter (obs_lma_density)."""

    def test_basic_plot(self, grid_lma_data):
        """Basic LMA density map renders without error."""
        from davinci_monet.plots.renderers.obs.obs_lma_density import ObsLMADensityPlotter

        plotter = ObsLMADensityPlotter()
        fig = plotter.plot(grid_lma_data, "flash_extent")

        assert fig is not None
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_registered_in_registry(self):
        """ObsLMADensityPlotter is registered as 'obs_lma_density'."""
        from davinci_monet.plots.renderers.obs.obs_lma_density import ObsLMADensityPlotter  # noqa: F401
        from davinci_monet.plots.registry import has_plotter

        assert has_plotter("obs_lma_density")

    def test_save_output(self, grid_lma_data, tmp_path):
        """LMA density map can be saved to disk."""
        from davinci_monet.plots.renderers.obs.obs_lma_density import ObsLMADensityPlotter

        plotter = ObsLMADensityPlotter()
        fig = plotter.plot(grid_lma_data, "flash_extent")

        output_path = tmp_path / "lma_density.png"
        saved_path = plotter.save(fig, output_path)

        assert saved_path.exists()
        assert saved_path.stat().st_size > 0
        plt.close(fig)
```

**Step 2: Run test to verify it fails**

Run: `pytest davinci_monet/tests/test_obs_plots.py -v -k "TestObsLMADensity"`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the renderer**

Create `davinci_monet/plots/renderers/obs/obs_lma_density.py`:

```python
"""LMA flash density map renderer.

Produces cartopy maps of Lightning Mapping Array gridded flash density data,
with optional aircraft flight track overlays.
"""

from __future__ import annotations

from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from davinci_monet.plots.obs_base import ObsPlotter
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.style import NCAR_PALETTE


@register_plotter("obs_lma_density")
class ObsLMADensityPlotter(ObsPlotter):
    """Plotter for LMA gridded flash density maps."""

    name: str = "obs_lma_density"
    default_figsize: tuple[float, float] = (10, 8)

    def plot(
        self,
        obs_data: xr.Dataset,
        variable: str,
        ax: matplotlib.axes.Axes | None = None,
        title: str | None = None,
        cmap: str = "YlOrRd",
        vmin: float | None = None,
        vmax: float | None = None,
        time_agg: str | None = None,
        map: dict[str, Any] | None = None,
        flight_tracks: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure | list[tuple[matplotlib.figure.Figure, str]]:
        """Generate LMA density map(s).

        Parameters
        ----------
        obs_data : xr.Dataset
            LMA gridded data with dims (time, latitude, longitude).
        variable : str
            Variable name to plot (e.g. 'flash_extent').
        title : str, optional
            Title template. Hour info is appended automatically.
        cmap : str
            Matplotlib colormap name.
        vmin, vmax : float, optional
            Colorbar limits. Auto-scaled if not provided.
        time_agg : str, optional
            If 'hourly', produce one figure per hour. Otherwise one figure
            summing all time steps.
        map : dict, optional
            Map configuration (projection, features).
        flight_tracks : dict, optional
            Mapping of {label: obs_key} for flight track overlays.
            Track data is resolved from kwargs['obs_datasets'].

        Returns
        -------
        Figure or list of (Figure, suffix) tuples for multi-hour output.
        """
        map_config = map or {}
        projection = self._get_projection(map_config)
        features = map_config.get("features", ["states"])

        lat = obs_data["latitude"].values
        lon = obs_data["longitude"].values

        if time_agg == "hourly":
            hourly_groups = self._aggregate_hourly(obs_data, variable)
            if not hourly_groups:
                # No data — return empty figure
                fig, _ = self.create_figure()
                return fig

            results = []
            # Compute global vmax across all hours for consistent colorbar
            if vmax is None:
                all_maxes = [float(data.max()) for _, data in hourly_groups]
                auto_vmax = max(all_maxes) if all_maxes else 1.0
            else:
                auto_vmax = vmax

            for hour_label, data_2d in hourly_groups:
                fig = self._render_map(
                    lat, lon, data_2d,
                    projection=projection,
                    features=features,
                    cmap=cmap,
                    vmin=vmin or 0,
                    vmax=auto_vmax,
                    title=f"{title} {hour_label}" if title else hour_label,
                    flight_tracks=flight_tracks,
                    hour_label=hour_label,
                    **kwargs,
                )
                suffix = f"_{hour_label.replace(':', '').replace(' ', '_').replace('–', '-')}"
                results.append((fig, suffix))
            return results
        else:
            # Sum all time steps
            data_2d = obs_data[variable].sum(dim="time").values
            fig = self._render_map(
                lat, lon, data_2d,
                projection=projection,
                features=features,
                cmap=cmap,
                vmin=vmin or 0,
                vmax=vmax or float(data_2d.max()),
                title=title or f"LMA {variable}",
                flight_tracks=flight_tracks,
                **kwargs,
            )
            return fig

    def _get_projection(self, map_config: dict[str, Any]) -> ccrs.Projection:
        """Create cartopy projection from config."""
        proj_name = map_config.get("projection", "LambertConformal")
        if proj_name == "LambertConformal":
            return ccrs.LambertConformal(
                central_longitude=-98.5, central_latitude=35.0,
            )
        elif proj_name == "PlateCarree":
            return ccrs.PlateCarree()
        else:
            return ccrs.PlateCarree()

    def _aggregate_hourly(
        self, ds: xr.Dataset, variable: str,
    ) -> list[tuple[str, np.ndarray]]:
        """Aggregate data into hourly sums, returning only hours with activity."""
        groups = ds[variable].resample(time="1h").sum()
        results = []
        for t in groups.time.values:
            data_2d = groups.sel(time=t).values
            if np.nansum(data_2d) > 0:
                ts = np.datetime_as_string(t, unit="h")
                # Format as "YYYY-MM-DD HH:00–HH+1:00 UTC"
                hour = int(ts[-2:]) if len(ts) >= 2 else 0
                date_str = str(t)[:10]
                label = f"{date_str} {hour:02d}:00\u2013{(hour + 1) % 24:02d}:00 UTC"
                results.append((label, data_2d))
        return results

    def _render_map(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        data_2d: np.ndarray,
        *,
        projection: ccrs.Projection,
        features: list[str],
        cmap: str,
        vmin: float,
        vmax: float,
        title: str,
        flight_tracks: dict[str, str] | None = None,
        hour_label: str | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a single density map."""
        text_cfg = self.config.text
        fig_cfg = self.config.figure

        fig = plt.figure(figsize=fig_cfg.figsize, dpi=fig_cfg.dpi)
        ax = fig.add_subplot(1, 1, 1, projection=projection)

        # Map extent with padding
        pad = 0.3
        ax.set_extent(
            [lon.min() - pad, lon.max() + pad, lat.min() - pad, lat.max() + pad],
            crs=ccrs.PlateCarree(),
        )

        # Map features
        ax.add_feature(cfeature.LAND, facecolor="#F0F0F0", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
        if "states" in features:
            ax.add_feature(
                cfeature.STATES, edgecolor="gray", linewidth=0.5, zorder=1,
            )
        if "counties" in features:
            ax.add_feature(
                cfeature.NaturalEarthFeature(
                    "cultural", "admin_2_counties_lakes_shp", "10m",
                    edgecolor="lightgray", facecolor="none", linewidth=0.3,
                ),
                zorder=1,
            )
        if "coastlines" in features:
            ax.add_feature(
                cfeature.COASTLINE, edgecolor="black", linewidth=0.5, zorder=1,
            )

        # Density pcolormesh
        mesh = ax.pcolormesh(
            lon, lat, data_2d,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            shading="auto",
            zorder=2,
        )

        # Flight track overlays
        if flight_tracks:
            self._overlay_tracks(ax, flight_tracks, hour_label, **kwargs)

        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Colorbar
        cbar = fig.colorbar(
            mesh, ax=ax, orientation="horizontal", shrink=0.7, pad=0.06,
        )
        cbar.set_label(
            "Flash extent density (flashes per grid cell)",
            fontsize=text_cfg.fontsize,
        )
        cbar.ax.tick_params(labelsize=text_cfg.tick_fontsize)

        # Title
        fig.suptitle(title, fontsize=text_cfg.title_fontsize, y=0.95)

        return fig

    def _overlay_tracks(
        self,
        ax: matplotlib.axes.Axes,
        flight_tracks: dict[str, str],
        hour_label: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Overlay aircraft flight tracks on the map."""
        obs_datasets: dict[str, xr.Dataset] = kwargs.get("obs_datasets", {})
        if not obs_datasets:
            return

        track_colors = {
            label: NCAR_PALETTE[i % len(NCAR_PALETTE)]
            for i, label in enumerate(flight_tracks.keys())
        }

        for label, obs_key in flight_tracks.items():
            if obs_key not in obs_datasets:
                continue
            track_ds = obs_datasets[obs_key]

            # Get coordinates
            if "latitude" not in track_ds.coords or "longitude" not in track_ds.coords:
                continue
            track_lat = track_ds["latitude"].values
            track_lon = track_ds["longitude"].values

            ax.plot(
                track_lon, track_lat,
                color=track_colors[label],
                linewidth=1.5,
                transform=ccrs.PlateCarree(),
                label=label.upper(),
                zorder=5,
            )

        ax.legend(
            loc="upper right",
            fontsize=self.config.text.legend_small,
            framealpha=0.8,
        )
```

**Step 4: Register in `__init__.py`**

Add to `davinci_monet/plots/renderers/obs/__init__.py`:

```python
from davinci_monet.plots.renderers.obs.obs_lma_density import ObsLMADensityPlotter
```

And add `"ObsLMADensityPlotter"` to `__all__`.

**Step 5: Run tests to verify they pass**

Run: `pytest davinci_monet/tests/test_obs_plots.py -v -k "TestObsLMADensity"`
Expected: 3 PASS

**Step 6: Commit**

```
feat: add obs_lma_density renderer for LMA flash density maps
```

---

### Task 4: Test hourly aggregation

**Files:**
- Modify: `davinci_monet/tests/test_obs_plots.py`

**Step 1: Write the failing tests**

```python
    def test_hourly_aggregation(self, grid_lma_data_multihour):
        """Hourly mode returns list of (fig, suffix) tuples."""
        from davinci_monet.plots.renderers.obs.obs_lma_density import ObsLMADensityPlotter

        plotter = ObsLMADensityPlotter()
        result = plotter.plot(
            grid_lma_data_multihour, "flash_extent",
            time_agg="hourly",
            title="Test LMA",
        )

        assert isinstance(result, list)
        assert len(result) == 3  # 3 hours of data
        for fig, suffix in result:
            assert fig is not None
            assert isinstance(suffix, str)
            assert "UTC" not in suffix  # suffix is filename-safe
            plt.close(fig)

    def test_hourly_consistent_colorbar(self, grid_lma_data_multihour):
        """All hourly maps use the same colorbar range."""
        from davinci_monet.plots.renderers.obs.obs_lma_density import ObsLMADensityPlotter

        plotter = ObsLMADensityPlotter()
        result = plotter.plot(
            grid_lma_data_multihour, "flash_extent",
            time_agg="hourly",
        )

        # All pcolormesh artists should share the same clim
        clims = []
        for fig, _ in result:
            for ax in fig.axes:
                for child in ax.get_children():
                    if hasattr(child, "get_clim"):
                        clims.append(child.get_clim())
                        break
            plt.close(fig)

        assert len(clims) >= 2
        for clim in clims[1:]:
            assert clim == clims[0], "Colorbar ranges should be consistent across hours"
```

**Step 2: Run tests**

Run: `pytest davinci_monet/tests/test_obs_plots.py -v -k "TestObsLMADensity"`
Expected: all PASS (implementation already handles this)

**Step 3: Commit**

```
test: add hourly aggregation tests for LMA density renderer
```

---

### Task 5: Test flight track overlay

**Files:**
- Modify: `davinci_monet/tests/test_obs_plots.py`

**Step 1: Write the tests**

```python
    def test_flight_track_overlay(self, grid_lma_data, track_obs_data):
        """Density map with flight track overlay renders without error."""
        from davinci_monet.plots.renderers.obs.obs_lma_density import ObsLMADensityPlotter

        plotter = ObsLMADensityPlotter()
        fig = plotter.plot(
            grid_lma_data, "flash_extent",
            flight_tracks={"dc8": "dc8"},
            obs_datasets={"dc8": track_obs_data},
        )

        assert fig is not None
        # Check legend exists with aircraft label
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert "DC8" in labels
        plt.close(fig)

    def test_flight_track_missing_dataset(self, grid_lma_data):
        """Overlay gracefully skips missing flight track datasets."""
        from davinci_monet.plots.renderers.obs.obs_lma_density import ObsLMADensityPlotter

        plotter = ObsLMADensityPlotter()
        fig = plotter.plot(
            grid_lma_data, "flash_extent",
            flight_tracks={"dc8": "dc8"},
            obs_datasets={},  # empty — no track data
        )

        assert fig is not None
        plt.close(fig)
```

**Step 2: Run tests**

Run: `pytest davinci_monet/tests/test_obs_plots.py -v -k "TestObsLMADensity"`
Expected: all PASS

**Step 3: Commit**

```
test: add flight track overlay tests for LMA density renderer
```

---

### Task 6: Pipeline integration — multi-figure support

**Files:**
- Modify: `davinci_monet/pipeline/stages.py` (ObsPlottingStage, ~line 1659)
- Test: `davinci_monet/tests/test_obs_plots.py`

**Step 1: Write a test for multi-figure pipeline handling**

```python
    def test_save_hourly_outputs(self, grid_lma_data_multihour, tmp_path):
        """Hourly output can be saved as multiple files."""
        from davinci_monet.plots.renderers.obs.obs_lma_density import ObsLMADensityPlotter

        plotter = ObsLMADensityPlotter()
        result = plotter.plot(
            grid_lma_data_multihour, "flash_extent",
            time_agg="hourly",
            title="Test",
        )

        assert isinstance(result, list)
        saved = []
        for fig, suffix in result:
            out_path = tmp_path / f"lma_density{suffix}.png"
            plotter.save(fig, out_path)
            assert out_path.exists()
            saved.append(out_path)
            plt.close(fig)

        assert len(saved) == 3
```

**Step 2: Modify ObsPlottingStage**

In `davinci_monet/pipeline/stages.py`, find the section where obs plots are saved (~line 1659). After the existing `fig = plotter.plot(...)` call, add handling for list return:

```python
result = plotter.plot(subset, variable, **flight_kwargs)

# Multi-figure support (e.g., hourly LMA density maps)
if isinstance(result, list):
    for fig, fig_suffix in result:
        out_path = output_dir / f"{plot_name}{suffix}{fig_suffix}.png"
        plotter.save(fig, out_path)
        plt.close(fig)
        plots_saved.append(str(out_path))
else:
    fig = result
    out_path = output_dir / f"{plot_name}{suffix}.png"
    plotter.save(fig, out_path)
    plt.close(fig)
    plots_saved.append(str(out_path))
```

**Step 3: Run full test suite**

Run: `pytest davinci_monet/tests/test_obs_plots.py -v -k "TestObsLMADensity"`
Expected: all PASS

Run: `pytest davinci_monet/tests/ -v --timeout=120`
Expected: no regressions

**Step 4: Commit**

```
feat: support multi-figure output in obs plotting pipeline stage
```

---

### Task 7: Pipeline integration — pass obs_datasets to renderer

**Files:**
- Modify: `davinci_monet/pipeline/stages.py` (ObsPlottingStage)

**Step 1: Identify the issue**

The `flight_tracks` config references other obs datasets by key (e.g., `dc8: dc8`). The pipeline stage needs to pass all loaded obs datasets to the renderer so it can resolve track data.

**Step 2: Modify ObsPlottingStage**

In the section where plot kwargs are assembled, add `obs_datasets` from the pipeline context:

```python
# Add obs_datasets for renderers that need cross-dataset access (e.g., flight track overlays)
if "flight_tracks" in plot_spec:
    flight_kwargs["obs_datasets"] = context.get("obs_data", {})
```

**Step 3: Run tests**

Run: `pytest davinci_monet/tests/ -v --timeout=120`
Expected: all PASS, no regressions

**Step 4: Commit**

```
feat: pass obs_datasets to renderers with flight_tracks config
```

---

### Task 8: Integration test — May 29 config validation

**Files:**
- Test: `davinci_monet/tests/test_obs_plots.py`

**Step 1: Write config validation test**

```python
    def test_yaml_config_parse(self):
        """The May 29 YAML config parses without error for LMA density entries."""
        import yaml
        from pathlib import Path

        config_path = Path("analyses/dc3/configs/dc3-may29-gemini.yaml")
        if not config_path.exists():
            pytest.skip("May 29 config not present")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify LMA density plot entries exist
        plots = config["plots"]
        assert "lma_density" in plots
        assert plots["lma_density"]["type"] == "obs_lma_density"
        assert plots["lma_density"]["time_agg"] == "hourly"

        assert "lma_density_with_tracks" in plots
        assert "flight_tracks" in plots["lma_density_with_tracks"]
        assert "dc8" in plots["lma_density_with_tracks"]["flight_tracks"]
```

**Step 2: Run test**

Run: `pytest davinci_monet/tests/test_obs_plots.py -v -k "test_yaml_config"`
Expected: PASS

**Step 3: Commit**

```
test: add integration test for May 29 LMA config validation
```

---

### Task 9: Visual verification — run on real data

**Step 1: Run the pipeline on May 29 config**

```bash
cd /Users/fillmore/EarthSystem/DAVINCI-MONET
conda activate davinci-monet
davinci-monet run analyses/dc3/configs/dc3-may29-gemini.yaml
```

**Step 2: Verify output**

Check `analyses/dc3/output/may29/` for:
- `lma_density_*.png` — standalone hourly maps (expect ~9 files for active hours)
- `lma_density_with_tracks_*.png` — maps with DC-8/GV overlay
- All existing plot types (flight tracks, profiles, histograms, time series)

**Step 3: Visual inspection**

Open a few PNGs and verify:
- Flash density hotspot visible in Oklahoma
- Colorbar range reasonable, consistent across hours
- State borders and gridlines present
- Flight tracks visible on overlay version (DC-8 and GV labeled)
- NCAR styling applied (fonts, colors)

**Step 4: Fix any issues found during visual inspection**

**Step 5: Commit**

```
feat: verified LMA density maps for DC3 May 29 case
```
