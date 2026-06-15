"""Programmatic rendered-label assertion tests (Task 17).

Builds synthetic paired data with realistic source_label / axis attrs and
mol/m2 units, then asserts the ACTUAL Axes text on the rendered figures.

Positive assertions: SI superscript units present, source names clean.
Negative assertions: no ALL-CAPS key leaks, no raw ``/m2`` solidus, no
``"y - x"`` / ``"(y"`` in colorbar labels.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from davinci_monet.plots import build_series

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _figure(
    result: matplotlib.figure.Figure | list[tuple[str, matplotlib.figure.Figure]],
) -> matplotlib.figure.Figure:
    """Narrow a render() result to a single Figure for assertion."""
    assert isinstance(result, matplotlib.figure.Figure)
    return result


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def paired_no2_mol_m2() -> xr.Dataset:
    """Paired dataset for NO2 column (mol/m2) with source_label/axis attrs.

    Simulates a cesm_no2_column vs pandora comparison so that:
    - x variable carries source_label='pandora', axis='x'
    - y variable carries source_label='cesm_no2_column', axis='y'
    - units are 'mol/m2' to exercise negative-exponent SI formatting.
    """
    rng = np.random.default_rng(0)
    n_times = 60
    n_sites = 4

    time = pd.date_range("2024-02-01", periods=n_times, freq="h")
    sites = [f"site_{i}" for i in range(n_sites)]

    obs = rng.uniform(1e-5, 4e-4, (n_times, n_sites))
    mod = obs * rng.uniform(0.8, 1.2, (n_times, n_sites))

    lats = np.linspace(20, 40, n_sites)
    lons = np.linspace(100, 130, n_sites)

    ds = xr.Dataset(
        {
            "pandora_no2_column": (
                ["time", "site"],
                obs,
                {
                    "units": "mol/m2",
                    "long_name": "NO2 Tropospheric Column",
                    "axis": "x",
                    "source_label": "pandora",
                },
            ),
            "cesm_no2_column": (
                ["time", "site"],
                mod,
                {
                    "units": "mol/m2",
                    "long_name": "NO2 Tropospheric Column",
                    "axis": "y",
                    "source_label": "cesm_no2_column",
                },
            ),
        },
        coords={
            "time": time,
            "site": sites,
            "latitude": ("site", lats),
            "longitude": ("site", lons),
        },
    )
    return ds


# ---------------------------------------------------------------------------
# Scatter label assertions
# ---------------------------------------------------------------------------


class TestScatterLabelsCleanliness:
    """Scatter axis labels must not contain ALL-CAPS keys, raw source keys, or
    un-formatted unit solidus ('/m2')."""

    def test_axis_labels_no_allcaps_key(self, paired_no2_mol_m2: xr.Dataset) -> None:
        """No ALL-CAPS raw key like 'CESM_NO2_COLUMN' or 'COLUMN' alone."""
        from davinci_monet.plots import ScatterPlotter

        fig = _figure(
            ScatterPlotter().render(
                build_series(
                    paired_no2_mol_m2,
                    "pandora_no2_column",
                    "cesm_no2_column",
                )
            )
        )
        ax = fig.axes[0]
        xl, yl = ax.get_xlabel(), ax.get_ylabel()
        for lbl in (xl, yl):
            assert "COLUMN" not in lbl, f"ALL-CAPS key 'COLUMN' leaked into label: {lbl!r}"
            assert "NO2_COLUMN" not in lbl, f"Raw key token in label: {lbl!r}"
        plt.close(fig)

    def test_axis_labels_no_raw_source_key(self, paired_no2_mol_m2: xr.Dataset) -> None:
        """Raw source key 'cesm_no2_column' must not appear in axis labels."""
        from davinci_monet.plots import ScatterPlotter

        fig = _figure(
            ScatterPlotter().render(
                build_series(
                    paired_no2_mol_m2,
                    "pandora_no2_column",
                    "cesm_no2_column",
                )
            )
        )
        ax = fig.axes[0]
        xl, yl = ax.get_xlabel(), ax.get_ylabel()
        for lbl in (xl, yl):
            assert "cesm_no2_column" not in lbl, f"Raw key in label: {lbl!r}"
        plt.close(fig)

    def test_axis_labels_no_solidus_m2(self, paired_no2_mol_m2: xr.Dataset) -> None:
        """Unit solidus '/m2' must not appear — units should be superscripted."""
        from davinci_monet.plots import ScatterPlotter

        fig = _figure(
            ScatterPlotter().render(
                build_series(
                    paired_no2_mol_m2,
                    "pandora_no2_column",
                    "cesm_no2_column",
                )
            )
        )
        ax = fig.axes[0]
        xl, yl = ax.get_xlabel(), ax.get_ylabel()
        for lbl in (xl, yl):
            assert "/m2" not in lbl, f"Un-formatted solidus '/m2' in label: {lbl!r}"
        plt.close(fig)

    def test_axis_labels_contain_superscript_units(self, paired_no2_mol_m2: xr.Dataset) -> None:
        """Axis labels must contain SI negative-exponent unit notation ('$^{-2}$')."""
        from davinci_monet.plots import ScatterPlotter

        fig = _figure(
            ScatterPlotter().render(
                build_series(
                    paired_no2_mol_m2,
                    "pandora_no2_column",
                    "cesm_no2_column",
                )
            )
        )
        ax = fig.axes[0]
        xl, yl = ax.get_xlabel(), ax.get_ylabel()
        for lbl in (xl, yl):
            assert "$^{-2}$" in lbl, f"Expected SI superscript '$^{{-2}}$' in label, got: {lbl!r}"
        plt.close(fig)

    def test_axis_labels_clean_source_names(self, paired_no2_mol_m2: xr.Dataset) -> None:
        """x-axis should display 'Pandora' and y-axis 'CESM' (not raw keys)."""
        from davinci_monet.plots import ScatterPlotter

        fig = _figure(
            ScatterPlotter().render(
                build_series(
                    paired_no2_mol_m2,
                    "pandora_no2_column",
                    "cesm_no2_column",
                )
            )
        )
        ax = fig.axes[0]
        xl, yl = ax.get_xlabel(), ax.get_ylabel()
        assert "Pandora" in xl, f"Expected 'Pandora' in xlabel, got: {xl!r}"
        assert "CESM" in yl, f"Expected 'CESM' in ylabel, got: {yl!r}"
        plt.close(fig)


# ---------------------------------------------------------------------------
# Spatial bias colorbar label assertions
# ---------------------------------------------------------------------------


class TestSpatialBiasColorbarlabel:
    """Spatial bias colorbar must not say 'y - x' or '(y'; must say 'Bias,'."""

    @pytest.mark.skipif(
        not pytest.importorskip("cartopy", reason="cartopy not available"),
        reason="cartopy not available",
    )
    def test_colorbar_no_xy_roles(self, paired_no2_mol_m2: xr.Dataset) -> None:
        """Colorbar label must not expose 'y - x' or '(y' role notation."""
        from davinci_monet.plots import SpatialBiasPlotter

        fig = SpatialBiasPlotter().render(
            build_series(
                paired_no2_mol_m2,
                "pandora_no2_column",
                "cesm_no2_column",
            )
        )
        # The colorbar is the last axes; its ylabel carries the label.
        cb_ax = fig.axes[-1]
        cb_label = cb_ax.get_ylabel()
        assert (
            "y - x" not in cb_label.lower()
        ), f"Colorbar must not expose x/y role notation, got: {cb_label!r}"
        assert (
            "(y" not in cb_label.lower()
        ), f"Colorbar must not expose '(y' role notation, got: {cb_label!r}"
        plt.close(fig)

    @pytest.mark.skipif(
        not pytest.importorskip("cartopy", reason="cartopy not available"),
        reason="cartopy not available",
    )
    def test_colorbar_contains_bias(self, paired_no2_mol_m2: xr.Dataset) -> None:
        """Colorbar label must contain 'Bias,'."""
        from davinci_monet.plots import SpatialBiasPlotter

        fig = SpatialBiasPlotter().render(
            build_series(
                paired_no2_mol_m2,
                "pandora_no2_column",
                "cesm_no2_column",
            )
        )
        cb_ax = fig.axes[-1]
        cb_label = cb_ax.get_ylabel()
        assert "Bias," in cb_label, f"Colorbar must contain 'Bias,', got: {cb_label!r}"
        plt.close(fig)

    @pytest.mark.skipif(
        not pytest.importorskip("cartopy", reason="cartopy not available"),
        reason="cartopy not available",
    )
    def test_colorbar_no_solidus_units(self, paired_no2_mol_m2: xr.Dataset) -> None:
        """Colorbar units must not use solidus '/m2'."""
        from davinci_monet.plots import SpatialBiasPlotter

        fig = SpatialBiasPlotter().render(
            build_series(
                paired_no2_mol_m2,
                "pandora_no2_column",
                "cesm_no2_column",
            )
        )
        cb_ax = fig.axes[-1]
        cb_label = cb_ax.get_ylabel()
        assert (
            "/m2" not in cb_label
        ), f"Colorbar units must be superscripted (no solidus), got: {cb_label!r}"
        plt.close(fig)


# ---------------------------------------------------------------------------
# Timeseries label assertions
# ---------------------------------------------------------------------------


class TestTimeseriesLabelsClean:
    """Timeseries y-label must include SI units; legend must have clean source names."""

    def test_ylabel_has_units(self, paired_no2_mol_m2: xr.Dataset) -> None:
        """Timeseries y-axis label must include SI units (not empty)."""
        from davinci_monet.plots import TimeSeriesPlotter

        fig = TimeSeriesPlotter().render(
            build_series(
                paired_no2_mol_m2,
                "pandora_no2_column",
                "cesm_no2_column",
            ),
            aggregate_dim="site",
        )
        ax = fig.axes[0]
        ylabel = ax.get_ylabel()
        assert ylabel, "Timeseries y-label must not be empty"
        assert (
            "$^{-2}$" in ylabel
        ), f"Timeseries y-label must contain SI superscript '$^{{-2}}$', got: {ylabel!r}"
        plt.close(fig)

    def test_ylabel_no_solidus_units(self, paired_no2_mol_m2: xr.Dataset) -> None:
        """Timeseries y-label must not use solidus '/m2'."""
        from davinci_monet.plots import TimeSeriesPlotter

        fig = TimeSeriesPlotter().render(
            build_series(
                paired_no2_mol_m2,
                "pandora_no2_column",
                "cesm_no2_column",
            ),
            aggregate_dim="site",
        )
        ax = fig.axes[0]
        ylabel = ax.get_ylabel()
        assert (
            "/m2" not in ylabel
        ), f"Timeseries y-label must not contain solidus '/m2', got: {ylabel!r}"
        plt.close(fig)

    def test_legend_clean_source_names(self, paired_no2_mol_m2: xr.Dataset) -> None:
        """Legend entries must show clean source names, not raw keys."""
        from davinci_monet.plots import TimeSeriesPlotter

        fig = TimeSeriesPlotter().render(
            build_series(
                paired_no2_mol_m2,
                "pandora_no2_column",
                "cesm_no2_column",
            ),
            aggregate_dim="site",
        )
        ax = fig.axes[0]
        legend = ax.get_legend()
        if legend is not None:
            legend_texts = [t.get_text() for t in legend.get_texts()]
            for text in legend_texts:
                assert "pandora_no2_column" not in text, f"Raw source key in legend: {text!r}"
                assert "cesm_no2_column" not in text, f"Raw source key in legend: {text!r}"
        plt.close(fig)

    def test_legend_no_xy_role_tokens(self, paired_no2_mol_m2: xr.Dataset) -> None:
        """Legend entries must not show '(x)' or '(y)' role tokens."""
        from davinci_monet.plots import TimeSeriesPlotter

        fig = TimeSeriesPlotter().render(
            build_series(
                paired_no2_mol_m2,
                "pandora_no2_column",
                "cesm_no2_column",
            ),
            aggregate_dim="site",
        )
        ax = fig.axes[0]
        legend = ax.get_legend()
        if legend is not None:
            legend_texts = [t.get_text() for t in legend.get_texts()]
            for text in legend_texts:
                assert "(x)" not in text.lower(), f"x/y role token in legend: {text!r}"
                assert "(y)" not in text.lower(), f"x/y role token in legend: {text!r}"
        plt.close(fig)
