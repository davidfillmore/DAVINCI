"""Smoke tests for radiative plot renderers.

Each test verifies that the plotter produces a Figure without errors.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_event_data():
    """Return (lats, lons, records) for a 10x10 grid over 5 days."""
    lats = np.linspace(30, 40, 10)
    lons = np.linspace(-120, -110, 10)
    rng = np.random.default_rng(42)

    records = []
    for day in range(5):
        records.append(
            {
                "date": f"2020-09-{10 + day:02d}",
                "aod": rng.uniform(0, 3, (10, 10)),
                "sw_all": rng.uniform(0, 350, (10, 10)),
                "sw_clr": rng.uniform(0, 350, (10, 10)),
                "toa_net": rng.uniform(-150, 150, (10, 10)),
                "cld_frac": rng.uniform(0, 100, (10, 10)),
                "tot_aod": rng.uniform(0, 3, (10, 10)),
                "smoke_aod": rng.uniform(0, 2, (10, 10)),
                "m2_sfc_effect": rng.uniform(-250, 250, (10, 10)),
                "semi_dimming": rng.uniform(-250, 250, (10, 10)),
            }
        )
    return lats, lons, records


@pytest.fixture()
def synthetic_aeronet():
    """DataFrame with 2 sites and 5 days of AOD data."""
    rng = np.random.default_rng(99)
    rows = []
    for site in ["Site_A", "Site_B"]:
        for day in range(5):
            for hour in range(6, 18):
                rows.append(
                    {
                        "siteid": site,
                        "time": pd.Timestamp(f"2020-09-{10 + day:02d} {hour:02d}:00"),
                        "aod": rng.uniform(0.1, 2.0),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_sites():
    """List of site tuples: (name, lat, lon, aeronet_name)."""
    return [
        ("Portland", 35.0, -115.0, "Site_A"),
        ("Eugene", 36.0, -116.0, "Site_B"),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEventFields:
    def test_produces_figure(self, synthetic_event_data):
        from davinci_monet.radiative.plots.event_fields import plot_event_fields

        lats, lons, records = synthetic_event_data
        fig = plot_event_fields(lats, lons, records[0], event_name="Test Event")
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestAnomalyMaps:
    def test_produces_figure(self, synthetic_event_data):
        from davinci_monet.radiative.plots.anomaly_maps import plot_anomaly_maps

        lats, lons, records = synthetic_event_data
        # Background from mean of first 3 records
        bg = {}
        for key in ("aod", "sw_all", "sw_clr", "toa_net"):
            bg[key] = np.mean([r[key] for r in records[:3]], axis=0)
        fig = plot_anomaly_maps(lats, lons, records[3], bg, event_name="Test Event")
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestScatter:
    def test_produces_figure(self, synthetic_event_data):
        from davinci_monet.radiative.plots.scatter import plot_sw_vs_aod_scatter

        lats, lons, records = synthetic_event_data
        fig = plot_sw_vs_aod_scatter(lats, lons, records, event_name="Test Event")
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestSiteTimeseries:
    def test_produces_figure(
        self, synthetic_event_data, synthetic_aeronet, synthetic_sites
    ):
        from davinci_monet.radiative.plots.site_timeseries import plot_site_timeseries

        lats, lons, records = synthetic_event_data
        # Compute a scalar background SW for each grid cell
        bg_sw = np.mean([r["sw_all"] for r in records[:3]], axis=0)
        fig = plot_site_timeseries(
            lats,
            lons,
            records,
            bg_sw,
            synthetic_sites,
            aeronet=synthetic_aeronet,
            event_name="Test Event",
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestSurfaceImpact:
    def test_produces_figure(self, synthetic_event_data):
        from davinci_monet.radiative.plots.surface_impact import plot_surface_impact

        lats, lons, records = synthetic_event_data
        fig = plot_surface_impact(lats, lons, records[0], event_name="Test Event")
        assert isinstance(fig, Figure)
        plt.close(fig)
