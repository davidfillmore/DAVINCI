"""Standalone bar chart of per-day Pearson r between MERRA-2 AOD and CERES SW.

Reproduces the daily correlation view from PlumeSentinelAI.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from davinci_monet.plots.style import NCAR_COLORS


def plot_daily_correlation(
    records: list[dict[str, Any]],
    event_name: str = "",
) -> Figure:
    """Bar chart: per-day Pearson r between MERRA-2 AOD and CERES SW.

    Parameters
    ----------
    records
        List of dicts, each with: date, tot_aod, smoke_aod, sw_all
        (all 2-D arrays to ravel).
    event_name
        Optional event label for the figure title.

    Returns
    -------
    Figure
    """
    n_days = len(records)
    dates = [r["date"] for r in records]
    r_total: list[float] = []
    r_smoke: list[float] = []

    for rec in records:
        tot_flat = np.asarray(rec["tot_aod"]).ravel()
        smoke_flat = np.asarray(rec["smoke_aod"]).ravel()
        sw_flat = np.asarray(rec["sw_all"]).ravel()

        mask_t = np.isfinite(tot_flat) & np.isfinite(sw_flat)
        mask_s = np.isfinite(smoke_flat) & np.isfinite(sw_flat)

        if mask_t.sum() > 10 and np.std(tot_flat[mask_t]) > 0:
            r_total.append(float(np.corrcoef(tot_flat[mask_t], sw_flat[mask_t])[0, 1]))
        else:
            r_total.append(0.0)

        if mask_s.sum() > 10 and np.std(smoke_flat[mask_s]) > 0:
            r_smoke.append(float(np.corrcoef(smoke_flat[mask_s], sw_flat[mask_s])[0, 1]))
        else:
            r_smoke.append(0.0)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(n_days)
    width = 0.35
    ax.bar(
        x - width / 2,
        r_total,
        width,
        label="Total AOD",
        color=NCAR_COLORS["ncar_blue"],
    )
    ax.bar(
        x + width / 2,
        r_smoke,
        width,
        label="Smoke AOD (OC+BC)",
        color=NCAR_COLORS["orange"],
    )

    # Format date labels like "Sep 05"
    try:
        import pandas as pd

        date_labels = [pd.Timestamp(d).strftime("%b %d") for d in dates]
    except Exception:
        date_labels = dates

    ax.set_xticks(x)
    ax.set_xticklabels(date_labels, rotation=45, ha="right")
    ax.set_ylabel("Pearson r (MERRA-2 AOD vs CERES SW)")
    ax.set_ylim(-0.3, 1.0)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend()

    ax.set_title(
        "Daily Correlation: MERRA-2 AOD vs CERES TOA SW Reflected",
        fontsize=14,
        fontweight="bold",
        color=NCAR_COLORS["space"],
    )
    fig.tight_layout()
    return fig
