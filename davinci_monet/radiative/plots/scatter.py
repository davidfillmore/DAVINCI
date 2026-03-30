"""SW flux vs AOD scatter plots with per-day correlation bars.

Produces a 1x3 panel figure: total AOD scatter, smoke AOD scatter,
and a bar chart of per-day Pearson r for both AOD types.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from davinci_monet.plots.style import NCAR_COLORS


def plot_sw_vs_aod_scatter(
    lats: np.ndarray,
    lons: np.ndarray,
    records: list[dict[str, Any]],
    event_name: str = "",
) -> Figure:
    """Plot SW vs AOD scatter and per-day correlation.

    Parameters
    ----------
    lats
        1-D latitude array.
    lons
        1-D longitude array.
    records
        List of dicts, each with: date, tot_aod, smoke_aod, sw_all.
    event_name
        Optional event label for the figure title.

    Returns
    -------
    Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    n_days = len(records)
    cmap = plt.get_cmap("coolwarm", max(n_days, 2))
    dates = [r["date"] for r in records]

    # --- Panel 1: Total AOD vs SW ---
    ax = axes[0]
    for i, rec in enumerate(records):
        ax.scatter(
            rec["tot_aod"].ravel(),
            rec["sw_all"].ravel(),
            s=4,
            alpha=0.5,
            color=cmap(i / max(n_days - 1, 1)),
            label=rec["date"],
        )
    ax.set_xlabel("Total AOD")
    ax.set_ylabel("SW All-Sky (W/m\u00b2)")
    ax.set_title("Total AOD vs SW")

    # --- Panel 2: Smoke AOD vs SW ---
    ax = axes[1]
    for i, rec in enumerate(records):
        ax.scatter(
            rec["smoke_aod"].ravel(),
            rec["sw_all"].ravel(),
            s=4,
            alpha=0.5,
            color=cmap(i / max(n_days - 1, 1)),
            label=rec["date"],
        )
    ax.set_xlabel("Smoke AOD")
    ax.set_ylabel("SW All-Sky (W/m\u00b2)")
    ax.set_title("Smoke AOD vs SW")

    # --- Panel 3: Per-day correlation bar chart ---
    ax = axes[2]
    r_total = []
    r_smoke = []
    for rec in records:
        tot_flat = rec["tot_aod"].ravel()
        smoke_flat = rec["smoke_aod"].ravel()
        sw_flat = rec["sw_all"].ravel()
        # Pearson r via np.corrcoef (handles constant arrays gracefully)
        mask_t = np.isfinite(tot_flat) & np.isfinite(sw_flat)
        mask_s = np.isfinite(smoke_flat) & np.isfinite(sw_flat)
        if mask_t.sum() > 2 and np.std(tot_flat[mask_t]) > 0:
            r_total.append(np.corrcoef(tot_flat[mask_t], sw_flat[mask_t])[0, 1])
        else:
            r_total.append(0.0)
        if mask_s.sum() > 2 and np.std(smoke_flat[mask_s]) > 0:
            r_smoke.append(np.corrcoef(smoke_flat[mask_s], sw_flat[mask_s])[0, 1])
        else:
            r_smoke.append(0.0)

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
        label="Smoke AOD",
        color=NCAR_COLORS["orange"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r")
    ax.set_title("AOD–SW Correlation")
    ax.legend(fontsize=8)

    suptitle = f"{event_name} — AOD vs SW" if event_name else "AOD vs SW"
    fig.suptitle(suptitle, fontsize=16, color=NCAR_COLORS["space"])
    fig.tight_layout()
    return fig
