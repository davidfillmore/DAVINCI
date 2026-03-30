"""AERONET AOD data loader."""

from __future__ import annotations

import glob

import pandas as pd

from davinci_monet.logging import get_logger

logger = get_logger(__name__)


def load_aeronet(
    files: str,
    domain: tuple[float, float, float, float],
    sites: list[str] | None = None,
) -> pd.DataFrame:
    """Load AERONET CSV file(s) and filter to a geographic domain.

    Parameters
    ----------
    files : str
        Glob pattern or single path to CSV file(s).
    domain : tuple
        ``(west, east, south, north)`` bounding box.
    sites : list[str] | None
        If given, keep only these site names.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with columns ``time``, ``site``, ``aod``,
        ``latitude``, ``longitude``.
    """
    paths = sorted(glob.glob(files))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {files}")
    logger.info("Loading %d AERONET file(s)", len(paths))

    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    # Create unified "aod" column from AOD_500nm if needed
    if "AOD_500nm" in df.columns and "aod" not in df.columns:
        df["aod"] = df["AOD_500nm"]

    # Filter to domain
    west, east, south, north = domain
    df = df[
        (df["longitude"] >= west)
        & (df["longitude"] <= east)
        & (df["latitude"] >= south)
        & (df["latitude"] <= north)
    ]

    # Filter to sites if specified
    if sites is not None:
        df = df[df["site"].isin(sites)]

    return df.reset_index(drop=True)
