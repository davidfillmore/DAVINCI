#!/usr/bin/env python
"""Download FIREX-AQ aircraft merge data from NASA ASDC.

Uses earthaccess library for Earthdata Login authentication.
Downloads ICARTT merge files to ~/Data/FIREX-AQ/aircraft/merge/.

Usage:
    python download_firex_aircraft.py [--data-dir ~/Data/FIREX-AQ]

Requirements:
    pip install earthaccess
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def download_firex_merge(data_dir: Path) -> None:
    """Download FIREX-AQ DC-8 merge data files."""
    try:
        import earthaccess
    except ImportError:
        print("ERROR: earthaccess not installed. Run: pip install earthaccess")
        sys.exit(1)

    print("Authenticating with Earthdata Login...")
    earthaccess.login()

    merge_dir = data_dir / "aircraft" / "merge"
    merge_dir.mkdir(parents=True, exist_ok=True)

    print("Searching for FIREXAQ_Merge_Data...")
    results = earthaccess.search_data(
        short_name="FIREXAQ_Merge_Data",
        temporal=("2019-07-01", "2019-10-01"),
    )

    if not results:
        print("No FIREX-AQ merge data found via earthaccess.")
        print("\nManual download alternatives:")
        print("  1. NASA ASDC: https://asdc.larc.nasa.gov/project/FIREX-AQ")
        print("  2. NASA Airborne Science: https://airbornescience.nasa.gov/content/FIREX-AQ")
        print("  3. NOAA CSL: https://csl.noaa.gov/projects/firex-aq/")
        print(f"  Download ICARTT files to: {merge_dir}")
        sys.exit(1)

    mrg60_granules = []
    for r in results:
        links = r.data_links()
        for link in links:
            if "MRG60" in link and link.endswith(".ict"):
                mrg60_granules.append(r)
                break

    if not mrg60_granules:
        mrg60_granules = results

    print(f"Found {len(mrg60_granules)} 60-second merge granules. Downloading to {merge_dir}...")
    downloaded = earthaccess.download(mrg60_granules, str(merge_dir))
    print(f"Downloaded {len(downloaded)} files to {merge_dir}")

    ict_files = sorted(merge_dir.glob("*.ict"))
    print(f"\nICARTT files available: {len(ict_files)}")
    for f in ict_files[:10]:
        print(f"  {f.name}")
    if len(ict_files) > 10:
        print(f"  ... and {len(ict_files) - 10} more")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FIREX-AQ aircraft merge data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "Data" / "FIREX-AQ",
        help="Root data directory (default: ~/Data/FIREX-AQ)",
    )
    args = parser.parse_args()
    download_firex_merge(args.data_dir)


if __name__ == "__main__":
    main()
