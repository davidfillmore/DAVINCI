#!/usr/bin/env python
"""Download DC3 aircraft merge data from NASA ASDC.

Uses earthaccess library for Earthdata Login authentication.
Downloads ICARTT merge files to ~/Data/DC3/aircraft/merge/.

Usage:
    python download_dc3_aircraft.py [--data-dir ~/Data/DC3]

Requirements:
    pip install earthaccess
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def download_dc3_merge(data_dir: Path) -> None:
    """Download DC3 merge data files."""
    try:
        import earthaccess
    except ImportError:
        print("ERROR: earthaccess not installed. Run: pip install earthaccess")
        sys.exit(1)

    print("Authenticating with Earthdata Login...")
    earthaccess.login()

    merge_dir = data_dir / "aircraft" / "merge"
    merge_dir.mkdir(parents=True, exist_ok=True)

    print("Searching for DC3_Merge_Data...")
    results = earthaccess.search_data(
        short_name="DC3_Merge_Data",
        temporal=("2012-05-01", "2012-07-01"),
    )

    if not results:
        print("No DC3 merge data found via earthaccess.")
        print("\nManual download alternative:")
        print("  1. Go to: https://asdc.larc.nasa.gov/project/DC3/DC3_Merge_Data_1")
        print("  2. Click 'Get Dataset'")
        print(f"  3. Download ICARTT files to: {merge_dir}")
        sys.exit(1)

    # Filter to 10-second merge ICARTT files only
    mrg10_granules = []
    for r in results:
        links = r.data_links()
        for link in links:
            if "MRG10" in link and link.endswith(".ict"):
                mrg10_granules.append(r)
                break

    if not mrg10_granules:
        mrg10_granules = results  # Fallback to all if filter fails

    print(f"Found {len(mrg10_granules)} 10-second merge granules. Downloading to {merge_dir}...")
    downloaded = earthaccess.download(mrg10_granules, str(merge_dir))
    print(f"Downloaded {len(downloaded)} files to {merge_dir}")

    ict_files = sorted(merge_dir.glob("*.ict"))
    print(f"\nICARTT files available: {len(ict_files)}")
    for f in ict_files[:10]:
        print(f"  {f.name}")
    if len(ict_files) > 10:
        print(f"  ... and {len(ict_files) - 10} more")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DC3 aircraft merge data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "Data" / "DC3",
        help="Root data directory (default: ~/Data/DC3)",
    )
    args = parser.parse_args()
    download_dc3_merge(args.data_dir)


if __name__ == "__main__":
    main()
