"""Fetch (or load from cache) OSM points of interest for the configured AOI,
classified into the vulnerability categories in app.vulnerable.poi.

HONESTY: these are a documented PROXY for vulnerable populations, not real
population/census data -- see app.vulnerable.poi.POPULATION_PROXY_NOTE.

Run from the backend/ directory so the app package is importable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.vulnerable.poi import POI_CATEGORIES, fetch_pois  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Fetch vulnerable-population-proxy POIs from OSM/Overpass")
    parser.add_argument("--output", default="../data/raw/vulnerable_pois.parquet")
    parser.add_argument("--force-refresh", action="store_true", help="re-hit Overpass even if the cache exists")
    args = parser.parse_args()

    settings = get_settings()
    pois_df = fetch_pois(settings.AOI_BBOX, output_path=Path(args.output), force_refresh=args.force_refresh)

    print(f"AOI: {settings.AOI_BBOX}")
    print(f"total POIs: {len(pois_df)}\n")

    print(f"{'category':<16} {'group':<20} {'weight':>7} {'count':>7}")
    print("-" * 52)
    counts = pois_df["category"].value_counts()
    for category in POI_CATEGORIES:
        n = int(counts.get(category, 0))
        group = POI_CATEGORIES[category]["vulnerability_group"]
        weight = POI_CATEGORIES[category]["weight"]
        print(f"{category:<16} {group:<20} {weight:>7.1f} {n:>7}")

    print(f"\noutput: {args.output}")


if __name__ == "__main__":
    main()
