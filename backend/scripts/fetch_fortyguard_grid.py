"""Fetch a live FortyGuard heatmap for the configured AOI and save it as a
TemperatureGrid snapshot under data/grids/, so app.api.deps.get_grid_for_timestamp
picks it up as real LIVE data instead of falling back to FIXTURE.

filter_type is a required *integer* 1-4 despite what the task brief implied --
confirmed against the real API on 2026-08-29 (a string like "Vehicle" gets a
422). Only 1 and 3 were confirmed to submit successfully against this
account's plan; 2 and 4 returned HTTP 500.

Not every AOI has FortyGuard coverage: an AOI outside FortyGuard's covered
regions completes successfully but returns n_cells=0. This script treats that
as a failure (raises rather than saving an empty, useless grid) so the app's
existing FIXTURE fallback stays in effect for uncovered AOIs.

Run from the backend/ directory so the app package is importable:
    python -m scripts.fetch_fortyguard_grid
    python -m scripts.fetch_fortyguard_grid --bbox -74.020,40.700,-73.995,40.726 --city "New York"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.core.geo import bbox_to_polygon  # noqa: E402
from app.fortyguard.client import DEFAULT_FILTER_TYPE, FortyGuardClient  # noqa: E402
from app.grid.adapter import SchemaMismatchError, parse_heatmap_response  # noqa: E402
from app.grid.store import save_grid  # noqa: E402

logger = logging.getLogger(__name__)


class NoCoverageError(Exception):
    """Raised when FortyGuard completes the request but returns zero cells for this AOI."""


async def fetch_and_save(bbox: str, granularity_m: int, filter_type: int, force_refresh: bool) -> Path:
    polygon = bbox_to_polygon(tuple(float(x) for x in bbox.split(",")))
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    async with FortyGuardClient() as client:
        raw = await client.get_heatmap(
            polygon,
            date_str,
            time_str,
            filter_type=filter_type,
            granularity=granularity_m,
            force_refresh=force_refresh,
        )

    activity_id = raw.get("data", {}).get("activity_id")
    n_cells = raw.get("data", {}).get("result", {}).get("stats_data", {}).get("n_cells")
    if n_cells == 0:
        raise NoCoverageError(f"FortyGuard completed activity {activity_id} but returned zero cells for bbox={bbox}")

    grid = parse_heatmap_response(
        raw,
        bbox=bbox,
        granularity_m=granularity_m,
        observed_at=now,
        api_activity_id=activity_id,
    )
    return save_grid(grid)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Fetch a live FortyGuard heatmap grid and save it to disk")
    parser.add_argument("--bbox", default=None, help="min_lon,min_lat,max_lon,max_lat (default: settings.AOI_BBOX)")
    parser.add_argument("--granularity", type=int, default=None, help="metres (default: settings.GRID_GRANULARITY_M)")
    parser.add_argument("--filter-type", type=int, default=DEFAULT_FILTER_TYPE, choices=[1, 2, 3, 4])
    parser.add_argument("--force-refresh", action="store_true", help="bypass the on-disk request cache")
    args = parser.parse_args()

    settings = get_settings()
    bbox = args.bbox or settings.AOI_BBOX
    granularity = args.granularity or settings.GRID_GRANULARITY_M

    start = time.monotonic()
    try:
        path = asyncio.run(fetch_and_save(bbox, granularity, args.filter_type, args.force_refresh))
    except NoCoverageError as exc:
        print(f"NO COVERAGE: {exc}")
        print("FortyGuard has no data for this AOI -- the app will keep using FIXTURE data for it.")
        raise SystemExit(1)
    except SchemaMismatchError as exc:
        print(f"SCHEMA MISMATCH -- FortyGuard's response shape changed:\n{exc}")
        raise SystemExit(1)

    elapsed_s = time.monotonic() - start
    print(f"Saved LIVE grid to {path} in {elapsed_s:.1f}s (bbox={bbox}, granularity={granularity}m)")


if __name__ == "__main__":
    main()
