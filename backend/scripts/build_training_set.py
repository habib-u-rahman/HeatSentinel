"""Build the RF training set: sample_points x surface_profiles x temperature
grid(s) x Open-Meteo weather.

By default, generates a FIXTURE grid series (source=FIXTURE, clearly labelled
as synthetic -- see app.fortyguard.fixtures) since no LIVE FortyGuard grids
are stored yet. Pass --use-live to load real stored grids instead via
app.grid.store.load_series.

Run from the backend/ directory so the app package is importable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.fortyguard.fixtures import generate_series  # noqa: E402
from app.grid import store as grid_store  # noqa: E402
from app.ingest.openmeteo import OpenMeteoClient  # noqa: E402
from app.ml.dataset import build_training_set  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build the RF training set")
    parser.add_argument("--sample-points", default="../data/raw/sample_points.parquet")
    parser.add_argument("--surface-profiles", default="../data/raw/surface_profiles.parquet")
    parser.add_argument("--output", default="../data/raw/training_set.parquet")
    parser.add_argument("--use-live", action="store_true", help="load stored LIVE grids instead of generating FIXTURE ones")
    parser.add_argument("--hours", type=int, default=24, help="fixture series length in hours (ignored with --use-live)")
    parser.add_argument("--granularity-m", type=int, default=100)
    parser.add_argument("--max-dist-m", type=float, default=150.0)
    args = parser.parse_args()

    settings = get_settings()
    bbox = settings.AOI_BBOX

    surface_profiles_path = Path(args.surface_profiles)
    surface_profiles_df = pd.read_parquet(surface_profiles_path) if surface_profiles_path.exists() else None

    if args.use_live:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=7)
        grids = grid_store.load_series(bbox, start, end)
        if not grids:
            raise SystemExit(f"--use-live given but no stored LIVE grids found for bbox={bbox} in the last 7 days")
        print(f"loaded {len(grids)} LIVE grids from the store")
    else:
        start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=args.hours - 1)
        grids = generate_series(
            bbox, start, end, freq="1h", granularity_m=args.granularity_m, seed=0, surface_profiles=surface_profiles_df
        )
        print(f"generated {len(grids)} FIXTURE grids ({start.isoformat()} .. {end.isoformat()}) -- SYNTHETIC data")

    training_df, _report_lines = build_training_set(
        Path(args.sample_points),
        surface_profiles_path,
        grids,
        openmeteo_client=OpenMeteoClient(),
        max_dist_m=args.max_dist_m,
        bbox=bbox,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_parquet(output_path, index=False)

    print(f"\nwrote {len(training_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
