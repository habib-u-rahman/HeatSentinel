"""Demo: heat-dose-aware routing between two hardcoded points in the AOI.

Prints the SHORTEST/BALANCED/COOLEST route family as a table plus the
plain-English shortest-vs-coolest comparison, and writes routes.geojson
(one Feature per route in the family) so you can drag it onto geojson.io
and eyeball it immediately.

Run from the backend/ directory so the app package is importable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import osmnx as ox  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.fortyguard.fixtures import generate_grid  # noqa: E402
from app.routing.graph import _annotate_edges  # noqa: E402
from app.routing.pareto import compare, compute_route_family  # noqa: E402

logger = logging.getLogger(__name__)

CONNECTED_GRAPH_FILENAME = "walk_graph_connected.graphml"

# Two hardcoded points in the Rawalpindi demo AOI, picked to actually show a
# shortest-vs-coolest divergence (many pairs in a dense street grid don't --
# see the module docstring in app.routing.pareto: that's fine too).
DEMO_START = (33.5820, 73.0330)
DEMO_END = (33.6080, 73.0670)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Demo heat-dose-aware routing")
    parser.add_argument("--start", nargs=2, type=float, default=DEMO_START, metavar=("LAT", "LON"))
    parser.add_argument("--end", nargs=2, type=float, default=DEMO_END, metavar=("LAT", "LON"))
    parser.add_argument("--output", default="routes.geojson")
    args = parser.parse_args()

    settings = get_settings()
    bbox = settings.AOI_BBOX
    graph_path = Path(settings.CACHE_DIR) / CONNECTED_GRAPH_FILENAME
    if not graph_path.exists():
        raise SystemExit(
            f"No connected walk graph at {graph_path}. Run `python scripts/build_graph.py` first (STEP 0)."
        )

    graph = ox.load_graphml(graph_path)
    _annotate_edges(graph)

    timestamp = datetime.now(timezone.utc).replace(hour=15, minute=0, second=0, microsecond=0)
    grid = generate_grid(bbox, settings.GRID_GRANULARITY_M, timestamp, seed=0)
    print(f"grid: source={grid.source.value} observed_at={grid.observed_at.isoformat()} ({len(grid.cells)} cells)")
    if grid.source.value == "FIXTURE":
        print("NOTE: routing on SYNTHETIC fixture temperatures -- not live FortyGuard data.\n")

    start = tuple(args.start)
    end = tuple(args.end)
    print(f"start: {start}")
    print(f"end:   {end}\n")

    family = compute_route_family(start, end, graph, grid)

    header = f"{'label':<10} {'lambda':>7} {'dist_m':>9} {'dur_min':>8} {'dose_degC_s':>12} {'mean_wbgt':>10} {'max_wbgt':>9} {'band':>10} {'orphaned':>9}"
    print(header)
    print("-" * len(header))
    for r in family:
        print(
            f"{(r.label or ''):<10} {r.lambda_heat:>7.2f} {r.total_distance_m:>9.0f} "
            f"{r.total_duration_s / 60:>8.1f} {r.total_heat_dose_degC_s:>12.0f} "
            f"{r.mean_wbgt_c:>10.1f} {r.max_wbgt_c:>9.1f} {r.peak_risk_band:>10} {r.n_edges_orphaned:>9}"
        )

    shortest = next((r for r in family if r.label == "SHORTEST"), family[0])
    coolest = next((r for r in family if r.label == "COOLEST"), family[-1])
    comparison = compare(shortest, coolest)

    print(f"\nSHORTEST vs COOLEST: {comparison.summary}")

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                **r.geojson,
                "properties": {
                    "label": r.label,
                    "lambda_heat": r.lambda_heat,
                    "total_distance_m": r.total_distance_m,
                    "total_duration_s": r.total_duration_s,
                    "total_heat_dose_degC_s": r.total_heat_dose_degC_s,
                    "mean_wbgt_c": r.mean_wbgt_c,
                    "max_wbgt_c": r.max_wbgt_c,
                    "peak_risk_band": r.peak_risk_band,
                    "data_source": r.data_source,
                },
            }
            for r in family
        ],
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    print(f"\nwrote {len(family)} route(s) to {output_path}")


if __name__ == "__main__":
    main()
