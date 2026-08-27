"""Build (or load from cache) the city walk graph and print stats.

STEP 0 of the routing pipeline: after building/loading the raw graph, reduce
it to its largest weakly-connected component (real street networks often
have disconnected islands -- piers, closed paths, isolated service roads)
and cache that REDUCED graph separately. app.routing.router always routes on
the connected graph, never the raw one, so a NetworkXNoPath traceback
mid-demo should never happen from a cross-island query.

Usage:
    python scripts/build_graph.py
    python scripts/build_graph.py --bbox min_lon,min_lat,max_lon,max_lat
    python scripts/build_graph.py --force   # skip the AOI size guard

Run from the backend/ directory so the app package is importable.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import osmnx as ox  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.core.geo import MAX_AREA_KM2, bbox_area_km2  # noqa: E402
from app.routing.graph import build_edge_index, build_walk_graph, largest_weakly_connected_component  # noqa: E402

logger = logging.getLogger(__name__)

CONNECTED_CACHE_FILENAME = "walk_graph_connected.graphml"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build the city walk graph")
    parser.add_argument("--bbox", default=None, help="min_lon,min_lat,max_lon,max_lat override")
    parser.add_argument("--cache", default=None, help="GraphML cache file path override")
    parser.add_argument("--connected-cache", default=None, help="reduced (largest-component) GraphML cache path override")
    parser.add_argument("--force", action="store_true", help="skip the AOI area guard")
    args = parser.parse_args()

    settings = get_settings()
    bbox = args.bbox or settings.AOI_BBOX
    area_km2 = bbox_area_km2(bbox)

    if area_km2 > MAX_AREA_KM2 and not args.force:
        raise SystemExit(
            f"AOI is {area_km2:.1f} km2, which exceeds the {MAX_AREA_KM2} km2 cap.\n"
            "A full-city walk graph is too slow to demo and will exhaust memory.\n"
            "Pass --bbox with a ~3km x 3km area, or --force to override."
        )

    cache_path = Path(args.cache) if args.cache else Path(settings.CACHE_DIR) / "walk_graph.graphml"
    connected_cache_path = (
        Path(args.connected_cache) if args.connected_cache else Path(settings.CACHE_DIR) / CONNECTED_CACHE_FILENAME
    )

    start = time.monotonic()
    graph = build_walk_graph(bbox, cache_path)
    build_edge_index(graph)  # exercise the KD-tree build path
    elapsed_s = time.monotonic() - start

    print(f"nodes: {graph.number_of_nodes()}")
    print(f"edges: {graph.number_of_edges()}")
    print(f"AOI area: {area_km2:.2f} km2")
    print(f"build time: {elapsed_s:.2f}s")

    # STEP 0: connectivity check + reduction, cached separately from the raw graph.
    connected_graph, n_dropped = largest_weakly_connected_component(graph)
    connected_cache_path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(connected_graph, connected_cache_path)

    print(f"\nconnected component: {connected_graph.number_of_nodes()} nodes ({n_dropped} dropped as disconnected islands)")
    print(f"connected component edges: {connected_graph.number_of_edges()}")
    print(f"connected graph cache: {connected_cache_path}")
    if n_dropped:
        print(f"NOTE: {n_dropped} nodes were unreachable from the main network and excluded from routing.")


if __name__ == "__main__":
    main()
