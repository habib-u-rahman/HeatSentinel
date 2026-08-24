"""Build the AOI sample-point set and write it to data/raw/sample_points.parquet.

n_target scales with AOI area (see DEFAULT_POINT_DENSITY_PER_KM2): ~300 points
for a 3km x 3km AOI, proportionally more/fewer for a different-sized bbox.

Run from the backend/ directory so the app package is importable.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.core.geo import bbox_area_km2  # noqa: E402
from app.sampling.points import DEFAULT_POINT_DENSITY_PER_KM2, generate_sample_points  # noqa: E402

logger = logging.getLogger(__name__)

MIN_N_TARGET = 10


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = get_settings()
    bbox = settings.AOI_BBOX
    area_km2 = bbox_area_km2(bbox)
    n_target = max(MIN_N_TARGET, round(area_km2 * DEFAULT_POINT_DENSITY_PER_KM2))

    # Prefer the STEP-0-reduced connected graph (see scripts/build_graph.py) so
    # nearest_edge_key always points at an edge that actually exists in the
    # graph app.routing routes on -- falling back to the raw graph (with a
    # warning) if build_graph.py's connectivity step hasn't been run yet.
    connected_cache_path = Path(settings.CACHE_DIR) / "walk_graph_connected.graphml"
    raw_cache_path = Path(settings.CACHE_DIR) / "walk_graph.graphml"
    if connected_cache_path.exists():
        graph_cache_path = connected_cache_path
    else:
        graph_cache_path = raw_cache_path
        if raw_cache_path.exists():
            logger.warning(
                "no connected-graph cache at %s -- sampling from the raw graph instead. "
                "Run `python scripts/build_graph.py` to build it (STEP 0) for routing-consistent nearest_edge_key values.",
                connected_cache_path,
            )

    output_path = Path("../data/raw/sample_points.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df, path_used = generate_sample_points(bbox, n_target, graph_cache_path=graph_cache_path)
    df.to_parquet(output_path, index=False)

    print(f"sampling path used: {path_used}")
    print(f"points written: {len(df)} (target was {n_target} for a {area_km2:.2f} km2 AOI)")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
