"""On-demand AOI build: given a bbox + city name, build everything a live AOI
needs by reusing the exact same functions the offline scripts/*.py CLIs use --
no new CV code, no new sampling/graph/POI logic.

Deliberately NOT included: a "build grid" step. app.api.deps.get_grid_for_timestamp
already calls app.fortyguard.fixtures.generate_grid(bbox, ...) fresh, per-request,
for ANY bbox whenever no stored LIVE grid exists at that timestamp (which is
always true today -- there's no live FortyGuard feed yet). The synthetic WBGT
grid needs zero precompute work here.

Also NOT included: retraining the RF intervention model. The pretrained bundle
(app.ml.intervention) scores generic surface-composition features, not
Rawalpindi-specific ones -- it's reused for inference on a new city with zero
retraining, exactly like a normal ML deployment. Retraining per-city is out of
scope: there's no ground-truth temp_c to train against for a freshly built
city, and app.ml.train_rf requires >=2 spatial blocks anyway.

This module is pure disk I/O + reused pipeline calls -- it never touches
app.state. app/api/routes/aoi.py does the state swap once this returns.

Run manually for a quick smoke test:
    python -c "from app.pipeline.build_aoi import run_aoi_build; \
        run_aoi_build('73.03,33.58,73.07,33.61', 'Test City', 10, Path('../data/aois/test'))"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from app.ingest.mapillary import MapillaryClient
from app.routing.graph import build_walk_graph, largest_weakly_connected_component
from app.sampling.points import generate_sample_points
from app.vision import segmenter
from app.vision.pipeline import analyze_image
from app.vulnerable.poi import fetch_pois

logger = logging.getLogger(__name__)

GRAPH_FILENAME = "walk_graph.graphml"
CONNECTED_GRAPH_FILENAME = "walk_graph_connected.graphml"
MAPILLARY_SEARCH_RADIUS_M = 30.0

# Images/overlays are written into the SAME directories main.py already mounts
# as /static/images and /static/overlays -- not a per-city subfolder -- so a
# freshly built city's photos are servable without adding new static mounts.
# Filenames are namespaced by point_id (a lat/lon-derived hash) + Mapillary
# image_id, so collisions across different cities are not a practical concern.
STATIC_IMAGES_DIR = Path("../data/street_images")
STATIC_OVERLAYS_DIR = Path("../data/overlays")


def slugify(city_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", city_name.lower()).strip("-")
    return slug or "city"


@dataclass
class AoiBuildResult:
    bbox: str
    city_name: str
    n_sample_points: int
    n_with_imagery: int
    mapillary_coverage_pct: float
    n_pois: int
    aoi_dir: Path
    graph_path: Path
    sample_points_path: Path
    surface_profiles_path: Path
    vulnerable_pois_path: Path


# (stage, message, current, total) -- called after every meaningfully-sized
# step so a polling job status always has a real, non-fabricated number to show.
ProgressFn = Callable[[str, str, int, int], None]


def _noop_progress(stage: str, message: str, current: int, total: int) -> None:
    return None


class NoImageryError(Exception):
    """Raised when zero sample points got a real Mapillary photo -- there is
    nothing honest to build a surface-composition profile from."""


def run_aoi_build(
    bbox: str,
    city_name: str,
    n_points: int,
    aoi_dir: Path,
    on_progress: Optional[ProgressFn] = None,
) -> AoiBuildResult:
    on_progress = on_progress or _noop_progress
    aoi_dir = Path(aoi_dir)
    cache_dir = aoi_dir / "cache"
    raw_dir = aoi_dir / "raw"
    for d in (cache_dir, raw_dir, STATIC_IMAGES_DIR, STATIC_OVERLAYS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # --- 1. walk network -------------------------------------------------------
    on_progress("graph", f"Downloading the street network for {city_name}…", 0, 1)
    graph_path = cache_dir / GRAPH_FILENAME
    graph = build_walk_graph(bbox, graph_path)

    connected_graph, n_dropped = largest_weakly_connected_component(graph)
    connected_path = cache_dir / CONNECTED_GRAPH_FILENAME
    import osmnx as ox

    ox.save_graphml(connected_graph, connected_path)
    on_progress(
        "graph",
        f"Street network ready ({connected_graph.number_of_nodes()} intersections, "
        f"{n_dropped} disconnected nodes dropped)",
        1,
        1,
    )

    # --- 2. sample points --------------------------------------------------------
    on_progress("sampling", f"Choosing {n_points} sample locations…", 0, 1)
    sample_points_df, _path_used = generate_sample_points(bbox, n_points, graph_cache_path=connected_path)
    n_total = len(sample_points_df)
    on_progress("sampling", f"{n_total} sample locations chosen", 1, 1)

    sample_points_path = raw_dir / "sample_points.parquet"
    sample_points_df.to_parquet(sample_points_path, index=False)

    # --- 3. real street imagery + CV analysis ------------------------------------
    manifest_rows: list[dict] = []
    profile_rows: list[dict] = []
    points_without_imagery: list[str] = []
    n_with_imagery = 0

    with MapillaryClient() as client:
        for i, row in enumerate(sample_points_df.itertuples(index=False), start=1):
            point_id, lat, lon = row.point_id, float(row.lat), float(row.lon)
            match = client.find_image_near(lat, lon, radius_m=MAPILLARY_SEARCH_RADIUS_M)
            if match is None:
                points_without_imagery.append(point_id)
                on_progress("imagery", f"Analyzed {n_with_imagery}/{n_total} street photos", i, n_total)
                continue

            filename = f"{point_id}_{match.image_id}.jpg"
            image_path = STATIC_IMAGES_DIR / filename
            client.download_image(match.image_id, image_path)
            manifest_rows.append(
                {
                    "filename": filename,
                    "lat": match.lat,
                    "lon": match.lon,
                    "point_id": point_id,
                    "image_id": match.image_id,
                    "captured_at": match.captured_at.isoformat(),
                }
            )

            profile = analyze_image(image_path, match.lat, match.lon)
            profile_row = profile.model_dump()
            profile_row["point_id"] = point_id
            profile_rows.append(profile_row)

            overlay_dest = STATIC_OVERLAYS_DIR / f"{image_path.stem}_overlay.png"
            segmenter.save_overlay(image_path, overlay_dest)

            n_with_imagery += 1
            on_progress("imagery", f"Analyzed {n_with_imagery}/{n_total} street photos", i, n_total)

    if n_with_imagery == 0:
        raise NoImageryError(
            f"No Mapillary street-level imagery found within this area for {city_name} "
            f"({n_total} sample points checked). Try a larger radius or a different city."
        )

    # manifest.csv lives under this city's own raw/ dir (not the shared images
    # dir, which images from every previously-built city also accumulate in) --
    # purely so scripts/analyze_images.py can be re-run offline against this
    # city's photos later for debugging, with --images-dir pointed at the
    # shared STATIC_IMAGES_DIR and --manifest pointed here.
    pd.DataFrame(manifest_rows).to_csv(raw_dir / "manifest.csv", index=False)
    if points_without_imagery:
        pd.DataFrame({"point_id": points_without_imagery}).to_csv(raw_dir / "points_without_imagery.csv", index=False)

    surface_profiles_path = raw_dir / "surface_profiles.parquet"
    pd.DataFrame(profile_rows).to_parquet(surface_profiles_path, index=False)

    coverage_pct = (n_with_imagery / n_total * 100.0) if n_total else 0.0

    # --- 4. POIs -------------------------------------------------------------------
    on_progress("pois", f"Finding points of interest in {city_name}…", 0, 1)
    vulnerable_pois_path = raw_dir / "vulnerable_pois.parquet"
    pois_df = fetch_pois(bbox, vulnerable_pois_path, force_refresh=True)
    on_progress("pois", f"{len(pois_df)} points of interest found", 1, 1)

    return AoiBuildResult(
        bbox=bbox,
        city_name=city_name,
        n_sample_points=n_total,
        n_with_imagery=n_with_imagery,
        mapillary_coverage_pct=coverage_pct,
        n_pois=len(pois_df),
        aoi_dir=aoi_dir,
        graph_path=connected_path,
        sample_points_path=sample_points_path,
        surface_profiles_path=surface_profiles_path,
        vulnerable_pois_path=vulnerable_pois_path,
    )
