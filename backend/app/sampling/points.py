"""Spatially-stratified sample-point generation over an AOI.

Two paths, in preference order:
  - graph_edge_midpoints: sample walk-graph edge midpoints (needs the OSMnx
    GraphML cache built by scripts/build_graph.py), stratified over a uniform
    grid so points spread across the AOI instead of clustering on dense
    street segments.
  - regular_grid_fallback: a plain lat/lon grid, used when no graph cache
    exists yet.

Independent of the FortyGuard client -- nothing here imports app.fortyguard.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import networkx as nx
import osmnx as ox
import pandas as pd

from app.core.geo import BBox, _parse_bbox, bbox_area_km2, grid_points
from app.routing.graph import EdgeKey, _annotate_edges

logger = logging.getLogger(__name__)

# Reference density used by scripts/build_sample_points.py to scale n_target
# with AOI size: ~300 points over a 3km x 3km (9 km2) AOI.
DEFAULT_POINT_DENSITY_PER_KM2 = 300 / 9.0

GRAPH_PATH = "graph_edge_midpoints"
GRID_PATH = "regular_grid_fallback"

_RANDOM_SEED = 0  # fixed seed so a given graph/bbox/n_target is reproducible


def _point_id(lat: float, lon: float) -> str:
    """Stable id: sha256 of the lat/lon rounded to ~1m precision (5 dp)."""
    canonical = f"{round(lat, 5)},{round(lon, 5)}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _cell_id(lat: float, lon: float, bbox: tuple[float, float, float, float], grid_dim: int) -> str:
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_span = max(max_lon - min_lon, 1e-12)
    lat_span = max(max_lat - min_lat, 1e-12)
    col = min(grid_dim - 1, max(0, int((lon - min_lon) / lon_span * grid_dim)))
    row = min(grid_dim - 1, max(0, int((lat - min_lat) / lat_span * grid_dim)))
    return f"r{row}_c{col}"


def _make_point(lat: float, lon: float, nearest_edge_key: Optional[EdgeKey], grid_cell_id: str) -> dict:
    return {
        "point_id": _point_id(lat, lon),
        "lat": lat,
        "lon": lon,
        "nearest_edge_key": "_".join(str(x) for x in nearest_edge_key) if nearest_edge_key else None,
        "grid_cell_id": grid_cell_id,
    }


def _sample_from_graph(bbox: BBox, n_target: int, graph_cache_path: Path) -> list[dict]:
    graph: nx.MultiDiGraph = ox.load_graphml(graph_cache_path)
    _annotate_edges(graph)

    edges = list(graph.edges(keys=True, data=True))
    if not edges:
        raise ValueError(f"Graph at {graph_cache_path} has no edges to sample from")

    parsed_bbox = _parse_bbox(bbox)
    grid_dim = max(1, round(math.sqrt(n_target)))

    cell_edges: dict[str, list] = defaultdict(list)
    for u, v, k, data in edges:
        cell = _cell_id(data["mid_lat"], data["mid_lon"], parsed_bbox, grid_dim)
        cell_edges[cell].append((u, v, k, data))

    rng = random.Random(_RANDOM_SEED)
    for cell_list in cell_edges.values():
        rng.shuffle(cell_list)

    total_cells = len(cell_edges)
    k_per_cell = max(1, math.ceil(n_target / total_cells))

    points: list[dict] = []
    remaining: dict[str, list] = {}
    for cell, cell_list in cell_edges.items():
        chosen, remaining[cell] = cell_list[:k_per_cell], cell_list[k_per_cell:]
        for u, v, k, data in chosen:
            points.append(_make_point(data["mid_lat"], data["mid_lon"], (u, v, k), cell))

    # Fill any shortfall (cells with few/no edges pulled the average down) with
    # a round-robin second pass over cells that still have unused edges.
    while len(points) < n_target and any(remaining.values()):
        for cell, cell_list in remaining.items():
            if len(points) >= n_target:
                break
            if cell_list:
                u, v, k, data = cell_list.pop(0)
                points.append(_make_point(data["mid_lat"], data["mid_lon"], (u, v, k), cell))

    if len(points) > n_target:
        rng.shuffle(points)
        points = points[:n_target]

    return points


def _sample_from_grid(bbox: BBox, n_target: int) -> list[dict]:
    parsed_bbox = _parse_bbox(bbox)

    area_m2 = max(bbox_area_km2(bbox), 1e-6) * 1_000_000.0
    spacing_m = max(math.sqrt(area_m2 / n_target), 1.0)

    grid_dim = max(1, round(math.sqrt(n_target)))
    points = []
    for lat, lon in grid_points(bbox, spacing_m):
        cell = _cell_id(lat, lon, parsed_bbox, grid_dim)
        points.append(_make_point(lat, lon, None, cell))

    if len(points) > n_target:
        rng = random.Random(_RANDOM_SEED)
        rng.shuffle(points)
        points = points[:n_target]

    return points


def generate_sample_points(
    bbox: BBox,
    n_target: int,
    graph_cache_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, str]:
    """Generate up to n_target sample points spread across bbox.

    Returns (dataframe, path_used) where path_used is GRAPH_PATH or GRID_PATH.
    Columns: point_id, lat, lon, nearest_edge_key, grid_cell_id.
    """
    if n_target < 1:
        raise ValueError(f"n_target must be >= 1, got {n_target}")

    if graph_cache_path is not None and Path(graph_cache_path).exists():
        logger.info(
            "sample_points_path=%s cache=%s n_target=%d", GRAPH_PATH, graph_cache_path, n_target
        )
        points = _sample_from_graph(bbox, n_target, Path(graph_cache_path))
        path_used = GRAPH_PATH
    else:
        logger.warning(
            "sample_points_path=%s reason=no_graph_cache cache=%s n_target=%d",
            GRID_PATH,
            graph_cache_path,
            n_target,
        )
        points = _sample_from_grid(bbox, n_target)
        path_used = GRID_PATH

    df = pd.DataFrame(points, columns=["point_id", "lat", "lon", "nearest_edge_key", "grid_cell_id"])
    return df, path_used
