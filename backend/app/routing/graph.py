"""City walk-network graph: build (or load from cache) and spatially index edges.

Independent of the FortyGuard client — nothing here imports app.fortyguard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np
import osmnx as ox
from scipy.spatial import cKDTree

from app.core.geo import BBox, _parse_bbox

logger = logging.getLogger(__name__)

EdgeKey = tuple[int, int, int]


def build_walk_graph(bbox: BBox, cache_path: str | Path) -> nx.MultiDiGraph:
    """Load the walk network for bbox from a GraphML cache, or download + cache it.

    Never re-downloads if cache_path already exists AND was built for this
    exact bbox. A cache hit for a DIFFERENT bbox (e.g. AOI_BBOX changed in
    .env since cache_path was last written) is treated as a miss and
    rebuilt -- silently routing/analysing the wrong city because a stale
    file happened to sit at the expected path would be far worse than a
    slower rebuild.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    requested_bbox = _parse_bbox(bbox)

    graph = None
    if cache_path.exists():
        candidate = ox.load_graphml(cache_path)
        cached_bbox_str = candidate.graph.get("aoi_bbox")
        cached_bbox = tuple(float(x) for x in cached_bbox_str.split(",")) if cached_bbox_str else None

        if cached_bbox is not None and all(abs(a - b) < 1e-6 for a, b in zip(cached_bbox, requested_bbox)):
            logger.info("graph_cache_hit path=%s bbox=%s", cache_path, requested_bbox)
            graph = candidate
        else:
            logger.warning(
                "graph_cache_bbox_mismatch path=%s cached_bbox=%s requested_bbox=%s -- "
                "treating as a cache miss and rebuilding (a stale graph for the wrong "
                "AOI would silently corrupt everything downstream)",
                cache_path,
                cached_bbox,
                requested_bbox,
            )

    if graph is None:
        west, south, east, north = requested_bbox
        logger.info(
            "graph_cache_miss path=%s bbox=(%.6f,%.6f,%.6f,%.6f) downloading_from_osm=True",
            cache_path,
            west,
            south,
            east,
            north,
        )
        graph = ox.graph_from_bbox((west, south, east, north), network_type="walk", simplify=True)
        graph.graph["aoi_bbox"] = ",".join(f"{v:.6f}" for v in requested_bbox)
        ox.save_graphml(graph, cache_path)

    _annotate_edges(graph)
    return graph


def largest_weakly_connected_component(graph: nx.MultiDiGraph) -> tuple[nx.MultiDiGraph, int]:
    """Reduce graph to its largest weakly-connected component.

    Real street networks often have disconnected islands (piers, closed-off
    paths, service roads with no walk connection to the mainland) -- routing
    across them raises networkx.NetworkXNoPath. Surfacing that HERE, loudly
    and with a count, is much better than a raw traceback mid-demo.
    """
    components = list(nx.weakly_connected_components(graph))
    largest = max(components, key=len)
    n_dropped = graph.number_of_nodes() - len(largest)

    reduced = graph.subgraph(largest).copy()
    logger.info(
        "graph_connectivity n_components=%d largest_component_nodes=%d dropped_nodes=%d",
        len(components),
        len(largest),
        n_dropped,
    )
    return reduced, n_dropped


def _annotate_edges(graph: nx.MultiDiGraph) -> None:
    """Add length_m and mid_lat/mid_lon to every edge in place.

    Uses the edge's simplified geometry (a LineString) when present for an
    accurate midpoint along curved roads; falls back to averaging the two
    endpoint nodes for straight edges.
    """
    for u, v, data in graph.edges(data=True):
        data["length_m"] = float(data.get("length", 0.0))

        geometry = data.get("geometry")
        if geometry is not None:
            mid_point = geometry.interpolate(0.5, normalized=True)
            mid_lon, mid_lat = mid_point.x, mid_point.y
        else:
            u_node, v_node = graph.nodes[u], graph.nodes[v]
            mid_lon = (u_node["x"] + v_node["x"]) / 2
            mid_lat = (u_node["y"] + v_node["y"]) / 2

        data["mid_lat"] = mid_lat
        data["mid_lon"] = mid_lon


@dataclass
class EdgeSpatialIndex:
    """cKDTree over edge midpoints, for fast nearest-edge lookups."""

    edge_keys: list[EdgeKey]
    tree: cKDTree
    lon_scale: float

    def nearest_edges(self, points: Sequence[tuple[float, float]]) -> list[EdgeKey]:
        """Return the nearest edge key (u, v, k) for each (lat, lon) point."""
        query = np.array([(lat, lon * self.lon_scale) for lat, lon in points])
        _, indices = self.tree.query(query)
        indices = np.atleast_1d(indices)
        return [self.edge_keys[i] for i in indices]


def build_edge_index(graph: nx.MultiDiGraph) -> EdgeSpatialIndex:
    """Build a cKDTree over every edge's midpoint (graph must already be annotated)."""
    edge_keys: list[EdgeKey] = []
    midpoints: list[tuple[float, float]] = []

    for u, v, k, data in graph.edges(keys=True, data=True):
        edge_keys.append((u, v, k))
        midpoints.append((data["mid_lat"], data["mid_lon"]))

    lats = np.array([p[0] for p in midpoints])
    mean_lat_rad = np.radians(lats.mean()) if len(lats) else 0.0
    lon_scale = max(np.cos(mean_lat_rad), 1e-6)

    points = np.array([(lat, lon * lon_scale) for lat, lon in midpoints])
    tree = cKDTree(points)

    return EdgeSpatialIndex(edge_keys=edge_keys, tree=tree, lon_scale=lon_scale)
