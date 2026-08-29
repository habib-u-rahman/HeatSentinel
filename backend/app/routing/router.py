"""Heat-dose-aware pedestrian routing.

Routes on the CONNECTED walk graph only (see app.routing.graph
largest_weakly_connected_component / scripts/build_graph.py's STEP 0) --
never the raw one -- and always fails with a clear, specific exception
instead of letting networkx.NetworkXNoPath surface as a raw traceback.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import osmnx as ox

from app.core.geo import haversine
from app.grid.schema import TemperatureGrid
from app.heat.thresholds import classify
from app.ingest.openmeteo import OpenMeteoClient
from app.routing.costs import WALK_SPEED_MS, EdgeCostResult, build_edge_costs
from app.routing.graph import EdgeKey

logger = logging.getLogger(__name__)

DEFAULT_MAX_SNAP_DIST_M = 100.0
ROUTE_WORK_INTENSITY = "light"  # walking; see app.heat.thresholds for the band cut points


class SnapDistanceExceededError(Exception):
    """Raised when the nearest graph node is farther than max_snap_dist_m from the query point."""


class RouteNotFoundError(Exception):
    """Raised when no path exists between the snapped start/end nodes (e.g. one-way
    street topology, or -- pre-STEP-0 -- disconnected graph components)."""


@dataclass
class RouteResult:
    node_path: list[int]
    edge_path: list[EdgeKey]
    geojson: dict
    total_distance_m: float
    total_duration_s: float
    total_heat_dose_degC_s: float
    mean_wbgt_c: float
    max_wbgt_c: float
    peak_risk_band: str
    n_edges_orphaned: int
    data_source: str
    lambda_heat: float
    label: Optional[str] = field(default=None)


def snap_to_graph(lat: float, lon: float, graph: nx.MultiDiGraph, max_snap_dist_m: float = DEFAULT_MAX_SNAP_DIST_M) -> int:
    """Nearest graph node to (lat, lon), raising SnapDistanceExceededError if it's
    farther than max_snap_dist_m away (likely outside the walkable AOI)."""
    node_id = ox.distance.nearest_nodes(graph, X=lon, Y=lat)
    node = graph.nodes[node_id]
    dist_m = haversine((lat, lon), (node["y"], node["x"]))
    if dist_m > max_snap_dist_m:
        raise SnapDistanceExceededError(
            f"Nearest graph node to ({lat}, {lon}) is {dist_m:.0f}m away, exceeding "
            f"max_snap_dist_m={max_snap_dist_m:.0f}m. This point is likely outside the "
            "walkable AOI, or on a disconnected island dropped by STEP 0's connectivity reduction."
        )
    return node_id


def _weight_fn(costs: dict[EdgeKey, float]):
    def weight(u, v, data: dict) -> float:
        # `data` is the dict of {key: edge_attrs} for ALL parallel u->v edges
        # (networkx's MultiGraph weight-callable convention) -- pick the
        # cheapest, matching what _reconstruct_edge_path will select.
        return min(costs.get((u, v, k), math.inf) for k in data)

    return weight


def _reconstruct_edge_path(node_path: list[int], graph: nx.MultiDiGraph, costs: dict[EdgeKey, float]) -> list[EdgeKey]:
    edge_path: list[EdgeKey] = []
    for u, v in zip(node_path[:-1], node_path[1:]):
        parallel = graph[u][v]
        best_k = min(parallel, key=lambda k: costs.get((u, v, k), math.inf))
        edge_path.append((u, v, best_k))
    return edge_path


def _route_geojson(node_path: list[int], graph: nx.MultiDiGraph) -> dict:
    """A GeoJSON LineString Feature for node_path. GeoJSON coordinate order is
    [lon, lat] -- NOT [lat, lon] -- or Leaflet/geojson.io silently draw nonsense."""
    coordinates = [[graph.nodes[n]["x"], graph.nodes[n]["y"]] for n in node_path]
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "LineString", "coordinates": coordinates},
    }


# Bounded: grid.observed_at is effectively unique per request (fixture grids
# stamp it with datetime.now()) and lambda_heat is a continuous UI-slider
# value, so an unbounded cache here grows forever -- each entry holds a full
# per-edge cost array (~7-8k edges), making this the largest contributor to
# a confirmed gradual OOM on Render's free tier. FIFO eviction via dict
# insertion order once the cap is hit.
_EDGE_COST_CACHE_MAX = 50
_edge_cost_cache: dict[tuple, EdgeCostResult] = {}


def get_or_build_edge_costs(
    graph: nx.MultiDiGraph,
    grid: TemperatureGrid,
    lambda_heat: float,
    sample_points_path: Optional[Path] = None,
    surface_profiles_path: Optional[Path] = None,
    max_dist_m: float = 150.0,
    openmeteo_client: Optional[OpenMeteoClient] = None,
) -> EdgeCostResult:
    """build_edge_costs, cached by (grid timestamp, lambda_heat, bbox) so repeated
    queries at the same lambda/grid don't recompute the whole edge-cost vector."""
    cache_key = (grid.observed_at, round(lambda_heat, 6), grid.bbox)
    cached = _edge_cost_cache.get(cache_key)
    if cached is not None:
        logger.info("edge_cost_cache_hit key=%s", cache_key)
        return cached

    kwargs = {"max_dist_m": max_dist_m, "openmeteo_client": openmeteo_client}
    if sample_points_path is not None:
        kwargs["sample_points_path"] = sample_points_path
    if surface_profiles_path is not None:
        kwargs["surface_profiles_path"] = surface_profiles_path

    result = build_edge_costs(graph, grid, lambda_heat, **kwargs)
    if len(_edge_cost_cache) >= _EDGE_COST_CACHE_MAX:
        _edge_cost_cache.pop(next(iter(_edge_cost_cache)))
    _edge_cost_cache[cache_key] = result
    return result


def clear_edge_cost_cache() -> None:
    _edge_cost_cache.clear()


def route(
    start: tuple[float, float],
    end: tuple[float, float],
    lambda_heat: float,
    graph: nx.MultiDiGraph,
    grid: TemperatureGrid,
    max_snap_dist_m: float = DEFAULT_MAX_SNAP_DIST_M,
    **edge_cost_kwargs,
) -> RouteResult:
    """The heat-dose-aware route between start=(lat,lon) and end=(lat,lon)."""
    start_node = snap_to_graph(start[0], start[1], graph, max_snap_dist_m=max_snap_dist_m)
    end_node = snap_to_graph(end[0], end[1], graph, max_snap_dist_m=max_snap_dist_m)

    edge_cost_result = get_or_build_edge_costs(graph, grid, lambda_heat, **edge_cost_kwargs)

    try:
        node_path = nx.shortest_path(graph, start_node, end_node, weight=_weight_fn(edge_cost_result.costs))
    except nx.NetworkXNoPath as exc:
        raise RouteNotFoundError(
            f"No walkable path exists between {start} (node {start_node}) and {end} (node {end_node}). "
            "They may be in different, disconnected parts of the walk graph -- see scripts/build_graph.py's "
            "STEP 0 connectivity reduction, or this may be a one-way-street routing dead end."
        ) from exc

    edge_path = _reconstruct_edge_path(node_path, graph, edge_cost_result.costs)

    total_distance_m = sum(edge_cost_result.edge_length_m[key] for key in edge_path)
    total_duration_s = total_distance_m / WALK_SPEED_MS
    total_heat_dose = sum(edge_cost_result.edge_dose_degC_s[key] for key in edge_path)
    wbgt_values = [
        edge_cost_result.edge_wbgt_c[key]
        for key in edge_path
        if not math.isnan(edge_cost_result.edge_wbgt_c[key])
    ]
    mean_wbgt_c = total_heat_dose / total_duration_s if total_duration_s else 0.0
    max_wbgt_c = max(wbgt_values) if wbgt_values else mean_wbgt_c
    n_edges_orphaned = sum(1 for key in edge_path if edge_cost_result.edge_orphaned_penalty.get(key, False))

    return RouteResult(
        node_path=node_path,
        edge_path=edge_path,
        geojson=_route_geojson(node_path, graph),
        total_distance_m=total_distance_m,
        total_duration_s=total_duration_s,
        total_heat_dose_degC_s=total_heat_dose,
        mean_wbgt_c=mean_wbgt_c,
        max_wbgt_c=max_wbgt_c,
        peak_risk_band=classify(max_wbgt_c, work_intensity=ROUTE_WORK_INTENSITY),
        n_edges_orphaned=n_edges_orphaned,
        data_source=grid.source.value.lower(),
        lambda_heat=lambda_heat,
    )
