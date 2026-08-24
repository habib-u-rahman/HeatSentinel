from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pytest

from app.grid.schema import GridSource, TemperatureCell, TemperatureGrid, cell_id_for
from app.heat.wbgt import wbgt_shade
from app.routing.costs import _fetch_aoi_humidity, _vectorized_wbgt_shade, build_edge_costs, clear_aoi_humidity_cache
from app.routing.graph import _annotate_edges
from app.routing.pareto import compare, compute_route_family
from app.routing.router import (
    RouteNotFoundError,
    SnapDistanceExceededError,
    clear_edge_cost_cache,
    route,
    snap_to_graph,
)

BBOX = "-74.020,40.700,-73.995,40.726"
_NONEXISTENT = Path("/nonexistent/does-not-exist.parquet")
CONNECTED_GRAPH_PATH = Path("../data/cache/walk_graph_connected.graphml")


@pytest.fixture(autouse=True)
def _clear_route_cache():
    # get_or_build_edge_costs / _fetch_aoi_humidity cache by (grid.observed_at,
    # bbox, ...); tests build many small graphs/grids that could otherwise
    # collide on those keys.
    clear_edge_cost_cache()
    clear_aoi_humidity_cache()
    yield
    clear_edge_cost_cache()
    clear_aoi_humidity_cache()


class _StubOpenMeteoClient:
    def fetch_hourly(self, lat, lon, start, end):
        return pd.DataFrame(
            {
                "time": ["2026-07-15T14:00", "2026-07-15T15:00", "2026-07-15T16:00"],
                "temperature_2m": [33.0, 34.0, 33.5],
                "relative_humidity_2m": [55.0, 50.0, 52.0],
                "wind_speed_10m": [2.0, 2.5, 2.2],
                "shortwave_radiation": [700.0, 800.0, 750.0],
            }
        )


def _cost_kwargs(**overrides) -> dict:
    kwargs = {
        "sample_points_path": _NONEXISTENT,
        "surface_profiles_path": _NONEXISTENT,
        "openmeteo_client": _StubOpenMeteoClient(),
    }
    kwargs.update(overrides)
    return kwargs


def _make_grid(cells: list[tuple[float, float, float]], bbox: str = BBOX, observed_at=None) -> TemperatureGrid:
    return TemperatureGrid(
        cells=[TemperatureCell(lat=lat, lon=lon, temp_c=temp, cell_id=cell_id_for(lat, lon)) for lat, lon, temp in cells],
        bbox=bbox,
        granularity_m=100,
        observed_at=observed_at or datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc),
        source=GridSource.FIXTURE,
    )


def _divergent_scenario() -> tuple[nx.MultiDiGraph, TemperatureGrid]:
    """A direct, SHORT hot edge (1->2, 100m) vs. a longer, 3-edge COOL detour
    (1->3->4->2, 40m each = 120m total) -- deliberately different edge COUNTS
    on the two candidate paths, which is exactly what would expose a min-max
    (rather than max-only) normalisation bug.
    """
    nodes = {
        1: (40.700, -74.000),
        2: (40.700, -74.0015),
        3: (40.7025, -74.000),
        4: (40.705, -74.0015),
    }
    graph = nx.MultiDiGraph(crs="epsg:4326")
    for node_id, (lat, lon) in nodes.items():
        graph.add_node(node_id, x=lon, y=lat)

    def mid(a: int, b: int) -> tuple[float, float]:
        return (nodes[a][0] + nodes[b][0]) / 2, (nodes[a][1] + nodes[b][1]) / 2

    edges = [(1, 2, 100.0), (1, 3, 40.0), (3, 4, 40.0), (4, 2, 40.0)]
    for u, v, length_m in edges:
        mid_lat, mid_lon = mid(u, v)
        graph.add_edge(u, v, key=0, length_m=length_m, mid_lat=mid_lat, mid_lon=mid_lon)

    hot_lat, hot_lon = mid(1, 2)
    grid_cells = [
        (hot_lat, hot_lon, 45.0),
        (*mid(1, 3), 25.0),
        (*mid(3, 4), 25.0),
        (*mid(4, 2), 25.0),
    ]
    grid = _make_grid(grid_cells)
    return graph, grid


# --- costs / normalisation -----------------------------------------------------


def test_lambda_zero_reproduces_plain_shortest_path_by_length():
    graph, grid = _divergent_scenario()
    result = route((40.700, -74.000), (40.700, -74.0015), 0.0, graph, grid, **_cost_kwargs())

    plain_path = nx.shortest_path(graph, 1, 2, weight="length_m")
    assert result.node_path == plain_path
    assert result.node_path == [1, 2]  # the short, hot, direct edge


def test_lambda_one_never_has_higher_dose_than_lambda_zero():
    graph, grid = _divergent_scenario()
    r0 = route((40.700, -74.000), (40.700, -74.0015), 0.0, graph, grid, **_cost_kwargs())
    r1 = route((40.700, -74.000), (40.700, -74.0015), 1.0, graph, grid, **_cost_kwargs())

    assert r1.total_heat_dose_degC_s <= r0.total_heat_dose_degC_s
    # in this scenario the divergence is real: lambda=1 detours onto the cooler path
    assert r1.node_path == [1, 3, 4, 2]
    assert r0.node_path != r1.node_path


def test_normalisation_puts_both_terms_in_unit_interval():
    graph, grid = _divergent_scenario()
    result = build_edge_costs(graph, grid, lambda_heat=0.5, **_cost_kwargs())

    for key in result.costs:
        length_m = result.edge_length_m[key]
        dose = result.edge_dose_degC_s[key]
        norm_distance = length_m / result.dist_max_m
        norm_dose = dose / result.dose_max_degC_s
        assert 0.0 <= norm_distance <= 1.0
        assert 0.0 <= norm_dose <= 1.0
        assert 0.0 <= result.costs[key] <= 1.0


def test_orphaned_edges_never_get_zero_cost():
    graph, _ = _divergent_scenario()
    # a grid with cells FAR from every edge -- nothing will match within max_dist_m,
    # and there ARE grid cells so interpolate_idw succeeds (fills, doesn't hit the
    # 75th-percentile fallback) -- either way, cost must never be 0.
    far_grid = _make_grid([(41.5, -75.5, 30.0), (41.51, -75.51, 31.0), (41.49, -75.49, 29.0)])

    result = build_edge_costs(graph, far_grid, lambda_heat=0.5, max_dist_m=50.0, **_cost_kwargs())

    assert result.n_orphaned == 4  # all edges in this tiny graph
    assert result.n_orphaned_idw_filled == 4
    for key, cost in result.costs.items():
        assert cost > 0.0
        assert result.edge_dose_degC_s[key] > 0.0


def test_orphaned_percentile_fallback_when_grid_has_no_cells():
    graph, _ = _divergent_scenario()
    empty_grid = _make_grid([])

    result = build_edge_costs(graph, empty_grid, lambda_heat=0.5, **_cost_kwargs())

    assert result.n_orphaned == 4
    assert result.n_orphaned_percentile_filled == 4
    assert all(result.edge_orphaned_penalty.values())
    for cost in result.costs.values():
        assert cost > 0.0


def test_aoi_humidity_is_cached_and_consistent_despite_a_flaky_first_call():
    """Regression test: a route family calls build_edge_costs once per lambda.
    If the humidity fetch is flaky and only SOME of those calls fail over to
    DEFAULT_RH_PCT while others get a real (different) value, different
    lambdas would silently compute WBGT from different humidities for what
    should be the same grid snapshot -- breaking the dose-monotonicity
    guarantee between lambda=0 and lambda=1. Humidity must be fetched once
    and reused for every lambda in a sweep.
    """

    class _FlakyThenDifferentClient:
        def __init__(self):
            self.calls = 0

        def fetch_hourly(self, lat, lon, start, end):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("simulated transient network failure")
            return pd.DataFrame(
                {
                    "time": ["2026-07-15T15:00"],
                    "temperature_2m": [34.0],
                    "relative_humidity_2m": [90.0],  # deliberately very different from DEFAULT_RH_PCT
                    "wind_speed_10m": [2.0],
                    "shortwave_radiation": [800.0],
                }
            )

    client = _FlakyThenDifferentClient()
    observed_at = datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc)

    first = _fetch_aoi_humidity(BBOX, observed_at, client)  # fails -> DEFAULT_RH_PCT, cached
    second = _fetch_aoi_humidity(BBOX, observed_at, client)  # would succeed with 90.0 if not cached

    assert client.calls == 1  # second call never hit the network -- served from cache
    assert first == second


def test_vectorized_wbgt_shade_matches_scalar_implementation():
    for temp_c, rh_pct in [(25.0, 40.0), (33.0, 55.0), (40.0, 70.0)]:
        scalar = wbgt_shade(temp_c, rh_pct).value
        vectorized = float(_vectorized_wbgt_shade(np.array([temp_c]), rh_pct)[0])
        assert vectorized == pytest.approx(scalar, abs=1e-9)


# --- snapping / error handling --------------------------------------------------


def test_snap_beyond_max_distance_raises_clear_error():
    graph, _ = _divergent_scenario()
    with pytest.raises(SnapDistanceExceededError, match="exceeding max_snap_dist_m"):
        snap_to_graph(41.5, -75.5, graph, max_snap_dist_m=100.0)


def test_routing_between_disconnected_components_raises_clear_error():
    graph = nx.MultiDiGraph(crs="epsg:4326")
    # island A
    graph.add_node(1, x=-74.000, y=40.700)
    graph.add_node(2, x=-74.001, y=40.700)
    graph.add_edge(1, 2, key=0, length_m=80.0, mid_lat=40.700, mid_lon=-74.0005)
    # island B -- no edge connects it to island A
    graph.add_node(3, x=-74.010, y=40.710)
    graph.add_node(4, x=-74.011, y=40.710)
    graph.add_edge(3, 4, key=0, length_m=80.0, mid_lat=40.710, mid_lon=-74.0105)

    grid = _make_grid([(40.700, -74.0005, 30.0), (40.710, -74.0105, 30.0)])

    with pytest.raises(RouteNotFoundError, match="disconnected"):
        route((40.700, -74.000), (40.710, -74.010), 0.0, graph, grid, **_cost_kwargs())


# --- GeoJSON ---------------------------------------------------------------------


def test_route_geojson_is_valid_and_coordinate_order_is_lon_lat():
    graph, grid = _divergent_scenario()
    result = route((40.700, -74.000), (40.700, -74.0015), 0.0, graph, grid, **_cost_kwargs())

    geojson = result.geojson
    assert geojson["type"] == "Feature"
    assert geojson["geometry"]["type"] == "LineString"
    coords = geojson["geometry"]["coordinates"]
    assert len(coords) == len(result.node_path)

    # node 1 is at lat=40.700, lon=-74.000 -- GeoJSON order is [lon, lat]
    first_lon, first_lat = coords[0]
    assert first_lon == pytest.approx(-74.000)
    assert first_lat == pytest.approx(40.700)
    for lon, lat in coords:
        assert -75.0 < lon < -73.0  # sanity: definitely longitude, not latitude
        assert 39.0 < lat < 42.0


# --- pareto family / comparison --------------------------------------------------


def test_compute_route_family_labels_and_dedup():
    graph, grid = _divergent_scenario()
    family = compute_route_family((40.700, -74.000), (40.700, -74.0015), graph, grid, **_cost_kwargs())

    labels = {r.label for r in family}
    assert "SHORTEST" in labels
    assert "COOLEST" in labels
    node_paths = [tuple(r.node_path) for r in family]
    assert len(node_paths) == len(set(node_paths))  # deduplicated


def test_compare_reports_same_path_explicitly():
    graph, grid = _divergent_scenario()
    r = route((40.700, -74.000), (40.700, -74.0015), 0.0, graph, grid, **_cost_kwargs())
    comparison = compare(r, r)
    assert comparison.same_path is True
    assert "SAME path" in comparison.summary


def test_compare_reports_real_tradeoff():
    graph, grid = _divergent_scenario()
    shortest = route((40.700, -74.000), (40.700, -74.0015), 0.0, graph, grid, **_cost_kwargs())
    coolest = route((40.700, -74.000), (40.700, -74.0015), 1.0, graph, grid, **_cost_kwargs())

    comparison = compare(shortest, coolest)
    assert comparison.same_path is False
    assert comparison.extra_distance_m > 0
    assert comparison.dose_reduction_degC_s > 0
    assert "longer" in comparison.summary and "less heat exposure" in comparison.summary


# --- performance (real cached graph) --------------------------------------------


@pytest.mark.skipif(not CONNECTED_GRAPH_PATH.exists(), reason="real connected graph cache not built yet")
def test_route_on_real_graph_completes_under_500ms():
    # Deliberately AOI-agnostic: whatever city happens to be cached (the AOI is
    # config-driven and can change), derive the bbox/start/end from the graph's
    # OWN node coordinates rather than hardcoding a specific city's lat/lon --
    # a hardcoded pair silently breaks the moment someone points AOI_BBOX
    # somewhere else (exactly the class of bug fixed in build_walk_graph's new
    # cache bbox-validation).
    from app.fortyguard.fixtures import generate_grid

    graph = ox.load_graphml(CONNECTED_GRAPH_PATH)
    _annotate_edges(graph)

    lats = [data["y"] for _, data in graph.nodes(data=True)]
    lons = [data["x"] for _, data in graph.nodes(data=True)]
    graph_bbox = f"{min(lons)},{min(lats)},{max(lons)},{max(lats)}"

    nodes_sorted = sorted(graph.nodes(data=True), key=lambda item: (item[1]["y"], item[1]["x"]))
    start_data, end_data = nodes_sorted[0][1], nodes_sorted[-1][1]
    start, end = (start_data["y"], start_data["x"]), (end_data["y"], end_data["x"])

    grid = generate_grid(graph_bbox, 100, datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc), seed=0)
    kwargs = _cost_kwargs()

    route(start, end, 0.5, graph, grid, **kwargs)  # warm the edge-cost cache

    t0 = time.monotonic()
    route(start, end, 0.5, graph, grid, **kwargs)
    elapsed_s = time.monotonic() - t0

    assert elapsed_s < 0.5
