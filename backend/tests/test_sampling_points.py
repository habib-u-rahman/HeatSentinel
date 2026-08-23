from __future__ import annotations

import networkx as nx
import osmnx as ox
import pytest

from app.sampling.points import GRAPH_PATH, GRID_PATH, generate_sample_points

BBOX = "-74.010,40.710,-74.000,40.720"  # ~1km x 1.1km, small AOI for fast tests


def test_grid_fallback_used_when_no_cache_path(tmp_path):
    df, path_used = generate_sample_points(BBOX, n_target=20, graph_cache_path=None)

    assert path_used == GRID_PATH
    assert len(df) > 0
    assert list(df.columns) == ["point_id", "lat", "lon", "nearest_edge_key", "grid_cell_id"]
    assert df["nearest_edge_key"].isna().all()


def test_grid_fallback_used_when_cache_file_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.graphml"
    df, path_used = generate_sample_points(BBOX, n_target=20, graph_cache_path=missing_path)
    assert path_used == GRID_PATH
    assert len(df) > 0


def test_point_id_is_stable_hash_of_rounded_latlon():
    df, _ = generate_sample_points(BBOX, n_target=20, graph_cache_path=None)
    # Same (rounded) coordinates must always hash to the same point_id.
    row = df.iloc[0]
    from app.sampling.points import _point_id

    assert row["point_id"] == _point_id(row["lat"], row["lon"])


def test_grid_fallback_respects_n_target_upper_bound():
    df, _ = generate_sample_points(BBOX, n_target=5, graph_cache_path=None)
    assert len(df) <= 5


def test_generate_sample_points_rejects_non_positive_target():
    with pytest.raises(ValueError):
        generate_sample_points(BBOX, n_target=0, graph_cache_path=None)


def _make_synthetic_graph_cache(path) -> None:
    """A tiny synthetic walk graph spread across BBOX, saved as GraphML."""
    graph = nx.MultiDiGraph(crs="epsg:4326")
    nodes = [
        (1, -74.009, 40.711),
        (2, -74.007, 40.712),
        (3, -74.005, 40.714),
        (4, -74.003, 40.716),
        (5, -74.001, 40.718),
        (6, -74.008, 40.719),
    ]
    for node_id, lon, lat in nodes:
        graph.add_node(node_id, x=lon, y=lat)
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (1, 4)]
    for u, v in edges:
        graph.add_edge(u, v, key=0, length=50.0)

    ox.save_graphml(graph, path)


def test_graph_path_used_when_cache_exists(tmp_path):
    cache_path = tmp_path / "walk_graph.graphml"
    _make_synthetic_graph_cache(cache_path)

    df, path_used = generate_sample_points(BBOX, n_target=5, graph_cache_path=cache_path)

    assert path_used == GRAPH_PATH
    assert len(df) > 0
    assert len(df) <= 7  # can't exceed the number of edges in the synthetic graph
    assert df["nearest_edge_key"].notna().all()
    # nearest_edge_key is a "u_v_k" string, not a raw tuple (parquet-safe).
    assert all(isinstance(x, str) and x.count("_") == 2 for x in df["nearest_edge_key"])
