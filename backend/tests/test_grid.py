from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from app.config import get_settings
from app.grid.adapter import SchemaMismatchError, parse_heatmap_response
from app.grid.join import attach_temps_to_edges, attach_temps_to_points, interpolate_idw
from app.grid.schema import GridSource, TemperatureCell, TemperatureGrid, cell_id_for
from app.grid.store import MixedSourceError, load_grid, load_series, save_grid

FIXTURE_JSON_PATH = Path(__file__).parent / "fixtures" / "fortyguard_heatmap_sample.json"
BBOX = "-74.020,40.700,-73.995,40.726"


def _make_grid(
    cells_latlon_temp: list[tuple[float, float, float]],
    bbox: str = BBOX,
    granularity_m: int = 100,
    source: GridSource = GridSource.FIXTURE,
    observed_at: datetime | None = None,
) -> TemperatureGrid:
    cells = [
        TemperatureCell(lat=lat, lon=lon, temp_c=temp, cell_id=cell_id_for(lat, lon))
        for lat, lon, temp in cells_latlon_temp
    ]
    return TemperatureGrid(
        cells=cells,
        bbox=bbox,
        granularity_m=granularity_m,
        observed_at=observed_at or datetime.now(timezone.utc),
        source=source,
    )


def _build_graph(edges: list[tuple[int, int, int, float, float]]) -> nx.MultiDiGraph:
    """edges: (u, v, k, mid_lat, mid_lon) -- pre-annotated, as app.routing.graph would leave it."""
    graph = nx.MultiDiGraph()
    for u, v, k, mid_lat, mid_lon in edges:
        graph.add_edge(u, v, key=k, mid_lat=mid_lat, mid_lon=mid_lon, length_m=10.0)
    return graph


@pytest.fixture
def _settings_env(monkeypatch):
    """Minimal required settings so get_settings() succeeds (autouse-style, but scoped
    to tests that actually need it -- store.load_grid only calls get_settings() for
    FIXTURE-sourced grids)."""
    monkeypatch.setenv("FORTYGUARD_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("CITY_NAME", "TestCity")
    monkeypatch.setenv("AOI_BBOX", "0,0,1,1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# --- adapter -----------------------------------------------------------------


def test_adapter_parses_fixture_json_correctly():
    raw = json.loads(FIXTURE_JSON_PATH.read_text(encoding="utf-8"))
    grid = parse_heatmap_response(
        raw,
        bbox=BBOX,
        granularity_m=100,
        observed_at=datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc),
        api_activity_id="act_9f31c2",
    )

    assert grid.source == GridSource.LIVE
    assert len(grid.cells) == 8
    assert grid.api_activity_id == "act_9f31c2"
    stats = grid.stats()
    assert stats["min"] == pytest.approx(30.5)
    assert stats["max"] == pytest.approx(36.0)
    # every temp landed in a physically sane range for the mixed-key-spelling records
    lats, lons, temps = grid.to_arrays()
    assert np.all((lats > 40) & (lats < 41))
    assert np.all(temps > 0)


def test_adapter_raises_schema_mismatch_with_context():
    bad_payload = {"status": "ok", "unexpected": {"foo": "bar"}}
    with pytest.raises(SchemaMismatchError) as exc_info:
        parse_heatmap_response(
            bad_payload, bbox=BBOX, granularity_m=100, observed_at=datetime.now(timezone.utc)
        )

    err = exc_info.value
    assert err.top_level_keys == ["status", "unexpected"]
    assert "unexpected" in err.payload_preview
    assert "foo" in err.payload_preview


def test_adapter_raises_on_records_missing_temperature():
    bad_payload = {"data": [{"lat": 40.71, "lon": -74.0}, {"lat": 40.72, "lon": -74.01}]}
    with pytest.raises(SchemaMismatchError):
        parse_heatmap_response(bad_payload, bbox=BBOX, granularity_m=100, observed_at=datetime.now(timezone.utc))


# --- join ---------------------------------------------------------------------


def test_attach_temps_to_edges_matches_hand_computed_nearest():
    grid = _make_grid(
        [
            (40.7100, -74.0100, 30.0),
            (40.7200, -74.0050, 32.0),
            (40.7300, -74.0000, 34.0),
        ]
    )
    graph = _build_graph(
        [
            (1, 2, 0, 40.7101, -74.0101),  # a few metres from cell 0 (30.0)
            (3, 4, 0, 40.7299, -73.9999),  # a few metres from cell 2 (34.0)
        ]
    )

    result = attach_temps_to_edges(graph, grid, max_dist_m=150)

    assert result[(1, 2, 0)] == pytest.approx(30.0)
    assert result[(3, 4, 0)] == pytest.approx(34.0)


def test_attach_temps_to_edges_orphans_beyond_max_dist_are_none():
    grid = _make_grid([(40.7100, -74.0100, 30.0)])
    graph = _build_graph(
        [
            (1, 2, 0, 40.7101, -74.0101),  # close
            (5, 6, 0, 41.5000, -75.5000),  # ~100+ km away -- must be orphaned
        ]
    )

    result = attach_temps_to_edges(graph, grid, max_dist_m=150)

    assert result[(1, 2, 0)] == pytest.approx(30.0)
    assert result[(5, 6, 0)] is None  # not silently 0.0


def test_attach_temps_to_points_orphans_beyond_max_dist():
    import pandas as pd

    grid = _make_grid([(40.7100, -74.0100, 30.0)])
    points_df = pd.DataFrame(
        {
            "point_id": ["near", "far"],
            "lat": [40.7101, 41.5000],
            "lon": [-74.0101, -75.5000],
        }
    )

    result = attach_temps_to_points(points_df, grid, max_dist_m=150)

    assert result.loc[result["point_id"] == "near", "temp_c"].iloc[0] == pytest.approx(30.0)
    assert np.isnan(result.loc[result["point_id"] == "far", "temp_c"].iloc[0])


def test_interpolate_idw_falls_between_k_neighbor_bounds():
    rng = np.random.default_rng(0)
    n = 30
    lats = 40.71 + rng.uniform(-0.01, 0.01, size=n)
    lons = -74.00 + rng.uniform(-0.01, 0.01, size=n)
    temps = rng.uniform(28, 36, size=n)
    grid = _make_grid(list(zip(lats.tolist(), lons.tolist(), temps.tolist())))

    target_lat, target_lon = 40.712, -74.003
    k = 4
    value = interpolate_idw(target_lat, target_lon, grid, k=k, power=2)

    dists = np.hypot(lats - target_lat, lons - target_lon)
    neighbor_temps = temps[np.argsort(dists)[:k]]

    assert neighbor_temps.min() - 1e-9 <= value <= neighbor_temps.max() + 1e-9


def test_edge_join_performance_5000_edges():
    rng = np.random.default_rng(1)
    n_edges = 5000
    lats = 40.71 + rng.uniform(-0.02, 0.02, size=n_edges)
    lons = -74.00 + rng.uniform(-0.02, 0.02, size=n_edges)
    graph = _build_graph([(i, i + 1, 0, float(lat), float(lon)) for i, (lat, lon) in enumerate(zip(lats, lons))])

    n_cells = 400
    grid_lats = 40.71 + rng.uniform(-0.02, 0.02, size=n_cells)
    grid_lons = -74.00 + rng.uniform(-0.02, 0.02, size=n_cells)
    grid_temps = rng.uniform(28, 36, size=n_cells)
    grid = _make_grid(list(zip(grid_lats.tolist(), grid_lons.tolist(), grid_temps.tolist())))

    start = time.monotonic()
    result = attach_temps_to_edges(graph, grid, max_dist_m=150)
    elapsed = time.monotonic() - start

    assert len(result) == n_edges
    assert elapsed < 2.0


# --- store ---------------------------------------------------------------------


def test_store_round_trips_grid_without_loss_and_preserves_source(tmp_path):
    grid = _make_grid(
        [(40.71, -74.01, 30.0), (40.72, -74.00, 31.5)],
        source=GridSource.LIVE,
        observed_at=datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc),
    )

    path = save_grid(grid, base_dir=tmp_path)
    assert path.exists()

    loaded = load_grid(path)

    assert loaded.source == GridSource.LIVE
    assert loaded.bbox == grid.bbox
    assert loaded.granularity_m == grid.granularity_m
    assert loaded.observed_at == grid.observed_at
    assert len(loaded.cells) == len(grid.cells)
    for original, roundtripped in zip(grid.cells, loaded.cells):
        assert roundtripped.lat == pytest.approx(original.lat)
        assert roundtripped.lon == pytest.approx(original.lon)
        assert roundtripped.temp_c == pytest.approx(original.temp_c)
        assert roundtripped.cell_id == original.cell_id


def test_load_fixture_grid_warns_and_respects_allow_fixture_flag(tmp_path, monkeypatch, caplog, _settings_env):
    grid = _make_grid([(40.71, -74.01, 30.0)], source=GridSource.FIXTURE)
    path = save_grid(grid, base_dir=tmp_path)

    monkeypatch.setenv("ALLOW_FIXTURE_DATA", "True")
    get_settings.cache_clear()
    with caplog.at_level(logging.WARNING):
        loaded = load_grid(path)
    assert loaded.source == GridSource.FIXTURE
    assert any("FIXTURE" in record.message for record in caplog.records)

    monkeypatch.setenv("ALLOW_FIXTURE_DATA", "False")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError):
        load_grid(path)


def test_load_series_raises_on_mixed_sources(tmp_path, _settings_env):
    live_grid = _make_grid(
        [(40.71, -74.01, 30.0)],
        source=GridSource.LIVE,
        observed_at=datetime(2026, 7, 15, 10, 0, tzinfo=timezone.utc),
    )
    fixture_grid = _make_grid(
        [(40.71, -74.01, 31.0)],
        source=GridSource.FIXTURE,
        observed_at=datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    )
    save_grid(live_grid, base_dir=tmp_path)
    save_grid(fixture_grid, base_dir=tmp_path)

    with pytest.raises(MixedSourceError):
        load_series(
            BBOX,
            datetime(2026, 7, 15, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 7, 15, 23, 59, tzinfo=timezone.utc),
            base_dir=tmp_path,
        )
