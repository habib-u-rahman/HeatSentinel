from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from shapely.geometry import Point

from app.alerts.engine import clear_dedup_state, evaluate
from app.grid.schema import GridSource, TemperatureCell, TemperatureGrid, cell_id_for
from app.main import app, lifespan
from app.vulnerable.poi import fetch_pois
from app.vulnerable.scoring import attach_risk_to_pois

BBOX = "73.03,33.58,73.07,33.61"
EMPTY_POIS = pd.DataFrame(columns=["poi_id", "name", "category", "vulnerability_group", "weight", "lat", "lon"])


def _make_grid(cells: list[tuple[float, float, float]], bbox: str = BBOX, observed_at=None) -> TemperatureGrid:
    return TemperatureGrid(
        cells=[TemperatureCell(lat=lat, lon=lon, temp_c=temp, cell_id=cell_id_for(lat, lon)) for lat, lon, temp in cells],
        bbox=bbox,
        granularity_m=100,
        observed_at=observed_at or datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc),
        source=GridSource.FIXTURE,
    )


@pytest.fixture(autouse=True)
def _clear_dedup():
    clear_dedup_state()
    yield
    clear_dedup_state()


# --- POI fetch / caching ---------------------------------------------------------


def test_fetch_pois_is_cached_and_never_hits_overpass_twice(tmp_path, monkeypatch):
    call_count = {"n": 0}

    def fake_features_from_bbox(bbox, tags):
        call_count["n"] += 1
        return gpd.GeoDataFrame(
            {"amenity": ["school"], "name": ["Test School"], "geometry": [Point(73.05, 33.59)]},
            index=pd.MultiIndex.from_tuples([("node", 1)], names=["element_type", "osmid"]),
        )

    import osmnx as ox

    monkeypatch.setattr(ox, "features_from_bbox", fake_features_from_bbox)

    output_path = tmp_path / "pois.parquet"
    df1 = fetch_pois(BBOX, output_path=output_path)
    assert call_count["n"] == 1
    assert len(df1) == 1
    assert df1.iloc[0]["category"] == "school"
    assert df1.iloc[0]["vulnerability_group"] == "children"

    df2 = fetch_pois(BBOX, output_path=output_path)
    assert call_count["n"] == 1  # cache hit -- no second Overpass call
    pd.testing.assert_frame_equal(df1, df2)


# --- exposure scoring -------------------------------------------------------------


def test_exposure_score_rises_with_weight_and_wbgt():
    grid_cool = _make_grid([(33.595, 73.048, 26.0)])
    grid_hot = _make_grid([(33.595, 73.048, 42.0)])

    pois_df = pd.DataFrame(
        {
            "poi_id": ["p_bus", "p_hospital"],
            "name": [None, None],
            "category": ["bus_stop", "hospital"],
            "vulnerability_group": ["waiting_exposure", "patients"],
            "weight": [0.5, 1.0],
            "lat": [33.595, 33.595],
            "lon": [73.048, 73.048],
        }
    )

    cool_scored = attach_risk_to_pois(pois_df, grid_cool, rh_pct=40.0)
    hot_scored = attach_risk_to_pois(pois_df, grid_hot, rh_pct=40.0)

    cool_bus = cool_scored.set_index("poi_id").loc["p_bus", "exposure_score"]
    hot_bus = hot_scored.set_index("poi_id").loc["p_bus", "exposure_score"]
    assert hot_bus > cool_bus  # same POI, hotter grid -> more exposure

    hot_hospital = hot_scored.set_index("poi_id").loc["p_hospital", "exposure_score"]
    assert hot_hospital > hot_bus  # same grid, higher weight -> more exposure


# --- alert evaluation --------------------------------------------------------------


def test_deduplication_suppresses_repeats():
    grid = _make_grid([(33.582, 73.033, 45.0), (33.6, 73.05, 45.0)])

    first = evaluate(grid, EMPTY_POIS, BBOX, rh_pct=40.0)
    second = evaluate(grid, EMPTY_POIS, BBOX, rh_pct=40.0)

    assert len(first) > 0
    assert len(second) == 0  # same conditions, still within the dedup window


def test_rapid_rise_skips_cleanly_without_prior_grid():
    grid = _make_grid([(33.59, 73.05, 45.0)])
    alerts = evaluate(grid, EMPTY_POIS, BBOX, rh_pct=40.0, previous_grid=None)
    assert all(a.category != "rapid_rise" for a in alerts)


def test_rapid_rise_fires_with_a_prior_grid():
    prev = _make_grid([(33.59, 73.05, 25.0)], observed_at=datetime(2026, 7, 15, 13, 0, tzinfo=timezone.utc))
    curr = _make_grid([(33.59, 73.05, 35.0)], observed_at=datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc))
    alerts = evaluate(curr, EMPTY_POIS, BBOX, rh_pct=40.0, previous_grid=prev)
    assert any(a.category == "rapid_rise" for a in alerts)


def test_alerts_carry_data_source_and_proxy_disclosure():
    grid = _make_grid([(33.59, 73.05, 45.0)])
    alerts = evaluate(grid, EMPTY_POIS, BBOX, rh_pct=40.0)
    assert len(alerts) > 0
    for alert in alerts:
        assert alert.data_source == "fixture"
        assert alert.population_proxy is True
        assert "proxy" in alert.proxy_note.lower() or "population" in alert.proxy_note.lower()


def test_poi_at_risk_groups_by_zone_and_category():
    grid = _make_grid([(33.595, 73.048, 45.0)])
    pois_df = pd.DataFrame(
        {
            "poi_id": ["s1", "s2", "s3"],
            "name": [None, None, None],
            "category": ["school", "school", "hospital"],
            "vulnerability_group": ["children", "children", "patients"],
            "weight": [0.8, 0.8, 1.0],
            "lat": [33.595, 33.5951, 33.5952],
            "lon": [73.048, 73.0481, 73.0482],
        }
    )
    alerts = evaluate(grid, pois_df, BBOX, rh_pct=40.0)
    poi_alerts = [a for a in alerts if a.category == "poi_at_risk"]
    assert any("2 school" in a.message for a in poi_alerts)
    assert any("hospital" in a.message for a in poi_alerts)


# --- API: envelope + static file 404 -----------------------------------------------


@pytest_asyncio.fixture
async def client():
    async with lifespan(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac


@pytest.mark.asyncio
async def test_vulnerable_endpoint_returns_valid_geojson_with_proxy_disclosure(client: AsyncClient):
    response = await client.get("/api/vulnerable")
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "FeatureCollection"
    assert body["population_proxy"] is True
    assert "proxy" in body["proxy_note"].lower() or "population" in body["proxy_note"].lower()
    assert "data_source" in body and "observed_at" in body


@pytest.mark.asyncio
async def test_alerts_endpoint_returns_ranked_alerts(client: AsyncClient):
    response = await client.get("/api/alerts")
    assert response.status_code == 200
    body = response.json()
    assert "data_source" in body and "observed_at" in body
    severities = [a["severity"] for a in body["alerts"]]
    band_rank = {"SAFE": 0, "CAUTION": 1, "DANGER": 2, "CRITICAL": 3}
    ranks = [band_rank.get(s, 0) for s in severities]
    assert ranks == sorted(ranks, reverse=True)


@pytest.mark.asyncio
async def test_missing_static_image_404s_cleanly(client: AsyncClient):
    response = await client.get("/static/images/this-file-does-not-exist.jpg")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_missing_static_overlay_404s_cleanly(client: AsyncClient):
    response = await client.get("/static/overlays/this-file-does-not-exist_overlay.png")
    assert response.status_code == 404
