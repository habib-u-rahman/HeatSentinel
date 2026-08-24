from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config import get_settings
from app.main import app, lifespan

SAMPLE_POINTS_PATH = "../data/raw/sample_points.parquet"
SURFACE_PROFILES_PATH = "../data/raw/surface_profiles.parquet"

pytestmark = pytest.mark.skipif(
    not Path(SURFACE_PROFILES_PATH).exists(),
    reason="requires the real Rawalpindi data build (surface_profiles.parquet) from earlier tasks",
)


@pytest_asyncio.fixture
async def client():
    # httpx's ASGITransport does not run lifespan events itself -- drive
    # app.main's lifespan directly so app.state.graph/model_bundle/etc. are
    # populated exactly as they would be under `uvicorn app.main:app`.
    async with lifespan(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac


@pytest.fixture(scope="module")
def aoi_bbox() -> str:
    return get_settings().AOI_BBOX


@pytest.fixture(scope="module")
def real_point_id() -> str:
    df = pd.read_parquet(SURFACE_PROFILES_PATH)
    return str(df.iloc[0]["point_id"])


@pytest.fixture(scope="module")
def aoi_start_end(aoi_bbox: str) -> tuple[dict, dict]:
    """Two points near opposite corners of the real AOI (inset slightly so
    they're likely to snap onto the graph), derived from config -- not
    hardcoded to whatever city happens to be configured."""
    min_lon, min_lat, max_lon, max_lat = (float(x) for x in aoi_bbox.split(","))
    inset_lon = (max_lon - min_lon) * 0.1
    inset_lat = (max_lat - min_lat) * 0.1
    start = {"lat": min_lat + inset_lat, "lon": min_lon + inset_lon}
    end = {"lat": max_lat - inset_lat, "lon": max_lon - inset_lon}
    return start, end


# --- health ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_returns_200_with_graph_loaded(client: AsyncClient):
    response = await client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["graph_loaded"] is True
    assert body["n_nodes"] > 0
    assert body["n_edges"] > 0
    assert "fortyguard_key_present" in body
    assert "data_source" in body and "observed_at" in body and "aoi" in body


# --- grid ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grid_respects_downsample(client: AsyncClient):
    full = await client.get("/api/grid")
    assert full.status_code == 200
    full_body = full.json()
    n_total = full_body["n_total"]

    downsampled = await client.get("/api/grid", params={"downsample": max(2, n_total // 10)})
    assert downsampled.status_code == 200
    body = downsampled.json()
    assert body["n_total"] == n_total
    assert body["n_returned"] < n_total or n_total <= 1
    assert body["n_returned"] == len(body["features"])


@pytest.mark.asyncio
async def test_grid_never_exceeds_feature_cap(client: AsyncClient):
    response = await client.get("/api/grid", params={"downsample": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["n_returned"] <= 3000


# --- route ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_family_returns_distinct_routes_and_comparison(client: AsyncClient, aoi_start_end):
    start, end = aoi_start_end
    response = await client.post("/api/route", json={"start": start, "end": end, "family": True})
    assert response.status_code == 200
    body = response.json()
    assert len(body["routes"]) >= 1
    assert body["comparison"] is not None
    assert "summary" in body["comparison"]


@pytest.mark.asyncio
async def test_route_geojson_coordinates_are_lon_lat(client: AsyncClient, aoi_start_end, aoi_bbox: str):
    start, end = aoi_start_end
    response = await client.post("/api/route", json={"start": start, "end": end, "lambda_heat": 0.5})
    assert response.status_code == 200
    route = response.json()["routes"][0]

    assert route["geojson"]["geometry"]["type"] == "LineString"
    coords = route["geojson"]["geometry"]["coordinates"]
    assert len(coords) >= 2

    # a snap-tolerance margin around the AOI, since a node can sit just outside
    # the exact bbox edge while still being the nearest walkable node
    min_lon, min_lat, max_lon, max_lat = (float(x) for x in aoi_bbox.split(","))
    margin = 0.01
    for lon, lat in coords:
        assert isinstance(lon, float) and isinstance(lat, float)
        # GeoJSON order is [lon, lat]: if these were swapped, this bounds
        # check would fail since our AOI's lat/lon ranges don't overlap
        assert min_lon - margin <= lon <= max_lon + margin, f"lon {lon} outside AOI -- coordinates may be [lat, lon]"
        assert min_lat - margin <= lat <= max_lat + margin, f"lat {lat} outside AOI -- coordinates may be [lat, lon]"


@pytest.mark.asyncio
async def test_out_of_aoi_coordinates_return_400_with_readable_message(client: AsyncClient):
    response = await client.post(
        "/api/route",
        json={"start": {"lat": 0.0, "lon": 0.0}, "end": {"lat": 1.0, "lon": 1.0}},
    )
    assert response.status_code == 400
    body = response.json()
    assert "detail" in body
    assert "outside" in body["detail"].lower() or "aoi" in body["detail"].lower()


@pytest.mark.asyncio
async def test_route_completes_under_one_second(client: AsyncClient, aoi_start_end):
    start, end = aoi_start_end
    # warm the edge-cost cache first, same rationale as tests/test_routing.py's perf test
    await client.post("/api/route", json={"start": start, "end": end, "lambda_heat": 0.5})

    t0 = time.monotonic()
    response = await client.post("/api/route", json={"start": start, "end": end, "lambda_heat": 0.5})
    elapsed_s = time.monotonic() - t0

    assert response.status_code == 200
    assert elapsed_s < 1.0


# --- points / interventions -----------------------------------------------------


@pytest.mark.asyncio
async def test_get_point_profile(client: AsyncClient, real_point_id: str):
    response = await client.get(f"/api/points/{real_point_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["point_id"] == real_point_id
    assert 0.0 <= body["buckets"]["road"] <= 1.0
    assert body["risk_band"] in {"SAFE", "CAUTION", "DANGER", "CRITICAL"}
    assert len(body["interventions"]) == 5


@pytest.mark.asyncio
async def test_unknown_point_id_returns_404(client: AsyncClient):
    response = await client.get("/api/points/not-a-real-point-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_interventions_catalog(client: AsyncClient):
    response = await client.get("/api/interventions")
    assert response.status_code == 200
    body = response.json()
    names = {item["name"] for item in body["interventions"]}
    assert names == {"plant_street_trees", "green_wall", "pocket_park", "cool_roof", "reflective_pavement"}
    literature_items = [i for i in body["interventions"] if i["method"] == "literature_offset"]
    assert {"cool_roof", "reflective_pavement"} == {i["name"] for i in literature_items}


@pytest.mark.asyncio
async def test_post_intervention(client: AsyncClient, real_point_id: str):
    response = await client.post("/api/intervention", json={"point_id": real_point_id, "intervention_name": "cool_roof"})
    assert response.status_code == 200
    body = response.json()
    assert body["result"]["method"] == "literature_offset"
    assert body["result"]["n_training_rows"] == 0


@pytest.mark.asyncio
async def test_unknown_intervention_name_returns_404(client: AsyncClient, real_point_id: str):
    response = await client.post("/api/intervention", json={"point_id": real_point_id, "intervention_name": "teleportation"})
    assert response.status_code == 404


# --- envelope -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_successful_response_carries_data_source_and_observed_at(client: AsyncClient, real_point_id, aoi_start_end):
    start, end = aoi_start_end
    calls = [
        ("GET", "/api/health", None),
        ("GET", "/api/grid", None),
        ("GET", "/api/zones", None),
        ("GET", "/api/interventions", None),
        ("GET", f"/api/points/{real_point_id}", None),
        ("POST", "/api/route", {"start": start, "end": end}),
        ("POST", "/api/intervention", {"point_id": real_point_id, "intervention_name": "green_wall"}),
    ]
    for method, path, json_body in calls:
        response = await client.request(method, path, json=json_body)
        assert response.status_code == 200, f"{method} {path} -> {response.status_code}: {response.text}"
        body = response.json()
        assert "data_source" in body and body["data_source"] in {"live", "fixture"}, path
        assert "observed_at" in body, path
        assert "aoi" in body, path
