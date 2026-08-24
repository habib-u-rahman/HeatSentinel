from __future__ import annotations

import respx
from httpx import Response

from app.ingest.mapillary import GRAPH_API_BASE, MapillaryClient

QUERY_LAT, QUERY_LON = 40.7128, -74.0060


def _search_payload(images: list[dict]) -> dict:
    return {"data": images}


def test_find_image_near_prefers_summer_and_recent():
    winter_far_lon = QUERY_LON  # same coords, different time only matters here
    images = [
        {
            "id": "winter_old",
            "captured_at": 1_600_000_000_000,  # 2020-09 (not summer)
            "compass_angle": 10.0,
            "geometry": {"type": "Point", "coordinates": [QUERY_LON + 0.0001, QUERY_LAT + 0.0001]},
        },
        {
            "id": "summer_older",
            "captured_at": 1_530_000_000_000,  # 2018-06 (summer)
            "compass_angle": 20.0,
            "geometry": {"type": "Point", "coordinates": [QUERY_LON + 0.0001, QUERY_LAT]},
        },
        {
            "id": "summer_newer",
            "captured_at": 1_594_000_000_000,  # 2020-07 (summer, most recent summer shot)
            "compass_angle": 30.0,
            "geometry": {"type": "Point", "coordinates": [QUERY_LON, QUERY_LAT + 0.0001]},
        },
    ]

    with respx.mock:
        respx.get(f"{GRAPH_API_BASE}/images").mock(return_value=Response(200, json=_search_payload(images)))
        with MapillaryClient(token="test-token") as client:
            match = client.find_image_near(QUERY_LAT, QUERY_LON, radius_m=30)

    assert match is not None
    assert match.image_id == "summer_newer"
    assert match.captured_at.month in (6, 7, 8)
    # Actual captured coordinates are returned, not the query point.
    assert (match.lat, match.lon) == (QUERY_LAT + 0.0001, QUERY_LON)


def test_find_image_near_falls_back_to_most_recent_when_no_summer_match():
    images = [
        {
            "id": "spring",
            "captured_at": 1_617_000_000_000,  # 2021-03
            "compass_angle": None,
            "geometry": {"type": "Point", "coordinates": [QUERY_LON, QUERY_LAT]},
        },
        {
            "id": "autumn_newer",
            "captured_at": 1_635_000_000_000,  # 2021-10, most recent overall
            "compass_angle": None,
            "geometry": {"type": "Point", "coordinates": [QUERY_LON, QUERY_LAT]},
        },
    ]

    with respx.mock:
        respx.get(f"{GRAPH_API_BASE}/images").mock(return_value=Response(200, json=_search_payload(images)))
        with MapillaryClient(token="test-token") as client:
            match = client.find_image_near(QUERY_LAT, QUERY_LON, radius_m=30)

    assert match.image_id == "autumn_newer"


def test_find_image_near_filters_out_of_radius_results():
    images = [
        {
            "id": "far_away",
            "captured_at": 1_600_000_000_000,
            "compass_angle": None,
            # ~1km away -- well outside a 30m radius
            "geometry": {"type": "Point", "coordinates": [QUERY_LON + 0.01, QUERY_LAT]},
        },
    ]

    with respx.mock:
        respx.get(f"{GRAPH_API_BASE}/images").mock(return_value=Response(200, json=_search_payload(images)))
        with MapillaryClient(token="test-token") as client:
            match = client.find_image_near(QUERY_LAT, QUERY_LON, radius_m=30)

    assert match is None


def test_find_image_near_returns_none_when_no_candidates():
    with respx.mock:
        respx.get(f"{GRAPH_API_BASE}/images").mock(return_value=Response(200, json=_search_payload([])))
        with MapillaryClient(token="test-token") as client:
            match = client.find_image_near(QUERY_LAT, QUERY_LON)

    assert match is None


def test_download_image_skips_when_already_on_disk(tmp_path):
    dest = tmp_path / "already_here.jpg"
    dest.write_bytes(b"cached bytes")

    with MapillaryClient(token="test-token") as client:
        downloaded = client.download_image("some_id", dest)

    assert downloaded is False
    assert dest.read_bytes() == b"cached bytes"


def test_download_image_fetches_thumb_url_and_saves(tmp_path):
    dest = tmp_path / "new_image.jpg"
    # Real Mapillary/Meta CDN thumb URLs carry a SIGNED query string (stp/oh/oe/...).
    # Regression coverage for a real bug: passing params={} to httpx.get(url, params=...)
    # silently STRIPS an existing query string (even when empty), which broke every
    # real download with a 403 "Bad URL hash" -- a query-string-free mock URL would
    # never have caught that. respx matches the exact URL, so this only passes if
    # the signed query string reaches the server untouched.
    thumb_url = "https://cdn.mapillary.example/thumb.jpg?stp=s2048x1536&oh=00_ABC123&oe=6AB36540"

    with respx.mock:
        respx.get(f"{GRAPH_API_BASE}/some_id").mock(
            return_value=Response(200, json={"thumb_2048_url": thumb_url})
        )
        respx.get(thumb_url).mock(return_value=Response(200, content=b"fake jpeg bytes"))

        with MapillaryClient(token="test-token") as client:
            downloaded = client.download_image("some_id", dest)

    assert downloaded is True
    assert dest.read_bytes() == b"fake jpeg bytes"
