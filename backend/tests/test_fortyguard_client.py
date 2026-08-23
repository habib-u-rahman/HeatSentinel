from __future__ import annotations

import pytest
import respx
from httpx import Response

from app.config import get_settings
from app.fortyguard.client import (
    POLL_ENDPOINT_TEMPLATE,
    SUBMIT_ENDPOINT,
    FortyGuardClient,
)

BASE_URL = "https://api.fortyguard.test/v1"
SAMPLE_POLYGON = {
    "type": "Polygon",
    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # Safety net so get_settings() never hits real network/env expectations
    # even if a code path forgets to pass an explicit override.
    monkeypatch.setenv("FORTYGUARD_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("CITY_NAME", "TestCity")
    monkeypatch.setenv("AOI_BBOX", "0,0,1,1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client(tmp_path):
    c = FortyGuardClient(api_key="test-key", base_url=BASE_URL, cache_dir=tmp_path / "cache")
    yield c


async def test_activity_id_parsed(client):
    with respx.mock:
        respx.post(f"{BASE_URL}{SUBMIT_ENDPOINT}").mock(
            return_value=Response(200, json={"data": {"activity_id": "abc123"}})
        )
        activity_id = await client.submit_heatmap(
            SAMPLE_POLYGON, "2026-08-23", "12:00", granularity=100
        )

    assert activity_id == "abc123"
    await client.aclose()


async def test_poll_retries_until_done(client):
    activity_id = "abc123"
    poll_url = f"{BASE_URL}{POLL_ENDPOINT_TEMPLATE.format(activity_id=activity_id)}"

    with respx.mock:
        route = respx.get(poll_url)
        route.side_effect = [
            Response(200, json={"data": {"status": "processing"}}),
            Response(200, json={"data": {"status": "processing"}}),
            Response(200, json={"data": {"status": "done", "result": {"foo": "bar"}}}),
        ]

        result = await client.poll_result(activity_id, timeout_s=5, interval_s=0.01)

        assert result["data"]["status"] == "done"
        assert route.call_count == 3

    await client.aclose()


async def test_cache_prevents_second_http_call(client):
    activity_id = "cache-1"
    poll_url = f"{BASE_URL}{POLL_ENDPOINT_TEMPLATE.format(activity_id=activity_id)}"

    with respx.mock:
        submit_route = respx.post(f"{BASE_URL}{SUBMIT_ENDPOINT}").mock(
            return_value=Response(200, json={"data": {"activity_id": activity_id}})
        )
        poll_route = respx.get(poll_url).mock(
            return_value=Response(200, json={"data": {"status": "done", "result": {"foo": "bar"}}})
        )

        result1 = await client.get_heatmap(
            SAMPLE_POLYGON, "2026-08-23", "12:00", granularity=100
        )
        assert submit_route.call_count == 1
        assert poll_route.call_count == 1

        # Second call with identical payload must be served entirely from cache.
        result2 = await client.get_heatmap(
            SAMPLE_POLYGON, "2026-08-23", "12:00", granularity=100
        )
        assert result2 == result1
        assert submit_route.call_count == 1
        assert poll_route.call_count == 1

    await client.aclose()


async def test_poll_timeout_raises(client):
    activity_id = "stuck"
    poll_url = f"{BASE_URL}{POLL_ENDPOINT_TEMPLATE.format(activity_id=activity_id)}"

    with respx.mock:
        respx.get(poll_url).mock(return_value=Response(200, json={"data": {"status": "processing"}}))

        with pytest.raises(TimeoutError):
            await client.poll_result(activity_id, timeout_s=0.05, interval_s=0.02)

    await client.aclose()
