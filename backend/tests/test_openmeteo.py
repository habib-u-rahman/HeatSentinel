from __future__ import annotations

import respx
from httpx import Response

from app.ingest.openmeteo import FORECAST_URL, OpenMeteoClient

SAMPLE_RESPONSE = {
    "hourly": {
        "time": ["2026-07-01T00:00", "2026-07-01T01:00"],
        "temperature_2m": [28.0, 27.5],
        "relative_humidity_2m": [55.0, 58.0],
        "wind_speed_10m": [3.2, 3.0],
        "shortwave_radiation": [0.0, 0.0],
    }
}


def test_fetch_hourly_parses_all_four_variables(tmp_path):
    with respx.mock:
        respx.get(FORECAST_URL).mock(return_value=Response(200, json=SAMPLE_RESPONSE))
        with OpenMeteoClient(cache_dir=tmp_path) as client:
            df = client.fetch_hourly(40.7128, -74.0060, "2026-07-01", "2026-07-01")

    assert list(df.columns) == ["time", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "shortwave_radiation"]
    assert len(df) == 2
    assert df["temperature_2m"].tolist() == [28.0, 27.5]


def test_fetch_hourly_second_call_hits_cache_not_network(tmp_path):
    with respx.mock:
        route = respx.get(FORECAST_URL).mock(return_value=Response(200, json=SAMPLE_RESPONSE))
        with OpenMeteoClient(cache_dir=tmp_path) as client:
            client.fetch_hourly(40.7128, -74.0060, "2026-07-01", "2026-07-01")
            assert route.call_count == 1

            client.fetch_hourly(40.7128, -74.0060, "2026-07-01", "2026-07-01")
            assert route.call_count == 1  # no second network call


def test_fetch_hourly_different_range_is_a_separate_cache_entry(tmp_path):
    with respx.mock:
        route = respx.get(FORECAST_URL).mock(return_value=Response(200, json=SAMPLE_RESPONSE))
        with OpenMeteoClient(cache_dir=tmp_path) as client:
            client.fetch_hourly(40.7128, -74.0060, "2026-07-01", "2026-07-01")
            client.fetch_hourly(40.7128, -74.0060, "2026-07-02", "2026-07-02")
            assert route.call_count == 2
