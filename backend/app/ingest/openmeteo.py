"""Open-Meteo forecast API client -- our humidity/wind/solar source.

No API key required. FortyGuard remains the temperature source of truth;
temperature_2m here is only a cross-check, never the primary reading.

Independent of the FortyGuard client -- nothing here imports app.fortyguard.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARIABLES = "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation"

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class OpenMeteoAPIError(Exception):
    """Raised on a non-retryable Open-Meteo API error."""


class _RetryableStatusError(Exception):
    """Internal signal used to drive tenacity retries; carries Retry-After if present."""

    def __init__(self, response: httpx.Response):
        self.response = response
        self.status_code = response.status_code
        retry_after_header = response.headers.get("Retry-After")
        self.retry_after: Optional[float] = None
        if retry_after_header:
            try:
                self.retry_after = float(retry_after_header)
            except ValueError:
                self.retry_after = None
        super().__init__(f"Retryable HTTP status {self.status_code}")


def _retry_after_wait(retry_state):
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, _RetryableStatusError) and exc.retry_after is not None:
        return exc.retry_after
    return wait_exponential(multiplier=1, min=1, max=30)(retry_state)


class OpenMeteoClient:
    """Synchronous, retrying, on-disk-cached Open-Meteo forecast client."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        http_client: Optional[httpx.Client] = None,
        request_timeout_s: float = 30.0,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path(get_settings().CACHE_DIR) / "openmeteo"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = http_client
        self._owns_client = http_client is None
        self.request_timeout_s = request_timeout_s

    def __enter__(self) -> "OpenMeteoClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.request_timeout_s)
        return self._client

    @staticmethod
    def _cache_key(lat: float, lon: float, start: str, end: str) -> str:
        # Round to ~110m precision -- plenty for hourly gridded forecast data.
        canonical = f"{round(lat, 3)}:{round(lon, 3)}:{start}:{end}"
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=_retry_after_wait,
        retry=retry_if_exception_type(_RetryableStatusError),
    )
    def _get(self, url: str, params: dict) -> httpx.Response:
        client = self._get_client()
        response = client.get(url, params=params)
        logger.info("openmeteo_http_call url=%s status=%s", url, response.status_code)
        if response.status_code in RETRYABLE_STATUS_CODES:
            raise _RetryableStatusError(response)
        if response.status_code >= 400:
            raise OpenMeteoAPIError(f"Open-Meteo API error {response.status_code}: {response.text}")
        return response

    def fetch_hourly(self, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
        """Hourly temperature_2m, relative_humidity_2m, wind_speed_10m, shortwave_radiation.

        start/end are "YYYY-MM-DD", inclusive. Cached on disk keyed by
        (rounded lat/lon, start, end) -- a repeated call for the same
        location/range never re-hits the network.
        """
        cache_key = self._cache_key(lat, lon, start, end)
        cache_path = self._cache_path(cache_key)

        if cache_path.exists():
            logger.info("openmeteo_cache_hit key=%s", cache_key)
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start,
                "end_date": end,
                "hourly": HOURLY_VARIABLES,
                "timezone": "UTC",
            }
            response = self._get(FORECAST_URL, params)
            payload = response.json()
            cache_path.write_text(json.dumps(payload), encoding="utf-8")

        hourly = payload.get("hourly", {})
        return pd.DataFrame(
            {
                "time": hourly.get("time", []),
                "temperature_2m": hourly.get("temperature_2m", []),
                "relative_humidity_2m": hourly.get("relative_humidity_2m", []),
                "wind_speed_10m": hourly.get("wind_speed_10m", []),
                "shortwave_radiation": hourly.get("shortwave_radiation", []),
            }
        )
