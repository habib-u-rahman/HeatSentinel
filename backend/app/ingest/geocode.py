"""Client for OpenStreetMap Nominatim -- resolves a free-text place name to a
lat/lon centroid, for the on-demand AOI builder (see app/pipeline/build_aoi.py).

Independent of the FortyGuard client -- nothing here imports app.fortyguard.

Nominatim is a free, shared, no-SLA public service. Its usage policy requires
a descriptive User-Agent (not a browser UA -- browsers can't set this on
fetch() anyway) and caps requests at ~1/second, and it doesn't reliably send
CORS headers for arbitrary origins. All of that means this MUST be called
from the backend, never directly from the browser -- see app/api/routes/aoi.py.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

NOMINATIM_BASE = "https://nominatim.openstreetmap.org"
USER_AGENT = "HeatSentinel/1.0 (hackathon urban-heat demo)"
MIN_INTERVAL_S = 1.1  # Nominatim's usage policy: max ~1 request/second

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Module-level (not per-instance): the throttle and cache need to hold across
# every NominatimClient created over the process's lifetime, not just within
# one request's client instance -- otherwise two AOI builds in a row would
# both cold-hit Nominatim for the same city and could double up on the
# 1 req/s budget.
_last_call_lock = threading.Lock()
_last_call_at = 0.0
_cache: dict[str, Optional["GeocodeMatch"]] = {}


class GeocodeAPIError(Exception):
    """Raised on a non-retryable Nominatim API error."""


class _RetryableStatusError(Exception):
    def __init__(self, response: httpx.Response):
        self.response = response
        self.status_code = response.status_code
        super().__init__(f"Retryable HTTP status {self.status_code}")


@dataclass
class GeocodeMatch:
    """A geocoded place -- just the centroid. We deliberately ignore
    Nominatim's own `boundingbox`: it's wildly inconsistent in scale (a POI
    match returns a tiny box, a country match returns a huge one), so the
    caller always derives its own fixed-radius bbox from (lat, lon) via
    app.core.geo.bbox_around instead.
    """

    lat: float
    lon: float
    display_name: str


def _throttle() -> None:
    global _last_call_at
    with _last_call_lock:
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.monotonic()


def _retry_after_wait(retry_state):
    return wait_exponential(multiplier=1, min=1, max=15)(retry_state)


class NominatimClient:
    """Synchronous, rate-limited, process-wide-cached Nominatim geocoding client."""

    def __init__(self, http_client: Optional[httpx.Client] = None, request_timeout_s: float = 15.0):
        self._client = http_client
        self._owns_client = http_client is None
        self.request_timeout_s = request_timeout_s

    def __enter__(self) -> "NominatimClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.request_timeout_s, headers={"User-Agent": USER_AGENT})
        return self._client

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=_retry_after_wait,
        retry=retry_if_exception_type(_RetryableStatusError),
    )
    def _get(self, url: str, params: dict) -> httpx.Response:
        _throttle()
        client = self._get_client()
        response = client.get(url, params=params)
        logger.info("nominatim_http_call url=%s status=%s", url, response.status_code)
        if response.status_code in RETRYABLE_STATUS_CODES:
            raise _RetryableStatusError(response)
        if response.status_code >= 400:
            raise GeocodeAPIError(f"Nominatim API error {response.status_code}: {response.text}")
        return response

    def geocode(self, query: str) -> Optional[GeocodeMatch]:
        """The best-guess centroid for a free-text place name, or None if
        Nominatim has no match. Cached process-wide by normalized query --
        re-typing the same city name during a demo never re-hits Nominatim.
        """
        key = " ".join(query.strip().lower().split())
        if not key:
            return None
        if key in _cache:
            return _cache[key]

        # accept-language=en: Nominatim otherwise returns the LOCAL-script
        # display_name (e.g. "لاہور" for Lahore), which is a real place name,
        # just not one an English-reading demo audience expects as a city label.
        response = self._get(
            f"{NOMINATIM_BASE}/search", {"q": query, "format": "json", "limit": 1, "accept-language": "en"}
        )
        results = response.json()
        if not results:
            _cache[key] = None
            return None

        top = results[0]
        match = GeocodeMatch(lat=float(top["lat"]), lon=float(top["lon"]), display_name=top["display_name"])
        _cache[key] = match
        return match
