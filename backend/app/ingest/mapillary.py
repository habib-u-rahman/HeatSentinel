"""Client for the Mapillary Graph API v4 -- finds and downloads street-level
imagery near a point.

Independent of the FortyGuard client -- nothing here imports app.fortyguard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core.geo import _parse_bbox, bbox_around, haversine

logger = logging.getLogger(__name__)

GRAPH_API_BASE = "https://graph.mapillary.com"
SEARCH_FIELDS = "id,captured_at,compass_angle,geometry"
IMAGE_FIELDS = "thumb_2048_url"
SEARCH_LIMIT = 50

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Northern-hemisphere summer months, used as a documented, simplified
# preference filter -- we don't attempt hemisphere-aware seasonality.
SUMMER_MONTHS = {6, 7, 8}


class MapillaryAPIError(Exception):
    """Raised on a non-retryable Mapillary API error."""


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


@dataclass
class ImageMatch:
    """Best Mapillary image found near a query point."""

    image_id: str
    lat: float  # ACTUAL captured latitude -- NOT the query point
    lon: float  # ACTUAL captured longitude -- NOT the query point
    captured_at: datetime
    compass_angle: Optional[float]


def _bbox_around(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    """(west, south, east, north) bbox spanning `radius_m` metres around a point."""
    return _parse_bbox(bbox_around(lat, lon, radius_m))


class MapillaryClient:
    """Synchronous, retrying, on-disk-cached Mapillary Graph API v4 client."""

    def __init__(
        self,
        token: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        request_timeout_s: float = 30.0,
    ):
        self.token = token if token is not None else get_settings().MAPILLARY_TOKEN
        self._client = http_client
        self._owns_client = http_client is None
        self.request_timeout_s = request_timeout_s

    def __enter__(self) -> "MapillaryClient":
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

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=_retry_after_wait,
        retry=retry_if_exception_type(_RetryableStatusError),
    )
    def _get(self, url: str, params: Optional[dict] = None) -> httpx.Response:
        client = self._get_client()
        # httpx treats an explicit `params=` as AUTHORITATIVE and strips any
        # existing query string from `url` -- even `params={}`. That's fatal
        # for the CDN thumb_2048_url download call, whose query string IS a
        # Meta-signed hash (stp/oh/oe/...); passing params={} there silently
        # stripped the signature and every download 403'd with "Bad URL hash".
        # Only pass params through when there's something real to merge.
        response = client.get(url, params=params) if params else client.get(url)
        logger.info("mapillary_http_call url=%s status=%s", url, response.status_code)
        if response.status_code in RETRYABLE_STATUS_CODES:
            raise _RetryableStatusError(response)
        if response.status_code >= 400:
            raise MapillaryAPIError(f"Mapillary API error {response.status_code}: {response.text}")
        return response

    def find_image_near(self, lat: float, lon: float, radius_m: float = 30.0) -> Optional[ImageMatch]:
        """Best image within radius_m of (lat, lon), or None if nothing is found.

        Selection: prefer images captured in a summer month if any exist within
        radius, then take the most recent among that preferred set (falling
        back to the most recent of ALL candidates if none were captured in
        summer). captured_at is recorded either way.
        """
        west, south, east, north = _bbox_around(lat, lon, radius_m)
        params = {
            "access_token": self.token,
            "fields": SEARCH_FIELDS,
            "bbox": f"{west},{south},{east},{north}",
            "limit": SEARCH_LIMIT,
        }
        response = self._get(f"{GRAPH_API_BASE}/images", params)
        candidates = response.json().get("data", [])

        matches: list[ImageMatch] = []
        for item in candidates:
            coordinates = (item.get("geometry") or {}).get("coordinates")
            captured_at_ms = item.get("captured_at")
            if not coordinates or captured_at_ms is None:
                continue

            img_lon, img_lat = coordinates  # GeoJSON Point is [lon, lat]
            if haversine((lat, lon), (img_lat, img_lon)) > radius_m:
                continue

            matches.append(
                ImageMatch(
                    image_id=item["id"],
                    lat=img_lat,
                    lon=img_lon,
                    captured_at=datetime.fromtimestamp(captured_at_ms / 1000, tz=timezone.utc),
                    compass_angle=item.get("compass_angle"),
                )
            )

        if not matches:
            return None

        summer_matches = [m for m in matches if m.captured_at.month in SUMMER_MONTHS]
        pool = summer_matches or matches
        return max(pool, key=lambda m: m.captured_at)

    def download_image(self, image_id: str, dest: Path) -> bool:
        """Download image_id's thumb_2048_url to dest. Returns False if already cached on disk."""
        dest = Path(dest)
        if dest.exists():
            logger.info("mapillary_download_cache_hit image_id=%s dest=%s", image_id, dest)
            return False

        response = self._get(f"{GRAPH_API_BASE}/{image_id}", {"access_token": self.token, "fields": IMAGE_FIELDS})
        thumb_url = response.json().get("thumb_2048_url")
        if not thumb_url:
            raise MapillaryAPIError(f"No thumb_2048_url returned for image {image_id}")

        image_response = self._get(thumb_url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(image_response.content)
        return True
