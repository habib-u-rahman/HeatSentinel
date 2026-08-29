"""Async client for the FortyGuard heatmap API.

THE API IS SUBMIT-AND-POLL: you POST an AOI + time filter and get back an
activity_id, then GET a poll endpoint until the job is done.

We do NOT have confirmed API docs yet. Every constant below marked
"VERIFY AGAINST DOCS" is a best guess based on the task brief and must be
corrected once real responses are seen (use `--probe`, see bottom of file).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core.geo import bbox_to_polygon

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VERIFY AGAINST DOCS: endpoint paths, response field locations, and status
# values are all guesses until confirmed against real FortyGuard responses.
# ---------------------------------------------------------------------------
SUBMIT_ENDPOINT = "/heatmap"  # POST
POLL_ENDPOINT_TEMPLATE = "/heatmap/{activity_id}"  # GET

ACTIVITY_ID_PATH = ("data", "activity_id")
STATUS_FIELD_PATH = ("data", "status")
RESULT_FIELD_PATH = ("data",)  # what we cache / return once done

STATUS_DONE_VALUES = {"done", "completed", "success", "finished"}
STATUS_FAILED_VALUES = {"failed", "error", "cancelled"}
STATUS_PENDING_VALUES = {"pending", "processing", "running", "queued", "in_progress"}
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_S = 180
DEFAULT_INTERVAL_S = 3.0
DEFAULT_FILTER_TYPE = "Vehicle"

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class FortyGuardAPIError(Exception):
    """Raised when the FortyGuard API returns a non-retryable error or a failed activity."""


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


def _get_path(data: dict, path: tuple[str, ...]) -> Any:
    node: Any = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _retry_after_wait(retry_state):
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, _RetryableStatusError) and exc.retry_after is not None:
        return exc.retry_after
    return wait_exponential(multiplier=1, min=1, max=30)(retry_state)


class FortyGuardClient:
    """Async, caching, retrying client for the FortyGuard heatmap API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        request_timeout_s: float = 30.0,
    ):
        if api_key is None or base_url is None or cache_dir is None:
            settings = get_settings()
            api_key = api_key if api_key is not None else settings.FORTYGUARD_API_KEY
            base_url = base_url if base_url is not None else settings.FORTYGUARD_BASE_URL
            cache_dir = cache_dir if cache_dir is not None else settings.CACHE_DIR

        self.api_key = api_key
        self.base_url = str(base_url).rstrip("/")
        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning("fortyguard_cache_dir_mkdir_failed dir=%s -- read-only filesystem", self.cache_dir)
        self.request_timeout_s = request_timeout_s

        self._client = http_client
        self._owns_client = http_client is None

    async def __aenter__(self) -> "FortyGuardClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.request_timeout_s)
        return self._client

    # -- cache -----------------------------------------------------------

    @staticmethod
    def _payload_hash(payload: dict) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _cache_path(self, payload_hash: str) -> Path:
        return self.cache_dir / f"{payload_hash}.json"

    def _read_cache(self, payload_hash: str) -> Optional[dict]:
        path = self._cache_path(payload_hash)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("fortyguard_cache_corrupt path=%s", path)
            return None

    def _write_cache(self, payload_hash: str, result: dict) -> None:
        path = self._cache_path(payload_hash)
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(result, f)
        except OSError:
            logger.warning("fortyguard_cache_write_failed path=%s -- read-only filesystem, skipping", path)

    # -- HTTP --------------------------------------------------------------

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=_retry_after_wait,
        retry=retry_if_exception_type(_RetryableStatusError),
    )
    async def _send(self, method: str, url: str, **kwargs) -> httpx.Response:
        client = self._get_client()
        start = time.monotonic()
        response = await client.request(method, url, **kwargs)
        latency_ms = (time.monotonic() - start) * 1000
        logger.info(
            "fortyguard_http_call method=%s url=%s status=%s latency_ms=%.1f",
            method,
            url,
            response.status_code,
            latency_ms,
        )
        if response.status_code in RETRYABLE_STATUS_CODES:
            raise _RetryableStatusError(response)
        if response.status_code >= 400:
            raise FortyGuardAPIError(
                f"FortyGuard API error {response.status_code} for {method} {url}: {response.text}"
            )
        return response

    async def _post(self, endpoint: str, payload: dict) -> httpx.Response:
        url = f"{self.base_url}{endpoint}"
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        return await self._send("POST", url, json=payload, headers=headers)

    async def _get(self, endpoint: str) -> httpx.Response:
        url = f"{self.base_url}{endpoint}"
        headers = {"api-key": self.api_key}
        return await self._send("GET", url, headers=headers)

    # -- public API ----------------------------------------------------------

    @staticmethod
    def _build_payload(
        polygon: dict,
        date: str,
        time_: str,
        filter_type: str,
        granularity: int,
    ) -> dict:
        return {
            "polygon_aoi": polygon,
            "date_time": {
                "start_date": date,
                "start_time": time_,
                "filter_type": filter_type,
            },
            "granularity": granularity,
        }

    async def submit_heatmap(
        self,
        polygon: dict,
        date: str,
        time_: str,
        filter_type: str = DEFAULT_FILTER_TYPE,
        granularity: Optional[int] = None,
    ) -> str:
        if granularity is None:
            granularity = get_settings().GRID_GRANULARITY_M
        payload = self._build_payload(polygon, date, time_, filter_type, granularity)
        payload_hash = self._payload_hash(payload)
        logger.info("fortyguard_submit payload_hash=%s", payload_hash)

        response = await self._post(SUBMIT_ENDPOINT, payload)
        data = response.json()
        activity_id = _get_path(data, ACTIVITY_ID_PATH)
        if not activity_id:
            raise FortyGuardAPIError(
                f"Could not find activity_id at {ACTIVITY_ID_PATH} in submit response: {data}"
            )
        return activity_id

    async def poll_result(
        self,
        activity_id: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        interval_s: float = DEFAULT_INTERVAL_S,
    ) -> dict:
        endpoint = POLL_ENDPOINT_TEMPLATE.format(activity_id=activity_id)
        deadline = time.monotonic() + timeout_s

        while True:
            response = await self._get(endpoint)
            data = response.json()
            status = _get_path(data, STATUS_FIELD_PATH)

            if status in STATUS_DONE_VALUES:
                return data
            if status in STATUS_FAILED_VALUES:
                raise FortyGuardAPIError(f"Activity {activity_id} failed: {data}")

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Polling for activity {activity_id} timed out after {timeout_s}s (last status={status!r})"
                )
            await asyncio.sleep(interval_s)

    async def get_heatmap(
        self,
        polygon: dict,
        date: str,
        time_: str,
        filter_type: str = DEFAULT_FILTER_TYPE,
        granularity: Optional[int] = None,
        force_refresh: bool = False,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        interval_s: float = DEFAULT_INTERVAL_S,
    ) -> dict:
        """Submit + poll, with an on-disk cache keyed by the request payload hash.

        We are rate limited: never re-fetch the same AOI/date/time/granularity
        combination twice unless force_refresh=True.
        """
        if granularity is None:
            granularity = get_settings().GRID_GRANULARITY_M
        payload = self._build_payload(polygon, date, time_, filter_type, granularity)
        payload_hash = self._payload_hash(payload)

        if not force_refresh:
            cached = self._read_cache(payload_hash)
            if cached is not None:
                logger.info("fortyguard_cache_hit payload_hash=%s", payload_hash)
                return cached

        activity_id = await self.submit_heatmap(polygon, date, time_, filter_type, granularity)
        result = await self.poll_result(activity_id, timeout_s=timeout_s, interval_s=interval_s)
        self._write_cache(payload_hash, result)
        return result


# ---------------------------------------------------------------------------
# --probe CLI: submits a tiny AOI and prints RAW submit + poll JSON so the
# schema guesses above can be corrected against a real response.
# ---------------------------------------------------------------------------


async def _run_probe() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()

    min_lon, min_lat, max_lon, max_lat = (float(x) for x in settings.AOI_BBOX.split(","))
    # Shrink to a tiny AOI around the bbox center to keep the probe cheap.
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    delta = 0.001
    tiny_bbox = (center_lon - delta, center_lat - delta, center_lon + delta, center_lat + delta)
    polygon = bbox_to_polygon(tiny_bbox)

    date = time.strftime("%Y-%m-%d")
    time_ = "12:00"

    async with FortyGuardClient() as client:
        payload = client._build_payload(polygon, date, time_, DEFAULT_FILTER_TYPE, settings.GRID_GRANULARITY_M)
        print(f"Submitting probe AOI: {tiny_bbox}")

        submit_response = await client._post(SUBMIT_ENDPOINT, payload)
        submit_json = submit_response.json()
        print("\n=== RAW SUBMIT RESPONSE ===")
        print(json.dumps(submit_json, indent=2))

        activity_id = _get_path(submit_json, ACTIVITY_ID_PATH)
        if not activity_id:
            print(
                f"\nWARNING: could not find activity_id at {ACTIVITY_ID_PATH}. "
                "Update ACTIVITY_ID_PATH in client.py once you see the real field."
            )
            return

        poll_endpoint = POLL_ENDPOINT_TEMPLATE.format(activity_id=activity_id)
        poll_response = await client._get(poll_endpoint)
        poll_json = poll_response.json()
        print("\n=== RAW POLL RESPONSE ===")
        print(json.dumps(poll_json, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="FortyGuard API client")
    parser.add_argument("--probe", action="store_true", help="Submit a tiny AOI and print raw JSON responses")
    args = parser.parse_args()

    if args.probe:
        asyncio.run(_run_probe())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
