"""GET /api/alerts -- ranked heat alerts (zone_critical / poi_at_risk / rapid_rise)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, Query, Request

from app.alerts.engine import evaluate
from app.api.deps import get_aoi_bbox, get_grid_for_timestamp, get_openmeteo_client, get_weather_context
from app.api.schemas import AlertSchema, AlertsResponse
from app.config import Settings, get_settings
from app.grid import store as grid_store
from app.heat.thresholds import BANDS_IN_ORDER

router = APIRouter()

_SEVERITY_RANK = {band: i for i, band in enumerate(BANDS_IN_ORDER)}
_LOOKBACK = timedelta(hours=6)


def _parse_at(at: Optional[str]) -> datetime:
    if at is None:
        return datetime.now(timezone.utc)
    dt = datetime.fromisoformat(at)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _find_previous_grid(bbox: str, observed_at: datetime):
    """The most recent STORED grid strictly before observed_at, for rapid_rise
    detection. None (not an error) if no prior grid exists yet."""
    available = grid_store.list_available(observed_at - _LOOKBACK, observed_at - timedelta(seconds=1))
    if not available:
        return None
    path = grid_store._grid_path(grid_store.DEFAULT_BASE_DIR, max(available))
    try:
        return grid_store.load_grid(path)
    except Exception:
        return None


@router.get("/alerts", response_model=AlertsResponse)
async def get_alerts(
    request: Request,
    at: Optional[str] = Query(None, description="ISO timestamp; defaults to now"),
    min_severity: Optional[str] = Query(None, description="SAFE|CAUTION|DANGER|CRITICAL -- only alerts at or above this band"),
    settings: Settings = Depends(get_settings),
) -> AlertsResponse:
    aoi_bbox = get_aoi_bbox(request)
    observed_at = _parse_at(at)
    grid = get_grid_for_timestamp(aoi_bbox, observed_at, settings.GRID_GRANULARITY_M, settings.ALLOW_FIXTURE_DATA)

    openmeteo_client = get_openmeteo_client(request)
    context = get_weather_context(aoi_bbox, grid.observed_at, openmeteo_client)

    pois_df = getattr(request.app.state, "vulnerable_pois_df", None)
    if pois_df is None:
        pois_df = pd.DataFrame(columns=["poi_id", "name", "category", "vulnerability_group", "weight", "lat", "lon"])

    previous_grid = _find_previous_grid(aoi_bbox, grid.observed_at)
    alerts = evaluate(grid, pois_df, aoi_bbox, context["relative_humidity"], previous_grid=previous_grid)

    if min_severity:
        min_rank = _SEVERITY_RANK.get(min_severity.upper())
        if min_rank is not None:
            alerts = [a for a in alerts if _SEVERITY_RANK.get(a.severity, 0) >= min_rank]

    return AlertsResponse(
        alerts=[AlertSchema(**a.__dict__) for a in alerts],
        data_source=grid.source.value.lower(),
        observed_at=grid.observed_at,
        aoi=aoi_bbox,
    )
