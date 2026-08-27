"""GET /api/vulnerable -- OSM POIs as a documented population-vulnerability PROXY.

HONESTY CONSTRAINT: population_proxy=True and proxy_note are on EVERY
response, success or empty, so nothing downstream can mistake this for real
population/census data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, Query, Request

from app.api.deps import get_aoi_bbox, get_grid_for_timestamp, get_openmeteo_client, get_weather_context
from app.api.schemas import VulnerablePoiFeature, VulnerablePoiProperties, VulnerableResponse
from app.config import Settings, get_settings
from app.vulnerable.poi import POPULATION_PROXY_NOTE
from app.vulnerable.scoring import attach_risk_to_pois

router = APIRouter()


def _parse_at(at: Optional[str]) -> datetime:
    if at is None:
        return datetime.now(timezone.utc)
    dt = datetime.fromisoformat(at)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@router.get("/vulnerable", response_model=VulnerableResponse)
async def get_vulnerable(
    request: Request,
    at: Optional[str] = Query(None, description="ISO timestamp; defaults to now"),
    settings: Settings = Depends(get_settings),
) -> VulnerableResponse:
    aoi_bbox = get_aoi_bbox(request)
    observed_at = _parse_at(at)
    grid = get_grid_for_timestamp(aoi_bbox, observed_at, settings.GRID_GRANULARITY_M, settings.ALLOW_FIXTURE_DATA)

    pois_df = getattr(request.app.state, "vulnerable_pois_df", None)
    if pois_df is None or pois_df.empty:
        return VulnerableResponse(
            features=[],
            population_proxy=True,
            proxy_note=POPULATION_PROXY_NOTE,
            data_source=grid.source.value.lower(),
            observed_at=grid.observed_at,
            aoi=aoi_bbox,
        )

    openmeteo_client = get_openmeteo_client(request)
    context = get_weather_context(aoi_bbox, grid.observed_at, openmeteo_client)
    scored = attach_risk_to_pois(pois_df, grid, context["relative_humidity"])

    features = [
        VulnerablePoiFeature(
            geometry={"type": "Point", "coordinates": [float(row["lon"]), float(row["lat"])]},
            properties=VulnerablePoiProperties(
                poi_id=row["poi_id"],
                name=row["name"] if pd.notna(row["name"]) else None,
                category=row["category"],
                vulnerability_group=row["vulnerability_group"],
                weight=float(row["weight"]),
                risk_band=row["risk_band"] if pd.notna(row["risk_band"]) else None,
                exposure_score=float(row["exposure_score"]) if pd.notna(row["exposure_score"]) else None,
            ),
        )
        for _, row in scored.iterrows()
    ]

    return VulnerableResponse(
        features=features,
        population_proxy=True,
        proxy_note=POPULATION_PROXY_NOTE,
        data_source=grid.source.value.lower(),
        observed_at=grid.observed_at,
        aoi=aoi_bbox,
    )
