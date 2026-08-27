"""/api/grid (raw cells, for detail) and /api/zones (aggregated, for map colour).

Both compute WBGT from temperature + humidity only (wbgt_shade) -- the
surface-adjusted WBGT used by routing is tied to specific graph EDGES via
nearest_edge_key, which doesn't extend to arbitrary grid cells. Grid/zones
are the meteorological background layer; routes carry the street-level detail.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, Query, Request

from app.api.deps import get_aoi_bbox, get_grid_for_timestamp, get_openmeteo_client, get_weather_context
from app.api.schemas import (
    GridCellFeature,
    GridCellProperties,
    GridResponse,
    ZoneFeature,
    ZoneProperties,
    ZonesResponse,
)
from app.config import Settings, get_settings
from app.grid.zones import compute_zones
from app.heat.thresholds import classify
from app.routing.costs import _vectorized_wbgt_shade
from app.routing.router import ROUTE_WORK_INTENSITY

router = APIRouter()

MAX_GRID_FEATURES = 3000  # Leaflet degrades badly past this many features


def _parse_at(at: Optional[str]) -> datetime:
    if at is None:
        return datetime.now(timezone.utc)
    dt = datetime.fromisoformat(at)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@router.get("/grid", response_model=GridResponse)
async def get_grid(
    request: Request,
    at: Optional[str] = Query(None, description="ISO timestamp; defaults to now"),
    downsample: Optional[int] = Query(None, ge=1, description="keep every Nth cell; auto-computed to stay under the feature cap if omitted"),
    settings: Settings = Depends(get_settings),
) -> GridResponse:
    aoi_bbox = get_aoi_bbox(request)
    observed_at = _parse_at(at)
    grid = get_grid_for_timestamp(aoi_bbox, observed_at, settings.GRID_GRANULARITY_M, settings.ALLOW_FIXTURE_DATA)

    openmeteo_client = get_openmeteo_client(request)
    context = get_weather_context(aoi_bbox, grid.observed_at, openmeteo_client)

    lats, lons, temps = grid.to_arrays()
    n_total = len(grid.cells)

    min_stride = max(1, math.ceil(n_total / MAX_GRID_FEATURES))
    stride = max(downsample, min_stride) if downsample else min_stride

    indices = np.arange(0, n_total, stride)
    wbgt_arr = _vectorized_wbgt_shade(temps, context["relative_humidity"])

    features = [
        GridCellFeature(
            geometry={"type": "Point", "coordinates": [float(lons[i]), float(lats[i])]},
            properties=GridCellProperties(
                cell_id=grid.cells[i].cell_id,
                temp_c=float(temps[i]),
                wbgt_c=float(wbgt_arr[i]),
                risk_band=classify(float(wbgt_arr[i]), ROUTE_WORK_INTENSITY),
            ),
        )
        for i in indices
    ]

    return GridResponse(
        features=features,
        n_returned=len(features),
        n_total=n_total,
        data_source=grid.source.value.lower(),
        observed_at=grid.observed_at,
        aoi=aoi_bbox,
    )


@router.get("/zones", response_model=ZonesResponse)
async def get_zones(
    request: Request,
    at: Optional[str] = Query(None, description="ISO timestamp; defaults to now"),
    settings: Settings = Depends(get_settings),
) -> ZonesResponse:
    aoi_bbox = get_aoi_bbox(request)
    observed_at = _parse_at(at)
    grid = get_grid_for_timestamp(aoi_bbox, observed_at, settings.GRID_GRANULARITY_M, settings.ALLOW_FIXTURE_DATA)

    openmeteo_client = get_openmeteo_client(request)
    context = get_weather_context(aoi_bbox, grid.observed_at, openmeteo_client)

    zones = compute_zones(grid, aoi_bbox, context["relative_humidity"])

    features = [
        ZoneFeature(
            geometry={
                "type": "Polygon",
                "coordinates": [
                    [
                        [zone.bounds[0], zone.bounds[1]],
                        [zone.bounds[2], zone.bounds[1]],
                        [zone.bounds[2], zone.bounds[3]],
                        [zone.bounds[0], zone.bounds[3]],
                        [zone.bounds[0], zone.bounds[1]],
                    ]
                ],
            },
            properties=ZoneProperties(
                zone_id=zone.zone_id,
                mean_wbgt_c=zone.mean_wbgt_c,
                max_wbgt_c=zone.max_wbgt_c,
                risk_band=zone.risk_band,
                n_cells=zone.n_cells,
            ),
        )
        for zone in zones
    ]

    return ZonesResponse(
        features=features,
        data_source=grid.source.value.lower(),
        observed_at=grid.observed_at,
        aoi=aoi_bbox,
    )
