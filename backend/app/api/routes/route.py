from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request

from app.api.deps import assert_within_aoi, get_aoi_bbox, get_grid_for_timestamp, get_graph
from app.api.schemas import RouteComparisonSchema, RouteRequestBody, RouteResponse, RouteResultSchema
from app.config import Settings, get_settings
from app.routing.pareto import compare, compute_route_family
from app.routing.router import RouteResult, route as compute_route

router = APIRouter()


def _to_schema(r: RouteResult) -> RouteResultSchema:
    return RouteResultSchema(
        label=r.label,
        lambda_heat=r.lambda_heat,
        geojson=r.geojson,
        total_distance_m=r.total_distance_m,
        total_duration_s=r.total_duration_s,
        total_heat_dose_degC_s=r.total_heat_dose_degC_s,
        mean_wbgt_c=r.mean_wbgt_c,
        max_wbgt_c=r.max_wbgt_c,
        peak_risk_band=r.peak_risk_band,
        n_edges_orphaned=r.n_edges_orphaned,
    )


@router.post("/route", response_model=RouteResponse)
async def post_route(
    body: RouteRequestBody,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> RouteResponse:
    aoi_bbox = get_aoi_bbox(request)
    graph = get_graph(request)
    assert_within_aoi(body.start.lat, body.start.lon, aoi_bbox)
    assert_within_aoi(body.end.lat, body.end.lon, aoi_bbox)

    now = datetime.now(timezone.utc)
    grid = get_grid_for_timestamp(aoi_bbox, now, settings.GRID_GRANULARITY_M, settings.ALLOW_FIXTURE_DATA)

    start = (body.start.lat, body.start.lon)
    end = (body.end.lat, body.end.lon)

    if body.family:
        family = compute_route_family(start, end, graph, grid)
        shortest = next((r for r in family if r.label == "SHORTEST"), family[0])
        coolest = next((r for r in family if r.label == "COOLEST"), family[-1])
        comparison_schema = RouteComparisonSchema(**compare(shortest, coolest).__dict__)
        routes = [_to_schema(r) for r in family]
    else:
        result = compute_route(start, end, body.lambda_heat, graph, grid)
        routes = [_to_schema(result)]
        comparison_schema = None

    return RouteResponse(
        routes=routes,
        comparison=comparison_schema,
        data_source=grid.source.value.lower(),
        observed_at=grid.observed_at,
        aoi=aoi_bbox,
    )
