from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request

from app.api.deps import get_aoi_bbox, get_grid_for_timestamp
from app.api.schemas import HealthResponse
from app.config import get_settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = get_settings()
    aoi_bbox = get_aoi_bbox(request)
    graph = getattr(request.app.state, "graph", None)
    model_bundle = getattr(request.app.state, "model_bundle", None)

    now = datetime.now(timezone.utc)
    try:
        grid = get_grid_for_timestamp(aoi_bbox, now, settings.GRID_GRANULARITY_M, settings.ALLOW_FIXTURE_DATA)
        data_source = grid.source.value.lower()
    except Exception:
        data_source = "fixture"

    return HealthResponse(
        status="ok",
        graph_loaded=graph is not None,
        model_loaded=model_bundle is not None,
        n_nodes=graph.number_of_nodes() if graph is not None else 0,
        n_edges=graph.number_of_edges() if graph is not None else 0,
        fortyguard_key_present=bool(settings.FORTYGUARD_API_KEY),
        data_source=data_source,
        observed_at=now,
        aoi=aoi_bbox,
    )
