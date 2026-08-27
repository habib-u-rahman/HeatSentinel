from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request

from app.api.deps import (
    UnknownInterventionError,
    UnknownPointIdError,
    get_aoi_bbox,
    get_openmeteo_client,
    get_surface_profiles_df,
    get_weather_context,
    surface_profile_dict_from_row,
)
from app.api.schemas import (
    DeltaTResultSchema,
    InterventionCatalogItem,
    InterventionRequestBody,
    InterventionResponse,
    InterventionsCatalogResponse,
)
from app.config import Settings, get_settings
from app.ml.intervention import INTERVENTIONS, predict_delta_t

router = APIRouter()


@router.get("/interventions", response_model=InterventionsCatalogResponse)
async def get_interventions_catalog(request: Request, settings: Settings = Depends(get_settings)) -> InterventionsCatalogResponse:
    now = datetime.now(timezone.utc)
    bundle = getattr(request.app.state, "model_bundle", None)
    data_source = bundle["metadata"]["source"].lower() if bundle else "fixture"

    items = [
        InterventionCatalogItem(
            name=name,
            description=spec["source"],
            deltas=spec["deltas"],
            literature_offset_c=spec["literature_offset_c"],
            method="model_prediction" if spec["deltas"] is not None else "literature_offset",
        )
        for name, spec in INTERVENTIONS.items()
    ]

    return InterventionsCatalogResponse(interventions=items, data_source=data_source, observed_at=now, aoi=get_aoi_bbox(request))


@router.post("/intervention", response_model=InterventionResponse)
async def post_intervention(
    body: InterventionRequestBody, request: Request, settings: Settings = Depends(get_settings)
) -> InterventionResponse:
    if body.intervention_name not in INTERVENTIONS:
        raise UnknownInterventionError(
            f"Unknown intervention_name {body.intervention_name!r}; choose one of {sorted(INTERVENTIONS)}"
        )

    surface_df = get_surface_profiles_df(request)
    matches = surface_df[surface_df["point_id"] == body.point_id]
    if matches.empty:
        raise UnknownPointIdError(f"Unknown point_id {body.point_id!r} -- no surface profile on record")
    row = matches.iloc[0]

    aoi_bbox = get_aoi_bbox(request)
    now = datetime.now(timezone.utc)
    openmeteo_client = get_openmeteo_client(request)
    context = get_weather_context(aoi_bbox, now, openmeteo_client)

    profile = surface_profile_dict_from_row(row)
    result = predict_delta_t(profile, body.intervention_name, context, aoi_bbox)

    return InterventionResponse(
        result=DeltaTResultSchema(**result.__dict__),
        data_source=result.data_source,
        observed_at=now,
        aoi=aoi_bbox,
    )
