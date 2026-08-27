from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, Query, Request

from app.api.deps import (
    UnknownPointIdError,
    get_aoi_bbox,
    get_grid_for_timestamp,
    get_openmeteo_client,
    get_surface_profiles_df,
    get_weather_context,
    surface_profile_dict_from_row,
)
from app.api.schemas import (
    BucketFractions,
    DeltaTResultSchema,
    PointProfileResponse,
    SamplePointFeature,
    SamplePointProperties,
    SamplePointsResponse,
    SurfaceMetrics,
)
from app.config import Settings, get_settings
from app.grid.join import attach_temps_to_points, interpolate_idw
from app.heat.thresholds import classify
from app.heat.wbgt import surface_adjusted_wbgt, wbgt_shade
from app.ml.features import BUCKET_NAMES
from app.ml.intervention import rank_interventions
from app.routing.costs import _vectorized_wbgt_shade
from app.routing.router import ROUTE_WORK_INTENSITY

router = APIRouter()

STATIC_IMAGES_DIR = Path("../data/street_images")
STATIC_OVERLAYS_DIR = Path("../data/overlays")


def _parse_at(at: Optional[str]) -> datetime:
    if at is None:
        return datetime.now(timezone.utc)
    dt = datetime.fromisoformat(at)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _image_url(image_path: str) -> tuple[str | None, str | None]:
    """(image_url, overlay_url) for a surface_profiles.parquet image_path -- null
    for whichever file isn't actually present on disk, never a broken link."""
    filename = Path(image_path).name
    image_url = f"/static/images/{filename}" if (STATIC_IMAGES_DIR / filename).exists() else None

    overlay_filename = f"{Path(filename).stem}_overlay.png"
    overlay_url = f"/static/overlays/{overlay_filename}" if (STATIC_OVERLAYS_DIR / overlay_filename).exists() else None

    return image_url, overlay_url


@router.get("/points", response_model=SamplePointsResponse)
async def list_points(
    request: Request,
    at: Optional[str] = Query(None, description="ISO timestamp; defaults to now"),
    settings: Settings = Depends(get_settings),
) -> SamplePointsResponse:
    """Every sample point with a surface profile (i.e. a real street photo),
    as GeoJSON -- the frontend's clickable map layer. Full detail (photo,
    overlay, interventions, ...) comes from GET /points/{point_id}.
    """
    aoi_bbox = get_aoi_bbox(request)
    surface_df = get_surface_profiles_df(request)
    observed_at = _parse_at(at)
    grid = get_grid_for_timestamp(aoi_bbox, observed_at, settings.GRID_GRANULARITY_M, settings.ALLOW_FIXTURE_DATA)
    openmeteo_client = get_openmeteo_client(request)
    context = get_weather_context(aoi_bbox, grid.observed_at, openmeteo_client)

    points_only = surface_df[["point_id", "lat", "lon"]].reset_index(drop=True)
    temp_joined = attach_temps_to_points(points_only, grid)
    wbgt_arr = _vectorized_wbgt_shade(temp_joined["temp_c"].fillna(0.0).to_numpy(), context["relative_humidity"])

    features = []
    for i, row in temp_joined.iterrows():
        risk_band = classify(float(wbgt_arr[i]), ROUTE_WORK_INTENSITY) if pd.notna(row["temp_c"]) else None
        features.append(
            SamplePointFeature(
                geometry={"type": "Point", "coordinates": [float(row["lon"]), float(row["lat"])]},
                properties=SamplePointProperties(point_id=row["point_id"], risk_band=risk_band, has_photo=True),
            )
        )

    return SamplePointsResponse(
        features=features,
        data_source=grid.source.value.lower(),
        observed_at=grid.observed_at,
        aoi=aoi_bbox,
    )


@router.get("/points/{point_id}", response_model=PointProfileResponse)
async def get_point(point_id: str, request: Request, settings: Settings = Depends(get_settings)) -> PointProfileResponse:
    surface_df = get_surface_profiles_df(request)
    matches = surface_df[surface_df["point_id"] == point_id]
    if matches.empty:
        raise UnknownPointIdError(f"Unknown point_id {point_id!r} -- no surface profile on record")
    row = matches.iloc[0]

    aoi_bbox = get_aoi_bbox(request)
    now = datetime.now(timezone.utc)
    grid = get_grid_for_timestamp(aoi_bbox, now, settings.GRID_GRANULARITY_M, settings.ALLOW_FIXTURE_DATA)
    openmeteo_client = get_openmeteo_client(request)
    context = get_weather_context(aoi_bbox, grid.observed_at, openmeteo_client)

    lat, lon = float(row["lat"]), float(row["lon"])
    temp_c = interpolate_idw(lat, lon, grid)
    base_wbgt = wbgt_shade(temp_c, context["relative_humidity"]).value
    wbgt_c = surface_adjusted_wbgt(base_wbgt, float(row["thermal_load_score"])).value
    risk_band = classify(wbgt_c, ROUTE_WORK_INTENSITY)

    profile = surface_profile_dict_from_row(row)
    interventions = rank_interventions(profile, context, aoi_bbox)
    image_url, overlay_url = _image_url(str(row["image_path"]))

    return PointProfileResponse(
        point_id=point_id,
        lat=lat,
        lon=lon,
        buckets=BucketFractions(**{name: float(row[name]) for name in BUCKET_NAMES}),
        metrics=SurfaceMetrics(
            impervious_fraction=float(row["impervious_fraction"]),
            green_view_index=float(row["green_view_index"]),
            sky_view_factor_proxy=float(row["sky_view_factor_proxy"]),
            thermal_load_score=float(row["thermal_load_score"]),
        ),
        temp_c=float(temp_c),
        wbgt_c=float(wbgt_c),
        risk_band=risk_band,
        interventions=[DeltaTResultSchema(**r.__dict__) for r in interventions],
        image_url=image_url,
        overlay_url=overlay_url,
        data_source=grid.source.value.lower(),
        observed_at=grid.observed_at,
        aoi=aoi_bbox,
    )
