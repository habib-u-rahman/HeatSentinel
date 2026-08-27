"""Pydantic request/response models for the HTTP API.

Every response inherits ResponseEnvelope (data_source, observed_at, aoi) --
the frontend renders a visible "SYNTHETIC DATA" banner whenever
data_source == "fixture". This is how we stay honest on stage.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# --- envelope ------------------------------------------------------------------


class ResponseEnvelope(BaseModel):
    data_source: Literal["live", "fixture"] = Field(..., description="Whether this response is backed by real FortyGuard data or synthetic fixtures")
    observed_at: datetime = Field(..., description="UTC timestamp the underlying temperature grid represents")
    aoi: str = Field(..., description="AOI bbox: min_lon,min_lat,max_lon,max_lat")


# --- health -----------------------------------------------------------------------


class HealthResponse(ResponseEnvelope):
    status: str
    graph_loaded: bool
    model_loaded: bool
    n_nodes: int
    n_edges: int
    fortyguard_key_present: bool = Field(..., description="Whether FORTYGUARD_API_KEY is configured -- never the key itself")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "graph_loaded": True,
                "model_loaded": True,
                "n_nodes": 2957,
                "n_edges": 7760,
                "fortyguard_key_present": True,
                "data_source": "fixture",
                "observed_at": "2026-08-24T15:00:00Z",
                "aoi": "73.03,33.58,73.07,33.61",
            }
        }
    )


# --- grid / zones -----------------------------------------------------------------


class GridCellProperties(BaseModel):
    cell_id: str
    temp_c: float
    wbgt_c: float
    risk_band: str


class GridCellFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: dict = Field(..., description='{"type": "Point", "coordinates": [lon, lat]}')
    properties: GridCellProperties


class GridResponse(ResponseEnvelope):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[GridCellFeature]
    n_returned: int
    n_total: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [73.048, 33.595]},
                        "properties": {"cell_id": "a1b2c3d4e5f6a7b8", "temp_c": 34.2, "wbgt_c": 33.6, "risk_band": "DANGER"},
                    }
                ],
                "n_returned": 1292,
                "n_total": 1292,
                "data_source": "fixture",
                "observed_at": "2026-08-24T15:00:00Z",
                "aoi": "73.03,33.58,73.07,33.61",
            }
        }
    )


class ZoneProperties(BaseModel):
    zone_id: str
    mean_wbgt_c: float
    max_wbgt_c: float
    risk_band: str = Field(..., description="Classified from max_wbgt_c -- the worst case in the zone, not the average")
    n_cells: int


class ZoneFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: dict = Field(..., description='{"type": "Polygon", "coordinates": [[[lon, lat], ...]]}')
    properties: ZoneProperties


class ZonesResponse(ResponseEnvelope):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[ZoneFeature]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [[[73.03, 33.58], [73.035, 33.58], [73.035, 33.5838], [73.03, 33.5838], [73.03, 33.58]]]},
                        "properties": {"zone_id": "r0_c0", "mean_wbgt_c": 33.1, "max_wbgt_c": 34.9, "risk_band": "DANGER", "n_cells": 21},
                    }
                ],
                "data_source": "fixture",
                "observed_at": "2026-08-24T15:00:00Z",
                "aoi": "73.03,33.58,73.07,33.61",
            }
        }
    )


# --- route -------------------------------------------------------------------------


class LatLon(BaseModel):
    lat: float
    lon: float


class RouteRequestBody(BaseModel):
    start: LatLon
    end: LatLon
    lambda_heat: float = Field(0.5, ge=0.0, le=1.0, description="0 = shortest distance, 1 = coolest (minimum heat dose)")
    family: bool = Field(False, description="If true, return the SHORTEST/BALANCED/COOLEST family + comparison instead of one route")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"start": {"lat": 33.5820, "lon": 73.0330}, "end": {"lat": 33.6080, "lon": 73.0670}, "lambda_heat": 0.5, "family": True}
        }
    )


class RouteResultSchema(BaseModel):
    label: Optional[str] = Field(None, description="SHORTEST / BALANCED / COOLEST when part of a family, else null")
    lambda_heat: float
    geojson: dict = Field(..., description="GeoJSON LineString Feature; coordinates are [lon, lat]")
    total_distance_m: float
    total_duration_s: float
    total_heat_dose_degC_s: float
    mean_wbgt_c: float
    max_wbgt_c: float
    peak_risk_band: str
    n_edges_orphaned: int


class RouteComparisonSchema(BaseModel):
    extra_distance_m: float
    extra_distance_pct: float
    extra_duration_s: float
    dose_reduction_degC_s: float
    dose_reduction_pct: float
    mean_wbgt_delta_c: float
    same_path: bool
    summary: str


class RouteResponse(ResponseEnvelope):
    routes: list[RouteResultSchema]
    comparison: Optional[RouteComparisonSchema] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "routes": [
                    {
                        "label": "SHORTEST",
                        "lambda_heat": 0.0,
                        "geojson": {"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": [[73.033, 33.582], [73.067, 33.608]]}},
                        "total_distance_m": 5243.0,
                        "total_duration_s": 3884.0,
                        "total_heat_dose_degC_s": 154199.0,
                        "mean_wbgt_c": 39.7,
                        "max_wbgt_c": 41.8,
                        "peak_risk_band": "CRITICAL",
                        "n_edges_orphaned": 0,
                    }
                ],
                "comparison": None,
                "data_source": "fixture",
                "observed_at": "2026-08-24T15:00:00Z",
                "aoi": "73.03,33.58,73.07,33.61",
            }
        }
    )


# --- points / interventions ---------------------------------------------------------


class BucketFractions(BaseModel):
    road: float
    sidewalk: float
    built: float
    vegetation: float
    sky: float
    other: float


class SurfaceMetrics(BaseModel):
    impervious_fraction: float
    green_view_index: float
    sky_view_factor_proxy: float
    thermal_load_score: float


class DeltaTResultSchema(BaseModel):
    intervention: str
    value: float = Field(..., description="Predicted temperature change in C; negative = cooling")
    confidence_interval: tuple[float, float]
    method: Literal["model_prediction", "literature_offset"]
    n_training_rows: int
    data_source: Literal["live", "fixture"]
    detail: str = ""


class PointProfileResponse(ResponseEnvelope):
    point_id: str
    lat: float
    lon: float
    buckets: BucketFractions
    metrics: SurfaceMetrics
    temp_c: float
    wbgt_c: float
    risk_band: str
    interventions: list[DeltaTResultSchema]
    image_url: Optional[str] = Field(None, description="/static/images/... ; null if the source photo isn't on disk")
    overlay_url: Optional[str] = Field(None, description="/static/overlays/... ; null if the segmentation overlay hasn't been generated")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "point_id": "c17cd19a87e1a1a6",
                "lat": 33.5951,
                "lon": 73.0488,
                "buckets": {"road": 0.15, "sidewalk": 0.01, "built": 0.29, "vegetation": 0.05, "sky": 0.32, "other": 0.18},
                "metrics": {"impervious_fraction": 0.45, "green_view_index": 0.05, "sky_view_factor_proxy": 0.32, "thermal_load_score": 0.58},
                "temp_c": 34.1,
                "wbgt_c": 33.5,
                "image_url": "/static/images/c17cd19a87e1a1a6_1065964548501710.jpg",
                "overlay_url": "/static/overlays/c17cd19a87e1a1a6_1065964548501710_overlay.png",
                "risk_band": "DANGER",
                "interventions": [
                    {
                        "intervention": "pocket_park",
                        "value": -0.99,
                        "confidence_interval": [-2.31, 0.28],
                        "method": "model_prediction",
                        "n_training_rows": 7560,
                        "data_source": "fixture",
                        "detail": "converting a small paved lot to a pocket park",
                    }
                ],
                "data_source": "fixture",
                "observed_at": "2026-08-24T15:00:00Z",
                "aoi": "73.03,33.58,73.07,33.61",
            }
        }
    )


class InterventionRequestBody(BaseModel):
    point_id: str
    intervention_name: str

    model_config = ConfigDict(json_schema_extra={"example": {"point_id": "c17cd19a87e1a1a6", "intervention_name": "pocket_park"}})


class InterventionResponse(ResponseEnvelope):
    result: DeltaTResultSchema


class InterventionCatalogItem(BaseModel):
    name: str
    description: str
    deltas: Optional[dict[str, float]] = Field(None, description="Bucket-fraction deltas; null for albedo-only interventions")
    literature_offset_c: Optional[float] = Field(None, description="Literature-derived point estimate; null for segmentation-visible interventions")
    method: Literal["model_prediction", "literature_offset"]


class InterventionsCatalogResponse(ResponseEnvelope):
    interventions: list[InterventionCatalogItem]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "interventions": [
                    {
                        "name": "cool_roof",
                        "description": "cool/reflective roofing raises albedo ~0.15-0.30; ~1-2C local air-temp reduction reported in the literature",
                        "deltas": None,
                        "literature_offset_c": -1.5,
                        "method": "literature_offset",
                    }
                ],
                "data_source": "fixture",
                "observed_at": "2026-08-24T15:00:00Z",
                "aoi": "73.03,33.58,73.07,33.61",
            }
        }
    )


# --- sample points list (for map markers -- detail comes from GET /points/{id}) ---


class SamplePointProperties(BaseModel):
    point_id: str
    risk_band: Optional[str] = Field(None, description="null if farther than the match radius from every grid cell")
    has_photo: bool = True


class SamplePointFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: dict = Field(..., description='{"type": "Point", "coordinates": [lon, lat]}')
    properties: SamplePointProperties


class SamplePointsResponse(ResponseEnvelope):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[SamplePointFeature]


# --- vulnerable POIs (documented population PROXY, never real census data) --------


class VulnerablePoiProperties(BaseModel):
    poi_id: str
    name: Optional[str] = None
    category: str
    vulnerability_group: str
    weight: float
    risk_band: Optional[str] = Field(None, description="null if this POI is farther than the match radius from every grid cell")
    exposure_score: Optional[float] = None


class VulnerablePoiFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: dict = Field(..., description='{"type": "Point", "coordinates": [lon, lat]}')
    properties: VulnerablePoiProperties


class VulnerableResponse(ResponseEnvelope):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[VulnerablePoiFeature]
    population_proxy: bool = Field(..., description="Always true -- these are OSM POIs, never real population/census data")
    proxy_note: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [73.048, 33.595]},
                        "properties": {
                            "poi_id": "node/123456",
                            "name": "Example Primary School",
                            "category": "school",
                            "vulnerability_group": "children",
                            "weight": 0.8,
                            "risk_band": "DANGER",
                            "exposure_score": 2.4,
                        },
                    }
                ],
                "population_proxy": True,
                "proxy_note": "No real population or census dataset exists for this AOI. These are OpenStreetMap points of interest used as a documented PROXY for where vulnerable people are likely to be -- NOT measured population or census data.",
                "data_source": "fixture",
                "observed_at": "2026-08-24T15:00:00Z",
                "aoi": "73.03,33.58,73.07,33.61",
            }
        }
    )


# --- alerts ------------------------------------------------------------------------


class AlertSchema(BaseModel):
    alert_id: str
    severity: str
    category: Literal["zone_critical", "poi_at_risk", "rapid_rise"]
    message: str
    lat: float
    lon: float
    poi_id: Optional[str] = None
    triggered_at: datetime
    wbgt_c: float
    risk_band: str
    data_source: Literal["live", "fixture"]
    population_proxy: bool = Field(..., description="Always true -- alerts blend zone and POI-proxy data")
    proxy_note: str
    exposure_score: Optional[float] = None


class AlertsResponse(ResponseEnvelope):
    alerts: list[AlertSchema]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "alerts": [
                    {
                        "alert_id": "a1b2c3d4e5f6a7b8",
                        "severity": "CRITICAL",
                        "category": "poi_at_risk",
                        "message": "3 schools in Zone r2_c5 are in CRITICAL heat stress (WBGT 33.2C)",
                        "lat": 33.595,
                        "lon": 73.048,
                        "poi_id": None,
                        "triggered_at": "2026-08-24T15:00:00Z",
                        "wbgt_c": 33.2,
                        "risk_band": "CRITICAL",
                        "data_source": "fixture",
                        "population_proxy": True,
                        "proxy_note": "No real population or census dataset exists for this AOI. These are OpenStreetMap points of interest used as a documented PROXY for where vulnerable people are likely to be -- NOT measured population or census data.",
                        "exposure_score": 3.1,
                    }
                ],
                "data_source": "fixture",
                "observed_at": "2026-08-24T15:00:00Z",
                "aoi": "73.03,33.58,73.07,33.61",
            }
        }
    )


# --- AOI build (location picker) ------------------------------------------------
#
# Not ResponseEnvelope subclasses: these describe BUILD ORCHESTRATION state, not
# a heat-data snapshot, so a "data_source: live|fixture" field would misrepresent
# rather than support the honesty framing above -- there's no temperature grid
# involved in "is the street network downloaded yet".


class AoiBuildRequest(BaseModel):
    query: Optional[str] = Field(None, description="Free-text place name, geocoded via Nominatim")
    bbox: Optional[str] = Field(None, description="Explicit min_lon,min_lat,max_lon,max_lat -- bypasses geocoding entirely")
    radius_km: float = Field(1.0, gt=0, description="Radius around the geocoded point; ignored if bbox is given")
    n_points: int = Field(60, ge=30, le=80, description="Sample points to build (one street photo + CV pass each)")


class AoiBuildQueuedResponse(BaseModel):
    job_id: str
    status: Literal["queued"]
    bbox: str
    city_name: str
    area_km2: float


class AoiBuildProgress(BaseModel):
    current: int
    total: int


class AoiBuildResultSchema(BaseModel):
    aoi_bbox: str
    city_name: str
    n_sample_points: int
    n_with_imagery: int
    mapillary_coverage_pct: float
    n_pois: int
    degraded: bool = Field(..., description="True when imagery coverage was real but sparse (<25%) -- shown as a persistent warning, never hidden")


class AoiBuildStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    stage: Optional[str] = Field(None, description="graph | sampling | imagery | pois")
    message: Optional[str] = Field(None, description="Human-readable progress, built from real counts -- never a fabricated animation")
    progress: Optional[AoiBuildProgress] = None
    result: Optional[AoiBuildResultSchema] = None
    error: Optional[str] = None


class AoiCurrentResponse(BaseModel):
    city_name: str
    aoi_bbox: str
    built_at: Optional[datetime] = Field(None, description="None if this is still the .env default (Rawalpindi), never overwritten by a build")
    n_sample_points: Optional[int] = None
    n_with_imagery: Optional[int] = None
    mapillary_coverage_pct: Optional[float] = None
    n_pois: Optional[int] = None
    degraded: bool = False


# --- errors --------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    detail: str
