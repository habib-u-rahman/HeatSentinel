"""Shared FastAPI dependencies: accessors for state preloaded once at startup
(see app.main's lifespan), plus small domain exceptions and helpers reused
across route modules.

Domain exceptions here are mapped to clean 4xx/503 JSON by exception handlers
registered in app.main -- route handlers just call these helpers and let
exceptions propagate; nothing here returns an HTTP response directly.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import networkx as nx
import pandas as pd
from fastapi import Request

from app.grid import store as grid_store
from app.grid.schema import TemperatureGrid
from app.heat.wbgt import DEFAULT_RH_PCT, DEFAULT_SOLAR_WM2, DEFAULT_WIND_MS
from app.ingest.openmeteo import OpenMeteoClient
from app.ml.features import BUCKET_NAMES

logger = logging.getLogger(__name__)


# --- domain exceptions (mapped to HTTP status codes in app.main) ---------------


class OutOfAOIError(Exception):
    """A requested lat/lon falls outside the configured AOI bbox."""


class NoGridAvailableError(Exception):
    """No temperature grid (stored LIVE or generated FIXTURE) is available for a request."""


class UnknownPointIdError(Exception):
    """A requested point_id has no surface profile on record."""


class UnknownInterventionError(Exception):
    """A requested intervention_name isn't in app.ml.intervention.INTERVENTIONS."""


class ServiceNotReadyError(Exception):
    """Server-side data/model that should have been preloaded at startup is missing."""


# --- app.state accessors ---------------------------------------------------------


def get_graph(request: Request) -> nx.MultiDiGraph:
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise ServiceNotReadyError("Walk graph is not loaded -- run scripts/build_graph.py, then restart the API.")
    return graph


def get_model_bundle(request: Request) -> dict:
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise ServiceNotReadyError("RF model is not loaded -- run scripts/train_rf.py, then restart the API.")
    return bundle


def get_sample_points_df(request: Request) -> pd.DataFrame:
    df = getattr(request.app.state, "sample_points_df", None)
    if df is None:
        raise ServiceNotReadyError("sample_points.parquet not found -- run scripts/build_sample_points.py, then restart the API.")
    return df


def get_surface_profiles_df(request: Request) -> pd.DataFrame:
    df = getattr(request.app.state, "surface_profiles_df", None)
    if df is None:
        raise ServiceNotReadyError("surface_profiles.parquet not found -- run scripts/analyze_images.py, then restart the API.")
    return df


def get_openmeteo_client(request: Request) -> OpenMeteoClient:
    return request.app.state.openmeteo_client


def get_aoi_bbox(request: Request) -> str:
    """The currently active AOI bbox -- the Rawalpindi `.env` default until a
    build_aoi.py run swaps it out (see app/api/routes/aoi.py). Read from
    app.state, never from Settings.AOI_BBOX directly, so a completed AOI
    build actually takes effect without a process restart.
    """
    return request.app.state.aoi_bbox


def get_city_name(request: Request) -> str:
    return request.app.state.city_name


# --- shared helpers ---------------------------------------------------------------


def assert_within_aoi(lat: float, lon: float, bbox: str) -> None:
    min_lon, min_lat, max_lon, max_lat = (float(x) for x in bbox.split(","))
    if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
        raise OutOfAOIError(f"({lat}, {lon}) is outside the configured AOI bbox {bbox}")


def get_grid_for_timestamp(bbox: str, at: datetime, granularity_m: int, allow_fixture: bool) -> TemperatureGrid:
    """A TemperatureGrid for `at`: a stored LIVE grid if one exists at that exact
    timestamp, else a freshly-generated FIXTURE grid (unless allow_fixture is
    False, in which case that's a clear error rather than a silent substitution).
    """
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)

    stored_path = grid_store._grid_path(grid_store.DEFAULT_BASE_DIR, at)
    if stored_path.exists():
        return grid_store.load_grid(stored_path)

    if not allow_fixture:
        raise NoGridAvailableError(
            f"No stored LIVE grid available at {at.isoformat()} for bbox={bbox}, and fixture data is "
            "disabled (ALLOW_FIXTURE_DATA=False)."
        )

    from app.fortyguard.fixtures import generate_grid  # local import: keeps fixtures optional at module load

    return generate_grid(bbox, granularity_m, at, seed=0)


_weather_context_cache: dict[tuple, dict] = {}


def get_weather_context(bbox: str, observed_at: datetime, openmeteo_client: OpenMeteoClient) -> dict:
    """hour/solar_wm2/wind_ms/relative_humidity at the AOI centre for observed_at,
    for feeding app.ml.features/app.ml.intervention. Cached in-process by
    (bbox, observed_at) -- same rationale as app.routing.costs._fetch_aoi_humidity:
    a flaky fetch must not give different requests for the same grid snapshot
    different weather.
    """
    cache_key = (bbox, observed_at)
    if cache_key in _weather_context_cache:
        return _weather_context_cache[cache_key]

    min_lon, min_lat, max_lon, max_lat = (float(x) for x in bbox.split(","))
    center_lat, center_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2

    try:
        date_str = observed_at.strftime("%Y-%m-%d")
        hourly = openmeteo_client.fetch_hourly(center_lat, center_lon, date_str, date_str)
        if hourly.empty:
            raise ValueError("Open-Meteo returned no hourly rows")
        times = pd.to_datetime(hourly["time"])
        target = pd.Timestamp(observed_at)
        if target.tzinfo is not None:
            target = target.tz_convert(None)
        idx = (times - target).abs().idxmin()
        row = hourly.loc[idx]
        context = {
            "hour": float(observed_at.hour),
            "solar_wm2": float(row["shortwave_radiation"]),
            "wind_ms": float(row["wind_speed_10m"]),
            "relative_humidity": float(row["relative_humidity_2m"]),
        }
    except Exception:
        logger.warning("Open-Meteo weather fetch failed for AOI centre -- falling back to documented defaults")
        context = {
            "hour": float(observed_at.hour),
            "solar_wm2": DEFAULT_SOLAR_WM2,
            "wind_ms": DEFAULT_WIND_MS,
            "relative_humidity": DEFAULT_RH_PCT,
        }

    _weather_context_cache[cache_key] = context
    return context


def surface_profile_dict_from_row(row: pd.Series) -> dict:
    """A row from surface_profiles.parquet -> the dict shape app.ml.intervention expects."""
    profile = {name: float(row[name]) for name in BUCKET_NAMES}
    profile.update(
        {
            "person_count": float(row["person_count"]),
            "bicycle_count": float(row["bicycle_count"]),
            "car_count": float(row["car_count"]),
            "motorcycle_count": float(row["motorcycle_count"]),
            "bus_count": float(row["bus_count"]),
            "truck_count": float(row["truck_count"]),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
        }
    )
    return profile
