from __future__ import annotations

import math
from typing import Sequence, Union

EARTH_RADIUS_M = 6_371_000.0
METERS_PER_DEGREE_LAT = 111_320.0

BBox = Union[str, Sequence[float]]


def _parse_bbox(bbox: BBox) -> tuple[float, float, float, float]:
    """Parse "min_lon,min_lat,max_lon,max_lat" (string or sequence) into floats."""
    if isinstance(bbox, str):
        parts = [float(x) for x in bbox.split(",")]
    else:
        parts = [float(x) for x in bbox]
    if len(parts) != 4:
        raise ValueError(f"bbox must have 4 values (min_lon,min_lat,max_lon,max_lat), got {parts}")
    return parts[0], parts[1], parts[2], parts[3]


def bbox_to_polygon(bbox: BBox) -> dict:
    """Convert a bbox into a closed GeoJSON Polygon ring (lon, lat order)."""
    min_lon, min_lat, max_lon, max_lat = _parse_bbox(bbox)
    ring = [
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


def haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in metres between (lat, lon) points a and b."""
    lat1, lon1 = a
    lat2, lon2 = b
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, h)))


def bbox_area_km2(bbox: BBox) -> float:
    """Approximate bbox area in km2 (flat rectangle: width at mid-lat x height)."""
    min_lon, min_lat, max_lon, max_lat = _parse_bbox(bbox)
    mid_lat = (min_lat + max_lat) / 2
    width_m = haversine((mid_lat, min_lon), (mid_lat, max_lon))
    height_m = haversine((min_lat, min_lon), (max_lat, min_lon))
    return (width_m * height_m) / 1_000_000.0


# A full-city walk graph is too slow/memory-heavy to build live during a demo
# (see scripts/build_graph.py, which enforces this same cap for the offline
# CLI path) -- shared here so the on-demand AOI builder enforces one ceiling.
MAX_AREA_KM2 = 12.0


def bbox_around(lat: float, lon: float, radius_m: float) -> str:
    """"min_lon,min_lat,max_lon,max_lat" bbox spanning `radius_m` metres around
    a point -- the string convention every bbox in this app uses. Mirrors
    app.ingest.mapillary._bbox_around's math (that one returns a plain tuple,
    which is what the Mapillary search API wants)."""
    lat_delta = radius_m / METERS_PER_DEGREE_LAT
    lon_delta = radius_m / (METERS_PER_DEGREE_LAT * max(math.cos(math.radians(lat)), 1e-6))
    min_lon, min_lat, max_lon, max_lat = lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta
    return f"{min_lon:.6f},{min_lat:.6f},{max_lon:.6f},{max_lat:.6f}"


def grid_points(bbox: BBox, spacing_m: float) -> list[tuple[float, float]]:
    """Generate a regular (lat, lon) grid covering bbox at ~spacing_m metre spacing."""
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")

    min_lon, min_lat, max_lon, max_lat = _parse_bbox(bbox)
    lat_step_deg = spacing_m / METERS_PER_DEGREE_LAT

    points: list[tuple[float, float]] = []
    lat = min_lat
    while lat <= max_lat:
        lon_step_deg = spacing_m / (METERS_PER_DEGREE_LAT * max(math.cos(math.radians(lat)), 1e-6))
        lon = min_lon
        while lon <= max_lon:
            points.append((lat, lon))
            lon += lon_step_deg
        lat += lat_step_deg
    return points
