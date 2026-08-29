"""Tolerant adapter: raw FortyGuard heatmap JSON -> normalized TemperatureGrid.

Confirmed against the real API on 2026-08-29: a heatmap response is a GeoJSON
FeatureCollection at data.result.map_data, one Polygon feature per tile, with
average_temperature (+ min/max_temperature) in `properties`. That shape is
handled directly by _try_geojson_polygons below (tile centroid -> point).

The original flat lat/lon/temperature-record search (arbitrary key spelling,
arbitrary container nesting) is kept as a fallback -- it's still exercised by
tests using synthetic fixtures in an older assumed shape, and costs nothing
to leave in place as a defensive fallback for any other payload shape.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from app.grid.schema import GridSource, TemperatureCell, TemperatureGrid, cell_id_for

logger = logging.getLogger(__name__)

# VERIFY AGAINST DOCS -- every plausible key spelling FortyGuard might use for
# each field. Order matters only as a first-match preference when a record
# happens to carry more than one.
LAT_KEYS = ["lat", "latitude", "y"]  # VERIFY AGAINST DOCS
LON_KEYS = ["lon", "lng", "long", "longitude", "x"]  # VERIFY AGAINST DOCS
TEMP_KEYS = ["temp", "temperature", "temp_c", "value", "t"]  # VERIFY AGAINST DOCS

# VERIFY AGAINST DOCS -- plausible container keys the record list might be
# nested under, searched in this preference order before falling back to a
# blind recursive scan of the whole payload.
CONTAINER_KEYS = ["data", "result", "results", "cells", "tiles", "points", "grid", "heatmap"]  # VERIFY AGAINST DOCS

# Confirmed real field names for the GeoJSON tile shape (properties.*).
GEOJSON_TEMP_KEYS = ["average_temperature", "temperature", "temp_c", "value"]

MAX_PRETTY_PRINT_CHARS = 2000
_MAX_SEARCH_DEPTH = 8


def _polygon_centroid(coordinates: list) -> Optional[tuple[float, float]]:
    """(lat, lon) centroid of a GeoJSON Polygon's exterior ring.

    coordinates[0] is the exterior ring: a list of [lon, lat] pairs, first
    and last identical (closed ring) -- a plain vertex average is a fine
    approximation for the small, roughly-square tiles FortyGuard returns.
    """
    if not coordinates or not isinstance(coordinates, list):
        return None
    ring = coordinates[0]
    if not isinstance(ring, list) or not ring:
        return None
    lons = [pt[0] for pt in ring if isinstance(pt, list) and len(pt) >= 2]
    lats = [pt[1] for pt in ring if isinstance(pt, list) and len(pt) >= 2]
    if not lons or not lats:
        return None
    return sum(lats) / len(lats), sum(lons) / len(lons)


def _find_geojson_feature_collection(node: Any, _depth: int = 0) -> Optional[dict]:
    """Recursively search node for a {"type": "FeatureCollection", "features": [...]}."""
    if _depth > _MAX_SEARCH_DEPTH:
        return None
    if isinstance(node, dict):
        if node.get("type") == "FeatureCollection" and isinstance(node.get("features"), list):
            return node
        for value in node.values():
            found = _find_geojson_feature_collection(value, _depth + 1)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_geojson_feature_collection(item, _depth + 1)
            if found is not None:
                return found
    return None


def _try_geojson_polygons(raw: Any) -> Optional[list[TemperatureCell]]:
    collection = _find_geojson_feature_collection(raw)
    if collection is None:
        return None

    cells: list[TemperatureCell] = []
    for feature in collection["features"]:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        if geometry.get("type") != "Polygon":
            continue
        centroid = _polygon_centroid(geometry.get("coordinates"))
        temp = _first_present(properties, GEOJSON_TEMP_KEYS)
        if centroid is None or temp is None:
            continue
        lat, lon = centroid
        cells.append(TemperatureCell(lat=lat, lon=lon, temp_c=float(temp), cell_id=cell_id_for(lat, lon)))
    return cells  # possibly [] -- a real FeatureCollection with zero tiles (no coverage), not "not found"


class SchemaMismatchError(Exception):
    """Raised when no lat/lon/temperature records list could be found in the payload.

    Carries the payload's top-level keys and a truncated pretty-print so the
    key-spelling / container-key lists above can be corrected in one pass
    instead of guessing blind.
    """

    def __init__(self, raw: Any):
        self.top_level_keys = list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__
        pretty = json.dumps(raw, indent=2, default=str)
        self.payload_preview = pretty[:MAX_PRETTY_PRINT_CHARS]
        if len(pretty) > MAX_PRETTY_PRINT_CHARS:
            self.payload_preview += f"\n... ({len(pretty) - MAX_PRETTY_PRINT_CHARS} more chars truncated)"
        super().__init__(
            "Could not find a lat/lon/temperature records list in the FortyGuard response.\n"
            f"Top-level keys: {self.top_level_keys}\n"
            f"Payload preview:\n{self.payload_preview}"
        )


def _first_present(record: dict, keys: list[str]) -> Optional[Any]:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _looks_like_cell_record(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    return (
        _first_present(item, LAT_KEYS) is not None
        and _first_present(item, LON_KEYS) is not None
        and _first_present(item, TEMP_KEYS) is not None
    )


def _find_records_list(node: Any, _depth: int = 0) -> Optional[list]:
    """Recursively search node for a non-empty list whose items look like cell records."""
    if _depth > _MAX_SEARCH_DEPTH:
        return None

    if isinstance(node, list):
        if node and all(_looks_like_cell_record(item) for item in node):
            return node
        for item in node:
            found = _find_records_list(item, _depth + 1)
            if found is not None:
                return found
        return None

    if isinstance(node, dict):
        for key in CONTAINER_KEYS:
            if key in node:
                found = _find_records_list(node[key], _depth + 1)
                if found is not None:
                    return found
        for value in node.values():
            found = _find_records_list(value, _depth + 1)
            if found is not None:
                return found
        return None

    return None


def parse_heatmap_response(
    raw: dict,
    *,
    bbox: str,
    granularity_m: int,
    observed_at: datetime,
    api_activity_id: Optional[str] = None,
) -> TemperatureGrid:
    """Parse a raw FortyGuard heatmap payload into a normalized TemperatureGrid.

    bbox/granularity_m/observed_at/api_activity_id come from the request side
    (what we asked FortyGuard for), since the response shape isn't confirmed
    to reliably echo them back. Raises SchemaMismatchError if no records list
    with lat/lon/temperature fields can be found anywhere in the payload.
    """
    cells = _try_geojson_polygons(raw)

    if cells is None:
        records = _find_records_list(raw)
        if not records:
            raise SchemaMismatchError(raw)

        cells = []
        for record in records:
            lat = _first_present(record, LAT_KEYS)
            lon = _first_present(record, LON_KEYS)
            temp = _first_present(record, TEMP_KEYS)
            if lat is None or lon is None or temp is None:
                continue
            lat, lon, temp = float(lat), float(lon), float(temp)
            cells.append(TemperatureCell(lat=lat, lon=lon, temp_c=temp, cell_id=cell_id_for(lat, lon)))

    if not cells:
        raise SchemaMismatchError(raw)

    return TemperatureGrid(
        cells=cells,
        bbox=bbox,
        granularity_m=granularity_m,
        observed_at=observed_at,
        source=GridSource.LIVE,
        api_activity_id=api_activity_id,
    )
