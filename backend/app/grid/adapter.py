"""Tolerant adapter: raw FortyGuard heatmap JSON -> normalized TemperatureGrid.

We do NOT have a confirmed FortyGuard response schema yet (see the
"VERIFY AGAINST DOCS" markers in app/fortyguard/client.py). Rather than guess
a shape and stall on every field-name mismatch, this parser searches the
payload for a list of records carrying lat/lon/temperature under any of
several plausible key spellings, nested under any of several plausible
container keys. Once the real schema is confirmed, only the key-spelling and
container-key lists below should need correcting.
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

MAX_PRETTY_PRINT_CHARS = 2000
_MAX_SEARCH_DEPTH = 8


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
    records = _find_records_list(raw)
    if not records:
        raise SchemaMismatchError(raw)

    cells: list[TemperatureCell] = []
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
