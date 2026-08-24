"""Aggregates TemperatureGrid cells into a coarse AOI grid of risk zones --
shared by GET /api/zones and app.alerts.engine, which needs the same per-zone
WBGT stats to detect zone_critical / rapid_rise conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.grid.schema import TemperatureGrid
from app.heat.thresholds import classify
from app.routing.costs import _vectorized_wbgt_shade
from app.routing.router import ROUTE_WORK_INTENSITY
from app.sampling.points import _cell_id

ZONE_GRID_DIM = 8  # an 8x8 AOI grid -- coarse enough for a legible choropleth, finer than the 4x4 ML spatial-CV blocks


@dataclass
class Zone:
    zone_id: str
    mean_wbgt_c: float
    max_wbgt_c: float
    risk_band: str  # classified from max_wbgt_c -- the worst case in the zone, not the average
    n_cells: int
    centroid_lat: float
    centroid_lon: float
    bounds: tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat


def compute_zones(grid: TemperatureGrid, bbox: str, rh_pct: float, zone_grid_dim: int = ZONE_GRID_DIM) -> list[Zone]:
    lats, lons, temps = grid.to_arrays()
    if lats.size == 0:
        return []

    wbgt_arr = _vectorized_wbgt_shade(temps, rh_pct)

    min_lon, min_lat, max_lon, max_lat = (float(x) for x in bbox.split(","))
    lon_step = (max_lon - min_lon) / zone_grid_dim
    lat_step = (max_lat - min_lat) / zone_grid_dim

    zone_wbgts: dict[str, list[float]] = {}
    for lat, lon, wbgt in zip(lats, lons, wbgt_arr):
        zone_id = _cell_id(float(lat), float(lon), (min_lon, min_lat, max_lon, max_lat), zone_grid_dim)
        zone_wbgts.setdefault(zone_id, []).append(float(wbgt))

    zones: list[Zone] = []
    for zone_id, values in zone_wbgts.items():
        row_idx, col_idx = (int(part[1:]) for part in zone_id.split("_"))
        z_min_lon, z_min_lat = min_lon + col_idx * lon_step, min_lat + row_idx * lat_step
        z_max_lon, z_max_lat = z_min_lon + lon_step, z_min_lat + lat_step
        max_wbgt = float(np.max(values))

        zones.append(
            Zone(
                zone_id=zone_id,
                mean_wbgt_c=float(np.mean(values)),
                max_wbgt_c=max_wbgt,
                risk_band=classify(max_wbgt, ROUTE_WORK_INTENSITY),
                n_cells=len(values),
                centroid_lat=(z_min_lat + z_max_lat) / 2,
                centroid_lon=(z_min_lon + z_max_lon) / 2,
                bounds=(z_min_lon, z_min_lat, z_max_lon, z_max_lat),
            )
        )
    return zones
