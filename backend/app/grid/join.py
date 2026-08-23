"""Grid-to-graph and grid-to-point spatial joins -- the performance-critical piece.

Every function here builds its cKDTree ONCE and queries it with a single
vectorized call over all targets (edges or points), never looping per-edge.
Lat/lon is projected to an equirectangular metre plane first so tree
distances are metric, not degrees.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from app.core.geo import METERS_PER_DEGREE_LAT
from app.grid.schema import TemperatureGrid
from app.routing.graph import EdgeKey

logger = logging.getLogger(__name__)


def _equirect_project(
    lats: np.ndarray, lons: np.ndarray, ref_lat: float, ref_lon: float
) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular (flat-earth) projection to metres centred at (ref_lat, ref_lon).

    Accurate enough for AOIs a few km across -- the same small-area approximation
    already used by app.core.geo (grid_points / bbox_area_km2) elsewhere in this
    codebase, just expressed as a forward projection instead of a distance calc.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    meters_per_deg_lon = METERS_PER_DEGREE_LAT * max(math.cos(math.radians(ref_lat)), 1e-6)
    x_m = (lons - ref_lon) * meters_per_deg_lon
    y_m = (lats - ref_lat) * METERS_PER_DEGREE_LAT
    return x_m, y_m


def _grid_tree(grid: TemperatureGrid) -> tuple[Optional[cKDTree], np.ndarray, float, float]:
    """cKDTree over the grid's cells (projected to metres), plus the temps array and the
    reference lat/lon the projection is centred on (the grid's own centroid)."""
    lats, lons, temps = grid.to_arrays()
    if lats.size == 0:
        return None, temps, 0.0, 0.0

    ref_lat = float(np.mean(lats))
    ref_lon = float(np.mean(lons))
    x_m, y_m = _equirect_project(lats, lons, ref_lat, ref_lon)
    tree = cKDTree(np.column_stack([x_m, y_m]))
    return tree, temps, ref_lat, ref_lon


def attach_temps_to_edges(
    graph: nx.MultiDiGraph, grid: TemperatureGrid, max_dist_m: float = 150.0
) -> dict[EdgeKey, Optional[float]]:
    """Nearest-neighbour temperature for every edge midpoint in graph.

    graph must already be annotated with mid_lat/mid_lon (see
    app.routing.graph._annotate_edges). Edges farther than max_dist_m from
    every grid cell map to None -- never silently to 0.0.
    """
    start = time.monotonic()

    edge_keys: list[EdgeKey] = []
    mid_lats: list[float] = []
    mid_lons: list[float] = []
    for u, v, k, data in graph.edges(keys=True, data=True):
        edge_keys.append((u, v, k))
        mid_lats.append(data["mid_lat"])
        mid_lons.append(data["mid_lon"])

    if not edge_keys:
        logger.warning("edge_temp_join: graph has no edges")
        return {}

    tree, temps, ref_lat, ref_lon = _grid_tree(grid)
    if tree is None:
        logger.warning("edge_temp_join: grid has no cells -- all %d edges orphaned", len(edge_keys))
        return {key: None for key in edge_keys}

    edge_x, edge_y = _equirect_project(np.array(mid_lats), np.array(mid_lons), ref_lat, ref_lon)
    distances, indices = tree.query(np.column_stack([edge_x, edge_y]))  # one vectorized call for ALL edges

    within = distances <= max_dist_m
    result: dict[EdgeKey, Optional[float]] = {
        key: (float(temps[idx]) if matched else None)
        for key, idx, matched in zip(edge_keys, indices, within)
    }

    n_matched = int(within.sum())
    n_orphaned = len(edge_keys) - n_matched
    mean_dist = float(np.mean(distances[within])) if n_matched else float("nan")
    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "edge_temp_join n_edges=%d n_matched=%d n_orphaned=%d mean_dist_m=%.1f elapsed_ms=%.1f",
        len(edge_keys),
        n_matched,
        n_orphaned,
        mean_dist,
        elapsed_ms,
    )
    return result


def attach_temps_to_points(
    points_df: pd.DataFrame, grid: TemperatureGrid, max_dist_m: float = 150.0
) -> pd.DataFrame:
    """Nearest-neighbour temperature for every row of points_df (needs lat/lon columns).

    Returns a copy of points_df with a new temp_c column (NaN where farther
    than max_dist_m from every grid cell). This is what builds the RF/LSTM
    training set.
    """
    start = time.monotonic()
    result_df = points_df.copy()

    if points_df.empty:
        result_df["temp_c"] = pd.Series(dtype=float)
        return result_df

    tree, temps, ref_lat, ref_lon = _grid_tree(grid)
    if tree is None:
        logger.warning("point_temp_join: grid has no cells -- all %d points orphaned", len(points_df))
        result_df["temp_c"] = np.nan
        return result_df

    point_x, point_y = _equirect_project(
        points_df["lat"].to_numpy(), points_df["lon"].to_numpy(), ref_lat, ref_lon
    )
    distances, indices = tree.query(np.column_stack([point_x, point_y]))  # one vectorized call for ALL points

    within = distances <= max_dist_m
    result_df["temp_c"] = np.where(within, temps[indices], np.nan)

    n_matched = int(within.sum())
    n_orphaned = len(points_df) - n_matched
    mean_dist = float(np.mean(distances[within])) if n_matched else float("nan")
    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "point_temp_join n_points=%d n_matched=%d n_orphaned=%d mean_dist_m=%.1f elapsed_ms=%.1f",
        len(points_df),
        n_matched,
        n_orphaned,
        mean_dist,
        elapsed_ms,
    )
    return result_df


def interpolate_idw(
    target_lat: float, target_lon: float, grid: TemperatureGrid, k: int = 4, power: int = 2
) -> float:
    """Inverse-distance-weighted temperature at (target_lat, target_lon) over the k
    nearest grid cells -- smoother than a bare nearest-neighbour lookup."""
    lats, lons, temps = grid.to_arrays()
    if lats.size == 0:
        raise ValueError("Cannot interpolate: grid has no cells")

    k = min(k, lats.size)
    x_m, y_m = _equirect_project(lats, lons, target_lat, target_lon)
    tree = cKDTree(np.column_stack([x_m, y_m]))

    distances, indices = tree.query([[0.0, 0.0]], k=k)
    distances = np.atleast_1d(np.asarray(distances).reshape(-1))
    indices = np.atleast_1d(np.asarray(indices).reshape(-1))

    exact = distances == 0
    if np.any(exact):
        return float(temps[indices[exact][0]])

    weights = 1.0 / np.power(distances, power)
    weights /= weights.sum()
    return float(np.sum(weights * temps[indices]))
