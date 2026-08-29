"""Per-edge heat-dose costing for the walk graph -- the performance-critical
piece routing builds on.

Heat DOSE, not raw temperature, is what routing minimises: exposure is
degrees x time, so a long cool street can beat a short scorching one. Costs
are vectorized with numpy over the whole edge list; nothing here loops in
Python per edge.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.grid.join import attach_temps_to_edges, interpolate_idw
from app.grid.schema import TemperatureGrid
from app.heat.wbgt import DEFAULT_RH_PCT
from app.ingest.openmeteo import OpenMeteoClient
from app.routing.graph import EdgeKey

logger = logging.getLogger(__name__)

WALK_SPEED_MS = 1.35  # ~4.9 km/h, a typical adult walking pace

SURFACE_ADJUSTED_MAX_DELTA_C = 2.0  # passed through to the surface_adjusted_wbgt formula
ORPHAN_PERCENTILE = 75.0  # dose assigned to edges where even IDW interpolation fails

DEFAULT_SAMPLE_POINTS_PATH = Path("../data/raw/sample_points.parquet")
DEFAULT_SURFACE_PROFILES_PATH = Path("../data/raw/surface_profiles.parquet")


def edge_traversal_seconds(length_m: float) -> float:
    return length_m / WALK_SPEED_MS


def edge_heat_dose(edge_temp_c: float, edge_wbgt_c: Optional[float], length_m: float) -> float:
    """Heat dose in degree-seconds: exposure = degrees x time.

    Prefers WBGT (the actual heat-stress metric) when available; falls back
    to raw temperature only if WBGT could not be computed for this edge.
    """
    degrees = edge_wbgt_c if edge_wbgt_c is not None else edge_temp_c
    return degrees * edge_traversal_seconds(length_m)


@dataclass
class EdgeCostResult:
    """Everything needed to route, explain a route, and reproduce a cost run."""

    costs: dict[EdgeKey, float]
    lambda_heat: float

    # normalisation constants -- costs are reproducible/explainable from these
    dist_min_m: float
    dist_max_m: float
    dose_min_degC_s: float
    dose_max_degC_s: float

    n_edges: int
    n_orphaned: int
    n_orphaned_idw_filled: int
    n_orphaned_percentile_filled: int
    wbgt_method_counts: dict[str, int]

    edge_length_m: dict[EdgeKey, float]
    edge_temp_c: dict[EdgeKey, float]
    edge_wbgt_c: dict[EdgeKey, float]
    edge_dose_degC_s: dict[EdgeKey, float]
    edge_wbgt_method: dict[EdgeKey, str] = field(default_factory=dict)
    edge_orphaned_penalty: dict[EdgeKey, bool] = field(default_factory=dict)


def _vectorized_wbgt_shade(temp_c: np.ndarray, rh_pct: float) -> np.ndarray:
    """Vectorized mirror of app.heat.wbgt.wbgt_shade (same Australian BoM formula/
    source -- WBGT = 0.567*Ta + 0.393*e + 3.94). Reimplemented with numpy because
    this module must vectorize over ~15k edges rather than loop per edge; keep in
    sync with app.heat.wbgt.wbgt_shade (tests/test_routing.py asserts numeric
    parity against the scalar version).
    """
    es = 6.112 * np.exp((17.62 * temp_c) / (243.12 + temp_c))
    e = es * rh_pct / 100.0
    return 0.567 * temp_c + 0.393 * e + 3.94


def _vectorized_surface_adjusted(
    base_wbgt: np.ndarray, thermal_load_score: np.ndarray, max_delta_c: float = SURFACE_ADJUSTED_MAX_DELTA_C
) -> np.ndarray:
    """Vectorized mirror of app.heat.wbgt.surface_adjusted_wbgt."""
    return base_wbgt + (thermal_load_score - 0.5) * 2 * max_delta_c


def _edge_key_str(u: int, v: int, k: int) -> str:
    """Matches app.sampling.points' nearest_edge_key string format exactly."""
    return f"{u}_{v}_{k}"


def _load_thermal_load_by_edge(
    sample_points_path: Path, surface_profiles_path: Path
) -> dict[str, float]:
    """edge_key_str -> thermal_load_score, for edges that are some sample point's
    nearest edge AND have a surface profile. Returns {} (never raises) if either
    file is missing -- callers fall back to wbgt_shade for every edge."""
    if not sample_points_path.exists() or not surface_profiles_path.exists():
        logger.warning(
            "no surface-profile linkage available (missing %s or %s) -- every edge will use the wbgt_shade fallback",
            sample_points_path,
            surface_profiles_path,
        )
        return {}

    points_df = pd.read_parquet(sample_points_path, columns=["point_id", "nearest_edge_key"])
    surface_df = pd.read_parquet(surface_profiles_path, columns=["point_id", "thermal_load_score"])
    joined = points_df.merge(surface_df, on="point_id", how="inner").dropna(subset=["nearest_edge_key"])
    # If multiple sample points share a nearest edge, keep the last -- rare, and
    # not worth a tie-break policy for a single scalar surface nudge.
    return dict(zip(joined["nearest_edge_key"], joined["thermal_load_score"]))


# Bounded for the same reason as app.api.deps._weather_context_cache:
# observed_at is effectively unique per request, so an unbounded dict here
# grows forever -- confirmed causing gradual OOM kills on Render's free tier.
_AOI_HUMIDITY_CACHE_MAX = 500
_aoi_humidity_cache: dict[tuple[str, object], float] = {}


def _fetch_aoi_humidity(bbox: str, observed_at, openmeteo_client: OpenMeteoClient) -> float:
    """A single AOI-representative relative humidity (%) at observed_at, used for
    every edge's wbgt_shade fallback -- fetching per-edge would be thousands of
    HTTP calls for one route. Falls back to DEFAULT_RH_PCT (logged) on failure.

    Cached in-process by (bbox, observed_at), independent of OpenMeteoClient's own
    on-disk cache: a route family calls build_edge_costs once per lambda, and a
    transient network failure on JUST ONE of those calls must not silently give a
    different lambda a different humidity (and therefore a different WBGT
    landscape) than its siblings -- that would break the dose-monotonicity
    guarantee between lambda values for what should be the SAME grid snapshot.
    """
    cache_key = (bbox, observed_at)
    if cache_key in _aoi_humidity_cache:
        return _aoi_humidity_cache[cache_key]

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
        rh_pct = float(hourly.loc[idx, "relative_humidity_2m"])
    except Exception:
        logger.warning(
            "Open-Meteo humidity fetch failed for AOI centre -- falling back to DEFAULT_RH_PCT=%s", DEFAULT_RH_PCT
        )
        rh_pct = DEFAULT_RH_PCT

    if len(_aoi_humidity_cache) >= _AOI_HUMIDITY_CACHE_MAX:
        _aoi_humidity_cache.pop(next(iter(_aoi_humidity_cache)))
    _aoi_humidity_cache[cache_key] = rh_pct
    return rh_pct


def clear_aoi_humidity_cache() -> None:
    _aoi_humidity_cache.clear()


def build_edge_costs(
    graph,
    grid: TemperatureGrid,
    lambda_heat: float,
    sample_points_path: Path = DEFAULT_SAMPLE_POINTS_PATH,
    surface_profiles_path: Path = DEFAULT_SURFACE_PROFILES_PATH,
    max_dist_m: float = 150.0,
    openmeteo_client: Optional[OpenMeteoClient] = None,
) -> EdgeCostResult:
    """Combined distance/heat-dose cost per edge:

        cost = (1 - lambda_heat) * norm_distance + lambda_heat * norm_dose

    Both terms are normalised to [0, 1] across the WHOLE graph before
    combining -- metres and degree-seconds are different units, so lambda is
    meaningless otherwise. Orphaned edges (no grid cell within max_dist_m)
    NEVER get cost 0 -- they're filled via IDW, or (if even that fails) get
    the graph's 75th-percentile dose plus a penalty flag.
    """
    if not (0.0 <= lambda_heat <= 1.0):
        raise ValueError(f"lambda_heat must be in [0, 1], got {lambda_heat}")

    start = time.monotonic()

    edge_keys: list[EdgeKey] = []
    lengths: list[float] = []
    mid_lats: list[float] = []
    mid_lons: list[float] = []
    for u, v, k, data in graph.edges(keys=True, data=True):
        edge_keys.append((u, v, k))
        lengths.append(float(data.get("length_m", 0.0)))
        mid_lats.append(data["mid_lat"])
        mid_lons.append(data["mid_lon"])

    n_edges = len(edge_keys)
    if n_edges == 0:
        raise ValueError("graph has no edges to cost")

    lengths_arr = np.array(lengths, dtype=float)
    mid_lats_arr = np.array(mid_lats, dtype=float)
    mid_lons_arr = np.array(mid_lons, dtype=float)

    # -- temperatures: attach_temps_to_edges is already a single vectorized
    # cKDTree query over all edges (see app.grid.join) --------------------
    raw_temps = attach_temps_to_edges(graph, grid, max_dist_m=max_dist_m)
    temps_arr = np.array([raw_temps[key] for key in edge_keys], dtype=object)
    orphaned_mask = np.array([t is None for t in temps_arr])
    n_orphaned = int(orphaned_mask.sum())

    n_idw_filled = 0
    n_percentile_filled = 0
    percentile_fill_mask = np.zeros(n_edges, dtype=bool)

    if n_orphaned:
        for i in np.nonzero(orphaned_mask)[0]:
            try:
                temps_arr[i] = interpolate_idw(mid_lats_arr[i], mid_lons_arr[i], grid)
                n_idw_filled += 1
            except ValueError:
                temps_arr[i] = None  # grid has zero cells; handled below via percentile fallback

    still_missing_mask = np.array([t is None for t in temps_arr])
    temps_arr = np.array([float(t) if t is not None else np.nan for t in temps_arr], dtype=float)

    # -- WBGT: surface-adjusted where we have a matched surface profile,
    # otherwise the shade fallback -- fully vectorized ---------------------
    thermal_load_by_edge = _load_thermal_load_by_edge(sample_points_path, surface_profiles_path)
    thermal_load_arr = np.array(
        [thermal_load_by_edge.get(_edge_key_str(*key), np.nan) for key in edge_keys], dtype=float
    )
    has_surface_profile = ~np.isnan(thermal_load_arr)

    if openmeteo_client is None:
        openmeteo_client = OpenMeteoClient()
    rh_pct = _fetch_aoi_humidity(grid.bbox, grid.observed_at, openmeteo_client)

    valid_mask = ~still_missing_mask
    temps_for_wbgt = np.where(valid_mask, temps_arr, 0.0).astype(float)  # placeholder value, masked out below
    base_shade = _vectorized_wbgt_shade(temps_for_wbgt, rh_pct)
    surface_adjusted = _vectorized_surface_adjusted(base_shade, np.nan_to_num(thermal_load_arr, nan=0.5))
    wbgt_arr = np.where(has_surface_profile, surface_adjusted, base_shade)
    wbgt_method_arr = np.where(has_surface_profile, "surface_adjusted", "shade_fallback").astype(object)

    # -- dose: wbgt * traversal_seconds, vectorized -------------------------
    traversal_s = lengths_arr / WALK_SPEED_MS
    dose_arr = wbgt_arr * traversal_s

    # -- edges where even IDW failed: dose = 75th percentile of everyone
    # else's dose, plus a penalty flag. NEVER cost 0. -----------------------
    if np.any(still_missing_mask):
        known_doses = dose_arr[~still_missing_mask]
        percentile_dose = (
            float(np.percentile(known_doses, ORPHAN_PERCENTILE)) if known_doses.size else 0.0
        )
        dose_arr[still_missing_mask] = percentile_dose
        wbgt_arr[still_missing_mask] = np.nan  # no honest WBGT value exists for these
        wbgt_method_arr[still_missing_mask] = "orphaned_percentile_fallback"
        percentile_fill_mask[still_missing_mask] = True
        n_percentile_filled = int(still_missing_mask.sum())

    # -- normalise BOTH terms to [0,1] across the graph, THEN combine ------
    # Scale by the graph MAXIMUM only (no min-subtraction). A full min-max
    # scale would subtract a per-edge constant, which breaks path-additivity:
    # summing (length_i - dist_min) over a path adds an extra -dist_min PER
    # EDGE, so two paths with the same raw distance but a different edge
    # count would rank differently -- lambda=0 would then no longer exactly
    # reproduce shortest-path-by-length. Dividing by the max is a pure scalar
    # rescale, so it preserves path ordering exactly while still bounding
    # every edge's contribution to [dist_min/dist_max, 1] subset of [0,1].
    dist_min, dist_max = float(lengths_arr.min()), float(lengths_arr.max())
    dose_min, dose_max = float(dose_arr.min()), float(dose_arr.max())
    dist_scale = max(dist_max, 1e-9)
    dose_scale = max(dose_max, 1e-9)

    norm_distance = lengths_arr / dist_scale
    norm_dose = dose_arr / dose_scale
    combined_cost = (1 - lambda_heat) * norm_distance + lambda_heat * norm_dose

    costs = dict(zip(edge_keys, combined_cost.tolist()))

    elapsed_ms = (time.monotonic() - start) * 1000
    method_counts = {
        "surface_adjusted": int(np.sum(wbgt_method_arr == "surface_adjusted")),
        "shade_fallback": int(np.sum(wbgt_method_arr == "shade_fallback")),
        "orphaned_percentile_fallback": n_percentile_filled,
    }
    logger.info(
        "edge_costs_built n_edges=%d lambda=%.2f n_orphaned=%d n_idw_filled=%d n_percentile_filled=%d "
        "methods=%s elapsed_ms=%.1f",
        n_edges,
        lambda_heat,
        n_orphaned,
        n_idw_filled,
        n_percentile_filled,
        method_counts,
        elapsed_ms,
    )

    return EdgeCostResult(
        costs=costs,
        lambda_heat=lambda_heat,
        dist_min_m=dist_min,
        dist_max_m=dist_max,
        dose_min_degC_s=dose_min,
        dose_max_degC_s=dose_max,
        n_edges=n_edges,
        n_orphaned=n_orphaned,
        n_orphaned_idw_filled=n_idw_filled,
        n_orphaned_percentile_filled=n_percentile_filled,
        wbgt_method_counts=method_counts,
        edge_length_m=dict(zip(edge_keys, lengths_arr.tolist())),
        edge_temp_c=dict(zip(edge_keys, temps_arr.tolist())),
        edge_wbgt_c=dict(zip(edge_keys, wbgt_arr.tolist())),
        edge_dose_degC_s=dict(zip(edge_keys, dose_arr.tolist())),
        edge_wbgt_method=dict(zip(edge_keys, wbgt_method_arr.tolist())),
        edge_orphaned_penalty=dict(zip(edge_keys, percentile_fill_mask.tolist())),
    )
