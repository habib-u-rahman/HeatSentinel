"""Physically-plausible synthetic temperature grids for development before a
real FortyGuard response schema is confirmed.

Garbage fixtures produce garbage models, so every grid here is built from four
additive, independently-tunable components:
  1. a smooth (asymmetric) diurnal curve -- cool ~05:00, peak ~15:00
  2. an urban-heat-island gradient -- hotter toward the AOI centre
  3. surface coupling -- warmer where street surfaces are impervious, cooler
     where they're vegetated (from app.vision surface profiles, if supplied)
  4. spatially-correlated (NOT white) noise -- a smoothed random perturbation

Every grid produced here carries source=FIXTURE (see app.grid.schema) and
must never reach a demo unnoticed -- see app.grid.store's ALLOW_FIXTURE_DATA
enforcement.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from app.core.geo import BBox, _parse_bbox, grid_points
from app.grid.join import _equirect_project
from app.grid.schema import GridSource, TemperatureCell, TemperatureGrid, cell_id_for

logger = logging.getLogger(__name__)

# --- component magnitudes (all in degrees C) --- documented so they're easy to tune.
BASE_TEMP_C = 28.0  # a mean summer-afternoon ambient baseline for the demo AOI
DIURNAL_AMPLITUDE_C = 10.0  # peak-to-trough swing across the day
DIURNAL_TROUGH_HOUR = 5.0  # local time, ~05:00
DIURNAL_PEAK_HOUR = 15.0  # local time, ~15:00
UHI_SPAN_C = 3.0  # AOI centre vs. AOI edge
SURFACE_COUPLING_SPAN_C = 4.0  # fully-impervious vs. fully-vegetated, at a matched point
SURFACE_COUPLING_MAX_DIST_M = 100.0  # beyond this, a grid cell gets no surface coupling
NOISE_AMPLITUDE_C = 0.6  # std-dev of the smoothed spatial noise
NOISE_CORRELATION_LENGTH_M = 150.0  # ~blob size of the smoothed noise field

DEMO_BBOX = "-74.020,40.700,-73.995,40.726"  # a real ~2.5km AOI (Lower Manhattan), used only as a fallback demo default
DEMO_SURFACE_PROFILES_PATH = Path("../data/raw/surface_profiles.parquet")


def _diurnal_offset_c(
    hour_frac: float,
    amplitude_c: float = DIURNAL_AMPLITUDE_C,
    trough_hour: float = DIURNAL_TROUGH_HOUR,
    peak_hour: float = DIURNAL_PEAK_HOUR,
) -> float:
    """Smooth, deliberately ASYMMETRIC diurnal curve: trough at trough_hour, peak at
    peak_hour, full peak-to-trough span = amplitude_c.

    Real diurnal cycles warm faster (post-dawn solar heating) than they cool
    (post-peak radiative loss), so the rising phase (trough_hour -> peak_hour)
    and falling phase (peak_hour -> next trough_hour) get different, but each
    individually smooth, half-cosine easings -- continuous at both joins.
    """
    h = hour_frac % 24
    rising_span = (peak_hour - trough_hour) % 24
    falling_span = 24 - rising_span

    if (h - trough_hour) % 24 <= rising_span:
        frac = ((h - trough_hour) % 24) / rising_span
        shape = -np.cos(np.pi * frac)  # -1 (trough) -> +1 (peak)
    else:
        frac = ((h - peak_hour) % 24) / falling_span
        shape = np.cos(np.pi * frac)  # +1 (peak) -> -1 (trough)

    return (amplitude_c / 2.0) * shape


def _smoothed_noise(coords_m: np.ndarray, sigma_m: float, amplitude_c: float, rng: np.random.Generator) -> np.ndarray:
    """Spatially-correlated noise: i.i.d. Gaussian noise smoothed with a Gaussian
    kernel of length-scale sigma_m over neighbouring points, then rescaled to
    the requested std-dev. NOT white noise -- points closer than ~sigma_m stay
    correlated, matching real micro-climate persistence (a given street stays
    a little hotter/cooler than its neighbours) instead of independent jitter.
    """
    n = len(coords_m)
    raw = rng.normal(0.0, 1.0, size=n)
    if n == 1:
        return raw * amplitude_c

    tree = cKDTree(coords_m)
    neighbor_lists = tree.query_ball_tree(tree, r=3 * sigma_m)

    smoothed = np.empty(n)
    for i, neighbors in enumerate(neighbor_lists):
        neighbors = np.asarray(neighbors)
        dists = np.linalg.norm(coords_m[neighbors] - coords_m[i], axis=1)
        weights = np.exp(-0.5 * (dists / sigma_m) ** 2)
        smoothed[i] = np.sum(weights * raw[neighbors]) / np.sum(weights)

    std = smoothed.std()
    if std < 1e-9:
        return np.zeros(n)
    return smoothed / std * amplitude_c


def _surface_offset_c(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    surface_profiles: Optional[pd.DataFrame],
    ref_lat: float,
    ref_lon: float,
) -> np.ndarray:
    """Warmer near impervious surface points, cooler near vegetated ones (see
    app.vision.metrics for impervious_fraction/vegetation). Cells farther than
    SURFACE_COUPLING_MAX_DIST_M from any surface-profile point get zero offset.
    """
    n = len(grid_x)
    if surface_profiles is None or surface_profiles.empty:
        return np.zeros(n)

    surf_x, surf_y = _equirect_project(
        surface_profiles["lat"].to_numpy(), surface_profiles["lon"].to_numpy(), ref_lat, ref_lon
    )
    tree = cKDTree(np.column_stack([surf_x, surf_y]))
    distances, indices = tree.query(np.column_stack([grid_x, grid_y]))

    impervious = surface_profiles["impervious_fraction"].to_numpy()
    vegetation = surface_profiles["vegetation"].to_numpy()
    coupling = (SURFACE_COUPLING_SPAN_C / 2.0) * (impervious[indices] - vegetation[indices])

    offset = np.zeros(n)
    within = distances <= SURFACE_COUPLING_MAX_DIST_M
    offset[within] = coupling[within]
    return offset


def generate_grid(
    bbox: BBox,
    granularity_m: int,
    timestamp: datetime,
    seed: int = 0,
    surface_profiles: Optional[pd.DataFrame] = None,
) -> TemperatureGrid:
    """A single physically-plausible synthetic TemperatureGrid (source=FIXTURE)."""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    points = grid_points(bbox, granularity_m)
    if not points:
        raise ValueError(f"No grid points generated for bbox={bbox} at granularity_m={granularity_m}")

    lats = np.array([p[0] for p in points])
    lons = np.array([p[1] for p in points])

    min_lon, min_lat, max_lon, max_lat = _parse_bbox(bbox)
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    x_m, y_m = _equirect_project(lats, lons, center_lat, center_lon)
    dist_from_center_m = np.hypot(x_m, y_m)
    max_dist_m = max(float(dist_from_center_m.max()), 1e-6)

    hour_frac = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
    diurnal = _diurnal_offset_c(hour_frac)  # same instant for the whole grid -> scalar
    uhi = UHI_SPAN_C * np.clip(1.0 - dist_from_center_m / max_dist_m, 0.0, 1.0)
    surface = _surface_offset_c(x_m, y_m, surface_profiles, center_lat, center_lon)

    rng = np.random.default_rng(seed)
    noise = _smoothed_noise(np.column_stack([x_m, y_m]), NOISE_CORRELATION_LENGTH_M, NOISE_AMPLITUDE_C, rng)

    temps = BASE_TEMP_C + diurnal + uhi + surface + noise

    cells = [
        TemperatureCell(lat=float(lat), lon=float(lon), temp_c=float(t), cell_id=cell_id_for(lat, lon))
        for lat, lon, t in zip(lats, lons, temps)
    ]

    return TemperatureGrid(
        cells=cells,
        bbox=bbox if isinstance(bbox, str) else ",".join(str(x) for x in bbox),
        granularity_m=granularity_m,
        observed_at=timestamp,
        source=GridSource.FIXTURE,
    )


def generate_series(
    bbox: BBox,
    start: datetime,
    end: datetime,
    freq: str = "1h",
    granularity_m: int = 100,
    seed: int = 0,
    surface_profiles: Optional[pd.DataFrame] = None,
) -> list[TemperatureGrid]:
    """A time series of grids for LSTM development.

    Uses the SAME seed for every timestamp, so the underlying spatially-
    correlated noise "texture" (a fixed micro-climate perturbation field) is
    stationary across the series -- only the diurnal curve moves frame to
    frame, matching real hour-to-hour continuity instead of independently
    re-randomizing every step.
    """
    timestamps = pd.date_range(start, end, freq=freq, tz="UTC")
    return [
        generate_grid(bbox, granularity_m, ts.to_pydatetime(), seed=seed, surface_profiles=surface_profiles)
        for ts in timestamps
    ]


def _print_ascii_heatmap(grid: TemperatureGrid, width: int = 50, height: int = 20) -> None:
    lats, lons, temps = grid.to_arrays()
    if lats.size == 0:
        print("(no cells)")
        return

    lon_bins = np.linspace(lons.min(), lons.max(), width + 1)
    lat_bins = np.linspace(lats.min(), lats.max(), height + 1)
    # +/- a hair so the max-valued row/col falls inside the last bin, not on its edge
    lon_bins[-1] += 1e-9
    lat_bins[-1] += 1e-9

    sum_grid, _, _ = np.histogram2d(lats, lons, bins=[lat_bins, lon_bins], weights=temps)
    count_grid, _, _ = np.histogram2d(lats, lons, bins=[lat_bins, lon_bins])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_grid = np.where(count_grid > 0, sum_grid / np.maximum(count_grid, 1), np.nan)

    ramp = " .:-=+*#%@"
    valid = mean_grid[~np.isnan(mean_grid)]
    if valid.size == 0:
        print("(no data)")
        return
    tmin, tmax = float(valid.min()), float(valid.max())
    span = max(tmax - tmin, 1e-6)

    for row in reversed(range(height)):  # reversed: higher latitude (north) prints on top
        chars = []
        for col in range(width):
            v = mean_grid[row, col]
            if np.isnan(v):
                chars.append(" ")
            else:
                idx = int((v - tmin) / span * (len(ramp) - 1))
                chars.append(ramp[idx])
        print("".join(chars))
    print(f"(scale: '{ramp[0]}'={tmin:.1f}C .. '{ramp[-1]}'={tmax:.1f}C)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Physically-plausible synthetic temperature-grid fixtures")
    parser.add_argument("--demo", action="store_true", help="print grid stats + an ASCII heat map")
    parser.add_argument("--bbox", default=DEMO_BBOX)
    parser.add_argument("--granularity-m", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--surface-profiles", default=str(DEMO_SURFACE_PROFILES_PATH))
    args = parser.parse_args()

    if not args.demo:
        parser.print_help()
        return

    surface_profiles = None
    surf_path = Path(args.surface_profiles)
    if surf_path.exists():
        surface_profiles = pd.read_parquet(surf_path)
        print(f"surface coupling: using {len(surface_profiles)} points from {surf_path}")
    else:
        print(f"surface coupling: no surface profiles found at {surf_path} -- generating without it")

    timestamp = datetime.now(timezone.utc).replace(hour=15, minute=0, second=0, microsecond=0)
    grid = generate_grid(args.bbox, args.granularity_m, timestamp, seed=args.seed, surface_profiles=surface_profiles)

    stats = grid.stats()
    print(
        f"\ngrid: {stats['count']} cells, granularity={grid.granularity_m}m, "
        f"observed_at={grid.observed_at.isoformat()}, source={grid.source.value}"
    )
    print(f"stats: min={stats['min']:.1f}C max={stats['max']:.1f}C mean={stats['mean']:.1f}C std={stats['std']:.2f}C\n")
    _print_ascii_heatmap(grid)


if __name__ == "__main__":
    main()
