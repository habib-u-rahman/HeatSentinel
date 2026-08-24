"""Assembles the RF training set by joining, on point_id:
  sample_points.parquet -> surface_profiles.parquet -> temperature grid(s)
  (via app.grid.join.attach_temps_to_points) -> Open-Meteo hourly weather.

Silent row loss here would quietly corrupt the model's honesty, so every join
stage reports exactly how many rows survived and why the rest were dropped.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from app.grid.join import attach_temps_to_points
from app.grid.schema import TemperatureGrid
from app.ingest.openmeteo import OpenMeteoClient

logger = logging.getLogger(__name__)

MIN_TRAINING_ROWS = 50

# Columns pulled from surface_profiles.parquet (see app.vision.pipeline.SurfaceProfile).
SURFACE_COLUMNS = [
    "point_id",
    "lat",
    "lon",
    "road",
    "sidewalk",
    "built",
    "vegetation",
    "sky",
    "other",
    "impervious_fraction",
    "green_view_index",
    "sky_view_factor_proxy",
    "thermal_load_score",
    "person_count",
    "bicycle_count",
    "car_count",
    "motorcycle_count",
    "bus_count",
    "truck_count",
]


class MixedSourceTrainingSetError(Exception):
    """Raised when build_training_set is asked to combine grids from multiple sources."""


class InsufficientTrainingDataError(Exception):
    """Raised when fewer than MIN_TRAINING_ROWS rows survive the joins."""


def _report(stage: str, before: int, after: int, reason: str) -> str:
    line = f"[join] {stage}: {before} -> {after} (dropped {before - after}: {reason})"
    logger.info(line)
    return line


def _match_hour(hourly_df: pd.DataFrame, observed_at) -> dict:
    """Nearest-hour row from an Open-Meteo hourly dataframe to observed_at (UTC)."""
    if hourly_df.empty:
        return {"relative_humidity": None, "wind_ms": None, "solar_wm2": None}

    times = pd.to_datetime(hourly_df["time"])
    target = pd.Timestamp(observed_at)
    if target.tzinfo is not None:
        target = target.tz_convert(None)  # Open-Meteo times are naive UTC wall-clock (we request timezone=UTC)

    idx = (times - target).abs().idxmin()
    row = hourly_df.loc[idx]
    return {
        "relative_humidity": row["relative_humidity_2m"],
        "wind_ms": row["wind_speed_10m"],
        "solar_wm2": row["shortwave_radiation"],
    }


def build_training_set(
    sample_points_path: Path,
    surface_profiles_path: Path,
    grids: list[TemperatureGrid],
    openmeteo_client: Optional[OpenMeteoClient] = None,
    max_dist_m: float = 150.0,
    bbox: Optional[str] = None,
    min_rows: int = MIN_TRAINING_ROWS,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the joined training set. Returns (training_df, report_lines).

    Raises MixedSourceTrainingSetError if grids mix LIVE and FIXTURE sources,
    or InsufficientTrainingDataError if fewer than min_rows rows survive --
    training a model on a near-empty set would be meaningless. min_rows
    defaults to MIN_TRAINING_ROWS; only override it for testing the join
    mechanics on a deliberately tiny input.
    """
    if not grids:
        raise ValueError("build_training_set needs at least one TemperatureGrid")

    sources = {g.source for g in grids}
    if len(sources) > 1:
        raise MixedSourceTrainingSetError(
            f"Refusing to build a training set mixing sources {sorted(s.value for s in sources)} -- "
            "LIVE and FIXTURE rows must never be trained on together."
        )
    source = grids[0].source
    bbox = bbox or grids[0].bbox

    report: list[str] = []

    points_df = pd.read_parquet(sample_points_path)
    n0 = len(points_df)

    surface_df = pd.read_parquet(surface_profiles_path)
    missing_cols = [c for c in SURFACE_COLUMNS if c not in surface_df.columns]
    if missing_cols:
        raise ValueError(f"surface_profiles.parquet is missing expected columns: {missing_cols}")

    # surface_df's lat/lon are the image's ACTUAL captured coordinates (see
    # app.ingest.mapillary), which is what we want to join to the temperature
    # grid; sample_points' lat/lon was only the query point, so we don't carry
    # it through -- only point_id/nearest_edge_key/grid_cell_id survive from it.
    joined = points_df[["point_id", "nearest_edge_key", "grid_cell_id"]].merge(
        surface_df[SURFACE_COLUMNS], on="point_id", how="inner"
    )
    n1 = len(joined)
    report.append(
        _report("sample_points -> surface_profiles", n0, n1, "no surface profile (image not fetched/analyzed)")
    )

    if openmeteo_client is None:
        openmeteo_client = OpenMeteoClient()

    n1_total = 0
    n2_total = 0
    n3_total = 0
    all_rows: list[pd.DataFrame] = []

    for grid in grids:
        per_grid = joined.copy()
        n1_total += len(per_grid)

        temp_joined = attach_temps_to_points(per_grid, grid, max_dist_m=max_dist_m)
        temp_joined = temp_joined.dropna(subset=["temp_c"])
        n2_total += len(temp_joined)
        if temp_joined.empty:
            continue

        temp_joined = temp_joined.copy()
        temp_joined["source"] = grid.source.value
        temp_joined["observed_at"] = grid.observed_at
        temp_joined["hour"] = grid.observed_at.hour

        weather_rows = [
            _match_hour(openmeteo_client.fetch_hourly(lat, lon, grid.observed_at.strftime("%Y-%m-%d"), grid.observed_at.strftime("%Y-%m-%d")), grid.observed_at)
            for lat, lon in zip(temp_joined["lat"], temp_joined["lon"])
        ]
        weather_df = pd.DataFrame(weather_rows, index=temp_joined.index)
        merged = pd.concat([temp_joined, weather_df], axis=1)
        merged = merged.dropna(subset=["relative_humidity", "wind_ms", "solar_wm2"])
        n3_total += len(merged)

        all_rows.append(merged)

    report.append(
        _report(
            "surface_profiles x grids -> temperature-matched",
            n1_total,
            n2_total,
            f"farther than {max_dist_m}m from every grid cell",
        )
    )
    report.append(_report("temperature-matched -> weather-matched", n2_total, n3_total, "Open-Meteo fetch/match failure"))

    training_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    n_final = len(training_df)
    report.append(
        f"[join] FINAL training set: {n_final} rows from {len(grids)} grid(s), source={source.value}, bbox={bbox}"
    )

    for line in report:
        print(line)

    if n_final < min_rows:
        raise InsufficientTrainingDataError(
            f"Only {n_final} rows survived the joins (need >= {min_rows}). "
            "Training a model on this would be meaningless.\n" + "\n".join(report)
        )

    training_df["bbox"] = bbox
    return training_df, report
