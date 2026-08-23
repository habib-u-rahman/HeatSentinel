"""On-disk store for TemperatureGrid snapshots.

Layout: data/grids/{YYYY-MM-DD}/{HHMM}.parquet, one file per grid snapshot
(observed_at, UTC). A companion data/grids/manifest.jsonl records path,
source, bbox, granularity_m, and cell_count for every saved grid, and
load_series refuses to silently blend LIVE and FIXTURE grids into one series.

CRITICAL SAFETY: loading a FIXTURE grid always logs a WARNING, and raises if
app.config.get_settings().ALLOW_FIXTURE_DATA is False -- fixture data must
never reach a demo unnoticed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from app.config import get_settings
from app.grid.schema import GridSource, TemperatureCell, TemperatureGrid

logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = Path("../data/grids")
MANIFEST_FILENAME = "manifest.jsonl"


class MixedSourceError(Exception):
    """Raised when a requested series would silently blend LIVE and FIXTURE grids."""


def _grid_path(base_dir: Path, observed_at: datetime) -> Path:
    observed_at = observed_at.astimezone(timezone.utc)
    return base_dir / observed_at.strftime("%Y-%m-%d") / f"{observed_at.strftime('%H%M')}.parquet"


def _append_manifest(base_dir: Path, entry: dict) -> None:
    manifest_path = base_dir / MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def save_grid(grid: TemperatureGrid, base_dir: Optional[Path] = None) -> Path:
    """Write grid to data/grids/{date}/{HHMM}.parquet and record it in manifest.jsonl."""
    base_dir = Path(base_dir) if base_dir is not None else DEFAULT_BASE_DIR
    path = _grid_path(base_dir, grid.observed_at)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = grid.to_dataframe()
    df["bbox"] = grid.bbox
    df["granularity_m"] = grid.granularity_m
    df["observed_at"] = grid.observed_at
    df["source"] = grid.source.value
    df["api_activity_id"] = grid.api_activity_id
    df.to_parquet(path, index=False)

    _append_manifest(
        base_dir,
        {
            "path": str(path),
            "observed_at": grid.observed_at.isoformat(),
            "source": grid.source.value,
            "bbox": grid.bbox,
            "granularity_m": grid.granularity_m,
            "cell_count": len(grid.cells),
        },
    )
    logger.info("grid_saved path=%s source=%s cells=%d", path, grid.source.value, len(grid.cells))
    return path


def load_grid(path: Path) -> TemperatureGrid:
    """Load a TemperatureGrid parquet.

    Logs a WARNING whenever the grid is FIXTURE-sourced, and raises RuntimeError
    if ALLOW_FIXTURE_DATA is False -- fixture data must never reach a demo
    unnoticed.
    """
    path = Path(path)
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Grid file {path} has no rows")

    first = df.iloc[0]
    source = GridSource(first["source"])

    if source == GridSource.FIXTURE:
        logger.warning(
            "FIXTURE temperature grid loaded from %s -- this is SYNTHETIC data, not live FortyGuard output",
            path,
        )
        if not get_settings().ALLOW_FIXTURE_DATA:
            raise RuntimeError(
                f"Refusing to load FIXTURE grid from {path}: ALLOW_FIXTURE_DATA is False. "
                "Set it True in dev, or load/generate a LIVE grid instead."
            )

    cells = [
        TemperatureCell(
            lat=record["lat"],
            lon=record["lon"],
            temp_c=record["temp_c"],
            cell_id=record["cell_id"],
            geohash6=record.get("geohash6"),
        )
        for record in df.to_dict("records")
    ]

    observed_at = pd.Timestamp(first["observed_at"]).to_pydatetime()
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)

    api_activity_id = first.get("api_activity_id")
    if api_activity_id is None or pd.isna(api_activity_id):
        api_activity_id = None

    return TemperatureGrid(
        cells=cells,
        bbox=first["bbox"],
        granularity_m=int(first["granularity_m"]),
        observed_at=observed_at,
        source=source,
        api_activity_id=api_activity_id,
    )


def list_available(start: datetime, end: datetime, base_dir: Optional[Path] = None) -> list[datetime]:
    """Timestamps (UTC) of grids saved under base_dir within [start, end]."""
    base_dir = Path(base_dir) if base_dir is not None else DEFAULT_BASE_DIR
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    timestamps: list[datetime] = []
    if not base_dir.exists():
        return timestamps

    for date_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        try:
            date_val = datetime.strptime(date_dir.name, "%Y-%m-%d").date()
        except ValueError:
            continue
        for file_path in sorted(date_dir.glob("*.parquet")):
            try:
                time_val = datetime.strptime(file_path.stem, "%H%M").time()
            except ValueError:
                continue
            ts = datetime.combine(date_val, time_val, tzinfo=timezone.utc)
            if start <= ts <= end:
                timestamps.append(ts)

    return sorted(timestamps)


def load_series(bbox: str, start: datetime, end: datetime, base_dir: Optional[Path] = None) -> list[TemperatureGrid]:
    """Load every stored grid for bbox within [start, end], sorted by time.

    Raises MixedSourceError if the matched grids span both LIVE and FIXTURE
    sources -- a time series must never silently blend the two.
    """
    base_dir = Path(base_dir) if base_dir is not None else DEFAULT_BASE_DIR
    timestamps = list_available(start, end, base_dir=base_dir)

    grids: list[TemperatureGrid] = []
    for ts in timestamps:
        path = _grid_path(base_dir, ts)
        if not path.exists():
            continue
        grid = load_grid(path)
        if grid.bbox == bbox:
            grids.append(grid)

    sources = {grid.source for grid in grids}
    if len(sources) > 1:
        raise MixedSourceError(
            f"load_series for bbox={bbox!r} between {start} and {end} matched grids from "
            f"multiple sources ({sorted(s.value for s in sources)}) -- refusing to silently "
            "blend LIVE and FIXTURE data into one series."
        )

    return grids
