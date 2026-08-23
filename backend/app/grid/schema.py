"""Normalized temperature-grid representation.

Everything downstream (fixtures, storage, the grid-to-graph join, ML
training) works against TemperatureGrid, never against raw FortyGuard JSON.
app.grid.adapter is the single place that maps raw API responses into this
shape -- when the real FortyGuard schema is confirmed, only that adapter
should need to change.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator


class GridSource(str, Enum):
    LIVE = "LIVE"  # a real, parsed FortyGuard API response
    FIXTURE = "FIXTURE"  # synthetic data from app.fortyguard.fixtures -- never demo-ready by itself


def cell_id_for(lat: float, lon: float) -> str:
    """Stable id derived purely from position (rounded to ~1m).

    Deliberately time-independent: the same physical cell must keep the same
    id across grids taken at different timestamps so per-cell time series
    (e.g. for LSTM training) can be assembled by joining on cell_id.
    """
    canonical = f"{round(lat, 5)},{round(lon, 5)}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class TemperatureCell(BaseModel):
    lat: float
    lon: float
    temp_c: float
    cell_id: str
    geohash6: Optional[str] = None


class TemperatureGrid(BaseModel):
    cells: list[TemperatureCell]
    bbox: str  # "min_lon,min_lat,max_lon,max_lat"
    granularity_m: int
    observed_at: datetime
    source: GridSource
    api_activity_id: Optional[str] = None

    @field_validator("observed_at")
    @classmethod
    def _ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.cells:
            return pd.DataFrame(columns=["lat", "lon", "temp_c", "cell_id", "geohash6"])
        return pd.DataFrame([c.model_dump() for c in self.cells])

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(lats, lons, temps) as numpy arrays, in cell order."""
        lats = np.array([c.lat for c in self.cells], dtype=float)
        lons = np.array([c.lon for c in self.cells], dtype=float)
        temps = np.array([c.temp_c for c in self.cells], dtype=float)
        return lats, lons, temps

    def stats(self) -> dict:
        if not self.cells:
            return {"min": None, "max": None, "mean": None, "std": None, "count": 0}
        _, _, temps = self.to_arrays()
        return {
            "min": float(np.min(temps)),
            "max": float(np.max(temps)),
            "mean": float(np.mean(temps)),
            "std": float(np.std(temps)),
            "count": int(temps.size),
        }
