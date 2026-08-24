"""Attaches current heat exposure to each vulnerable-population-proxy POI.

exposure_score = vulnerability_weight * max(0, wbgt_c - EXPOSURE_THRESHOLD_C)

A documented, simple linear formula: exposure only accrues once WBGT crosses
a baseline "no longer SAFE" threshold (the light-work SAFE/CAUTION boundary
from app.heat.thresholds), scaled by how much more vulnerable this group is
assumed to be. This is a RANKING heuristic for prioritising outreach, not a
calibrated epidemiological risk model.
"""

from __future__ import annotations

import pandas as pd

from app.grid.join import attach_temps_to_points
from app.grid.schema import TemperatureGrid
from app.heat.thresholds import WBGT_THRESHOLDS_C, classify
from app.heat.wbgt import wbgt_shade
from app.routing.router import ROUTE_WORK_INTENSITY

EXPOSURE_THRESHOLD_C = WBGT_THRESHOLDS_C[ROUTE_WORK_INTENSITY]["SAFE"]
DEFAULT_MAX_DIST_M = 150.0


def attach_risk_to_pois(pois_df: pd.DataFrame, grid: TemperatureGrid, rh_pct: float, max_dist_m: float = DEFAULT_MAX_DIST_M) -> pd.DataFrame:
    """pois_df + temp_c, wbgt_c, risk_band, exposure_score.

    Rows farther than max_dist_m from every grid cell get NaN temp/wbgt/
    exposure/risk_band -- never a fake 0 or a fake SAFE.
    """
    joined = attach_temps_to_points(pois_df, grid, max_dist_m=max_dist_m).copy()

    wbgt_values = [
        wbgt_shade(float(temp_c), rh_pct).value if pd.notna(temp_c) else float("nan") for temp_c in joined["temp_c"]
    ]
    joined["wbgt_c"] = wbgt_values
    joined["risk_band"] = [classify(w, ROUTE_WORK_INTENSITY) if pd.notna(w) else None for w in wbgt_values]
    joined["exposure_score"] = [
        float(weight) * max(0.0, w - EXPOSURE_THRESHOLD_C) if pd.notna(w) else float("nan")
        for weight, w in zip(joined["weight"], wbgt_values)
    ]
    return joined


def rank_by_exposure(pois_with_risk: pd.DataFrame) -> pd.DataFrame:
    """POIs most at risk right now, highest exposure_score first. Rows with no
    matched grid cell (NaN exposure_score) sort last, not first."""
    return pois_with_risk.sort_values("exposure_score", ascending=False, na_position="last").reset_index(drop=True)
