"""Real-time heat-alert evaluation over the current grid + vulnerable-POI
proxy layer + zone aggregation.

HONESTY CONSTRAINT: every alert (not just poi_at_risk ones -- the whole
/api/alerts feed blends zone- and POI-derived data) carries
population_proxy=True and POPULATION_PROXY_NOTE, so nothing downstream can
mistake an alert for a verified-population signal.

DEDUPLICATION: alerts are keyed by (category, zone_id[, sub-key], risk_band)
and suppressed for DEDUP_WINDOW_S after they first fire -- a poller hitting
/api/alerts every few seconds must not re-announce the same condition on
every call.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from app.grid.schema import TemperatureGrid
from app.grid.zones import ZONE_GRID_DIM, compute_zones
from app.heat.thresholds import BANDS_IN_ORDER
from app.sampling.points import _cell_id
from app.vulnerable.poi import POPULATION_PROXY_NOTE

DEDUP_WINDOW_S = 15 * 60  # 15 minutes
RAPID_RISE_THRESHOLD_C = 2.0
ALERT_RISK_BANDS = {"DANGER", "CRITICAL"}  # poi_at_risk only fires at these bands

_SEVERITY_RANK = {band: i for i, band in enumerate(BANDS_IN_ORDER)}


@dataclass
class Alert:
    alert_id: str
    severity: str
    category: str  # "zone_critical" | "poi_at_risk" | "rapid_rise"
    message: str
    lat: float
    lon: float
    poi_id: Optional[str]
    triggered_at: datetime
    wbgt_c: float
    risk_band: str
    data_source: str
    population_proxy: bool
    proxy_note: str
    exposure_score: Optional[float] = None


# --- deduplication ---------------------------------------------------------------

_dedup_seen: dict[str, float] = {}  # dedup_key -> last-fired monotonic time


def _dedup_key(category: str, sub_key: str, risk_band: str) -> str:
    return f"{category}:{sub_key}:{risk_band}"


def _should_suppress(category: str, sub_key: str, risk_band: str, window_s: float) -> bool:
    key = _dedup_key(category, sub_key, risk_band)
    now = time.monotonic()
    last = _dedup_seen.get(key)
    if last is not None and (now - last) < window_s:
        return True
    _dedup_seen[key] = now
    return False


def clear_dedup_state() -> None:
    _dedup_seen.clear()


def _alert_id(category: str, sub_key: str, risk_band: str, triggered_at: datetime) -> str:
    canonical = f"{category}:{sub_key}:{risk_band}:{triggered_at.isoformat()}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# --- evaluation ------------------------------------------------------------------


def evaluate(
    grid: TemperatureGrid,
    pois_df: pd.DataFrame,
    bbox: str,
    rh_pct: float,
    previous_grid: Optional[TemperatureGrid] = None,
    dedup_window_s: float = DEDUP_WINDOW_S,
) -> list[Alert]:
    """Evaluate zone_critical / poi_at_risk / rapid_rise alerts for the current grid.

    previous_grid=None skips rapid_rise cleanly -- there's nothing wrong with
    not having a prior reading yet, it just means that check can't run.
    """
    from app.vulnerable.scoring import attach_risk_to_pois  # local import: avoids a module-load-time cycle with app.api

    zones = compute_zones(grid, bbox, rh_pct)
    now = grid.observed_at
    data_source = grid.source.value.lower()
    alerts: list[Alert] = []

    # -- zone_critical --------------------------------------------------------
    for zone in zones:
        if zone.risk_band != "CRITICAL":
            continue
        if _should_suppress("zone_critical", zone.zone_id, zone.risk_band, dedup_window_s):
            continue
        alerts.append(
            Alert(
                alert_id=_alert_id("zone_critical", zone.zone_id, zone.risk_band, now),
                severity=zone.risk_band,
                category="zone_critical",
                message=f"Zone {zone.zone_id} is in CRITICAL heat stress (max WBGT {zone.max_wbgt_c:.1f}C)",
                lat=zone.centroid_lat,
                lon=zone.centroid_lon,
                poi_id=None,
                triggered_at=now,
                wbgt_c=zone.max_wbgt_c,
                risk_band=zone.risk_band,
                data_source=data_source,
                population_proxy=True,
                proxy_note=POPULATION_PROXY_NOTE,
            )
        )

    # -- poi_at_risk -- grouped by (zone_id, category, risk_band) so we report
    # "3 schools in Zone r2_c5 are in CRITICAL", not one alert per school -----
    if not pois_df.empty:
        risky = attach_risk_to_pois(pois_df, grid, rh_pct)
        risky = risky[risky["risk_band"].isin(ALERT_RISK_BANDS)].copy()

        if not risky.empty:
            min_lon, min_lat, max_lon, max_lat = (float(x) for x in bbox.split(","))
            risky["zone_id"] = [
                _cell_id(float(lat), float(lon), (min_lon, min_lat, max_lon, max_lat), ZONE_GRID_DIM)
                for lat, lon in zip(risky["lat"], risky["lon"])
            ]

            for (zone_id, category, risk_band), group in risky.groupby(["zone_id", "category", "risk_band"]):
                sub_key = f"{zone_id}:{category}"
                if _should_suppress("poi_at_risk", sub_key, risk_band, dedup_window_s):
                    continue
                n = len(group)
                label = category.replace("_", " ") + ("s" if n != 1 else "")
                verb = "is" if n == 1 else "are"
                mean_wbgt = float(group["wbgt_c"].mean())
                alerts.append(
                    Alert(
                        alert_id=_alert_id("poi_at_risk", sub_key, risk_band, now),
                        severity=risk_band,
                        category="poi_at_risk",
                        message=f"{n} {label} in Zone {zone_id} {verb} in {risk_band} heat stress (WBGT {mean_wbgt:.1f}C)",
                        lat=float(group["lat"].mean()),
                        lon=float(group["lon"].mean()),
                        poi_id=str(group.iloc[0]["poi_id"]) if n == 1 else None,
                        triggered_at=now,
                        wbgt_c=mean_wbgt,
                        risk_band=risk_band,
                        data_source=data_source,
                        population_proxy=True,
                        proxy_note=POPULATION_PROXY_NOTE,
                        exposure_score=float(group["exposure_score"].mean()),
                    )
                )

    # -- rapid_rise -------------------------------------------------------------
    if previous_grid is not None:
        prev_zones_by_id = {z.zone_id: z for z in compute_zones(previous_grid, bbox, rh_pct)}
        for zone in zones:
            prev = prev_zones_by_id.get(zone.zone_id)
            if prev is None:
                continue
            delta = zone.mean_wbgt_c - prev.mean_wbgt_c
            if delta <= RAPID_RISE_THRESHOLD_C:
                continue
            if _should_suppress("rapid_rise", zone.zone_id, zone.risk_band, dedup_window_s):
                continue
            alerts.append(
                Alert(
                    alert_id=_alert_id("rapid_rise", zone.zone_id, zone.risk_band, now),
                    severity=zone.risk_band,
                    category="rapid_rise",
                    message=f"Zone {zone.zone_id} WBGT rose {delta:.1f}C since the last reading (now {zone.mean_wbgt_c:.1f}C)",
                    lat=zone.centroid_lat,
                    lon=zone.centroid_lon,
                    poi_id=None,
                    triggered_at=now,
                    wbgt_c=zone.mean_wbgt_c,
                    risk_band=zone.risk_band,
                    data_source=data_source,
                    population_proxy=True,
                    proxy_note=POPULATION_PROXY_NOTE,
                )
            )

    alerts.sort(key=lambda a: (-_SEVERITY_RANK.get(a.severity, 0), -(a.exposure_score if a.exposure_score is not None else a.wbgt_c)))
    return alerts
