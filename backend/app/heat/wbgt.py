"""Wet Bulb Globe Temperature (WBGT) estimation -- pure functions, no I/O.

True outdoor WBGT needs air temperature, humidity, wind speed, and solar
radiation. Our primary data source (FortyGuard) provides temperature only;
humidity/wind/solar come from app.ingest.openmeteo when available. This
module is deliberately explicit about which of those inputs it actually had
for a given call -- every function here returns a HeatCalcResult carrying the
value, the method/formula used, which inputs were supplied, and a confidence
tier -- rather than silently filling gaps and reporting a bare number.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Confidence(str, Enum):
    HIGH = "HIGH"  # full inputs: temp + humidity + wind + solar
    MEDIUM = "MEDIUM"  # temp + humidity only
    LOW = "LOW"  # temp only


@dataclass
class HeatCalcResult:
    value: float
    method_used: str
    inputs_available: list[str] = field(default_factory=list)
    confidence: Confidence = Confidence.LOW


# Formula validity window for Stull (2011) / the Magnus-Tetens vapour-pressure
# fit. Outside this, results are logged as a warning rather than rejected --
# the approximation degrades gracefully near the edges.
_VALID_TEMP_RANGE_C = (-20.0, 50.0)
_VALID_RH_RANGE_PCT = (5.0, 99.0)

# Physically-implausible bounds -- these ARE rejected outright.
_HARD_TEMP_RANGE_C = (-90.0, 60.0)
_HARD_RH_RANGE_PCT = (0.0, 100.0)


def _validate_temp_rh(temp_c: float, rh_pct: float) -> None:
    if not (_HARD_RH_RANGE_PCT[0] <= rh_pct <= _HARD_RH_RANGE_PCT[1]):
        raise ValueError(f"rh_pct must be within {_HARD_RH_RANGE_PCT}, got {rh_pct}")
    if not (_HARD_TEMP_RANGE_C[0] <= temp_c <= _HARD_TEMP_RANGE_C[1]):
        raise ValueError(f"temp_c is physically implausible: {temp_c}")

    if not (_VALID_TEMP_RANGE_C[0] <= temp_c <= _VALID_TEMP_RANGE_C[1]) or not (
        _VALID_RH_RANGE_PCT[0] <= rh_pct <= _VALID_RH_RANGE_PCT[1]
    ):
        logger.warning(
            "temp_c=%.1f rh_pct=%.1f is outside the validated fit range "
            "(temp_c %s, rh_pct %s) -- result may be less accurate",
            temp_c,
            rh_pct,
            _VALID_TEMP_RANGE_C,
            _VALID_RH_RANGE_PCT,
        )


def stull_wet_bulb(temp_c: float, rh_pct: float) -> HeatCalcResult:
    """Stull (2011) empirical wet-bulb temperature from dry-bulb temp + RH.

    Source: Stull, R. (2011), "Wet-Bulb Temperature from Relative Humidity
    and Air Temperature", J. Appl. Meteor. Climatol., 50(11), 2267-2269.
    Claimed accuracy ~0.3 C within temp_c in [-20, 50] and rh_pct in [5, 99].
    """
    _validate_temp_rh(temp_c, rh_pct)

    tw = (
        temp_c * math.atan(0.151977 * math.sqrt(rh_pct + 8.313659))
        + math.atan(temp_c + rh_pct)
        - math.atan(rh_pct - 1.676331)
        + 0.00391838 * rh_pct**1.5 * math.atan(0.023101 * rh_pct)
        - 4.686035
    )
    return HeatCalcResult(
        value=tw,
        method_used="Stull (2011) empirical wet-bulb approximation",
        inputs_available=["temp_c", "rh_pct"],
        confidence=Confidence.MEDIUM,
    )


def vapour_pressure(temp_c: float, rh_pct: float) -> HeatCalcResult:
    """Actual vapour pressure (hPa) via Magnus-Tetens saturation vapour pressure x RH.

    es(T) = 6.112 * exp(17.62*T / (243.12+T)) hPa -- WMO (2008) Magnus-Tetens
    coefficients over water. e = es * RH/100.
    """
    _validate_temp_rh(temp_c, rh_pct)

    es = 6.112 * math.exp((17.62 * temp_c) / (243.12 + temp_c))
    e = es * rh_pct / 100.0
    return HeatCalcResult(
        value=e,
        method_used="Magnus-Tetens saturation vapour pressure (WMO 2008 coefficients) x RH/100",
        inputs_available=["temp_c", "rh_pct"],
        confidence=Confidence.MEDIUM,
    )


def wbgt_shade(temp_c: float, rh_pct: float) -> HeatCalcResult:
    """Australian BoM shade WBGT approximation (no solar, no wind).

    WBGT = 0.567*Ta + 0.393*e + 3.94, where e is vapour pressure in hPa.
    Source: Australian Bureau of Meteorology, "Thermal Comfort observations".
    """
    e = vapour_pressure(temp_c, rh_pct).value
    value = 0.567 * temp_c + 0.393 * e + 3.94
    return HeatCalcResult(
        value=value,
        method_used="Australian BoM shade WBGT approximation: 0.567*Ta + 0.393*e + 3.94 (shade only)",
        inputs_available=["temp_c", "rh_pct"],
        confidence=Confidence.MEDIUM,
    )


# --- outdoor (solar/wind-adjusted) WBGT --------------------------------------

# Documented fallback defaults used by wbgt_outdoor when a real measurement
# isn't available. Each is a deliberate, named assumption -- never a silent
# gap-fill -- and the returned confidence is downgraded whenever any of these
# get used instead of a real value.
DEFAULT_RH_PCT = 50.0  # a generic mid-range humidity assumption
DEFAULT_WIND_MS = 1.0  # light air (Beaufort 1); a low-wind assumption is conservative: less wind -> less evaporative cooling -> higher estimated heat stress
DEFAULT_SOLAR_WM2 = 700.0  # a typical clear-sky midday shortwave irradiance; a moderate-high assumption is conservative for a daytime worst-case estimate

# Globe-temperature-excess model constants (see _globe_temp_excess docstring).
SOLAR_NORMALIZATION_WM2 = 1000.0  # ~clear-sky peak shortwave irradiance, used to scale solar_wm2 to a ~0..1 fraction
GLOBE_SOLAR_GAIN_C = 25.0  # max globe-temp excess above air temp at full sun & zero wind (coarse, documented assumption)
WIND_HALVING_MS = 3.0  # wind speed at which the solar-driven globe excess is halved (forced-convective cooling)
ISO7243_GLOBE_WEIGHT = 0.2  # ISO 7243's WBGT weighting coefficient applied to globe temperature

_ALL_OUTDOOR_INPUTS = ("rh_pct", "wind_ms", "solar_wm2")


def _globe_temp_excess(solar_wm2: float, wind_ms: float) -> float:
    """Estimated globe-temperature excess (C) above air temp from solar loading, damped by wind.

    This is NOT the Liljegren et al. (2008) black-globe energy-balance model
    (that requires an iterative solver over globe emissivity/convection
    coefficients we do not reimplement here). Instead this is a coarse,
    transparently-documented stand-in with the same qualitative physics:
    globe temperature rises with incoming solar radiation (radiative heating)
    and falls back toward air temperature as wind increases (forced
    convective cooling). Swap in a verified globe-temperature solver before
    relying on this for anything beyond a directional estimate.
    """
    solar_wm2 = max(0.0, solar_wm2)
    wind_ms = max(0.0, wind_ms)
    return GLOBE_SOLAR_GAIN_C * (solar_wm2 / SOLAR_NORMALIZATION_WM2) / (1 + wind_ms / WIND_HALVING_MS)


def wbgt_outdoor(
    temp_c: float,
    rh_pct: Optional[float] = None,
    wind_ms: Optional[float] = None,
    solar_wm2: Optional[float] = None,
) -> HeatCalcResult:
    """Estimated outdoor WBGT including a solar/wind correction on top of shade WBGT.

    Method: start from the Australian BoM shade WBGT (wbgt_shade), which
    already captures temperature + humidity, then add a solar-loading
    correction representing the extra globe-temperature heating from direct
    sun (damped by wind, see _globe_temp_excess), weighted by the ISO 7243
    globe-temperature coefficient (0.2):

        wbgt_outdoor = wbgt_shade(Ta, RH) + 0.2 * globe_temp_excess(solar, wind)

    This guarantees wbgt_outdoor >= wbgt_shade for any non-negative solar
    input (more sun can only add heat load, never remove it), without
    re-deriving the full Liljegren et al. (2008) globe-temperature model.

    Any of rh_pct, wind_ms, solar_wm2 not supplied fall back to a documented
    default (DEFAULT_RH_PCT / DEFAULT_WIND_MS / DEFAULT_SOLAR_WM2) and the
    returned confidence is downgraded accordingly -- this function never
    silently pretends to have a measurement it doesn't.
    """
    inputs_available = ["temp_c"]

    rh_used = rh_pct if rh_pct is not None else DEFAULT_RH_PCT
    if rh_pct is not None:
        inputs_available.append("rh_pct")

    wind_used = wind_ms if wind_ms is not None else DEFAULT_WIND_MS
    if wind_ms is not None:
        inputs_available.append("wind_ms")

    solar_used = solar_wm2 if solar_wm2 is not None else DEFAULT_SOLAR_WM2
    if solar_wm2 is not None:
        inputs_available.append("solar_wm2")

    shade = wbgt_shade(temp_c, rh_used)
    excess = _globe_temp_excess(solar_used, wind_used)
    value = shade.value + ISO7243_GLOBE_WEIGHT * excess

    missing = [name for name in _ALL_OUTDOOR_INPUTS if name not in inputs_available]
    if not missing:
        confidence = Confidence.HIGH
    elif "rh_pct" in inputs_available:
        confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.LOW

    method_used = "BoM shade WBGT + ISO7243-weighted solar/wind globe-temp-excess correction"
    if missing:
        method_used += f"; defaulted missing inputs: {missing}"

    return HeatCalcResult(
        value=value,
        method_used=method_used,
        inputs_available=inputs_available,
        confidence=confidence,
    )


def surface_adjusted_wbgt(base_wbgt: float, thermal_load_score: float, max_delta_c: float = 2.0) -> HeatCalcResult:
    """Nudge a meteorological WBGT by the vision module's street-level thermal_load_score.

    thermal_load_score in [0, 1] (see app.vision.metrics.thermal_load_score;
    0 = coolest surface mix, 1 = hottest). Linear, symmetric around the 0.5
    midpoint (neither warmer nor cooler than the met-only estimate):

        delta_c = (thermal_load_score - 0.5) * 2 * max_delta_c   in [-max_delta_c, +max_delta_c]
        adjusted = base_wbgt + delta_c

    This is our own contribution, not a literature formula: it captures
    surface effects (asphalt vs. tree canopy, etc.) that temperature/humidity/
    wind/solar alone miss, bounded to a modest +/- max_delta_c so it nudges
    rather than dominates the meteorological estimate.
    """
    delta_c = (thermal_load_score - 0.5) * 2 * max_delta_c
    return HeatCalcResult(
        value=base_wbgt + delta_c,
        method_used=f"linear surface adjustment: delta_c = (thermal_load_score - 0.5) * 2 * {max_delta_c}",
        inputs_available=["base_wbgt", "thermal_load_score"],
        confidence=Confidence.HIGH,
    )


def _demo_table() -> None:
    """Print WBGT + risk band for temp 30/35/40/45 C at RH 20/40/60% (moderate work intensity)."""
    from app.heat.thresholds import classify

    demo_wind_ms = 2.0
    demo_solar_wm2 = 800.0

    header = f"{'Temp C':>8} {'RH %':>6} {'WBGT_shade':>11} {'WBGT_outdoor':>13} {'Band (moderate)':>16}"
    print(header)
    print("-" * len(header))
    for temp_c in (30, 35, 40, 45):
        for rh_pct in (20, 40, 60):
            shade = wbgt_shade(temp_c, rh_pct)
            outdoor = wbgt_outdoor(temp_c, rh_pct, wind_ms=demo_wind_ms, solar_wm2=demo_solar_wm2)
            band = classify(outdoor.value, "moderate")
            print(f"{temp_c:>8} {rh_pct:>6} {shade.value:>11.1f} {outdoor.value:>13.1f} {band:>16}")


if __name__ == "__main__":
    _demo_table()
