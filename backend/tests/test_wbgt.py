from __future__ import annotations

import pytest

from app.heat.thresholds import BANDS_IN_ORDER, WBGT_THRESHOLDS_C, classify
from app.heat.wbgt import (
    Confidence,
    stull_wet_bulb,
    surface_adjusted_wbgt,
    vapour_pressure,
    wbgt_outdoor,
    wbgt_shade,
)


@pytest.mark.parametrize("rh_pct", [20, 40, 60, 80, 95])
@pytest.mark.parametrize("temp_c", [10, 20, 30, 40])
def test_stull_wet_bulb_never_exceeds_dry_bulb(temp_c, rh_pct):
    result = stull_wet_bulb(temp_c, rh_pct)
    assert result.value <= temp_c + 1e-6


def test_stull_wet_bulb_equals_dry_bulb_at_saturation():
    for temp_c in (10, 20, 30, 40):
        result = stull_wet_bulb(temp_c, 100.0)
        assert result.value == pytest.approx(temp_c, abs=0.5)


def test_stull_wet_bulb_rejects_invalid_rh():
    with pytest.raises(ValueError):
        stull_wet_bulb(30.0, -5.0)
    with pytest.raises(ValueError):
        stull_wet_bulb(30.0, 150.0)


def test_vapour_pressure_increases_with_rh():
    low = vapour_pressure(30.0, 20.0)
    high = vapour_pressure(30.0, 80.0)
    assert high.value > low.value


def test_wbgt_shade_confidence_is_medium():
    result = wbgt_shade(30.0, 50.0)
    assert result.confidence == Confidence.MEDIUM
    assert set(result.inputs_available) == {"temp_c", "rh_pct"}


@pytest.mark.parametrize("temp_c,rh_pct", [(25, 30), (30, 50), (35, 70), (40, 40)])
def test_wbgt_outdoor_gte_shade_for_positive_solar(temp_c, rh_pct):
    shade = wbgt_shade(temp_c, rh_pct)
    outdoor = wbgt_outdoor(temp_c, rh_pct, wind_ms=2.0, solar_wm2=800.0)
    assert outdoor.value >= shade.value


def test_wbgt_outdoor_equals_shade_at_zero_solar():
    temp_c, rh_pct = 30.0, 50.0
    shade = wbgt_shade(temp_c, rh_pct)
    outdoor = wbgt_outdoor(temp_c, rh_pct, wind_ms=2.0, solar_wm2=0.0)
    assert outdoor.value == pytest.approx(shade.value, abs=1e-9)


def test_confidence_downgrades_as_inputs_are_dropped():
    full = wbgt_outdoor(30.0, rh_pct=50.0, wind_ms=2.0, solar_wm2=800.0)
    temp_and_humidity = wbgt_outdoor(30.0, rh_pct=50.0)
    temp_only = wbgt_outdoor(30.0)

    assert full.confidence == Confidence.HIGH
    assert temp_and_humidity.confidence == Confidence.MEDIUM
    assert temp_only.confidence == Confidence.LOW
    assert temp_only.inputs_available == ["temp_c"]
    assert set(full.inputs_available) == {"temp_c", "rh_pct", "wind_ms", "solar_wm2"}


def test_surface_adjusted_wbgt_bounds_and_direction():
    base = 30.0
    hottest = surface_adjusted_wbgt(base, thermal_load_score=1.0, max_delta_c=2.0)
    coolest = surface_adjusted_wbgt(base, thermal_load_score=0.0, max_delta_c=2.0)
    neutral = surface_adjusted_wbgt(base, thermal_load_score=0.5, max_delta_c=2.0)

    assert hottest.value == pytest.approx(base + 2.0)
    assert coolest.value == pytest.approx(base - 2.0)
    assert neutral.value == pytest.approx(base)


def test_classify_band_boundaries_are_monotonic():
    for intensity in WBGT_THRESHOLDS_C:
        values = [10, 20, 26, 27, 28, 29, 30, 31, 32, 34, 36, 40]
        indices = [BANDS_IN_ORDER.index(classify(v, intensity)) for v in values]
        assert indices == sorted(indices)


def test_classify_rejects_unknown_work_intensity():
    with pytest.raises(ValueError):
        classify(30.0, work_intensity="sprinting")
