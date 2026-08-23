from __future__ import annotations

import pytest

from app.vision.metrics import (
    green_view_index,
    impervious_fraction,
    sky_view_factor_proxy,
    thermal_load_score,
)


def _buckets(**overrides: float) -> dict[str, float]:
    base = {"road": 0.0, "sidewalk": 0.0, "built": 0.0, "vegetation": 0.0, "sky": 0.0, "other": 0.0}
    base.update(overrides)
    return base


def test_buckets_fixture_sums_to_one():
    buckets = _buckets(road=0.4, sidewalk=0.1, built=0.2, vegetation=0.15, sky=0.1, other=0.05)
    assert sum(buckets.values()) == pytest.approx(1.0)


def test_green_view_index():
    assert green_view_index(_buckets(vegetation=0.42)) == pytest.approx(0.42)


def test_sky_view_factor_proxy():
    assert sky_view_factor_proxy(_buckets(sky=0.2)) == pytest.approx(0.2)


def test_impervious_fraction():
    buckets = _buckets(road=0.3, sidewalk=0.1, built=0.2, vegetation=0.3, sky=0.1)
    assert impervious_fraction(buckets) == pytest.approx(0.6)


def test_thermal_load_score_all_road_is_max():
    assert thermal_load_score(_buckets(road=1.0)) == pytest.approx(1.0)


def test_thermal_load_score_all_vegetation_is_min():
    assert thermal_load_score(_buckets(vegetation=1.0)) == pytest.approx(0.0)


def test_thermal_load_score_stays_in_bounds_for_mixed_buckets():
    buckets = _buckets(road=0.4, sidewalk=0.1, built=0.2, vegetation=0.15, sky=0.1, other=0.05)
    score = thermal_load_score(buckets)
    assert 0.0 <= score <= 1.0


def test_thermal_load_score_monotonic_in_road_vs_vegetation():
    low = thermal_load_score(_buckets(road=0.2, vegetation=0.8))
    high = thermal_load_score(_buckets(road=0.8, vegetation=0.2))
    assert high > low
