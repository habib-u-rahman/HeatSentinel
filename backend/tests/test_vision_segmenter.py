from __future__ import annotations

import pytest

from app.vision.segmenter import BUCKETS, to_buckets

# All 19 Cityscapes classes, fractions summing to 1.0.
FULL_FRACTIONS = {
    "road": 0.30,
    "sidewalk": 0.05,
    "building": 0.15,
    "wall": 0.02,
    "fence": 0.01,
    "pole": 0.01,
    "traffic light": 0.005,
    "traffic sign": 0.005,
    "vegetation": 0.20,
    "terrain": 0.05,
    "sky": 0.15,
    "person": 0.01,
    "rider": 0.0,
    "car": 0.035,
    "truck": 0.0,
    "bus": 0.0,
    "train": 0.0,
    "motorcycle": 0.0,
    "bicycle": 0.005,
}


def test_full_fractions_fixture_sums_to_one():
    assert sum(FULL_FRACTIONS.values()) == pytest.approx(1.0)


def test_to_buckets_has_exactly_six_buckets():
    buckets = to_buckets(FULL_FRACTIONS)
    assert set(buckets.keys()) == set(BUCKETS)


def test_to_buckets_sums_to_one():
    buckets = to_buckets(FULL_FRACTIONS)
    assert sum(buckets.values()) == pytest.approx(1.0)


def test_to_buckets_groups_classes_correctly():
    buckets = to_buckets(FULL_FRACTIONS)
    assert buckets["road"] == pytest.approx(0.30)
    assert buckets["sidewalk"] == pytest.approx(0.05)
    assert buckets["built"] == pytest.approx(0.15 + 0.02 + 0.01)
    assert buckets["vegetation"] == pytest.approx(0.20 + 0.05)
    assert buckets["sky"] == pytest.approx(0.15)
    expected_other = FULL_FRACTIONS["pole"] + FULL_FRACTIONS["traffic light"] + FULL_FRACTIONS["traffic sign"]
    expected_other += FULL_FRACTIONS["person"] + FULL_FRACTIONS["rider"] + FULL_FRACTIONS["car"]
    expected_other += FULL_FRACTIONS["truck"] + FULL_FRACTIONS["bus"] + FULL_FRACTIONS["train"]
    expected_other += FULL_FRACTIONS["motorcycle"] + FULL_FRACTIONS["bicycle"]
    assert buckets["other"] == pytest.approx(expected_other)


def test_to_buckets_handles_unknown_class_name():
    fractions = {"road": 0.5, "some_unlisted_class": 0.5}
    buckets = to_buckets(fractions)
    assert buckets["road"] == pytest.approx(0.5)
    assert buckets["other"] == pytest.approx(0.5)
    assert sum(buckets.values()) == pytest.approx(1.0)
