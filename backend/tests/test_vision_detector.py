from __future__ import annotations

import pytest

from app.vision.detector import _summarize


def test_summarize_counts_and_mean_confidence():
    detections = [(0, 0.9), (0, 0.5), (2, 0.8), (99, 0.99)]  # class 99 is untracked and ignored
    result = _summarize(detections)

    assert result["counts"]["person"] == 2
    assert result["counts"]["car"] == 1
    assert result["counts"]["bicycle"] == 0
    assert result["mean_confidence"] == pytest.approx((0.9 + 0.5 + 0.8) / 3)


def test_summarize_empty_detections():
    result = _summarize([])
    assert all(count == 0 for count in result["counts"].values())
    assert result["mean_confidence"] == 0.0


def test_summarize_all_tracked_classes_present():
    result = _summarize([])
    assert set(result["counts"].keys()) == {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
