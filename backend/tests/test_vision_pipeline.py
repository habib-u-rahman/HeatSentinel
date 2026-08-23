from __future__ import annotations

import pytest

from app.vision import detector, metrics, segmenter
from app.vision.pipeline import SurfaceProfile, analyze_image


def test_analyze_image_assembles_surface_profile(tmp_path, monkeypatch):
    fake_image = tmp_path / "fake.jpg"
    fake_image.write_bytes(b"not a real image, just needs to exist on disk")

    fake_fractions = {"road": 0.5, "sky": 0.5}
    fake_buckets = {"road": 0.4, "sidewalk": 0.1, "built": 0.1, "vegetation": 0.2, "sky": 0.15, "other": 0.05}
    fake_detections = {
        "counts": {"person": 2, "bicycle": 0, "car": 1, "motorcycle": 0, "bus": 0, "truck": 0},
        "mean_confidence": 0.77,
    }

    monkeypatch.setattr(segmenter, "segment", lambda path: fake_fractions)
    monkeypatch.setattr(segmenter, "to_buckets", lambda fractions: fake_buckets)
    monkeypatch.setattr(detector, "detect", lambda path: fake_detections)

    profile = analyze_image(fake_image, lat=40.7128, lon=-74.0060)

    assert isinstance(profile, SurfaceProfile)
    assert profile.lat == pytest.approx(40.7128)
    assert profile.lon == pytest.approx(-74.0060)
    assert profile.image_path == str(fake_image)

    bucket_sum = profile.road + profile.sidewalk + profile.built + profile.vegetation + profile.sky + profile.other
    assert bucket_sum == pytest.approx(1.0)

    assert profile.green_view_index == pytest.approx(metrics.green_view_index(fake_buckets))
    assert profile.sky_view_factor_proxy == pytest.approx(metrics.sky_view_factor_proxy(fake_buckets))
    assert profile.impervious_fraction == pytest.approx(metrics.impervious_fraction(fake_buckets))
    assert profile.thermal_load_score == pytest.approx(metrics.thermal_load_score(fake_buckets))

    assert profile.person_count == 2
    assert profile.car_count == 1
    assert profile.bicycle_count == 0
    assert profile.detection_mean_confidence == pytest.approx(0.77)

    assert profile.segmenter_model == segmenter.MODEL_NAME
    assert profile.detector_model == detector.MODEL_NAME
