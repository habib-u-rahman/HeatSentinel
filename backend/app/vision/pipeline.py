from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from pydantic import BaseModel, Field

from app.vision import detector, metrics, segmenter


class SurfaceProfile(BaseModel):
    """Per-location surface composition + exposure profile for one street-level image."""

    lat: float
    lon: float
    image_path: str

    # Surface buckets (fractions, sum to 1.0)
    road: float
    sidewalk: float
    built: float
    vegetation: float
    sky: float
    other: float

    # Urban-climate metrics derived from the buckets
    green_view_index: float
    sky_view_factor_proxy: float
    impervious_fraction: float
    thermal_load_score: float

    # Exposure proxy: COCO detection counts + mean confidence
    person_count: int
    bicycle_count: int
    car_count: int
    motorcycle_count: int
    bus_count: int
    truck_count: int
    detection_mean_confidence: float

    segmenter_model: str
    detector_model: str
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def analyze_image(path: Union[str, Path], lat: float, lon: float) -> SurfaceProfile:
    """Run the full vision pipeline (segmentation + detection) on a single image."""
    path = Path(path)

    class_fractions = segmenter.segment(path)
    buckets = segmenter.to_buckets(class_fractions)
    detections = detector.detect(path)
    counts = detections["counts"]

    return SurfaceProfile(
        lat=lat,
        lon=lon,
        image_path=str(path),
        road=buckets["road"],
        sidewalk=buckets["sidewalk"],
        built=buckets["built"],
        vegetation=buckets["vegetation"],
        sky=buckets["sky"],
        other=buckets["other"],
        green_view_index=metrics.green_view_index(buckets),
        sky_view_factor_proxy=metrics.sky_view_factor_proxy(buckets),
        impervious_fraction=metrics.impervious_fraction(buckets),
        thermal_load_score=metrics.thermal_load_score(buckets),
        person_count=counts["person"],
        bicycle_count=counts["bicycle"],
        car_count=counts["car"],
        motorcycle_count=counts["motorcycle"],
        bus_count=counts["bus"],
        truck_count=counts["truck"],
        detection_mean_confidence=detections["mean_confidence"],
        segmenter_model=segmenter.MODEL_NAME,
        detector_model=detector.MODEL_NAME,
    )
