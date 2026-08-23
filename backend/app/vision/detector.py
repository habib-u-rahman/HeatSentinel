from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Union

from ultralytics import YOLO

logger = logging.getLogger(__name__)

MODEL_NAME = "yolov8n.pt"
CONF_THRESHOLD = 0.35

# COCO class id -> tracked bucket name. These are the people/vehicle classes
# used as an exposure proxy (who/what is present at street level).
TRACKED_CLASSES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@lru_cache(maxsize=1)
def _get_model() -> YOLO:
    """Load the YOLOv8n weights exactly once (lazy singleton)."""
    logger.info("Loading YOLOv8n model %s (first call only)...", MODEL_NAME)
    return YOLO(MODEL_NAME)


def _summarize(detections: list[tuple[int, float]]) -> dict:
    """Pure aggregation step: (class_id, confidence) pairs -> counts + mean confidence."""
    counts = {name: 0 for name in TRACKED_CLASSES.values()}
    confidences: list[float] = []

    for class_id, conf in detections:
        name = TRACKED_CLASSES.get(class_id)
        if name is not None:
            counts[name] += 1
            confidences.append(conf)

    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return {"counts": counts, "mean_confidence": mean_confidence}


def detect(image: Union[str, Path]) -> dict:
    """Run YOLOv8n on `image` and return tracked-class counts + mean confidence."""
    model = _get_model()
    results = model.predict(source=str(image), conf=CONF_THRESHOLD, verbose=False, device="cpu")
    result = results[0]

    detections: list[tuple[int, float]] = []
    if result.boxes is not None:
        for box in result.boxes:
            detections.append((int(box.cls.item()), float(box.conf.item())))

    return _summarize(detections)
