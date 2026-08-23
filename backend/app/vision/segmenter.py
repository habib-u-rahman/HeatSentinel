from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

logger = logging.getLogger(__name__)

MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

ImageInput = Union[str, Path, Image.Image]

# Cityscapes class name -> surface bucket. Anything not listed here (pole,
# traffic light/sign, person, rider, car, truck, bus, train, motorcycle,
# bicycle, ...) falls into "other".
BUCKET_MAP: dict[str, str] = {
    "road": "road",
    "sidewalk": "sidewalk",
    "building": "built",
    "wall": "built",
    "fence": "built",
    "vegetation": "vegetation",
    "terrain": "vegetation",
    "sky": "sky",
}
BUCKETS: tuple[str, ...] = ("road", "sidewalk", "built", "vegetation", "sky", "other")

# Official Cityscapes palette, indexed by class id (0..18), used for overlays.
CITYSCAPES_PALETTE: list[tuple[int, int, int]] = [
    (128, 64, 128),  # road
    (244, 35, 232),  # sidewalk
    (70, 70, 70),  # building
    (102, 102, 156),  # wall
    (190, 153, 153),  # fence
    (153, 153, 153),  # pole
    (250, 170, 30),  # traffic light
    (220, 220, 0),  # traffic sign
    (107, 142, 35),  # vegetation
    (152, 251, 152),  # terrain
    (70, 130, 180),  # sky
    (220, 20, 60),  # person
    (255, 0, 0),  # rider
    (0, 0, 142),  # car
    (0, 0, 70),  # truck
    (0, 60, 100),  # bus
    (0, 80, 100),  # train
    (0, 0, 230),  # motorcycle
    (119, 11, 32),  # bicycle
]


@lru_cache(maxsize=1)
def _get_model_and_processor() -> tuple[SegformerImageProcessor, SegformerForSemanticSegmentation]:
    """Load the SegFormer processor + model exactly once (lazy singleton)."""
    logger.info("Loading SegFormer model %s (first call only)...", MODEL_NAME)
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


def _load_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _predict_mask(img: Image.Image) -> np.ndarray:
    """Run the model and return a (H, W) int array of class ids at the image's own resolution."""
    processor, model = _get_model_and_processor()

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Logits come out at a coarser resolution than the input; upsample to the
    # ORIGINAL image size before argmax, otherwise the pixel fractions are wrong.
    upsampled = F.interpolate(
        outputs.logits,
        size=img.size[::-1],  # PIL gives (W, H); interpolate wants (H, W)
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.argmax(dim=1)[0].numpy()


def segment(image: ImageInput) -> dict[str, float]:
    """Run SegFormer and return {cityscapes_class_name: pixel_fraction} over all 19 classes."""
    img = _load_image(image)
    pred = _predict_mask(img)
    _, model = _get_model_and_processor()

    id2label = model.config.id2label
    total = pred.size
    return {label: int(np.count_nonzero(pred == int(class_id))) / total for class_id, label in id2label.items()}


def to_buckets(fractions: dict[str, float]) -> dict[str, float]:
    """Group the 19 Cityscapes class fractions into the 6 surface buckets; sums to 1.0."""
    buckets = {b: 0.0 for b in BUCKETS}
    for class_name, frac in fractions.items():
        buckets[BUCKET_MAP.get(class_name, "other")] += frac
    return buckets


def save_overlay(image: ImageInput, path: Union[str, Path]) -> None:
    """Run segmentation and write a colourised mask blended over the original image as a PNG."""
    img = _load_image(image)
    pred = _predict_mask(img)

    color_mask = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(CITYSCAPES_PALETTE):
        color_mask[pred == class_id] = color

    mask_img = Image.fromarray(color_mask).convert("RGB")
    blended = Image.blend(img, mask_img, alpha=0.5)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(out_path)
