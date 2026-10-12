"""Shared types for box annotation: the class list, pixel boxes, and upright image loading.

Every dataset this package builds (Kaggle close-ups, counter sessions) uses this class
order, so exported labels from different sources can be trained together.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

CLASS_NAMES: tuple[str, ...] = ("cup", "fork", "glass", "knife", "plate", "spoon")
"""Dataset class order. A class's position here is its YOLO class index; append new classes
at the end so existing indices stay valid."""

EXIF_ORIENTATION_TAG = 0x0112
UPRIGHT_EXIF_ORIENTATION = 1


@dataclass(frozen=True)
class PixelBox:
    """Axis-aligned box in upright-image pixels (x_min, y_min, x_max, y_max)."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if not (0 <= self.x_min < self.x_max and 0 <= self.y_min < self.y_max):
            raise ValueError(f"invalid box {self}")

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def intersection_area(self, other: "PixelBox") -> float:
        width = min(self.x_max, other.x_max) - max(self.x_min, other.x_min)
        height = min(self.y_max, other.y_max) - max(self.y_min, other.y_min)
        return max(width, 0.0) * max(height, 0.0)

    def fits_within(self, width_px: int, height_px: int) -> bool:
        return self.x_max <= width_px and self.y_max <= height_px


def yolo_label_line(class_index: int, box: PixelBox, width_px: int, height_px: int) -> str:
    """One YOLO label line: class index and box center/size normalized to the image size."""
    center_x = (box.x_min + box.x_max) / 2 / width_px
    center_y = (box.y_min + box.y_max) / 2 / height_px
    box_width = (box.x_max - box.x_min) / width_px
    box_height = (box.y_max - box.y_min) / height_px
    return f"{class_index} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_upright_rgb(path: Path) -> tuple[Image.Image, int]:
    """Load an image rotated by its EXIF orientation, with the original orientation tag.

    Boxes are always expressed in the upright frame, which is also how browsers display
    the image, so review overlays and exported labels agree.
    """
    with Image.open(path) as image:
        orientation = int(image.getexif().get(EXIF_ORIENTATION_TAG, UPRIGHT_EXIF_ORIENTATION))
        upright = ImageOps.exif_transpose(image).convert("RGB")
    return upright, orientation
