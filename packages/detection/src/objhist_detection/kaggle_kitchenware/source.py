"""Read and validate the Kaggle Kitchenware Classification training labels.

Source: https://www.kaggle.com/competitions/kitchenware-classification. `train.csv` maps each
image ID to one image-level class; `test.csv` has no labels and is not used.
"""

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

CLASS_NAMES: tuple[str, ...] = ("cup", "fork", "glass", "knife", "plate", "spoon")
"""Dataset class order. A class's position here is its YOLO class index."""

LABELS_FILENAME = "train.csv"
EXPECTED_COLUMNS = ["Id", "label"]
# IDs become file names in exported datasets, so only allow characters safe in paths.
IMAGE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
EXIF_ORIENTATION_TAG = 0x0112
UPRIGHT_EXIF_ORIENTATION = 1


class SourceDatasetError(ValueError):
    """The Kaggle folder does not match the expected layout or contents."""


@dataclass(frozen=True)
class LabeledImage:
    image_id: str
    class_name: str
    path: Path


def read_labeled_images(source_dir: Path) -> list[LabeledImage]:
    """Return every labeled image in `train.csv`, checking that each image file exists."""
    labels_path = source_dir / LABELS_FILENAME
    if not labels_path.is_file():
        raise SourceDatasetError(
            f"{labels_path} not found; pass the Kaggle folder that holds train.csv and images/"
        )
    images: list[LabeledImage] = []
    seen_ids: set[str] = set()
    with labels_path.open(newline="", encoding="utf-8") as labels_file:
        reader = csv.DictReader(labels_file)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise SourceDatasetError(
                f"{labels_path} has columns {reader.fieldnames}, expected {EXPECTED_COLUMNS}"
            )
        for line_number, row in enumerate(reader, start=2):
            image_id = row["Id"].strip()
            class_name = row["label"].strip()
            if not IMAGE_ID_PATTERN.fullmatch(image_id):
                raise SourceDatasetError(f"{labels_path}:{line_number}: invalid Id {image_id!r}")
            if image_id in seen_ids:
                raise SourceDatasetError(f"{labels_path}:{line_number}: duplicate Id {image_id}")
            if class_name not in CLASS_NAMES:
                raise SourceDatasetError(
                    f"{labels_path}:{line_number}: unknown label {class_name!r}; "
                    f"expected one of {', '.join(CLASS_NAMES)}"
                )
            path = source_dir / "images" / f"{image_id}.jpg"
            if not path.is_file():
                raise SourceDatasetError(f"{labels_path}:{line_number}: missing image {path}")
            seen_ids.add(image_id)
            images.append(LabeledImage(image_id=image_id, class_name=class_name, path=path))
    if not images:
        raise SourceDatasetError(f"{labels_path} has no labeled images")
    return images


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
