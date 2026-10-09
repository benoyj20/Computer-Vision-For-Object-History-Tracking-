from pathlib import Path

import pytest
from PIL import Image

from objhist_detection.kaggle_kitchenware.source import (
    EXIF_ORIENTATION_TAG,
    SourceDatasetError,
    load_upright_rgb,
    read_labeled_images,
)


def test_reads_labels_and_keeps_leading_zeros_in_ids(kaggle_dir: Path) -> None:
    images = read_labeled_images(kaggle_dir)

    assert [(image.image_id, image.class_name) for image in images] == [
        ("0001", "cup"),
        ("0002", "fork"),
        ("0003", "plate"),
    ]
    assert images[0].path == kaggle_dir / "images" / "0001.jpg"


@pytest.mark.parametrize(
    ("csv_text", "message"),
    [
        ("Id,class\n0001,cup\n", "columns"),
        ("Id,label\n0001,bowl\n", "unknown label 'bowl'"),
        ("Id,label\n0001,cup\n0001,cup\n", "duplicate Id 0001"),
        ("Id,label\n../0001,cup\n", "invalid Id"),
        ("Id,label\n0009,cup\n", "missing image"),
        ("Id,label\n", "no labeled images"),
    ],
)
def test_rejects_invalid_labels(kaggle_dir: Path, csv_text: str, message: str) -> None:
    (kaggle_dir / "train.csv").write_text(csv_text, encoding="utf-8")

    with pytest.raises(SourceDatasetError, match=message):
        read_labeled_images(kaggle_dir)


def test_missing_labels_file_names_the_expected_layout(tmp_path: Path) -> None:
    with pytest.raises(SourceDatasetError, match="train.csv and images/"):
        read_labeled_images(tmp_path)


def test_loads_exif_rotated_photo_upright(tmp_path: Path) -> None:
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[EXIF_ORIENTATION_TAG] = 6  # stored sideways, displayed rotated 90 degrees
    Image.new("RGB", (100, 200), "white").save(path, exif=exif)

    upright, orientation = load_upright_rgb(path)

    assert orientation == 6
    assert upright.size == (200, 100)
