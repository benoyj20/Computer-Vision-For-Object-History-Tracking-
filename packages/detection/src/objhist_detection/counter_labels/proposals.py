"""Box proposals for multi-object counter photos.

Unlike a Kaggle close-up, a counter photo shows several objects, so every detector box above
a confidence floor becomes a proposal. The detector's class is only a starting point: a
reviewer confirms or corrects each box's class, rejects false boxes, and draws missed objects.
"""

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from objhist_detection.annotation import CLASS_NAMES, PixelBox

PROPOSAL_CONFIDENCE_FLOOR = 0.15
"""Boxes below this are dropped before review. On the first counter photos, YOLOE's boxes
below it were background or duplicates; a missed object can still be drawn by the reviewer."""

LOW_CONFIDENCE_HINT = 0.4
"""The review page marks boxes below this confidence so reviewers look at them closely."""


@dataclass(frozen=True)
class PhotoRecord:
    key: str
    session_id: str
    filename: str
    width_px: int
    height_px: int
    """Upright image size, after applying the EXIF orientation."""
    exif_orientation: int
    sha256: str


@dataclass(frozen=True)
class ProposedBox:
    key: str
    """Photo key (`<session>/<filename>`)."""
    box_id: str
    """Unique within the photo, for example `p1`; reviews refer to it."""
    class_name: str
    """The detector's class; a reviewer confirms or replaces it."""
    confidence: float
    box: PixelBox


@dataclass(frozen=True)
class Candidate:
    class_name: str
    confidence: float
    box: PixelBox


def proposals_for_photo(key: str, candidates: Sequence[Candidate]) -> list[ProposedBox]:
    """Keep candidates at or above the floor, most confident first, numbered `p1`, `p2`, ..."""
    kept = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.confidence >= PROPOSAL_CONFIDENCE_FLOOR
        ),
        key=lambda candidate: (-candidate.confidence, candidate.box.x_min, candidate.box.y_min),
    )
    return [
        ProposedBox(key, f"p{index}", candidate.class_name, candidate.confidence, candidate.box)
        for index, candidate in enumerate(kept, start=1)
    ]


PHOTO_COLUMNS = [
    "key",
    "session_id",
    "filename",
    "width_px",
    "height_px",
    "exif_orientation",
    "sha256",
]
BOX_COLUMNS = [
    "key",
    "box_id",
    "class_name",
    "confidence",
    "x_min_px",
    "y_min_px",
    "x_max_px",
    "y_max_px",
]


def write_photos(path: Path, photos: Iterable[PhotoRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(PHOTO_COLUMNS)
        for photo in photos:
            writer.writerow(
                [
                    photo.key,
                    photo.session_id,
                    photo.filename,
                    photo.width_px,
                    photo.height_px,
                    photo.exif_orientation,
                    photo.sha256,
                ]
            )


def write_boxes(path: Path, boxes: Iterable[ProposedBox]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(BOX_COLUMNS)
        for proposed in boxes:
            box = proposed.box
            writer.writerow(
                [
                    proposed.key,
                    proposed.box_id,
                    proposed.class_name,
                    f"{proposed.confidence:.4f}",
                    f"{box.x_min:.2f}",
                    f"{box.y_min:.2f}",
                    f"{box.x_max:.2f}",
                    f"{box.y_max:.2f}",
                ]
            )


def _read_rows(path: Path, columns: list[str]) -> Iterable[tuple[int, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != columns:
            raise ValueError(f"{path} has columns {reader.fieldnames}, expected {columns}")
        yield from enumerate(reader, start=2)


def read_photos(path: Path) -> dict[str, PhotoRecord]:
    photos: dict[str, PhotoRecord] = {}
    for line_number, row in _read_rows(path, PHOTO_COLUMNS):
        try:
            photo = PhotoRecord(
                key=row["key"],
                session_id=row["session_id"],
                filename=row["filename"],
                width_px=int(row["width_px"]),
                height_px=int(row["height_px"]),
                exif_orientation=int(row["exif_orientation"]),
                sha256=row["sha256"],
            )
            if photo.key != f"{photo.session_id}/{photo.filename}":
                raise ValueError(f"key {photo.key!r} does not match session and filename")
            if photo.width_px <= 0 or photo.height_px <= 0:
                raise ValueError("image size must be positive")
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
        if photo.key in photos:
            raise ValueError(f"{path}:{line_number}: duplicate photo {photo.key}")
        photos[photo.key] = photo
    return photos


def read_boxes(path: Path, photos: dict[str, PhotoRecord]) -> dict[str, list[ProposedBox]]:
    """Read proposed boxes grouped by photo key; every photo appears, possibly with none."""
    boxes: dict[str, list[ProposedBox]] = {key: [] for key in photos}
    seen: set[tuple[str, str]] = set()
    for line_number, row in _read_rows(path, BOX_COLUMNS):
        try:
            key = row["key"]
            if key not in photos:
                raise ValueError(f"box for unknown photo {key!r}")
            if row["class_name"] not in CLASS_NAMES:
                raise ValueError(f"unknown class {row['class_name']!r}")
            box = PixelBox(
                float(row["x_min_px"]),
                float(row["y_min_px"]),
                float(row["x_max_px"]),
                float(row["y_max_px"]),
            )
            photo = photos[key]
            if not box.fits_within(photo.width_px, photo.height_px):
                raise ValueError(f"box {box} exceeds the {photo.width_px}x{photo.height_px} image")
            proposed = ProposedBox(
                key, row["box_id"], row["class_name"], float(row["confidence"]), box
            )
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
        if (key, proposed.box_id) in seen:
            raise ValueError(f"{path}:{line_number}: duplicate box {proposed.box_id} in {key}")
        seen.add((key, proposed.box_id))
        boxes[key].append(proposed)
    return boxes
