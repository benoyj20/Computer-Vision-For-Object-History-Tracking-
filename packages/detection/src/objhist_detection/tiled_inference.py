"""Tiled inference: find small objects in wide counter photos.

The detector was trained on close-ups where an object fills the frame. A wide 4000 px phone
photo shrunk to the model's input size leaves cutlery a few pixels wide, so it is missed. On
crops around each object the same model found every one (2026-10-12, both test photos).

Tiled inference runs the model on overlapping tiles, so objects keep close to their real
size, plus one pass over the whole photo for objects larger than a tile, and then merges the
results. Detections are returned as `objhist_core.Detection` in full-photo pixels.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from PIL import Image

from objhist_core import BoundingBox, Detection
from objhist_detection.annotation import PixelBox


@dataclass(frozen=True)
class TilingConfig:
    """Tiling settings, recorded with results so they can be traced (see AGENTS.md)."""

    tile_px: int = 1280
    """Square tile side in photo pixels. On the 4000x2252 test photos this keeps a table
    knife about 60x310 px at the model input, the size it was trained on."""
    overlap: float = 0.25
    """Fraction of a tile shared with its neighbor, so an object cut by one tile's edge lies
    whole inside the next."""
    tile_inference_px: int = 640
    full_photo_inference_px: int = 1280
    confidence_threshold: float = 0.25
    edge_margin_px: float = 4.0
    """A tile box within this distance of an inner tile edge is a cut-off object and is
    dropped; a neighboring tile or the full-photo pass sees it whole."""
    same_class_iou: float = 0.5
    same_class_containment: float = 0.7
    """A same-class box with this fraction of its area inside a stronger one is a duplicate
    (typically a partial view from one tile)."""
    cross_class_iou: float = 0.6
    """Boxes of different classes merge only when they largely coincide (one object named two
    ways, for example a mug as "glass" from a tile and "cup" from the full pass: IoU 0.72 on
    the 2026-10-12 test photo). A spoon lying on a plate overlaps far less (IoU about 0.03)
    and stays. Set after seeing those two photos; re-check on held-out sessions."""

    def __post_init__(self) -> None:
        if self.tile_px <= 0 or not 0 <= self.overlap < 1:
            raise ValueError(f"invalid tiling: tile_px={self.tile_px}, overlap={self.overlap}")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(f"confidence_threshold {self.confidence_threshold} is not in [0, 1]")

    def as_record(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class ScoredBox:
    class_name: str
    confidence: float
    box: PixelBox


Predictor = Callable[[Image.Image, int, float], list[ScoredBox]]
"""Runs a detector on an image at an input size and confidence threshold; boxes are in the
pixels of the image passed in."""


def tile_origins(length_px: int, tile_px: int, overlap: float) -> list[int]:
    """Evenly spaced tile start offsets along one axis, from the first to the last pixel.

    Uses the fewest tiles whose neighbors share at least `overlap` of a tile.
    """
    if length_px <= tile_px:
        return [0]
    max_step_px = max(1.0, tile_px * (1 - overlap))
    gaps = math.ceil((length_px - tile_px) / max_step_px)
    return [round(index * (length_px - tile_px) / gaps) for index in range(gaps + 1)]


def is_cut_by_inner_edge(
    box: PixelBox,
    tile: PixelBox,
    width_px: int,
    height_px: int,
    margin_px: float,
) -> bool:
    """Whether `box` touches a tile edge that is not also the photo edge."""
    return (
        (tile.x_min > 0 and box.x_min - tile.x_min < margin_px)
        or (tile.y_min > 0 and box.y_min - tile.y_min < margin_px)
        or (tile.x_max < width_px and tile.x_max - box.x_max < margin_px)
        or (tile.y_max < height_px and tile.y_max - box.y_max < margin_px)
    )


def _iou(a: PixelBox, b: PixelBox) -> float:
    intersection = a.intersection_area(b)
    return intersection / (a.area + b.area - intersection)


def _is_duplicate(candidate: ScoredBox, kept: ScoredBox, config: TilingConfig) -> bool:
    iou = _iou(candidate.box, kept.box)
    if candidate.class_name != kept.class_name:
        return iou > config.cross_class_iou
    containment = candidate.box.intersection_area(kept.box) / candidate.box.area
    return iou > config.same_class_iou or containment > config.same_class_containment


def merge_boxes(boxes: Sequence[ScoredBox], config: TilingConfig) -> list[ScoredBox]:
    """Keep the most confident box of each group of duplicates, most confident first.

    Ties are broken by position and class so the result does not depend on input order.
    """
    ranked = sorted(
        boxes,
        key=lambda scored: (
            -scored.confidence,
            scored.box.x_min,
            scored.box.y_min,
            scored.class_name,
        ),
    )
    kept: list[ScoredBox] = []
    for candidate in ranked:
        if not any(_is_duplicate(candidate, existing, config) for existing in kept):
            kept.append(candidate)
    return kept


def to_detection(scored: ScoredBox, width_px: int, height_px: int) -> Detection:
    """Convert to the shared detector contract, clipped to the photo."""
    return Detection(
        class_name=scored.class_name,
        confidence=scored.confidence,
        box=BoundingBox(
            x_min=scored.box.x_min,
            y_min=scored.box.y_min,
            x_max=min(scored.box.x_max, width_px),
            y_max=min(scored.box.y_max, height_px),
        ),
    )


def detect_tiled(image: Image.Image, predict: Predictor, config: TilingConfig) -> list[Detection]:
    """Detect objects in `image` with one full-photo pass and overlapping tiles."""
    width_px, height_px = image.size
    candidates = predict(image, config.full_photo_inference_px, config.confidence_threshold)
    for y in tile_origins(height_px, config.tile_px, config.overlap):
        for x in tile_origins(width_px, config.tile_px, config.overlap):
            tile = PixelBox(
                x, y, min(x + config.tile_px, width_px), min(y + config.tile_px, height_px)
            )
            crop = image.crop((int(tile.x_min), int(tile.y_min), int(tile.x_max), int(tile.y_max)))
            for scored in predict(crop, config.tile_inference_px, config.confidence_threshold):
                box = PixelBox(
                    scored.box.x_min + x,
                    scored.box.y_min + y,
                    min(scored.box.x_max + x, tile.x_max),
                    min(scored.box.y_max + y, tile.y_max),
                )
                if not is_cut_by_inner_edge(box, tile, width_px, height_px, config.edge_margin_px):
                    candidates.append(ScoredBox(scored.class_name, scored.confidence, box))
    return [to_detection(scored, width_px, height_px) for scored in merge_boxes(candidates, config)]
