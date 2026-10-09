"""Box proposals for single-object photos and the reasons a proposal needs human review.

Each Kaggle photo shows one object, so the highest-confidence detector box is taken as that
object and labeled with the class from `train.csv`. The detector's own class guess is kept
for reference only. On the full set it disagreed with the label on 29% of images, almost all
systematically (glass read as cup, spoons and forks as knife), so it is not a review reason.
"""

import csv
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from objhist_detection.kaggle_kitchenware.source import CLASS_NAMES

LOW_CONFIDENCE_THRESHOLD = 0.4
"""Top boxes below this confidence go to review; chosen from the probe, not from test data."""

SECOND_OBJECT_CONFIDENCE_THRESHOLD = 0.4
"""A further box at least this confident may be a second object that would go unlabeled."""

SAME_OBJECT_CONTAINMENT = 0.5
"""A further box with at least this fraction of its area inside the top box is treated as
part of the same object (for example a cup's handle), not as a second object."""


class ReviewReason(StrEnum):
    NO_BOX = "no_box"
    LOW_CONFIDENCE = "low_confidence"
    SECOND_OBJECT = "second_object"
    DUPLICATE_IMAGE = "duplicate_image"


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


@dataclass(frozen=True)
class BoxCandidate:
    class_name: str
    confidence: float
    box: PixelBox


@dataclass(frozen=True)
class BoxProposal:
    image_id: str
    class_name: str
    """Label from `train.csv`."""
    width_px: int
    height_px: int
    """Upright image size, after applying the EXIF orientation."""
    exif_orientation: int
    sha256: str
    box: PixelBox | None
    confidence: float | None
    detector_class: str | None
    """The detector's class for the top box; for reference, never the label."""
    second_object_confidence: float | None
    review_reasons: tuple[ReviewReason, ...]

    @property
    def needs_review(self) -> bool:
        return bool(self.review_reasons)


def _candidate_rank(candidate: BoxCandidate) -> tuple[float, str, float, float]:
    # Highest confidence first; ties broken deterministically by class and position.
    return (-candidate.confidence, candidate.class_name, candidate.box.x_min, candidate.box.y_min)


def select_top_box(
    candidates: Sequence[BoxCandidate],
) -> tuple[BoxCandidate | None, float | None]:
    """Return the most confident candidate and the confidence of the strongest second object.

    A candidate counts as a second object when it is confident enough and mostly lies
    outside the top box. Candidates mostly inside the top box are parts of the same object.
    """
    if not candidates:
        return None, None
    ranked = sorted(candidates, key=_candidate_rank)
    top = ranked[0]
    for other in ranked[1:]:
        if other.confidence < SECOND_OBJECT_CONFIDENCE_THRESHOLD:
            break
        if top.box.intersection_area(other.box) / other.box.area < SAME_OBJECT_CONTAINMENT:
            return top, other.confidence
    return top, None


def review_reasons_for(
    top: BoxCandidate | None, second_object_confidence: float | None
) -> tuple[ReviewReason, ...]:
    if top is None:
        return (ReviewReason.NO_BOX,)
    reasons: list[ReviewReason] = []
    if top.confidence < LOW_CONFIDENCE_THRESHOLD:
        reasons.append(ReviewReason.LOW_CONFIDENCE)
    if second_object_confidence is not None:
        reasons.append(ReviewReason.SECOND_OBJECT)
    return tuple(reasons)


def flag_duplicate_images(proposals: Sequence[BoxProposal]) -> list[BoxProposal]:
    """Flag every image whose bytes are identical to another image's.

    In the Kaggle set every such pair carries two different labels, so at most one of them
    can be right and a reviewer has to decide.
    """
    ids_by_hash: dict[str, list[str]] = defaultdict(list)
    for proposal in proposals:
        ids_by_hash[proposal.sha256].append(proposal.image_id)
    flagged: list[BoxProposal] = []
    for proposal in proposals:
        is_duplicate = len(ids_by_hash[proposal.sha256]) > 1
        if is_duplicate and ReviewReason.DUPLICATE_IMAGE not in proposal.review_reasons:
            reasons = (*proposal.review_reasons, ReviewReason.DUPLICATE_IMAGE)
            proposal = replace(proposal, review_reasons=reasons)
        flagged.append(proposal)
    return flagged


PROPOSAL_COLUMNS = [
    "image_id",
    "class_name",
    "width_px",
    "height_px",
    "exif_orientation",
    "sha256",
    "x_min_px",
    "y_min_px",
    "x_max_px",
    "y_max_px",
    "confidence",
    "detector_class",
    "second_object_confidence",
    "review_reasons",
]
_REASON_SEPARATOR = "|"


def _optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def write_proposals(path: Path, proposals: Iterable[BoxProposal]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PROPOSAL_COLUMNS)
        writer.writeheader()
        for proposal in proposals:
            box = proposal.box
            writer.writerow(
                {
                    "image_id": proposal.image_id,
                    "class_name": proposal.class_name,
                    "width_px": proposal.width_px,
                    "height_px": proposal.height_px,
                    "exif_orientation": proposal.exif_orientation,
                    "sha256": proposal.sha256,
                    "x_min_px": "" if box is None else f"{box.x_min:.2f}",
                    "y_min_px": "" if box is None else f"{box.y_min:.2f}",
                    "x_max_px": "" if box is None else f"{box.x_max:.2f}",
                    "y_max_px": "" if box is None else f"{box.y_max:.2f}",
                    "confidence": _optional_float(proposal.confidence),
                    "detector_class": proposal.detector_class or "",
                    "second_object_confidence": _optional_float(proposal.second_object_confidence),
                    "review_reasons": _REASON_SEPARATOR.join(proposal.review_reasons),
                }
            )


def _parse_optional_float(value: str) -> float | None:
    return float(value) if value else None


def _parse_row(row: dict[str, str]) -> BoxProposal:
    class_name = row["class_name"]
    if class_name not in CLASS_NAMES:
        raise ValueError(f"unknown class {class_name!r}")
    width_px = int(row["width_px"])
    height_px = int(row["height_px"])
    corners = [row["x_min_px"], row["y_min_px"], row["x_max_px"], row["y_max_px"]]
    box = None
    if all(corners):
        box = PixelBox(*(float(value) for value in corners))
        if box.x_max > width_px or box.y_max > height_px:
            raise ValueError(f"box {box} exceeds the {width_px}x{height_px} image")
    elif any(corners):
        raise ValueError("box corners must be all present or all empty")
    reasons_text = row["review_reasons"]
    reasons = tuple(ReviewReason(value) for value in reasons_text.split(_REASON_SEPARATOR) if value)
    if (box is None) != (ReviewReason.NO_BOX in reasons):
        raise ValueError("no_box must be listed exactly when the box is empty")
    return BoxProposal(
        image_id=row["image_id"],
        class_name=class_name,
        width_px=width_px,
        height_px=height_px,
        exif_orientation=int(row["exif_orientation"]),
        sha256=row["sha256"],
        box=box,
        confidence=_parse_optional_float(row["confidence"]),
        detector_class=row["detector_class"] or None,
        second_object_confidence=_parse_optional_float(row["second_object_confidence"]),
        review_reasons=reasons,
    )


def read_proposals(path: Path) -> dict[str, BoxProposal]:
    """Read proposals keyed by image ID, rejecting malformed rows with their line number."""
    proposals: dict[str, BoxProposal] = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != PROPOSAL_COLUMNS:
            raise ValueError(f"{path} has columns {reader.fieldnames}, expected {PROPOSAL_COLUMNS}")
        for line_number, row in enumerate(reader, start=2):
            try:
                proposal = _parse_row(row)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if proposal.image_id in proposals:
                raise ValueError(f"{path}:{line_number}: duplicate image {proposal.image_id}")
            proposals[proposal.image_id] = proposal
    return proposals
