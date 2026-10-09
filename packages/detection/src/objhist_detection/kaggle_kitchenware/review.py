"""Human review of proposed boxes: the review queue, the reviews file, and the review page.

The queue holds every validation and test image (all must be checked), every flagged
training image, and a seeded random spot-check sample of unflagged training images. The
spot-check measures the error rate of the training labels that nobody looked at.
"""

import csv
import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from importlib import resources
from pathlib import Path

from objhist_detection.kaggle_kitchenware.proposals import BoxProposal
from objhist_detection.kaggle_kitchenware.source import CLASS_NAMES
from objhist_detection.kaggle_kitchenware.splits import Split, SplitAssignment

SPOT_CHECK_SEED = 5330
DEFAULT_SPOT_CHECK_SIZE = 300
"""About ±2.5 percentage points (95%) on a 5% error rate."""


class ReviewCategory(StrEnum):
    HELD_OUT = "held_out"
    FLAGGED = "flagged"
    SPOT_CHECK = "spot_check"


class Verdict(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"


@dataclass(frozen=True)
class QueueItem:
    image_id: str
    category: ReviewCategory


@dataclass(frozen=True)
class Review:
    image_id: str
    verdict: Verdict
    class_name: str
    """The reviewer's class; differs from the `train.csv` label when relabeled."""
    reviewer: str
    reviewed_at: datetime


def build_review_queue(
    proposals: Mapping[str, BoxProposal],
    splits: Mapping[str, SplitAssignment],
    spot_check_size: int = DEFAULT_SPOT_CHECK_SIZE,
    seed: int = SPOT_CHECK_SEED,
) -> list[QueueItem]:
    if proposals.keys() != splits.keys():
        raise ValueError("proposals and splits cover different images; regenerate one of them")
    queue: list[QueueItem] = []
    unflagged_train: list[str] = []
    for image_id in sorted(proposals):
        if splits[image_id].split is not Split.TRAIN:
            queue.append(QueueItem(image_id, ReviewCategory.HELD_OUT))
        elif proposals[image_id].needs_review:
            queue.append(QueueItem(image_id, ReviewCategory.FLAGGED))
        else:
            unflagged_train.append(image_id)
    if spot_check_size > len(unflagged_train):
        raise ValueError(
            f"spot-check size {spot_check_size} exceeds the {len(unflagged_train)} "
            "unflagged training images"
        )
    sample = random.Random(seed).sample(unflagged_train, spot_check_size)
    queue.extend(QueueItem(image_id, ReviewCategory.SPOT_CHECK) for image_id in sorted(sample))
    return queue


QUEUE_COLUMNS = ["image_id", "category"]


def write_queue(path: Path, queue: Iterable[QueueItem]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(QUEUE_COLUMNS)
        writer.writerows([item.image_id, item.category] for item in queue)


def read_queue(path: Path) -> dict[str, ReviewCategory]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != QUEUE_COLUMNS:
            raise ValueError(f"{path} has columns {reader.fieldnames}, expected {QUEUE_COLUMNS}")
        queue: dict[str, ReviewCategory] = {}
        for line_number, row in enumerate(reader, start=2):
            if row["image_id"] in queue:
                raise ValueError(f"{path}:{line_number}: duplicate image {row['image_id']}")
            queue[row["image_id"]] = ReviewCategory(row["category"])
    return queue


REVIEW_COLUMNS = ["image_id", "verdict", "class_name", "reviewer", "reviewed_at"]


def _parse_review(row: dict[str, str]) -> Review:
    class_name = row["class_name"]
    if class_name not in CLASS_NAMES:
        raise ValueError(f"unknown class {class_name!r}")
    reviewer = row["reviewer"].strip()
    if not reviewer:
        raise ValueError("reviewer is empty")
    reviewed_at = datetime.fromisoformat(row["reviewed_at"])
    if reviewed_at.tzinfo is None:
        raise ValueError("reviewed_at must include a UTC offset")
    return Review(row["image_id"], Verdict(row["verdict"]), class_name, reviewer, reviewed_at)


def read_reviews(path: Path, queue: Mapping[str, ReviewCategory]) -> dict[str, Review]:
    """Read the reviews downloaded from the review page; every image must be in the queue."""
    reviews: dict[str, Review] = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != REVIEW_COLUMNS:
            raise ValueError(f"{path} has columns {reader.fieldnames}, expected {REVIEW_COLUMNS}")
        for line_number, row in enumerate(reader, start=2):
            try:
                review = _parse_review(row)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if review.image_id not in queue:
                raise ValueError(f"{path}:{line_number}: {review.image_id} is not in the queue")
            if review.image_id in reviews:
                raise ValueError(f"{path}:{line_number}: duplicate review of {review.image_id}")
            reviews[review.image_id] = review
    return reviews


def queue_fingerprint(queue: Sequence[QueueItem]) -> str:
    """Short hash of the queue; the page keys its saved progress by it."""
    text = "\n".join(f"{item.image_id},{item.category}" for item in queue)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _page_item(
    item: QueueItem,
    proposal: BoxProposal,
    split: Split,
    image_uri: str,
    review: Review | None,
) -> dict[str, object]:
    box = proposal.box
    return {
        "id": item.image_id,
        "src": image_uri,
        "category": item.category,
        "split": split,
        "label": proposal.class_name,
        "width": proposal.width_px,
        "height": proposal.height_px,
        "box": None if box is None else [box.x_min, box.y_min, box.x_max, box.y_max],
        "confidence": proposal.confidence,
        "detectorClass": proposal.detector_class,
        "reasons": list(proposal.review_reasons),
        "review": None
        if review is None
        else {
            "verdict": review.verdict,
            "className": review.class_name,
            "reviewer": review.reviewer,
            "reviewedAt": review.reviewed_at.isoformat(),
        },
    }


def render_review_page(
    queue: Sequence[QueueItem],
    proposals: Mapping[str, BoxProposal],
    splits: Mapping[str, SplitAssignment],
    image_paths: Mapping[str, Path],
    reviews: Mapping[str, Review],
) -> str:
    """Return a self-contained HTML page that shows each queued image with its box."""
    data = {
        "queueId": queue_fingerprint(queue),
        "classes": list(CLASS_NAMES),
        "items": [
            _page_item(
                item,
                proposals[item.image_id],
                splits[item.image_id].split,
                image_paths[item.image_id].resolve().as_uri(),
                reviews.get(item.image_id),
            )
            for item in queue
        ],
    }
    # Escape "<" so no value can close the <script> element that embeds the data.
    data_json = json.dumps(data).replace("<", "\\u003c")
    template = (
        resources.files("objhist_detection.kaggle_kitchenware")
        .joinpath("review_page.html")
        .read_text("utf-8")
    )
    return template.replace("__REVIEW_DATA__", data_json)
