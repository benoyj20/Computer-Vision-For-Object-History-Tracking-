"""Write reviewed proposals as a YOLO-format detection dataset.

Inclusion rules, applied per image:

- Validation and test images must all have a review; the export stops otherwise.
- Flagged training images are included only when a reviewer accepted them.
- Unflagged training images are included unless a reviewer rejected them.
- Images without a box are never included, and a review cannot accept one.

A reviewer's class replaces the `train.csv` label. Byte-identical images that end up with
different classes stop the export, because one of the two labels must be wrong.
"""

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from objhist_detection import annotation
from objhist_detection.annotation import CLASS_NAMES, file_sha256
from objhist_detection.kaggle_kitchenware.proposals import BoxProposal
from objhist_detection.kaggle_kitchenware.review import Review, ReviewCategory, Verdict
from objhist_detection.kaggle_kitchenware.splits import Split, SplitAssignment
from objhist_detection.yolo_dataset import (
    MANIFEST_FILENAME,
    SUMMARY_FILENAME,
    ExportError,
    place_upright_image,
    prepare_output_dir,
    write_data_yaml,
)


class LabelSource(StrEnum):
    UNREVIEWED = "unreviewed"
    REVIEWED = "reviewed"
    RELABELED = "relabeled"


class Exclusion(StrEnum):
    NO_BOX = "no_box"
    REJECTED = "rejected"
    FLAGGED_UNREVIEWED = "flagged_unreviewed"


@dataclass(frozen=True)
class ExportItem:
    proposal: BoxProposal
    split: Split
    class_name: str
    label_source: LabelSource


@dataclass(frozen=True)
class ExportPlan:
    items: list[ExportItem]
    exclusions: dict[str, Exclusion]


def plan_export(
    proposals: Mapping[str, BoxProposal],
    splits: Mapping[str, SplitAssignment],
    queue: Mapping[str, ReviewCategory],
    reviews: Mapping[str, Review],
) -> ExportPlan:
    if proposals.keys() != splits.keys():
        raise ExportError("proposals and splits cover different images")
    unreviewed_held_out = sorted(
        image_id
        for image_id, category in queue.items()
        if category is ReviewCategory.HELD_OUT and image_id not in reviews
    )
    if unreviewed_held_out:
        raise ExportError(
            f"{len(unreviewed_held_out)} validation/test images have no review, for example "
            f"{', '.join(unreviewed_held_out[:5])}; finish them on the review page"
        )

    items: list[ExportItem] = []
    exclusions: dict[str, Exclusion] = {}
    for image_id in sorted(proposals):
        proposal = proposals[image_id]
        review = reviews.get(image_id)
        if review is not None and review.verdict is Verdict.ACCEPT and proposal.box is None:
            raise ExportError(f"{image_id} was accepted but has no proposed box")
        if proposal.box is None:
            exclusions[image_id] = Exclusion.NO_BOX
        elif review is not None and review.verdict is Verdict.REJECT:
            exclusions[image_id] = Exclusion.REJECTED
        elif review is None and proposal.needs_review:
            exclusions[image_id] = Exclusion.FLAGGED_UNREVIEWED
        else:
            if review is None:
                class_name, source = proposal.class_name, LabelSource.UNREVIEWED
            elif review.class_name != proposal.class_name:
                class_name, source = review.class_name, LabelSource.RELABELED
            else:
                class_name, source = review.class_name, LabelSource.REVIEWED
            items.append(ExportItem(proposal, splits[image_id].split, class_name, source))

    classes_by_hash: dict[str, set[str]] = defaultdict(set)
    for item in items:
        classes_by_hash[item.proposal.sha256].add(item.class_name)
    conflicts = [
        sorted(item.proposal.image_id for item in items if item.proposal.sha256 == sha256)
        for sha256, classes in classes_by_hash.items()
        if len(classes) > 1
    ]
    if conflicts:
        raise ExportError(
            f"identical images with different classes: {conflicts}; reject or relabel one of each"
        )
    return ExportPlan(items=items, exclusions=exclusions)


def yolo_label_line(item: ExportItem) -> str:
    """One YOLO label line: class index and box center/size normalized to the upright image."""
    box = item.proposal.box
    if box is None:
        raise ExportError(f"{item.proposal.image_id} has no box")
    class_index = CLASS_NAMES.index(item.class_name)
    return annotation.yolo_label_line(
        class_index, box, item.proposal.width_px, item.proposal.height_px
    )


def write_export(
    plan: ExportPlan,
    queue: Mapping[str, ReviewCategory],
    reviews: Mapping[str, Review],
    image_paths: Mapping[str, Path],
    out_dir: Path,
    input_files: Mapping[str, Path],
    overwrite: bool = False,
) -> dict[str, object]:
    """Write images, labels, `data.yaml`, the manifest, and the summary; return the summary."""
    prepare_output_dir(out_dir, overwrite)
    for item in plan.items:
        image_id = item.proposal.image_id
        place_upright_image(
            image_paths[image_id],
            out_dir / "images" / item.split / f"{image_id}.jpg",
            item.proposal.exif_orientation,
        )
        label_path = out_dir / "labels" / item.split / f"{image_id}.txt"
        label_path.write_text(yolo_label_line(item), encoding="utf-8")

    write_data_yaml(out_dir)

    with (out_dir / MANIFEST_FILENAME).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "split", "class_name", "label_source", "source_sha256"])
        for item in plan.items:
            writer.writerow(
                [
                    item.proposal.image_id,
                    item.split,
                    item.class_name,
                    item.label_source,
                    item.proposal.sha256,
                ]
            )

    summary = summarize(plan, queue, reviews)
    summary["created_at"] = datetime.now(UTC).isoformat()
    summary["inputs_sha256"] = {name: file_sha256(path) for name, path in input_files.items()}
    summary["manifest_sha256"] = file_sha256(out_dir / MANIFEST_FILENAME)
    (out_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    return summary


def summarize(
    plan: ExportPlan,
    queue: Mapping[str, ReviewCategory],
    reviews: Mapping[str, Review],
) -> dict[str, object]:
    """Counts per split and class, exclusions, and how often reviewers changed auto-labels."""
    counts: dict[str, dict[str, int]] = {
        split: {name: 0 for name in CLASS_NAMES} for split in Split
    }
    label_sources: dict[str, LabelSource] = {}
    for item in plan.items:
        counts[item.split][item.class_name] += 1
        label_sources[item.proposal.image_id] = item.label_source
    review_outcomes: dict[str, dict[str, int]] = {}
    for category in ReviewCategory:
        queued = [image_id for image_id, queued_as in queue.items() if queued_as is category]
        reviewed = [reviews[image_id] for image_id in queued if image_id in reviews]
        review_outcomes[category] = {
            "queued": len(queued),
            "reviewed": len(reviewed),
            "rejected": sum(review.verdict is Verdict.REJECT for review in reviewed),
            "relabeled": sum(
                label_sources.get(review.image_id) is LabelSource.RELABELED for review in reviewed
            ),
        }
    return {
        "class_names": list(CLASS_NAMES),
        "images_per_split": counts,
        "exclusions": dict(Counter(plan.exclusions.values())),
        "review_outcomes": review_outcomes,
    }
