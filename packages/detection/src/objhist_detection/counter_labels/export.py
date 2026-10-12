"""Write reviewed counter photos as a YOLO-format detection dataset.

Inclusion rules:

- Every photo of a validation or test session must be reviewed; the export stops otherwise.
- Unreviewed training photos are left out (their labels would be incomplete).
- A reviewed photo with no objects is kept with an empty label file: an empty counter is a
  valid example of background.
- A photo whose bytes changed since its boxes were proposed stops the export.
"""

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from objhist_detection.annotation import CLASS_NAMES, file_sha256, yolo_label_line
from objhist_detection.counter_labels.proposals import PhotoRecord, ProposedBox
from objhist_detection.counter_labels.review import PhotoReview
from objhist_detection.counter_labels.sessions import SessionPhoto
from objhist_detection.yolo_dataset import (
    MANIFEST_FILENAME,
    SPLITS,
    SUMMARY_FILENAME,
    ExportError,
    place_upright_image,
    prepare_output_dir,
    write_data_yaml,
)


@dataclass(frozen=True)
class ExportItem:
    photo: PhotoRecord
    split: str
    review: PhotoReview


@dataclass(frozen=True)
class ExportPlan:
    items: list[ExportItem]
    unreviewed_train: list[str]


def plan_export(
    photos: Mapping[str, PhotoRecord],
    splits: Mapping[str, str],
    reviews: Mapping[str, PhotoReview],
) -> ExportPlan:
    unassigned = sorted({photo.session_id for photo in photos.values()} - set(splits))
    if unassigned:
        raise ExportError(f"sessions without a split: {', '.join(unassigned)}; run the split step")
    items: list[ExportItem] = []
    unreviewed_train: list[str] = []
    unreviewed_held_out: list[str] = []
    for key in sorted(photos):
        photo = photos[key]
        split = splits[photo.session_id]
        review = reviews.get(key)
        if review is not None:
            items.append(ExportItem(photo, split, review))
        elif split == "train":
            unreviewed_train.append(key)
        else:
            unreviewed_held_out.append(key)
    if unreviewed_held_out:
        raise ExportError(
            f"{len(unreviewed_held_out)} validation/test photos are not reviewed, for example "
            f"{', '.join(unreviewed_held_out[:5])}; finish them on the review page"
        )
    return ExportPlan(items=items, unreviewed_train=unreviewed_train)


def label_text(item: ExportItem) -> str:
    return "".join(
        yolo_label_line(
            CLASS_NAMES.index(reviewed.class_name),
            reviewed.pixel_box(),
            item.photo.width_px,
            item.photo.height_px,
        )
        for reviewed in item.review.objects
    )


def review_counts(review: PhotoReview, proposals: Sequence[ProposedBox]) -> dict[str, int]:
    """How the reviewer changed the proposals of one photo."""
    detector_class = {proposed.box_id: proposed.class_name for proposed in proposals}
    kept = [reviewed for reviewed in review.objects if reviewed.proposal_id is not None]
    relabeled = sum(
        reviewed.class_name != detector_class[reviewed.proposal_id] for reviewed in kept
    )
    return {
        "proposed": len(proposals),
        "kept_same_class": len(kept) - relabeled,
        "kept_relabeled": relabeled,
        "rejected": len(review.rejected_proposals),
        "drawn": len(review.objects) - len(kept),
    }


def write_export(
    plan: ExportPlan,
    proposals: Mapping[str, Sequence[ProposedBox]],
    session_photos: Mapping[str, SessionPhoto],
    out_dir: Path,
    input_files: Mapping[str, Path],
    overwrite: bool = False,
) -> dict[str, object]:
    """Write images, labels, `data.yaml`, the manifest, and the summary; return the summary."""
    for item in plan.items:
        source = session_photos.get(item.photo.key)
        if source is None:
            raise ExportError(f"{item.photo.key} is reviewed but its photo file is missing")
        if file_sha256(source.path) != item.photo.sha256:
            raise ExportError(f"{source.path} changed since its boxes were proposed; propose again")
    prepare_output_dir(out_dir, overwrite)

    objects_per_split = {split: Counter[str]() for split in SPLITS}
    photos_per_split: Counter[str] = Counter()
    review_totals: Counter[str] = Counter()
    manifest_rows: list[list[object]] = []
    for item in plan.items:
        source = session_photos[item.photo.key]
        export_name = source.export_stem + source.path.suffix.lower()
        place_upright_image(
            source.path, out_dir / "images" / item.split / export_name, item.photo.exif_orientation
        )
        label_path = out_dir / "labels" / item.split / f"{source.export_stem}.txt"
        label_path.write_text(label_text(item), encoding="utf-8")
        counts = review_counts(item.review, proposals[item.photo.key])
        review_totals.update(counts)
        photos_per_split[item.split] += 1
        objects_per_split[item.split].update(
            reviewed.class_name for reviewed in item.review.objects
        )
        manifest_rows.append(
            [
                item.photo.key,
                item.split,
                export_name,
                len(item.review.objects),
                counts["kept_relabeled"],
                counts["drawn"],
                counts["rejected"],
                item.review.reviewer,
                item.photo.sha256,
            ]
        )

    write_data_yaml(out_dir)
    with (out_dir / MANIFEST_FILENAME).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "key",
                "split",
                "export_name",
                "objects",
                "relabeled",
                "drawn",
                "rejected_proposals",
                "reviewer",
                "source_sha256",
            ]
        )
        writer.writerows(manifest_rows)

    summary: dict[str, object] = {
        "class_names": list(CLASS_NAMES),
        "photos_per_split": {split: photos_per_split[split] for split in SPLITS},
        "objects_per_split": {
            split: {name: objects_per_split[split][name] for name in CLASS_NAMES}
            for split in SPLITS
        },
        "unreviewed_train_photos_excluded": len(plan.unreviewed_train),
        "review_totals": {
            name: review_totals[name]
            for name in ("proposed", "kept_same_class", "kept_relabeled", "rejected", "drawn")
        },
        "reviewers": sorted({item.review.reviewer for item in plan.items}),
        "created_at": datetime.now(UTC).isoformat(),
        "inputs_sha256": {name: file_sha256(path) for name, path in input_files.items()},
        "manifest_sha256": file_sha256(out_dir / MANIFEST_FILENAME),
    }
    (out_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    return summary
