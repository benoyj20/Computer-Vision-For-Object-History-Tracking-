"""Class-stratified train/val/test split that keeps byte-identical images together.

The split depends only on the labels and image hashes, never on detector output, so it can
be frozen before any box is proposed or reviewed.
"""

import csv
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from objhist_detection.kaggle_kitchenware.source import CLASS_NAMES

SPLIT_SEED = 5330


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


SPLIT_FRACTIONS: dict[Split, float] = {Split.TRAIN: 0.70, Split.VAL: 0.15, Split.TEST: 0.15}


@dataclass(frozen=True)
class SplitAssignment:
    image_id: str
    class_name: str
    sha256: str
    split: Split


def assign_splits(
    images: Sequence[tuple[str, str, str]], seed: int = SPLIT_SEED
) -> list[SplitAssignment]:
    """Split `(image_id, class_name, sha256)` rows 70/15/15 within each class.

    Images with identical bytes form one group and always share a split; a group is
    stratified under the class of its lowest image ID. Within each class, groups are
    shuffled with `seed` and filled into train, then val, then test until each split's
    cumulative share of the class's images is reached.
    """
    groups: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for row in images:
        groups[row[2]].append(row)
    groups_by_class: dict[str, list[list[tuple[str, str, str]]]] = defaultdict(list)
    for members in groups.values():
        members.sort()
        groups_by_class[members[0][1]].append(members)

    rng = random.Random(seed)
    assignments: list[SplitAssignment] = []
    for class_name in CLASS_NAMES:
        class_groups = sorted(groups_by_class.get(class_name, []))
        rng.shuffle(class_groups)
        class_total = sum(len(group) for group in class_groups)
        cumulative_limits: list[tuple[Split, float]] = []
        cumulative_fraction = 0.0
        for split, fraction in SPLIT_FRACTIONS.items():
            cumulative_fraction += fraction
            cumulative_limits.append((split, cumulative_fraction * class_total))
        assigned = 0
        for group in class_groups:
            split = next(
                (split for split, limit in cumulative_limits if assigned < round(limit)),
                Split.TEST,
            )
            for image_id, image_class, sha256 in group:
                assignments.append(SplitAssignment(image_id, image_class, sha256, split))
            assigned += len(group)
    assignments.sort(key=lambda assignment: assignment.image_id)
    return assignments


def split_counts(assignments: Iterable[SplitAssignment]) -> dict[Split, Counter[str]]:
    counts: dict[Split, Counter[str]] = {split: Counter() for split in Split}
    for assignment in assignments:
        counts[assignment.split][assignment.class_name] += 1
    return counts


SPLIT_COLUMNS = ["image_id", "class_name", "sha256", "split"]


def write_splits(path: Path, assignments: Iterable[SplitAssignment]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(SPLIT_COLUMNS)
        for assignment in assignments:
            writer.writerow(
                [assignment.image_id, assignment.class_name, assignment.sha256, assignment.split]
            )


def read_splits(path: Path) -> dict[str, SplitAssignment]:
    assignments: dict[str, SplitAssignment] = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != SPLIT_COLUMNS:
            raise ValueError(f"{path} has columns {reader.fieldnames}, expected {SPLIT_COLUMNS}")
        for line_number, row in enumerate(reader, start=2):
            try:
                assignment = SplitAssignment(
                    image_id=row["image_id"],
                    class_name=row["class_name"],
                    sha256=row["sha256"],
                    split=Split(row["split"]),
                )
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if assignment.image_id in assignments:
                raise ValueError(f"{path}:{line_number}: duplicate image {assignment.image_id}")
            assignments[assignment.image_id] = assignment
    return assignments
