"""Assign whole counter sessions to train, val, or test.

Photos of one session share the counter, lighting, and objects, so splitting a session would
leak near-duplicates into evaluation. Assignments are frozen once written: new sessions are
added, existing ones never move unless the file is edited deliberately.
"""

import csv
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from objhist_detection.yolo_dataset import SPLITS

TARGET_PHOTO_FRACTIONS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_COLUMNS = ["session_id", "split"]


def assign_new_sessions(
    existing: Mapping[str, str],
    photos_per_session: Mapping[str, int],
    overrides: Mapping[str, str],
) -> dict[str, str]:
    """Return all assignments: existing ones unchanged, overrides, then new sessions.

    New sessions are taken in name order (session IDs start with the date) and each goes to
    the split furthest below its target share of photos once that session is counted. Ties
    go to train, then val, then test. The rule is deterministic, so rerunning it after adding
    a session never reshuffles earlier ones.
    """
    for session_id, split in overrides.items():
        if split not in SPLITS:
            raise ValueError(
                f"unknown split {split!r} for {session_id}; use one of {', '.join(SPLITS)}"
            )
        if session_id in existing and existing[session_id] != split:
            raise ValueError(
                f"{session_id} is already frozen in {existing[session_id]}; "
                "edit the splits file deliberately to move it"
            )
        if session_id not in photos_per_session:
            raise ValueError(f"no photos for session {session_id}")
    assignments = dict(existing) | dict(overrides)
    photos_per_split: Counter[str] = Counter()
    for session_id, split in assignments.items():
        photos_per_split[split] += photos_per_session.get(session_id, 0)
    for session_id in sorted(set(photos_per_session) - set(assignments)):
        total = sum(photos_per_split.values()) + photos_per_session[session_id]
        split = max(
            SPLITS,
            key=lambda name: (
                TARGET_PHOTO_FRACTIONS[name] * total - photos_per_split[name],
                -SPLITS.index(name),
            ),
        )
        assignments[session_id] = split
        photos_per_split[split] += photos_per_session[session_id]
    return assignments


HELD_OUT_SPLITS = frozenset({"val", "test"})


def check_held_out_proposals(
    assignments: Mapping[str, str], sessions_with_trained_proposals: set[str]
) -> None:
    """Refuse validation or test sessions whose boxes were proposed by a trained detector.

    Their labels measure that detector, so they must start from an independent proposer.
    """
    bad = sorted(
        session
        for session in sessions_with_trained_proposals
        if assignments.get(session) in HELD_OUT_SPLITS
    )
    if bad:
        raise ValueError(
            f"{', '.join(bad)} would be held out but has trained-detector proposals; assign it "
            "to train (--assign SESSION=train) or propose it again with YOLOE (--force)"
        )


def write_splits(path: Path, assignments: Mapping[str, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(SPLIT_COLUMNS)
        writer.writerows(sorted(assignments.items()))


def read_splits(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    assignments: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != SPLIT_COLUMNS:
            raise ValueError(f"{path} has columns {reader.fieldnames}, expected {SPLIT_COLUMNS}")
        for line_number, row in enumerate(reader, start=2):
            if row["split"] not in SPLITS:
                raise ValueError(f"{path}:{line_number}: unknown split {row['split']!r}")
            if row["session_id"] in assignments:
                raise ValueError(f"{path}:{line_number}: duplicate session {row['session_id']}")
            assignments[row["session_id"]] = row["split"]
    return assignments
