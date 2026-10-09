from collections import Counter
from pathlib import Path

from objhist_detection.kaggle_kitchenware.splits import (
    Split,
    assign_splits,
    read_splits,
    split_counts,
    write_splits,
)


def rows_for(class_name: str, count: int, offset: int = 0) -> list[tuple[str, str, str]]:
    return [(f"{class_name}{offset + i:04}", class_name, f"{class_name}-{i}") for i in range(count)]


def test_each_class_is_split_70_15_15() -> None:
    rows = rows_for("cup", 100) + rows_for("fork", 20)

    counts = split_counts(assign_splits(rows))

    assert counts[Split.TRAIN] == Counter({"cup": 70, "fork": 14})
    assert counts[Split.VAL] == Counter({"cup": 15, "fork": 3})
    assert counts[Split.TEST] == Counter({"cup": 15, "fork": 3})


def test_identical_images_share_a_split_even_with_different_labels() -> None:
    rows = rows_for("cup", 40) + rows_for("glass", 40)
    duplicates = [("dup-a", "cup", "same-bytes"), ("dup-b", "glass", "same-bytes")]

    for seed in range(20):
        by_id = {a.image_id: a for a in assign_splits(rows + duplicates, seed=seed)}
        assert by_id["dup-a"].split == by_id["dup-b"].split


def test_split_is_deterministic_and_independent_of_input_order() -> None:
    rows = rows_for("plate", 50) + rows_for("spoon", 30)

    assert assign_splits(rows) == assign_splits(list(reversed(rows)))
    assert assign_splits(rows, seed=1) != assign_splits(rows, seed=2)


def test_splits_round_trip_through_csv(tmp_path: Path) -> None:
    assignments = assign_splits(rows_for("knife", 10))
    path = tmp_path / "splits.csv"

    write_splits(path, assignments)

    assert list(read_splits(path).values()) == assignments
