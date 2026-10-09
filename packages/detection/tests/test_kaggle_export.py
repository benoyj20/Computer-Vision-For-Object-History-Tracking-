from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from PIL import Image

from objhist_detection.kaggle_kitchenware.export import (
    MANIFEST_FILENAME,
    Exclusion,
    ExportError,
    ExportItem,
    LabelSource,
    plan_export,
    write_export,
    yolo_label_line,
)
from objhist_detection.kaggle_kitchenware.proposals import BoxProposal, PixelBox, ReviewReason
from objhist_detection.kaggle_kitchenware.review import Review, ReviewCategory, Verdict
from objhist_detection.kaggle_kitchenware.source import EXIF_ORIENTATION_TAG
from objhist_detection.kaggle_kitchenware.splits import Split, SplitAssignment

ProposalFactory = Callable[..., BoxProposal]
REVIEWED_AT = datetime(2026, 10, 9, 15, 0, tzinfo=UTC)

Dataset = tuple[dict[str, BoxProposal], dict[str, SplitAssignment], dict[str, ReviewCategory]]


def review(image_id: str, verdict: Verdict, class_name: str = "cup") -> Review:
    return Review(image_id, verdict, class_name, "Benoy", REVIEWED_AT)


@pytest.fixture
def dataset(make_proposal: ProposalFactory) -> Dataset:
    """A held-out image, a flagged and a clean training image, and a held-out image
    without a box."""
    entries = [
        (make_proposal("val1"), Split.VAL, ReviewCategory.HELD_OUT),
        (make_proposal("nobox", box=None), Split.TEST, ReviewCategory.HELD_OUT),
        (
            make_proposal("flag", reasons=(ReviewReason.LOW_CONFIDENCE,)),
            Split.TRAIN,
            ReviewCategory.FLAGGED,
        ),
        (make_proposal("clean"), Split.TRAIN, None),
    ]
    proposals = {proposal.image_id: proposal for proposal, _, _ in entries}
    splits = {
        proposal.image_id: SplitAssignment(
            proposal.image_id, proposal.class_name, proposal.sha256, split
        )
        for proposal, split, _ in entries
    }
    queue = {proposal.image_id: category for proposal, _, category in entries if category}
    return proposals, splits, queue


def held_out_reviews() -> dict[str, Review]:
    return {"val1": review("val1", Verdict.ACCEPT), "nobox": review("nobox", Verdict.REJECT)}


def test_unreviewed_validation_or_test_image_stops_the_export(dataset: Dataset) -> None:
    reviews = {"val1": review("val1", Verdict.ACCEPT)}

    with pytest.raises(ExportError, match="1 validation/test images have no review.*nobox"):
        plan_export(*dataset, reviews)


def test_unreviewed_flagged_training_image_is_excluded(dataset: Dataset) -> None:
    plan = plan_export(*dataset, held_out_reviews())

    assert [item.proposal.image_id for item in plan.items] == ["clean", "val1"]
    assert plan.exclusions == {"flag": Exclusion.FLAGGED_UNREVIEWED, "nobox": Exclusion.NO_BOX}
    assert plan.items[0].label_source is LabelSource.UNREVIEWED


def test_reviewer_class_replaces_the_csv_label(dataset: Dataset) -> None:
    reviews = held_out_reviews() | {"flag": review("flag", Verdict.ACCEPT, "glass")}

    items = {item.proposal.image_id: item for item in plan_export(*dataset, reviews).items}

    assert items["flag"].class_name == "glass"
    assert items["flag"].label_source is LabelSource.RELABELED
    assert items["val1"].label_source is LabelSource.REVIEWED


def test_rejected_clean_training_image_is_excluded(dataset: Dataset) -> None:
    proposals, splits, queue = dataset
    queue = queue | {"clean": ReviewCategory.SPOT_CHECK}
    reviews = held_out_reviews() | {"clean": review("clean", Verdict.REJECT)}

    plan = plan_export(proposals, splits, queue, reviews)

    assert plan.exclusions["clean"] is Exclusion.REJECTED


def test_accepting_an_image_without_a_box_is_an_error(dataset: Dataset) -> None:
    reviews = held_out_reviews() | {"nobox": review("nobox", Verdict.ACCEPT)}

    with pytest.raises(ExportError, match="nobox was accepted but has no proposed box"):
        plan_export(*dataset, reviews)


def test_identical_images_with_different_classes_stop_the_export(
    make_proposal: ProposalFactory,
) -> None:
    proposals = {
        "a": make_proposal("a", "knife", sha256="same"),
        "b": make_proposal("b", "fork", sha256="same"),
    }
    splits = {
        key: SplitAssignment(key, value.class_name, "same", Split.TRAIN)
        for key, value in proposals.items()
    }

    with pytest.raises(ExportError, match=r"\[\['a', 'b'\]\]"):
        plan_export(proposals, splits, {}, {})


def test_label_line_is_normalized_center_and_size(make_proposal: ProposalFactory) -> None:
    item = ExportItem(
        make_proposal("a", "glass", box=PixelBox(10, 50, 60, 150)),
        Split.TRAIN,
        "glass",
        LabelSource.UNREVIEWED,
    )

    assert yolo_label_line(item) == "2 0.350000 0.500000 0.500000 0.500000\n"


def make_source_images(directory: Path, image_ids: list[str]) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for image_id in image_ids:
        paths[image_id] = directory / f"{image_id}.jpg"
        Image.new("RGB", (100, 200), "white").save(paths[image_id])
    return paths


def test_writes_yolo_dataset_with_manifest_and_summary(tmp_path: Path, dataset: Dataset) -> None:
    proposals, splits, queue = dataset
    reviews = held_out_reviews()
    plan = plan_export(proposals, splits, queue, reviews)
    image_paths = make_source_images(tmp_path / "source", list(proposals))
    out_dir = tmp_path / "out"

    summary = write_export(plan, queue, reviews, image_paths, out_dir, input_files={})

    assert (out_dir / "images" / "val" / "val1.jpg").read_bytes() == image_paths[
        "val1"
    ].read_bytes()
    assert (out_dir / "labels" / "train" / "clean.txt").read_text().startswith("0 ")
    assert not (out_dir / "images" / "train" / "flag.jpg").exists()
    config = yaml.safe_load((out_dir / "data.yaml").read_text())
    assert config["names"] == {0: "cup", 1: "fork", 2: "glass", 3: "knife", 4: "plate", 5: "spoon"}
    assert config["val"] == "images/val"
    manifest_lines = (out_dir / MANIFEST_FILENAME).read_text().splitlines()
    assert manifest_lines[1:] == [
        "clean,train,cup,unreviewed,hash-clean",
        "val1,val,cup,reviewed,hash-val1",
    ]
    assert summary["exclusions"] == {"flagged_unreviewed": 1, "no_box": 1}
    assert summary["review_outcomes"]["held_out"] == {
        "queued": 2,
        "reviewed": 2,
        "rejected": 1,
        "relabeled": 0,
    }


def test_rotated_photo_is_stored_upright(tmp_path: Path, make_proposal: ProposalFactory) -> None:
    source = tmp_path / "source" / "rot.jpg"
    source.parent.mkdir()
    exif = Image.Exif()
    exif[EXIF_ORIENTATION_TAG] = 6
    Image.new("RGB", (200, 100), "white").save(source, exif=exif)
    proposal = make_proposal("rot", exif_orientation=6)
    plan = plan_export(
        {"rot": proposal},
        {"rot": SplitAssignment("rot", "cup", proposal.sha256, Split.TRAIN)},
        {},
        {},
    )

    write_export(plan, {}, {}, {"rot": source}, tmp_path / "out", input_files={})

    with Image.open(tmp_path / "out" / "images" / "train" / "rot.jpg") as stored:
        assert stored.size == (100, 200)
        assert stored.getexif().get(EXIF_ORIENTATION_TAG, 1) == 1


def test_overwrite_only_replaces_a_previous_export(tmp_path: Path, dataset: Dataset) -> None:
    proposals, splits, queue = dataset
    reviews = held_out_reviews()
    plan = plan_export(proposals, splits, queue, reviews)
    image_paths = make_source_images(tmp_path / "source", list(proposals))
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "notes.txt").write_text("keep me")

    with pytest.raises(ExportError, match="not empty"):
        write_export(plan, queue, reviews, image_paths, unrelated, input_files={})
    with pytest.raises(ExportError, match="not a previous export"):
        write_export(plan, queue, reviews, image_paths, unrelated, {}, overwrite=True)
    assert (unrelated / "notes.txt").read_text() == "keep me"

    out_dir = tmp_path / "out"
    write_export(plan, queue, reviews, image_paths, out_dir, input_files={})
    write_export(plan, queue, reviews, image_paths, out_dir, {}, overwrite=True)
    assert (out_dir / MANIFEST_FILENAME).is_file()
