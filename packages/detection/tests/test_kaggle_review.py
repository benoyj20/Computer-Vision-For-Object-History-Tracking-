import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from objhist_detection.kaggle_kitchenware.proposals import BoxProposal, ReviewReason
from objhist_detection.kaggle_kitchenware.review import (
    REVIEW_COLUMNS,
    QueueItem,
    ReviewCategory,
    build_review_queue,
    read_queue,
    read_reviews,
    render_review_page,
    write_queue,
)
from objhist_detection.kaggle_kitchenware.splits import Split, SplitAssignment

ProposalFactory = Callable[..., BoxProposal]


def split_of(proposal: BoxProposal, split: Split) -> SplitAssignment:
    return SplitAssignment(proposal.image_id, proposal.class_name, proposal.sha256, split)


@pytest.fixture
def dataset(
    make_proposal: ProposalFactory,
) -> tuple[dict[str, BoxProposal], dict[str, SplitAssignment]]:
    """Two held-out images, one flagged training image, and ten clean training images."""
    entries = [
        (make_proposal("v1"), Split.VAL),
        (make_proposal("t1"), Split.TEST),
        (make_proposal("f1", reasons=(ReviewReason.LOW_CONFIDENCE,)), Split.TRAIN),
        *((make_proposal(f"c{i}"), Split.TRAIN) for i in range(10)),
    ]
    proposals = {proposal.image_id: proposal for proposal, _ in entries}
    splits = {proposal.image_id: split_of(proposal, split) for proposal, split in entries}
    return proposals, splits


def test_queue_holds_all_held_out_flagged_and_a_clean_training_sample(
    dataset: tuple[dict[str, BoxProposal], dict[str, SplitAssignment]],
) -> None:
    proposals, splits = dataset

    queue = build_review_queue(proposals, splits, spot_check_size=3)

    by_category = {category: [] for category in ReviewCategory}
    for item in queue:
        by_category[item.category].append(item.image_id)
    assert sorted(by_category[ReviewCategory.HELD_OUT]) == ["t1", "v1"]
    assert by_category[ReviewCategory.FLAGGED] == ["f1"]
    assert len(by_category[ReviewCategory.SPOT_CHECK]) == 3
    assert all(image_id.startswith("c") for image_id in by_category[ReviewCategory.SPOT_CHECK])
    assert queue == build_review_queue(proposals, splits, spot_check_size=3)


def test_spot_check_larger_than_clean_training_set_is_an_error(
    dataset: tuple[dict[str, BoxProposal], dict[str, SplitAssignment]],
) -> None:
    with pytest.raises(ValueError, match="exceeds the 10 unflagged"):
        build_review_queue(*dataset, spot_check_size=11)


def test_queue_round_trips_through_csv(tmp_path: Path) -> None:
    queue = [QueueItem("a", ReviewCategory.HELD_OUT), QueueItem("b", ReviewCategory.SPOT_CHECK)]
    path = tmp_path / "queue.csv"

    write_queue(path, queue)

    assert read_queue(path) == {"a": ReviewCategory.HELD_OUT, "b": ReviewCategory.SPOT_CHECK}


def write_reviews(path: Path, *rows: str) -> Path:
    path.write_text("\n".join([",".join(REVIEW_COLUMNS), *rows]) + "\n", encoding="utf-8")
    return path


def test_reads_reviews_downloaded_from_the_page(tmp_path: Path) -> None:
    path = write_reviews(tmp_path / "reviews.csv", "a,reject,cup,Benoy,2026-10-09T15:04:05.123Z")

    review = read_reviews(path, {"a": ReviewCategory.HELD_OUT})["a"]

    assert review.verdict == "reject"
    assert review.reviewer == "Benoy"
    assert review.reviewed_at.utcoffset() is not None


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("z,accept,cup,Benoy,2026-10-09T15:04:05Z", "not in the queue"),
        ("a,maybe,cup,Benoy,2026-10-09T15:04:05Z", "'maybe' is not a valid Verdict"),
        ("a,accept,bowl,Benoy,2026-10-09T15:04:05Z", "unknown class 'bowl'"),
        ("a,accept,cup,,2026-10-09T15:04:05Z", "reviewer is empty"),
        ("a,accept,cup,Benoy,2026-10-09T15:04:05", "UTC offset"),
    ],
)
def test_rejects_invalid_reviews(tmp_path: Path, row: str, message: str) -> None:
    path = write_reviews(tmp_path / "reviews.csv", row)

    with pytest.raises(ValueError, match=re.escape(message)):
        read_reviews(path, {"a": ReviewCategory.HELD_OUT})


def test_page_embeds_data_without_letting_values_close_the_script(
    tmp_path: Path, make_proposal: ProposalFactory
) -> None:
    hostile_id = "v1</script><b>"
    proposal = make_proposal(hostile_id)
    queue = [QueueItem(hostile_id, ReviewCategory.HELD_OUT)]

    page = render_review_page(
        queue,
        {hostile_id: proposal},
        {hostile_id: split_of(proposal, Split.VAL)},
        {hostile_id: tmp_path / "v1.jpg"},
        reviews={},
    )

    embedded = page.split('id="review-data">', 1)[1].split("</script>", 1)[0]
    item = json.loads(embedded)["items"][0]
    assert item["id"] == hostile_id
    assert item["box"] == [20, 40, 80, 160]
    assert item["src"].startswith("file:///")
    assert "__REVIEW_DATA__" not in page
