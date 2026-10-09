from collections.abc import Callable
from pathlib import Path

import pytest

from objhist_detection.kaggle_kitchenware.proposals import (
    BoxCandidate,
    BoxProposal,
    PixelBox,
    ReviewReason,
    flag_duplicate_images,
    read_proposals,
    review_reasons_for,
    select_top_box,
    write_proposals,
)

ProposalFactory = Callable[..., BoxProposal]


def candidate(class_name: str, confidence: float, box: PixelBox) -> BoxCandidate:
    return BoxCandidate(class_name=class_name, confidence=confidence, box=box)


OBJECT_BOX = PixelBox(10, 10, 90, 90)


def test_no_candidates_means_no_box() -> None:
    assert select_top_box([]) == (None, None)
    assert review_reasons_for(None, None) == (ReviewReason.NO_BOX,)


def test_selects_most_confident_box_and_breaks_ties_by_class() -> None:
    weaker = candidate("cup", 0.6, OBJECT_BOX)
    tied_glass = candidate("glass", 0.8, OBJECT_BOX)
    tied_cup = candidate("cup", 0.8, OBJECT_BOX)

    top, second = select_top_box([weaker, tied_glass, tied_cup])

    assert top == tied_cup
    assert second is None


def test_box_inside_the_top_box_is_part_of_the_same_object() -> None:
    handle = candidate("cup", 0.7, PixelBox(70, 30, 95, 60))

    _, second = select_top_box([candidate("cup", 0.9, OBJECT_BOX), handle])

    assert second is None


def test_confident_box_outside_the_top_box_is_a_second_object() -> None:
    other = candidate("knife", 0.5, PixelBox(100, 10, 140, 90))

    top, second = select_top_box([candidate("fork", 0.9, OBJECT_BOX), other])

    assert second == 0.5
    assert review_reasons_for(top, second) == (ReviewReason.SECOND_OBJECT,)


def test_weak_box_outside_the_top_box_is_ignored() -> None:
    other = candidate("knife", 0.39, PixelBox(100, 10, 140, 90))

    assert select_top_box([candidate("fork", 0.9, OBJECT_BOX), other])[1] is None


def test_low_confidence_is_a_review_reason_but_detector_class_is_not() -> None:
    assert review_reasons_for(candidate("cup", 0.39, OBJECT_BOX), None) == (
        ReviewReason.LOW_CONFIDENCE,
    )
    assert review_reasons_for(candidate("knife", 0.4, OBJECT_BOX), None) == ()


def test_flags_every_image_with_identical_bytes(make_proposal: ProposalFactory) -> None:
    proposals = [
        make_proposal("1", "knife", sha256="same"),
        make_proposal("2", "fork", sha256="same"),
        make_proposal("3", sha256="unique"),
    ]

    flagged = flag_duplicate_images(proposals)

    assert [proposal.review_reasons for proposal in flagged] == [
        (ReviewReason.DUPLICATE_IMAGE,),
        (ReviewReason.DUPLICATE_IMAGE,),
        (),
    ]


def test_proposals_round_trip_through_csv(tmp_path: Path, make_proposal: ProposalFactory) -> None:
    path = tmp_path / "proposals.csv"
    proposals = [
        make_proposal("0001", reasons=(ReviewReason.LOW_CONFIDENCE,)),
        make_proposal("0002", "spoon", box=None),
    ]

    write_proposals(path, proposals)

    assert read_proposals(path) == {proposal.image_id: proposal for proposal in proposals}


def test_reading_rejects_a_box_outside_the_image(
    tmp_path: Path, make_proposal: ProposalFactory
) -> None:
    path = tmp_path / "proposals.csv"
    write_proposals(path, [make_proposal("0001", box=PixelBox(20, 40, 101, 160))])

    with pytest.raises(ValueError, match=r"proposals.csv:2: .*exceeds the 100x200 image"):
        read_proposals(path)


def test_invalid_box_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid box"):
        PixelBox(50, 10, 50, 90)
