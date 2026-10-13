import pytest

from objhist_evaluation.detector_scoring import (
    LabeledObject,
    NormalizedBox,
    ScoredPrediction,
    score_photo,
)

FORK_BOX = NormalizedBox(0.10, 0.10, 0.20, 0.50)
CUP_BOX = NormalizedBox(0.60, 0.20, 0.80, 0.50)
LABELS = [LabeledObject("fork", FORK_BOX), LabeledObject("cup", CUP_BOX)]


def predict(class_name: str, confidence: float, box: NormalizedBox) -> ScoredPrediction:
    return ScoredPrediction(class_name, confidence, box)


def test_invalid_boxes_are_rejected() -> None:
    with pytest.raises(ValueError, match="invalid normalized box"):
        NormalizedBox(0.5, 0.1, 0.5, 0.2)
    with pytest.raises(ValueError, match="invalid normalized box"):
        NormalizedBox(0.1, 0.1, 1.2, 0.2)


def test_every_object_found() -> None:
    counts = score_photo(
        LABELS, [predict("fork", 0.9, FORK_BOX), predict("cup", 0.8, CUP_BOX)], 0.25
    )

    assert counts.totals() == {
        "objects": 2,
        "correct": 2,
        "wrong_class": 0,
        "missed": 0,
        "false_boxes": 0,
    }


def test_wrong_class_is_counted_under_the_labeled_class_with_its_confusion() -> None:
    counts = score_photo(LABELS, [predict("spoon", 0.9, FORK_BOX)], 0.25)

    assert counts.wrong_class == {"fork": 1}
    assert counts.confusions == {("fork", "spoon"): 1}
    assert counts.missed == {"cup": 1}
    assert counts.false_boxes == {}


def test_a_second_box_on_a_found_object_is_a_false_box() -> None:
    counts = score_photo(
        LABELS, [predict("fork", 0.9, FORK_BOX), predict("fork", 0.7, FORK_BOX)], 0.25
    )

    assert counts.correct == {"fork": 1}
    assert counts.false_boxes == {"fork": 1}


def test_same_class_match_wins_over_a_closer_box_of_another_class() -> None:
    labels = [
        LabeledObject("spoon", NormalizedBox(0.10, 0.10, 0.20, 0.50)),
        LabeledObject("fork", NormalizedBox(0.10, 0.12, 0.20, 0.52)),
    ]

    counts = score_photo(
        labels, [predict("fork", 0.9, NormalizedBox(0.10, 0.10, 0.20, 0.50))], 0.25
    )

    assert counts.correct == {"fork": 1}
    assert counts.missed == {"spoon": 1}


def test_iou_threshold_is_inclusive() -> None:
    # Half the label's width at full height gives IoU 0.5; binary fractions keep it exact.
    label = LabeledObject("fork", NormalizedBox(0.0, 0.0, 0.5, 0.5))
    half = NormalizedBox(0.0, 0.0, 0.25, 0.5)
    below = NormalizedBox(0.0, 0.0, 0.2499, 0.5)

    assert score_photo([label], [predict("fork", 0.9, half)], 0.25).correct == {"fork": 1}
    assert score_photo([label], [predict("fork", 0.9, below)], 0.25).false_boxes == {"fork": 1}


def test_more_confident_prediction_takes_the_object() -> None:
    weak_exact = predict("fork", 0.4, FORK_BOX)
    strong_loose = predict("fork", 0.9, NormalizedBox(0.10, 0.10, 0.20, 0.40))

    counts = score_photo([LABELS[0]], [weak_exact, strong_loose], 0.25)

    assert counts.correct == {"fork": 1}
    assert counts.false_boxes == {"fork": 1}


def test_predictions_below_the_threshold_are_ignored() -> None:
    counts = score_photo(LABELS, [predict("fork", 0.2, FORK_BOX)], 0.25)

    assert counts.missed == {"fork": 1, "cup": 1}
    assert counts.false_boxes == {}


def test_empty_photo_counts_every_prediction_as_false() -> None:
    counts = score_photo([], [predict("knife", 0.6, FORK_BOX)], 0.25)

    assert counts.totals()["objects"] == 0
    assert counts.false_boxes == {"knife": 1}


def test_result_does_not_depend_on_prediction_order() -> None:
    predictions = [
        predict("fork", 0.8, FORK_BOX),
        predict("spoon", 0.8, NormalizedBox(0.10, 0.11, 0.20, 0.51)),
        predict("cup", 0.8, CUP_BOX),
    ]

    forward = score_photo(LABELS, predictions, 0.25)
    backward = score_photo(LABELS, list(reversed(predictions)), 0.25)

    assert forward == backward
