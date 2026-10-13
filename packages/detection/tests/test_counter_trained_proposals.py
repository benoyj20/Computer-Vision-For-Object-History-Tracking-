from pathlib import Path

import pytest
from PIL import Image

from objhist_detection.annotation import CLASS_NAMES, EXIF_ORIENTATION_TAG, PixelBox
from objhist_detection.counter_labels import __main__ as counter_cli
from objhist_detection.counter_labels.detector_boxes import (
    check_class_names,
    trained_detector_candidate_finder,
)
from objhist_detection.counter_labels.proposals import (
    PROPOSAL_CONFIDENCE_FLOOR,
    TRAINED_DETECTOR_PROPOSER,
    YOLOE_PROPOSER,
    Candidate,
    propose_photos,
    proposer_by_photo,
)
from objhist_detection.counter_labels.sessions import find_session_photos
from objhist_detection.counter_labels.splits import check_held_out_proposals, write_splits
from objhist_detection.tiled_inference import ScoredBox

ROTATE_90_CLOCKWISE = 6


def test_photos_are_proposed_in_upright_pixels(tmp_path: Path) -> None:
    exif = Image.Exif()
    exif[EXIF_ORIENTATION_TAG] = ROTATE_90_CLOCKWISE
    (tmp_path / "s1").mkdir()
    Image.new("RGB", (200, 100), "white").save(tmp_path / "s1" / "a.jpg", exif=exif)
    seen_sizes = []

    def find(image: Image.Image) -> list[Candidate]:
        seen_sizes.append(image.size)
        return [
            Candidate("fork", 0.5, PixelBox(10, 10, 20, 90)),
            Candidate("cup", 0.9, PixelBox(50, 100, 90, 150)),
            Candidate("spoon", PROPOSAL_CONFIDENCE_FLOOR / 2, PixelBox(0, 0, 5, 5)),
        ]

    ((record, boxes),) = list(propose_photos(find, find_session_photos(tmp_path)))

    assert seen_sizes == [(100, 200)]
    assert (record.width_px, record.height_px, record.exif_orientation) == (100, 200, 6)
    assert [(b.box_id, b.class_name) for b in boxes] == [("p1", "cup"), ("p2", "fork")]


def test_latest_run_decides_each_photos_proposer() -> None:
    runs = [
        {"photos": ["s1/a.jpg", "s1/b.jpg"]},
        {"photos": ["s1/b.jpg"], "proposer": TRAINED_DETECTOR_PROPOSER},
    ]

    assert proposer_by_photo(runs) == {
        "s1/a.jpg": YOLOE_PROPOSER,
        "s1/b.jpg": TRAINED_DETECTOR_PROPOSER,
    }
    with pytest.raises(ValueError, match="no photo list"):
        proposer_by_photo([{"proposer": YOLOE_PROPOSER}])


def test_held_out_sessions_cannot_have_trained_detector_proposals() -> None:
    check_held_out_proposals({"night": "train", "day": "test"}, {"night"})

    with pytest.raises(ValueError, match="day would be held out"):
        check_held_out_proposals({"night": "train", "day": "test"}, {"night", "day"})


def test_detector_must_use_the_project_class_order() -> None:
    check_class_names(dict(enumerate(CLASS_NAMES)))

    with pytest.raises(ValueError, match="differ from the project classes"):
        check_class_names(dict(enumerate(reversed(CLASS_NAMES))))


def test_trained_detector_proposals_use_the_floor_and_merge_duplicates() -> None:
    calls = []

    def predict(image: Image.Image, inference_px: int, confidence: float) -> list[ScoredBox]:
        calls.append((inference_px, confidence))
        return [
            ScoredBox("fork", 0.6, PixelBox(10, 10, 30, 90)),
            ScoredBox("spoon", 0.8, PixelBox(10, 12, 30, 90)),
        ]

    candidates = trained_detector_candidate_finder(predict)(Image.new("RGB", (100, 100)))

    assert calls == [(1280, PROPOSAL_CONFIDENCE_FLOOR)]
    assert [(c.class_name, c.confidence) for c in candidates] == [("spoon", 0.8)]


def test_propose_refuses_a_trained_detector_for_a_test_session(tmp_path: Path) -> None:
    sequences, work = tmp_path / "sequences", tmp_path / "work"
    (sequences / "day").mkdir(parents=True)
    work.mkdir()
    Image.new("RGB", (40, 20)).save(sequences / "day" / "a.jpg")
    write_splits(work / counter_cli.SPLITS_FILENAME, {"day": "test"})
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"never loaded")
    argv = ["--sequences-dir", str(sequences), "--work-dir", str(work), "propose"]

    assert counter_cli.main([*argv, "--detector", str(weights)]) == 1
    assert not (work / counter_cli.BOXES_FILENAME).exists()
