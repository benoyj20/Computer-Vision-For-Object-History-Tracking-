from pathlib import Path

import pytest
from PIL import Image

from objhist_detection.annotation import PixelBox
from objhist_detection.counter_labels.proposals import (
    PROPOSAL_CONFIDENCE_FLOOR,
    Candidate,
    PhotoRecord,
    proposals_for_photo,
    read_boxes,
    read_photos,
    write_boxes,
    write_photos,
)
from objhist_detection.counter_labels.sessions import SessionError, find_session_photos


def make_photo(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 20), "white").save(path)


def test_finds_photos_per_session_and_skips_other_files(tmp_path: Path) -> None:
    make_photo(tmp_path / "2026-10-12-a" / "b.jpg")
    make_photo(tmp_path / "2026-10-12-a" / "a.JPG")
    (tmp_path / "2026-10-12-a" / "changes.xlsx").write_bytes(b"not a photo")
    make_photo(tmp_path / "2026-10-13-b" / "c.png")

    photos = find_session_photos(tmp_path)

    assert [photo.key for photo in photos] == [
        "2026-10-12-a/a.JPG",
        "2026-10-12-a/b.jpg",
        "2026-10-13-b/c.png",
    ]
    assert photos[0].export_stem == "2026-10-12-a__a"
    assert [photo.key for photo in find_session_photos(tmp_path, ["2026-10-13-b"])] == [
        "2026-10-13-b/c.png"
    ]


def test_heic_photos_get_an_actionable_error(tmp_path: Path) -> None:
    (tmp_path / "s1").mkdir()
    (tmp_path / "s1" / "IMG_1.HEIC").write_bytes(b"heic")

    with pytest.raises(SessionError, match="save photos as JPEG"):
        find_session_photos(tmp_path)


@pytest.mark.parametrize("name", ["my photo.jpg", "-starts-with-dash.jpg"])
def test_unsafe_photo_names_are_rejected(tmp_path: Path, name: str) -> None:
    make_photo(tmp_path / "s1" / name)

    with pytest.raises(SessionError, match="unsafe characters"):
        find_session_photos(tmp_path)


def test_missing_session_and_empty_folder_are_errors(tmp_path: Path) -> None:
    (tmp_path / "s1").mkdir()

    with pytest.raises(SessionError, match="no photos found"):
        find_session_photos(tmp_path)
    with pytest.raises(SessionError, match="no session folder for nope"):
        find_session_photos(tmp_path, ["nope"])


def test_proposals_drop_weak_boxes_and_number_by_confidence() -> None:
    box = PixelBox(1, 1, 10, 10)
    candidates = [
        Candidate("cup", 0.5, box),
        Candidate("spoon", PROPOSAL_CONFIDENCE_FLOOR - 0.01, box),
        Candidate("fork", 0.9, PixelBox(5, 1, 10, 10)),
        Candidate("knife", PROPOSAL_CONFIDENCE_FLOOR, box),
    ]

    proposed = proposals_for_photo("s/p.jpg", candidates)

    assert [(p.box_id, p.class_name) for p in proposed] == [
        ("p1", "fork"),
        ("p2", "cup"),
        ("p3", "knife"),
    ]


def photo_record(key: str = "s1/p.jpg") -> PhotoRecord:
    session_id, filename = key.split("/")
    return PhotoRecord(key, session_id, filename, 400, 200, 1, "hash-" + filename)


def test_photos_and_boxes_round_trip_through_csv(tmp_path: Path) -> None:
    photos = [photo_record("s1/a.jpg"), photo_record("s1/b.jpg")]
    boxes = proposals_for_photo("s1/a.jpg", [Candidate("plate", 0.8, PixelBox(10, 20, 300, 190))])
    write_photos(tmp_path / "photos.csv", photos)
    write_boxes(tmp_path / "boxes.csv", boxes)

    read_back = read_photos(tmp_path / "photos.csv")

    assert read_back == {photo.key: photo for photo in photos}
    assert read_boxes(tmp_path / "boxes.csv", read_back) == {"s1/a.jpg": boxes, "s1/b.jpg": []}


def test_reading_rejects_boxes_outside_the_photo_or_for_unknown_photos(tmp_path: Path) -> None:
    photos = {"s1/p.jpg": photo_record()}
    write_photos(tmp_path / "photos.csv", photos.values())
    write_boxes(
        tmp_path / "boxes.csv",
        proposals_for_photo("s1/p.jpg", [Candidate("cup", 0.9, PixelBox(10, 10, 401, 50))]),
    )
    with pytest.raises(ValueError, match=r"boxes.csv:2: .*exceeds the 400x200 image"):
        read_boxes(tmp_path / "boxes.csv", photos)

    write_boxes(
        tmp_path / "boxes.csv",
        proposals_for_photo("s2/q.jpg", [Candidate("cup", 0.9, PixelBox(10, 10, 50, 50))]),
    )
    with pytest.raises(ValueError, match="unknown photo 's2/q.jpg'"):
        read_boxes(tmp_path / "boxes.csv", photos)
