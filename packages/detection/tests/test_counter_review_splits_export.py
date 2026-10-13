import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from objhist_detection.annotation import PixelBox, file_sha256
from objhist_detection.counter_labels.export import (
    ExportItem,
    label_text,
    plan_export,
    review_counts,
    write_export,
)
from objhist_detection.counter_labels.proposals import Candidate, PhotoRecord, proposals_for_photo
from objhist_detection.counter_labels.review import (
    REVIEWS_FORMAT,
    ReviewError,
    read_reviews,
    render_review_page,
)
from objhist_detection.counter_labels.sessions import find_session_photos
from objhist_detection.counter_labels.splits import assign_new_sessions, read_splits, write_splits
from objhist_detection.yolo_dataset import MANIFEST_FILENAME, ExportError

KEY = "s1/p.jpg"
PHOTO = PhotoRecord(KEY, "s1", "p.jpg", 400, 200, 1, "hash")
PROPOSALS = {
    KEY: proposals_for_photo(
        KEY,
        [
            Candidate("cup", 0.9, PixelBox(10, 10, 110, 110)),
            Candidate("spoon", 0.3, PixelBox(300, 20, 390, 190)),
        ],
    )
}


def proposed(box_id: str, class_name: str) -> dict[str, object]:
    box = next(p.box for p in PROPOSALS[KEY] if p.box_id == box_id)
    return {
        "class_name": class_name,
        "box": [box.x_min, box.y_min, box.x_max, box.y_max],
        "source": "proposed",
        "proposal_id": box_id,
    }


def photo_review(**changes: object) -> dict[str, object]:
    review: dict[str, object] = {
        "key": KEY,
        "reviewer": "Benoy",
        "reviewed_at": "2026-10-12T18:00:00Z",
        "objects": [
            proposed("p1", "glass"),
            {"class_name": "fork", "box": [150, 40, 200, 190], "source": "drawn"},
        ],
        "rejected_proposals": ["p2"],
    }
    review.update(changes)
    return review


def write_reviews(path: Path, *photos: dict[str, object]) -> Path:
    path.write_text(json.dumps({"format": REVIEWS_FORMAT, "photos": list(photos)}))
    return path


def test_valid_review_is_read(tmp_path: Path) -> None:
    reviews = read_reviews(
        write_reviews(tmp_path / "r.json", photo_review()), {KEY: PHOTO}, PROPOSALS
    )

    review = reviews[KEY]
    assert [(o.class_name, o.source) for o in review.objects] == [
        ("glass", "proposed"),
        ("fork", "drawn"),
    ]
    assert review_counts(review, PROPOSALS[KEY]) == {
        "proposed": 2,
        "kept_same_class": 0,
        "kept_relabeled": 1,
        "kept_adjusted": 0,
        "rejected": 1,
        "drawn": 1,
    }


def test_adjusted_proposal_keeps_the_reviewers_box(tmp_path: Path) -> None:
    tighter = {**proposed("p1", "cup"), "box": [14, 12, 104, 108], "adjusted": True}
    reviews = read_reviews(
        write_reviews(tmp_path / "r.json", photo_review(objects=[tighter])), {KEY: PHOTO}, PROPOSALS
    )

    (cup,) = reviews[KEY].objects
    assert (cup.box, cup.adjusted) == ([14, 12, 104, 108], True)
    assert review_counts(reviews[KEY], PROPOSALS[KEY])["kept_adjusted"] == 1
    # The label uses the adjusted box (center 59,60; size 90x96 in a 400x200 photo).
    assert label_text(ExportItem(PHOTO, "train", reviews[KEY])) == (
        "0 0.147500 0.300000 0.225000 0.480000\n"
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"rejected_proposals": []}, r"proposals without a verdict \['p2'\]"),
        ({"rejected_proposals": ["p1", "p2"]}, "kept or rejected more than once"),
        (
            {"objects": [{**proposed("p1", "cup"), "box": [10, 10, 120, 110]}]},
            "p1 box was changed without being marked adjusted",
        ),
        (
            {
                "objects": [
                    {**proposed("p1", "cup"), "box": [10, 10, 15, 110], "adjusted": True},
                ]
            },
            "adjusted box .* is too small",
        ),
        (
            {
                "objects": [
                    proposed("p1", "cup"),
                    {
                        "class_name": "fork",
                        "box": [150, 40, 200, 190],
                        "source": "drawn",
                        "adjusted": True,
                    },
                ]
            },
            "only proposed boxes can be marked adjusted",
        ),
        (
            {
                "objects": [
                    proposed("p1", "cup"),
                    {"class_name": "fork", "box": [1, 1, 5, 50], "source": "drawn"},
                ]
            },
            "too small",
        ),
        (
            {
                "objects": [
                    proposed("p1", "cup"),
                    {"class_name": "fork", "box": [300, 10, 401, 50], "source": "drawn"},
                ]
            },
            "exceeds the 400x200 image",
        ),
        ({"objects": [proposed("p1", "bowl")]}, "unknown class 'bowl'"),
        ({"reviewed_at": "2026-10-12T18:00:00"}, "timezone"),
        ({"reviewer": "   "}, "reviewer is empty"),
    ],
)
def test_invalid_reviews_are_rejected(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    path = write_reviews(tmp_path / "r.json", photo_review(**changes))

    with pytest.raises(ReviewError, match=message):
        read_reviews(path, {KEY: PHOTO}, PROPOSALS)


def test_review_of_unknown_photo_is_rejected(tmp_path: Path) -> None:
    path = write_reviews(tmp_path / "r.json", photo_review(key="s9/x.jpg"))

    with pytest.raises(ReviewError, match="unknown photo"):
        read_reviews(path, {KEY: PHOTO}, PROPOSALS)


def test_review_page_embeds_data_without_closing_the_script(tmp_path: Path) -> None:
    hostile = PhotoRecord("s1/</script>.jpg", "s1", "</script>.jpg", 400, 200, 1, "h")

    page = render_review_page(
        {hostile.key: hostile}, {hostile.key: []}, {hostile.key: tmp_path / "x.jpg"}, {}
    )

    assert page.count("</script>") == 2
    assert "\\u003c/script>.jpg" in page


def test_first_sessions_fill_train_then_val_and_test() -> None:
    photos = {
        "2026-10-01": 60,
        "2026-10-02": 60,
        "2026-10-03": 60,
        "2026-10-04": 60,
        "2026-10-05": 60,
    }

    assignments = assign_new_sessions({}, photos, {})

    assert assignments == {
        "2026-10-01": "train",
        "2026-10-02": "train",
        "2026-10-03": "val",
        "2026-10-04": "train",
        "2026-10-05": "test",
    }


def test_frozen_sessions_never_move(tmp_path: Path) -> None:
    existing = {"2026-10-01": "test"}
    assignments = assign_new_sessions(
        existing, {"2026-10-01": 60, "2026-10-02": 60}, {"2026-10-02": "val"}
    )
    write_splits(tmp_path / "splits.csv", assignments)

    assert read_splits(tmp_path / "splits.csv") == {"2026-10-01": "test", "2026-10-02": "val"}
    with pytest.raises(ValueError, match="already frozen in test"):
        assign_new_sessions(existing, {"2026-10-01": 60}, {"2026-10-01": "train"})
    with pytest.raises(ValueError, match="unknown split 'dev'"):
        assign_new_sessions({}, {"2026-10-02": 60}, {"2026-10-02": "dev"})


def make_session(tmp_path: Path) -> tuple[dict[str, PhotoRecord], dict[str, object]]:
    sequences = tmp_path / "sequences"
    photos = {}
    for session, name in [("train1", "a.jpg"), ("train1", "empty.jpg"), ("val1", "b.jpg")]:
        path = sequences / session / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (400, 200), "white").save(path)
        key = f"{session}/{name}"
        photos[key] = PhotoRecord(key, session, name, 400, 200, 1, file_sha256(path))
    session_photos = {photo.key: photo for photo in find_session_photos(sequences)}
    return photos, session_photos


def review_for(key: str, objects: list[dict[str, object]]) -> dict[str, object]:
    return {
        "key": key,
        "reviewer": "Benoy",
        "reviewed_at": "2026-10-12T18:00:00+00:00",
        "objects": objects,
    }


def drawn(class_name: str, box: list[float]) -> dict[str, object]:
    return {"class_name": class_name, "box": box, "source": "drawn"}


def test_export_writes_labels_and_keeps_empty_counters(tmp_path: Path) -> None:
    photos, session_photos = make_session(tmp_path)
    proposals = {key: [] for key in photos}
    reviews = read_reviews(
        write_reviews(
            tmp_path / "r.json",
            review_for(
                "train1/a.jpg",
                [drawn("plate", [100, 50, 300, 150]), drawn("knife", [0, 0, 40, 200])],
            ),
            review_for("train1/empty.jpg", []),
            review_for("val1/b.jpg", [drawn("cup", [10, 10, 50, 50])]),
        ),
        photos,
        proposals,
    )
    plan = plan_export(photos, {"train1": "train", "val1": "val"}, reviews)
    out = tmp_path / "out"

    summary = write_export(plan, proposals, session_photos, out, input_files={})

    labels = (out / "labels" / "train" / "train1__a.txt").read_text().splitlines()
    assert labels == [
        "4 0.500000 0.500000 0.500000 0.500000",
        "3 0.050000 0.500000 0.100000 1.000000",
    ]
    assert (out / "labels" / "train" / "train1__empty.txt").read_text() == ""
    assert (out / "images" / "val" / "val1__b.jpg").is_file()
    assert yaml.safe_load((out / "data.yaml").read_text())["names"][2] == "glass"
    assert summary["photos_per_split"] == {"train": 2, "val": 1, "test": 0}
    assert summary["objects_per_split"]["train"]["plate"] == 1
    assert len((out / MANIFEST_FILENAME).read_text().splitlines()) == 4


def test_export_requires_every_held_out_photo_and_skips_unreviewed_train(tmp_path: Path) -> None:
    photos, _ = make_session(tmp_path)
    proposals = {key: [] for key in photos}
    only_train = read_reviews(
        write_reviews(tmp_path / "r.json", review_for("train1/a.jpg", [])), photos, proposals
    )

    with pytest.raises(ExportError, match="1 validation/test photos are not reviewed.*val1/b.jpg"):
        plan_export(photos, {"train1": "train", "val1": "val"}, only_train)
    with pytest.raises(ExportError, match="sessions without a split: val1"):
        plan_export(photos, {"train1": "train"}, only_train)

    plan = plan_export(photos, {"train1": "train", "val1": "train"}, only_train)
    assert plan.unreviewed_train == ["train1/empty.jpg", "val1/b.jpg"]


def test_export_stops_when_a_photo_changed_after_proposing(tmp_path: Path) -> None:
    photos, session_photos = make_session(tmp_path)
    proposals = {key: [] for key in photos}
    reviews = read_reviews(
        write_reviews(tmp_path / "r.json", *(review_for(key, []) for key in photos)),
        photos,
        proposals,
    )
    plan = plan_export(photos, {"train1": "train", "val1": "val"}, reviews)
    Image.new("RGB", (400, 200), "black").save(session_photos["val1/b.jpg"].path)

    with pytest.raises(ExportError, match="changed since its boxes were proposed"):
        write_export(plan, proposals, session_photos, tmp_path / "out", input_files={})
    assert not (tmp_path / "out").exists()
