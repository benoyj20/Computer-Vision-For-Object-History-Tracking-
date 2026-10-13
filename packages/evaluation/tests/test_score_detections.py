import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from objhist_core import BoundingBox, Detection, DetectorRun, FrameDetections, SessionDetections
from objhist_evaluation import score_detections
from objhist_evaluation.yolo_labels import LabelError, parse_label_file, read_split_labels

CLASSES = ("cup", "fork")
SESSION = "2026-10-12-1643"
WIDTH_PX, HEIGHT_PX = 4000, 2000
START = datetime(2026, 10, 12, 16, 43, tzinfo=UTC)


def make_dataset(tmp_path: Path) -> Path:
    """Two test photos (a cup and a fork; an empty counter) and one train photo."""
    dataset = tmp_path / "counter"
    (dataset / "labels" / "test").mkdir(parents=True)
    (dataset / "labels" / "train").mkdir(parents=True)
    (dataset / "data.yaml").write_text("names:\n  0: cup\n  1: fork\n", encoding="utf-8")
    rows = [
        (
            "a.jpg",
            "test",
            "0 0.250000 0.250000 0.100000 0.100000\n1 0.700000 0.500000 0.050000 0.400000\n",
        ),
        ("b.jpg", "test", ""),
        ("c.jpg", "train", "0 0.5 0.5 0.1 0.1\n"),
    ]
    lines = ["key,split,export_name,objects"]
    for name, split, text in rows:
        export_name = f"{SESSION}__{name}"
        lines.append(f"{SESSION}/{name},{split},{export_name},0")
        (dataset / "labels" / split / f"{Path(export_name).stem}.txt").write_text(text, "utf-8")
    (dataset / "manifest.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dataset


def detector(version: str, threshold: float = 0.25) -> DetectorRun:
    return DetectorRun(
        version=version,
        weights_sha256="0" * 64,
        class_names=CLASSES,
        confidence_threshold=threshold,
        settings={"mode": "single"},
        framework_version="test",
    )


def pixel_box(x_min: float, y_min: float, x_max: float, y_max: float) -> BoundingBox:
    return BoundingBox(
        x_min=x_min * WIDTH_PX,
        y_min=y_min * HEIGHT_PX,
        x_max=x_max * WIDTH_PX,
        y_max=y_max * HEIGHT_PX,
    )


def write_detections(
    folder: Path, run: DetectorRun, frames: dict[str, tuple[Detection, ...]]
) -> Path:
    folder.mkdir(parents=True)
    session = SessionDetections(
        session_id=SESSION,
        created_at=START,
        capture_time_source="exif",
        detector=run,
        frames=tuple(
            FrameDetections(
                frame_id=f"{SESSION}/{name}",
                captured_at=START + timedelta(seconds=index),
                image_width_px=WIDTH_PX,
                image_height_px=HEIGHT_PX,
                detector_version=run.version,
                detections=detections,
            )
            for index, (name, detections) in enumerate(frames.items())
        ),
    )
    (folder / f"{SESSION}.json").write_text(session.model_dump_json(), encoding="utf-8")
    return folder


CUP_FOUND = Detection(class_name="cup", confidence=0.9, box=pixel_box(0.2, 0.2, 0.3, 0.3))
FORK_AS_CUP = Detection(class_name="cup", confidence=0.3, box=pixel_box(0.675, 0.3, 0.725, 0.7))
FALSE_FORK = Detection(class_name="fork", confidence=0.5, box=pixel_box(0.0, 0.8, 0.1, 0.9))


def test_split_labels_are_read_by_photo_key(tmp_path: Path) -> None:
    labels = read_split_labels(make_dataset(tmp_path), "test")

    assert sorted(labels) == [f"{SESSION}/a.jpg", f"{SESSION}/b.jpg"]
    assert [obj.class_name for obj in labels[f"{SESSION}/a.jpg"]] == ["cup", "fork"]
    assert labels[f"{SESSION}/b.jpg"] == []


def test_label_rounding_past_the_edge_is_clipped(tmp_path: Path) -> None:
    path = tmp_path / "edge.txt"
    path.write_text("0 0.950001 0.5 0.100003 0.2\n", encoding="utf-8")

    (cup,) = parse_label_file(path, CLASSES)

    assert cup.box.x_max == 1.0


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("7 0.5 0.5 0.1 0.1", "class index 7"),
        ("0 0.5 0.5", "cannot parse"),
        ("0 1.5 0.5 0.2 0.2", "outside"),
    ],
)
def test_invalid_label_lines_are_rejected(tmp_path: Path, line: str, message: str) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(LabelError, match=message):
        parse_label_file(path, CLASSES)


def test_cli_scores_each_detector_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = make_dataset(tmp_path)
    good = write_detections(
        tmp_path / "good",
        detector("good"),
        {"a.jpg": (CUP_FOUND, FORK_AS_CUP), "b.jpg": (FALSE_FORK,)},
    )
    empty = write_detections(tmp_path / "empty", detector("empty"), {"a.jpg": (), "b.jpg": ()})
    out = tmp_path / "scores.json"

    assert (
        score_detections.main(
            ["--dataset", str(dataset), str(good), str(empty), "--json-out", str(out)]
        )
        == 0
    )

    record = json.loads(out.read_text(encoding="utf-8"))
    good_result, empty_result = record["results"]
    assert good_result["totals"] == {
        "objects": 2,
        "correct": 1,
        "wrong_class": 1,
        "missed": 0,
        "false_boxes": 1,
    }
    assert good_result["confusions"] == [{"labeled": "fork", "predicted": "cup", "count": 1}]
    assert empty_result["totals"]["missed"] == 2
    assert record["photos"] == 2
    assert "fork as cup x1" in capsys.readouterr().out


def test_higher_confidence_threshold_drops_weak_detections(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    folder = write_detections(
        tmp_path / "good", detector("good"), {"a.jpg": (CUP_FOUND, FORK_AS_CUP), "b.jpg": ()}
    )
    output = score_detections.load_detector_output(folder)

    counts, threshold = score_detections.score_output(
        output, read_split_labels(dataset, "test"), 0.5
    )

    assert threshold == 0.5
    assert counts.totals()["wrong_class"] == 0
    assert counts.missed == {"fork": 1}


def test_threshold_below_the_recorded_one_is_refused(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    folder = write_detections(
        tmp_path / "d", detector("d", threshold=0.25), {"a.jpg": (), "b.jpg": ()}
    )

    with pytest.raises(score_detections.ScoringError, match="cannot be scored at 0.1"):
        score_detections.score_output(
            score_detections.load_detector_output(folder), read_split_labels(dataset, "test"), 0.1
        )


def test_unprocessed_photo_is_an_error_not_a_miss(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    folder = write_detections(tmp_path / "partial", detector("partial"), {"a.jpg": (CUP_FOUND,)})

    assert score_detections.main(["--dataset", str(dataset), str(folder)]) == 1


def test_folder_mixing_detectors_is_refused(tmp_path: Path) -> None:
    folder = write_detections(tmp_path / "mixed", detector("one"), {"a.jpg": (), "b.jpg": ()})
    other = write_detections(tmp_path / "other", detector("two"), {"a.jpg": (), "b.jpg": ()})
    (other / f"{SESSION}.json").rename(folder / "second.json")

    with pytest.raises(score_detections.ScoringError, match="keep one detector configuration"):
        score_detections.load_detector_output(folder)
