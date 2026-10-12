from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

from objhist_core import DetectorRun, SessionDetections
from objhist_detection import detect
from objhist_detection.annotation import EXIF_ORIENTATION_TAG, PixelBox, file_sha256
from objhist_detection.counter_labels.sessions import find_session_photos
from objhist_detection.inference_settings import InferenceSettings
from objhist_detection.tiled_inference import ScoredBox

SESSION_ID = "2026-10-12-evening"
SINGLE_PASS = InferenceSettings(tiled=False)
CREATED_AT = datetime(2026, 10, 12, 20, 0, tzinfo=UTC)
ROTATE_90_CLOCKWISE = 6
"""EXIF orientation of a phone held upright: stored sideways, displayed rotated 90 degrees."""


def make_session_dir(tmp_path: Path, manifest_rows: list[tuple[str, str]]) -> Path:
    """Photos named in the manifest: `rotated.jpg` is stored 200x100 with orientation 6."""
    session_dir = tmp_path / "sequences" / SESSION_ID
    session_dir.mkdir(parents=True)
    for filename, _ in manifest_rows:
        exif = Image.Exif()
        if filename == "rotated.jpg":
            exif[EXIF_ORIENTATION_TAG] = ROTATE_90_CLOCKWISE
        Image.new("RGB", (200, 100), "white").save(session_dir / filename, exif=exif)
    lines = ["filename,captured_at", *(",".join(row) for row in manifest_rows)]
    (session_dir / "manifest.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return session_dir.parent


def detector_run(version: str = "yolo26n-kitchen-0123456789ab-single1280") -> DetectorRun:
    return DetectorRun(
        version=version,
        weights_sha256="0" * 64,
        class_names=("cup", "fork"),
        confidence_threshold=SINGLE_PASS.confidence_threshold,
        settings=SINGLE_PASS.as_record(),
        framework_version="ultralytics test",
    )


def fork_on_tall_photos(
    image: Image.Image, inference_px: int, confidence_threshold: float
) -> list[ScoredBox]:
    """Fake detector: one fork in the lower half of portrait photos, nothing on landscape."""
    width_px, height_px = image.size
    if height_px <= width_px:
        return []
    return [ScoredBox("fork", 0.8, PixelBox(10, height_px / 2, 30, height_px))]


def test_session_frames_are_in_capture_order_in_upright_pixels(tmp_path: Path) -> None:
    sequences_dir = make_session_dir(
        tmp_path,
        [("plain.jpg", "2026-10-12T14:05:00-04:00"), ("rotated.jpg", "2026-10-12T14:06:00-04:00")],
    )
    (session,) = detect.plan_sessions(find_session_photos(sequences_dir))

    result = detect.detect_session(
        session, fork_on_tall_photos, SINGLE_PASS, detector_run(), CREATED_AT
    )

    assert result.capture_time_source == "manifest"
    plain, rotated = result.frames
    assert plain.frame_id == f"{SESSION_ID}/plain.jpg"
    assert (plain.image_width_px, plain.image_height_px) == (200, 100)
    assert plain.detections == ()  # Ran and found nothing, still a frame.
    assert (rotated.image_width_px, rotated.image_height_px) == (100, 200)
    (fork,) = rotated.detections
    assert (fork.box.x_min, fork.box.y_min, fork.box.x_max, fork.box.y_max) == (10, 100, 30, 200)
    assert rotated.captured_at == datetime.fromisoformat("2026-10-12T14:06:00-04:00")


def test_sessions_are_ordered_by_capture_time_not_filename(tmp_path: Path) -> None:
    sequences_dir = make_session_dir(
        tmp_path,
        [("b.jpg", "2026-10-12T14:05:00-04:00"), ("a.jpg", "2026-10-12T14:06:00-04:00")],
    )

    (session,) = detect.plan_sessions(find_session_photos(sequences_dir))

    assert [photo.filename for photo, _ in session.photos] == ["b.jpg", "a.jpg"]


def test_unreadable_photo_stops_the_run(tmp_path: Path) -> None:
    sequences_dir = make_session_dir(tmp_path, [("a.jpg", "2026-10-12T14:05:00-04:00")])
    (sequences_dir / SESSION_ID / "a.jpg").write_bytes(b"not a jpeg")
    (session,) = detect.plan_sessions(find_session_photos(sequences_dir))

    with pytest.raises(detect.DetectError, match="cannot read"):
        detect.detect_session(session, fork_on_tall_photos, SINGLE_PASS, detector_run(), CREATED_AT)


def test_written_session_reads_back(tmp_path: Path) -> None:
    sequences_dir = make_session_dir(tmp_path, [("a.jpg", "2026-10-12T14:05:00-04:00")])
    (session,) = detect.plan_sessions(find_session_photos(sequences_dir))
    result = detect.detect_session(
        session, fork_on_tall_photos, SINGLE_PASS, detector_run(), CREATED_AT
    )
    path = tmp_path / "out" / "v1" / f"{SESSION_ID}.json"

    detect.write_session(path, result)

    assert SessionDetections.model_validate_json(path.read_text(encoding="utf-8")) == result
    assert list(path.parent.iterdir()) == [path]


def test_detector_version_names_the_run_weights_and_mode(tmp_path: Path) -> None:
    weights = tmp_path / "runs" / "detect" / "yolo26n-kitchen" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    loose = tmp_path / "model.pt"
    loose.write_bytes(b"weights")
    sha = file_sha256(weights)

    assert detect.detector_version(weights, sha, InferenceSettings()) == (
        f"yolo26n-kitchen-{sha[:12]}-tiled"
    )
    assert detect.detector_version(loose, sha, SINGLE_PASS) == f"model-{sha[:12]}-single1280"


def test_cli_refuses_to_replace_detections_before_loading_a_model(tmp_path: Path) -> None:
    sequences_dir = make_session_dir(tmp_path, [("a.jpg", "2026-10-12T14:05:00-04:00")])
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not loaded")
    version = detect.detector_version(weights, file_sha256(weights), InferenceSettings())
    existing = tmp_path / "out" / version / f"{SESSION_ID}.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("{}", encoding="utf-8")
    argv = ["--weights", str(weights), "--sequences-dir", str(sequences_dir)]

    assert detect.main([*argv, "--out-dir", str(tmp_path / "out")]) == 1
    assert existing.read_text(encoding="utf-8") == "{}"


def test_cli_reports_missing_weights_and_bad_thresholds(tmp_path: Path) -> None:
    sequences_dir = make_session_dir(tmp_path, [("a.jpg", "2026-10-12T14:05:00-04:00")])
    argv = ["--sequences-dir", str(sequences_dir)]

    assert detect.main([*argv, "--weights", str(tmp_path / "missing.pt")]) == 1
    assert detect.main([*argv, "--weights", str(tmp_path / "missing.pt"), "--conf", "1.5"]) == 1
