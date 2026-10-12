import math
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from objhist_core import BoundingBox, Detection, DetectorRun, FrameDetections, SessionDetections

CAPTURED_AT = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)


def make_detection(**box: float) -> Detection:
    coords = {"x_min": 10, "y_min": 20, "x_max": 110, "y_max": 220} | box
    return Detection(class_name="cup", confidence=0.9, box=BoundingBox(**coords))


def make_frame(
    *detections: Detection,
    captured_at: datetime = CAPTURED_AT,
    frame_id: str = "frame-0001",
    detector_version: str = "yolo26n-test",
) -> FrameDetections:
    return FrameDetections(
        frame_id=frame_id,
        captured_at=captured_at,
        image_width_px=640,
        image_height_px=480,
        detector_version=detector_version,
        detections=detections,
    )


def test_valid_frame_round_trips_through_json() -> None:
    frame = make_frame(make_detection())

    assert FrameDetections.model_validate_json(frame.model_dump_json()) == frame


def test_empty_frame_is_valid() -> None:
    assert make_frame().detections == ()


@pytest.mark.parametrize(
    "box",
    [
        {"x_max": 10},  # zero width
        {"y_max": 5},  # inverted height
        {"x_min": -1},  # outside the image
        {"x_max": math.nan},
        {"y_max": math.inf},
    ],
)
def test_invalid_boxes_are_rejected(box: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        make_detection(**box)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, math.nan])
def test_confidence_must_be_a_probability(confidence: float) -> None:
    with pytest.raises(ValidationError):
        Detection(class_name="cup", confidence=confidence, box=make_detection().box)


def test_naive_capture_time_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_frame(captured_at=datetime(2026, 10, 8, 12, 0))  # noqa: DTZ001  # naive on purpose


def test_box_beyond_image_bounds_is_rejected() -> None:
    with pytest.raises(ValidationError, match="exceeds the 640x480 image"):
        make_frame(make_detection(x_max=641))


def test_contracts_are_immutable() -> None:
    detection = make_detection()

    with pytest.raises(ValidationError):
        detection.confidence = 0.1  # type: ignore[misc]  # asserting frozen behavior


DETECTOR = DetectorRun(
    version="yolo26n-test",
    weights_sha256="0" * 64,
    class_names=("cup", "fork"),
    confidence_threshold=0.25,
    settings={"mode": "tiled", "tile_fraction": 0.32, "tile_inference_px": 640},
    framework_version="ultralytics 8.4.174",
)


def make_session(*frames: FrameDetections) -> SessionDetections:
    return SessionDetections(
        session_id="2026-10-12-evening",
        created_at=CAPTURED_AT,
        capture_time_source="exif",
        detector=DETECTOR,
        frames=frames,
    )


def session_frame(name: str, minutes: int, *detections: Detection) -> FrameDetections:
    return make_frame(
        *detections,
        frame_id=f"2026-10-12-evening/{name}",
        captured_at=CAPTURED_AT + timedelta(minutes=minutes),
    )


def test_session_round_trips_through_json_with_its_offset() -> None:
    offset_time = datetime.fromisoformat("2026-10-12T13:47:54.365-04:00")
    session = make_session(
        make_frame(frame_id="2026-10-12-evening/a.jpg", captured_at=offset_time),
        make_frame(
            make_detection(),
            frame_id="2026-10-12-evening/b.jpg",
            captured_at=offset_time + timedelta(seconds=43),
        ),
    )

    restored = SessionDetections.model_validate_json(session.model_dump_json())

    assert restored == session
    assert restored.frames[0].captured_at.utcoffset() == timedelta(hours=-4)


def test_session_needs_at_least_one_frame() -> None:
    with pytest.raises(ValidationError):
        make_session()


@pytest.mark.parametrize("minutes", [0, -1])
def test_session_frames_must_be_in_strict_capture_order(minutes: int) -> None:
    with pytest.raises(ValidationError, match="is not captured after"):
        make_session(session_frame("a.jpg", 0), session_frame("b.jpg", minutes))


def test_session_rejects_repeated_frames() -> None:
    with pytest.raises(ValidationError, match="appears twice"):
        make_session(session_frame("a.jpg", 0), session_frame("a.jpg", 1))


def test_session_rejects_frames_from_another_session() -> None:
    with pytest.raises(ValidationError, match="is not in session"):
        make_session(make_frame(frame_id="2026-10-13-morning/a.jpg"))


def test_session_rejects_frames_from_another_detector() -> None:
    other = make_frame(frame_id="2026-10-12-evening/a.jpg", detector_version="yoloe-26s")

    with pytest.raises(ValidationError, match="has detector yoloe-26s"):
        make_session(other)


def test_session_rejects_classes_the_detector_does_not_list() -> None:
    plate = Detection(class_name="plate", confidence=0.9, box=make_detection().box)

    with pytest.raises(ValidationError, match="'plate'"):
        make_session(session_frame("a.jpg", 0, plate))


def test_detector_run_needs_a_full_weights_hash() -> None:
    with pytest.raises(ValidationError):
        DETECTOR.model_validate(DETECTOR.model_dump() | {"weights_sha256": "abc123"})
