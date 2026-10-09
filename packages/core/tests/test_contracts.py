import math
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from objhist_core import BoundingBox, Detection, FrameDetections

CAPTURED_AT = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)


def make_detection(**box: float) -> Detection:
    coords = {"x_min": 10, "y_min": 20, "x_max": 110, "y_max": 220} | box
    return Detection(class_name="cup", confidence=0.9, box=BoundingBox(**coords))


def make_frame(*detections: Detection, captured_at: datetime = CAPTURED_AT) -> FrameDetections:
    return FrameDetections(
        frame_id="frame-0001",
        captured_at=captured_at,
        image_width_px=640,
        image_height_px=480,
        detector_version="yolo26n-test",
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
