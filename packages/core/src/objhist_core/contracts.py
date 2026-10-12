"""Detector output for captured frames and the session file handed to the tracker.

Coordinates are pixels of the upright image (rotated by its EXIF orientation, as
browsers display it) with the origin at the top-left corner. Capture times are
timezone-aware so frames from different sources can be ordered and subtracted
safely.
"""

from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

PixelCoordinate = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BoundingBox(_Contract):
    """Axis-aligned box in pixel coordinates (x_min, y_min, x_max, y_max)."""

    x_min: PixelCoordinate
    y_min: PixelCoordinate
    x_max: PixelCoordinate
    y_max: PixelCoordinate

    @model_validator(mode="after")
    def _has_positive_area(self) -> Self:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("box must have x_max > x_min and y_max > y_min")
        return self


class Detection(_Contract):
    """One detected object instance. Identity is assigned later by the tracker."""

    class_name: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    box: BoundingBox


class FrameDetections(_Contract):
    """All detections for one counter snapshot.

    The camera is handheld and may move between snapshots, so the same object
    can have different pixel positions in two frames without having moved.

    An empty `detections` tuple means the detector ran and found nothing; it is
    not the same as a frame that was never processed.
    """

    frame_id: str = Field(min_length=1)
    captured_at: AwareDatetime
    image_width_px: int = Field(gt=0)
    image_height_px: int = Field(gt=0)
    detector_version: str = Field(min_length=1)
    detections: tuple[Detection, ...] = ()

    @model_validator(mode="after")
    def _boxes_inside_image(self) -> Self:
        for detection in self.detections:
            box = detection.box
            if box.x_max > self.image_width_px or box.y_max > self.image_height_px:
                raise ValueError(
                    f"{detection.class_name} box {box} exceeds the "
                    f"{self.image_width_px}x{self.image_height_px} image"
                )
        return self


CaptureTimeSource = Literal["manifest", "exif"]
"""Where capture times came from: the session's `manifest.csv`, or each photo's EXIF
original time with its UTC offset. One session uses one source, so its times share a clock."""

InferenceSetting = str | int | float | bool | None


class DetectorRun(_Contract):
    """The detector configuration that produced a session's detections.

    Recorded with every generated history so results trace back to the detector.
    """

    version: str = Field(min_length=1)
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    class_names: tuple[str, ...] = Field(min_length=1)
    confidence_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    settings: dict[str, InferenceSetting]
    """Inference settings (input sizes, tiling), flat so any detector can describe itself."""
    framework_version: str = Field(min_length=1)


class SessionDetections(_Contract):
    """Detector output for every photo of one capture session, in capture order.

    This is the file the detector writes and the tracker reads. Every photo of the
    session is a frame: a photo the detector could not read fails the run instead of
    being left out, so a frame missing from this file is never "no objects found".
    Frame IDs are `<session_id>/<photo filename>`, the photo's path relative to the
    sequences folder.
    """

    format_version: Literal[1] = 1
    session_id: str = Field(min_length=1)
    created_at: AwareDatetime
    capture_time_source: CaptureTimeSource
    detector: DetectorRun
    frames: tuple[FrameDetections, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _frames_are_consistent(self) -> Self:
        known_classes = set(self.detector.class_names)
        seen_ids: set[str] = set()
        for index, frame in enumerate(self.frames):
            if not frame.frame_id.startswith(f"{self.session_id}/"):
                raise ValueError(f"frame {frame.frame_id} is not in session {self.session_id}")
            if frame.frame_id in seen_ids:
                raise ValueError(f"frame {frame.frame_id} appears twice")
            seen_ids.add(frame.frame_id)
            if frame.detector_version != self.detector.version:
                raise ValueError(
                    f"frame {frame.frame_id} has detector {frame.detector_version}, "
                    f"not {self.detector.version}"
                )
            unknown = {d.class_name for d in frame.detections} - known_classes
            if unknown:
                raise ValueError(
                    f"frame {frame.frame_id} has classes {sorted(unknown)} "
                    "that the detector does not list"
                )
            # Placement intervals run from one frame to the next, so they must be positive.
            if index > 0 and frame.captured_at <= self.frames[index - 1].captured_at:
                raise ValueError(
                    f"frame {frame.frame_id} is not captured after "
                    f"{self.frames[index - 1].frame_id}"
                )
        return self
