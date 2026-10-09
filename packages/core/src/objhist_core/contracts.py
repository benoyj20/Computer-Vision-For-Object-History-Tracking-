"""Detector output for one captured frame.

Coordinates are image pixels with the origin at the top-left corner. Capture
times are timezone-aware so frames from different sources can be ordered and
subtracted safely.
"""

from typing import Annotated, Self

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
    """All detections for one snapshot from the fixed counter camera.

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
