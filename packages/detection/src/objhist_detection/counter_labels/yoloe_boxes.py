"""Propose every object box in counter photos with text-prompted YOLOE."""

from collections.abc import Iterator, Sequence

from ultralytics import YOLOE

from objhist_detection.annotation import CLASS_NAMES, PixelBox, file_sha256, load_upright_rgb
from objhist_detection.counter_labels.proposals import (
    PROPOSAL_CONFIDENCE_FLOOR,
    Candidate,
    PhotoRecord,
    ProposedBox,
    proposals_for_photo,
)
from objhist_detection.counter_labels.sessions import SessionPhoto

INFERENCE_IMAGE_SIZE_PX = 1280
"""Counter photos are wide shots where cutlery is small. At 640 px the first two photos
lost most cutlery; at 1280 px YOLOE boxed every object."""

AGNOSTIC_NMS = True
"""Suppress overlapping boxes across classes, so one glass is not proposed both as a cup
and as a glass. The reviewer picks the class."""


def propose_photo_boxes(
    model: YOLOE, photos: Sequence[SessionPhoto], device: str
) -> Iterator[tuple[PhotoRecord, list[ProposedBox]]]:
    """Yield each photo's record and its proposals, one photo at a time to bound GPU memory."""
    for photo in photos:
        upright, orientation = load_upright_rgb(photo.path)
        result = model.predict(
            upright,
            conf=PROPOSAL_CONFIDENCE_FLOOR,
            imgsz=INFERENCE_IMAGE_SIZE_PX,
            agnostic_nms=AGNOSTIC_NMS,
            device=device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None:
            raise RuntimeError(f"YOLOE returned no box tensor for {photo.path}")
        candidates = [
            Candidate(CLASS_NAMES[int(class_index)], float(confidence), PixelBox(*corners))
            for class_index, confidence, corners in zip(
                boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True
            )
        ]
        record = PhotoRecord(
            key=photo.key,
            session_id=photo.session_id,
            filename=photo.filename,
            width_px=upright.width,
            height_px=upright.height,
            exif_orientation=orientation,
            sha256=file_sha256(photo.path),
        )
        yield record, proposals_for_photo(photo.key, candidates)
