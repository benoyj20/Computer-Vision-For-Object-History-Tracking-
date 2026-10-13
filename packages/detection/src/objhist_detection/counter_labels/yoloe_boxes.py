"""Propose every object box in counter photos with text-prompted YOLOE."""

from PIL import Image
from ultralytics import YOLOE

from objhist_detection.annotation import CLASS_NAMES, PixelBox
from objhist_detection.counter_labels.proposals import (
    PROPOSAL_CONFIDENCE_FLOOR,
    Candidate,
    CandidateFinder,
)

INFERENCE_IMAGE_SIZE_PX = 1280
"""Counter photos are wide shots where cutlery is small. At 640 px the first two photos
lost most cutlery; at 1280 px YOLOE boxed every object."""

AGNOSTIC_NMS = True
"""Suppress overlapping boxes across classes, so one glass is not proposed both as a cup
and as a glass. The reviewer picks the class."""


def yoloe_candidate_finder(model: YOLOE, device: str) -> CandidateFinder:
    """Candidates from YOLOE prompted with the class list (see `yoloe_prompts`)."""

    def find(image: Image.Image) -> list[Candidate]:
        result = model.predict(
            image,
            conf=PROPOSAL_CONFIDENCE_FLOOR,
            imgsz=INFERENCE_IMAGE_SIZE_PX,
            agnostic_nms=AGNOSTIC_NMS,
            device=device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None:
            raise RuntimeError("YOLOE returned no box tensor")
        return [
            Candidate(CLASS_NAMES[int(class_index)], float(confidence), PixelBox(*corners))
            for class_index, confidence, corners in zip(
                boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True
            )
        ]

    return find
