"""Propose one box per Kaggle photo with text-prompted YOLOE (built into Ultralytics).

YOLOE is prompted with all six class names and its class for the top box is recorded, but
the label always comes from `train.csv`. Weights come from official Ultralytics
releases; Ultralytics downloads them to its `weights_dir` setting on first use.
"""

from collections.abc import Iterator, Sequence
from itertools import batched

from ultralytics import YOLOE

from objhist_detection.kaggle_kitchenware.proposals import (
    BoxCandidate,
    BoxProposal,
    PixelBox,
    review_reasons_for,
    select_top_box,
)
from objhist_detection.kaggle_kitchenware.source import (
    CLASS_NAMES,
    LabeledImage,
    file_sha256,
    load_upright_rgb,
)

DEFAULT_YOLOE_MODEL = "yoloe-26s-seg.pt"
YOLOE_TEXT_PROMPTS: dict[str, str] = {
    "cup": "cup",
    "fork": "fork",
    # "glass" alone also matches windows and glass surfaces.
    "glass": "drinking glass",
    "knife": "knife",
    "plate": "plate",
    "spoon": "spoon",
}
CANDIDATE_CONFIDENCE_FLOOR = 0.05
"""YOLOE boxes below this confidence are discarded before selecting the top box."""
INFERENCE_IMAGE_SIZE_PX = 640


def load_prompted_model(model_path: str) -> YOLOE:
    """Load YOLOE with the class prompts; `model.ckpt_path` is the resolved weights file."""
    model = YOLOE(model_path)
    model.set_classes([YOLOE_TEXT_PROMPTS[name] for name in CLASS_NAMES])
    return model


def propose_boxes(
    model: YOLOE, images: Sequence[LabeledImage], device: str, batch_size: int
) -> Iterator[BoxProposal]:
    """Yield one proposal per image, in input order. Duplicate flags are added later."""
    for batch in batched(images, batch_size):
        loaded = [load_upright_rgb(image.path) for image in batch]
        results = model.predict(
            [upright for upright, _ in loaded],
            conf=CANDIDATE_CONFIDENCE_FLOOR,
            imgsz=INFERENCE_IMAGE_SIZE_PX,
            device=device,
            verbose=False,
        )
        for image, (upright, orientation), result in zip(batch, loaded, results, strict=True):
            boxes = result.boxes
            if boxes is None:
                raise RuntimeError(f"YOLOE returned no box tensor for {image.path}")
            candidates = [
                BoxCandidate(
                    class_name=CLASS_NAMES[int(class_index)],
                    confidence=float(confidence),
                    box=PixelBox(*corners),
                )
                for class_index, confidence, corners in zip(
                    boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True
                )
            ]
            top, second_object_confidence = select_top_box(candidates)
            yield BoxProposal(
                image_id=image.image_id,
                class_name=image.class_name,
                width_px=upright.width,
                height_px=upright.height,
                exif_orientation=orientation,
                sha256=file_sha256(image.path),
                box=None if top is None else top.box,
                confidence=None if top is None else top.confidence,
                detector_class=None if top is None else top.class_name,
                second_object_confidence=second_object_confidence,
                review_reasons=review_reasons_for(top, second_object_confidence),
            )
