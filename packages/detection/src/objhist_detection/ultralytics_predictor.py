"""Adapt an Ultralytics detector to the `Predictor` interface used by tiled inference."""

from PIL import Image
from ultralytics import YOLO

from objhist_detection.annotation import PixelBox
from objhist_detection.tiled_inference import Predictor, ScoredBox


def ultralytics_predictor(model: YOLO, device: str) -> Predictor:
    """Wrap `model`; boxes are clipped to the image passed to each call."""

    def predict(
        image: Image.Image, inference_px: int, confidence_threshold: float
    ) -> list[ScoredBox]:
        result = model.predict(
            image, imgsz=inference_px, conf=confidence_threshold, device=device, verbose=False
        )[0]
        boxes = result.boxes
        if boxes is None:
            raise RuntimeError("the detector returned no box tensor")
        width_px, height_px = image.size
        scored: list[ScoredBox] = []
        for class_index, confidence, (x_min, y_min, x_max, y_max) in zip(
            boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True
        ):
            x_min, y_min = max(x_min, 0.0), max(y_min, 0.0)
            x_max, y_max = min(x_max, width_px), min(y_max, height_px)
            if x_max <= x_min or y_max <= y_min:
                continue  # Degenerate after clipping: no visible area to report.
            scored.append(
                ScoredBox(
                    model.names[int(class_index)],
                    float(confidence),
                    PixelBox(x_min, y_min, x_max, y_max),
                )
            )
        return scored

    return predict
