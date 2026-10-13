"""How the trained detector is run on a photo: tiled (default) or a single pass.

Shared by the `preview` and `detect` commands so both run, and record, the same inference.
"""

import argparse
from dataclasses import dataclass

from PIL import Image

from objhist_core import Detection
from objhist_detection.tiled_inference import (
    Predictor,
    TilingConfig,
    detect_tiled,
    merge_boxes,
    to_detection,
)

DEFAULT_SINGLE_PASS_INFERENCE_PX = 1280


@dataclass(frozen=True)
class InferenceSettings:
    confidence_threshold: float = TilingConfig().confidence_threshold
    tiled: bool = True
    single_pass_inference_px: int = DEFAULT_SINGLE_PASS_INFERENCE_PX
    """Model input size when `tiled` is off; tiled inference uses the `TilingConfig` sizes."""
    merge_duplicates: bool = True
    """Merge overlapping single-pass boxes with the tiled merge rule. YOLO26 does not suppress
    overlaps and put a second box, often of another class, on the same spoon or fork. On the
    validation session `2026-10-12-1624` merging cut false boxes from 8 to 3 with no object
    lost (2026-10-12). Tiled inference always merges."""

    def __post_init__(self) -> None:
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(f"confidence threshold {self.confidence_threshold} is not in [0, 1]")
        if self.single_pass_inference_px <= 0:
            raise ValueError(f"inference size {self.single_pass_inference_px} px is not positive")

    @property
    def tiling(self) -> TilingConfig:
        return TilingConfig(confidence_threshold=self.confidence_threshold)

    @property
    def mode_label(self) -> str:
        """Short name for the inference mode, part of the detector version."""
        if self.tiled:
            return "tiled"
        merged = "-merged" if self.merge_duplicates else ""
        return f"single{self.single_pass_inference_px}{merged}"

    def detect(self, image: Image.Image, predict: Predictor) -> list[Detection]:
        """Detections in the pixels of `image`."""
        if self.tiled:
            return detect_tiled(image, predict, self.tiling)
        scored = predict(image, self.single_pass_inference_px, self.confidence_threshold)
        if self.merge_duplicates:
            scored = merge_boxes(scored, self.tiling)
        return [to_detection(box, image.width, image.height) for box in scored]

    def as_record(self) -> dict[str, str | int | float | bool | None]:
        """Flat settings record, stored with results so they can be traced (see AGENTS.md)."""
        record: dict[str, str | int | float | bool | None] = {
            "mode": "tiled" if self.tiled else "single",
            "confidence_threshold": self.confidence_threshold,
        }
        if self.tiled:
            record.update(self.tiling.as_record())
        else:
            record["single_pass_inference_px"] = self.single_pass_inference_px
            record["merge_duplicates"] = self.merge_duplicates
            if self.merge_duplicates:
                tiling = self.tiling
                record["same_class_iou"] = tiling.same_class_iou
                record["same_class_containment"] = tiling.same_class_containment
                record["cross_class_iou"] = tiling.cross_class_iou
        return record


def add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-tiled", action="store_true", help="single pass over the whole photo")
    parser.add_argument(
        "--inference-px",
        type=int,
        default=DEFAULT_SINGLE_PASS_INFERENCE_PX,
        help="input size for --no-tiled",
    )
    parser.add_argument(
        "--no-merge-duplicates",
        dest="merge_duplicates",
        action="store_false",
        help="with --no-tiled, keep overlapping boxes instead of merging them",
    )
    parser.add_argument("--conf", type=float, default=TilingConfig().confidence_threshold)


def inference_settings_from_args(args: argparse.Namespace) -> InferenceSettings:
    return InferenceSettings(
        confidence_threshold=args.conf,
        tiled=not args.no_tiled,
        single_pass_inference_px=args.inference_px,
        merge_duplicates=args.merge_duplicates,
    )
