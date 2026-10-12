"""How the trained detector is run on a photo: tiled (default) or a single pass.

Shared by the `preview` and `detect` commands so both run, and record, the same inference.
"""

import argparse
from dataclasses import dataclass

from PIL import Image

from objhist_core import Detection
from objhist_detection.tiled_inference import Predictor, TilingConfig, detect_tiled, to_detection

DEFAULT_SINGLE_PASS_INFERENCE_PX = 1280


@dataclass(frozen=True)
class InferenceSettings:
    confidence_threshold: float = TilingConfig().confidence_threshold
    tiled: bool = True
    single_pass_inference_px: int = DEFAULT_SINGLE_PASS_INFERENCE_PX
    """Model input size when `tiled` is off; tiled inference uses the `TilingConfig` sizes."""

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
        return "tiled" if self.tiled else f"single{self.single_pass_inference_px}"

    def detect(self, image: Image.Image, predict: Predictor) -> list[Detection]:
        """Detections in the pixels of `image`."""
        if self.tiled:
            return detect_tiled(image, predict, self.tiling)
        return [
            to_detection(scored, image.width, image.height)
            for scored in predict(image, self.single_pass_inference_px, self.confidence_threshold)
        ]

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
        return record


def add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-tiled", action="store_true", help="single pass over the whole photo")
    parser.add_argument(
        "--inference-px",
        type=int,
        default=DEFAULT_SINGLE_PASS_INFERENCE_PX,
        help="input size for --no-tiled",
    )
    parser.add_argument("--conf", type=float, default=TilingConfig().confidence_threshold)


def inference_settings_from_args(args: argparse.Namespace) -> InferenceSettings:
    return InferenceSettings(
        confidence_threshold=args.conf,
        tiled=not args.no_tiled,
        single_pass_inference_px=args.inference_px,
    )
