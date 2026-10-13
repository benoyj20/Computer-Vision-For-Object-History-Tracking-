"""Propose counter-photo boxes with a detector trained on this project's labels.

A trained detector proposes far better boxes than YOLOE on counter photos (98 of 112 objects
right on test session `2026-10-12-1643`, against 67), so review becomes mostly confirming. Use
it for training sessions only: `propose` and `split` refuse it for validation and test
sessions, whose labels measure that same detector.
"""

from collections.abc import Mapping

from PIL import Image

from objhist_detection.annotation import CLASS_NAMES, PixelBox
from objhist_detection.counter_labels.proposals import (
    PROPOSAL_CONFIDENCE_FLOOR,
    Candidate,
    CandidateFinder,
)
from objhist_detection.inference_settings import InferenceSettings
from objhist_detection.tiled_inference import Predictor

PROPOSAL_INFERENCE = InferenceSettings(confidence_threshold=PROPOSAL_CONFIDENCE_FLOOR, tiled=False)
"""Single pass at 1280 px with duplicates merged, as the detector runs in use; the lower
proposal floor lets the reviewer see weak boxes too."""


def check_class_names(model_names: Mapping[int, str]) -> None:
    """The detector must use the project's class list in the same order."""
    names = tuple(model_names[index] for index in sorted(model_names))
    if names != CLASS_NAMES:
        raise ValueError(
            f"detector classes {list(names)} differ from the project classes {list(CLASS_NAMES)}"
        )


def trained_detector_candidate_finder(predict: Predictor) -> CandidateFinder:
    def find(image: Image.Image) -> list[Candidate]:
        return [
            Candidate(
                detection.class_name,
                detection.confidence,
                PixelBox(
                    detection.box.x_min,
                    detection.box.y_min,
                    detection.box.x_max,
                    detection.box.y_max,
                ),
            )
            for detection in PROPOSAL_INFERENCE.detect(image, predict)
        ]

    return find
