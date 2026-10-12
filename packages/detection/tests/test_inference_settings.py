import pytest
from PIL import Image

from objhist_detection.annotation import PixelBox
from objhist_detection.inference_settings import InferenceSettings
from objhist_detection.tiled_inference import ScoredBox


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="not in"):
        InferenceSettings(confidence_threshold=1.5)
    with pytest.raises(ValueError, match="not positive"):
        InferenceSettings(tiled=False, single_pass_inference_px=0)


def test_single_pass_runs_once_at_its_size_and_clips_to_the_photo() -> None:
    calls: list[tuple[int, float]] = []

    def predict(image: Image.Image, inference_px: int, confidence: float) -> list[ScoredBox]:
        calls.append((inference_px, confidence))
        return [ScoredBox("cup", 0.9, PixelBox(10, 10, 120, 50))]

    settings = InferenceSettings(
        confidence_threshold=0.3, tiled=False, single_pass_inference_px=960
    )
    (cup,) = settings.detect(Image.new("RGB", (100, 60)), predict)

    assert calls == [(960, 0.3)]
    assert cup.box.x_max == 100


def test_records_name_only_the_settings_in_use() -> None:
    tiled = InferenceSettings().as_record()
    single = InferenceSettings(tiled=False).as_record()

    assert tiled["mode"] == "tiled"
    assert tiled["tile_fraction"] == 0.32
    assert "single_pass_inference_px" not in tiled
    assert single == {
        "mode": "single",
        "confidence_threshold": 0.25,
        "single_pass_inference_px": 1280,
    }
