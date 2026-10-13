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
        "merge_duplicates": True,
        "same_class_iou": 0.5,
        "same_class_containment": 0.7,
        "cross_class_iou": 0.6,
    }
    unmerged = InferenceSettings(tiled=False, merge_duplicates=False).as_record()
    assert unmerged["merge_duplicates"] is False
    assert "cross_class_iou" not in unmerged


def test_single_pass_merges_a_second_class_on_the_same_object_unless_asked_not_to() -> None:
    def predict(image: Image.Image, inference_px: int, confidence: float) -> list[ScoredBox]:
        return [
            ScoredBox("fork", 0.6, PixelBox(10, 10, 30, 90)),
            ScoredBox("spoon", 0.8, PixelBox(10, 12, 30, 90)),
            ScoredBox("cup", 0.9, PixelBox(50, 10, 90, 50)),
        ]

    image = Image.new("RGB", (100, 100))
    merged = InferenceSettings(tiled=False).detect(image, predict)
    kept = InferenceSettings(tiled=False, merge_duplicates=False).detect(image, predict)

    assert [d.class_name for d in merged] == ["cup", "spoon"]
    assert len(kept) == 3


def test_mode_label_names_the_merge() -> None:
    assert InferenceSettings(tiled=False).mode_label == "single1280-merged"
    assert InferenceSettings(tiled=False, merge_duplicates=False).mode_label == "single1280"
    assert InferenceSettings().mode_label == "tiled"
