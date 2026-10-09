from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

from objhist_detection.kaggle_kitchenware.proposals import BoxProposal, PixelBox, ReviewReason

DEFAULT_BOX = PixelBox(20, 40, 80, 160)


@pytest.fixture
def make_proposal() -> Callable[..., BoxProposal]:
    """Build a synthetic proposal; by default an unflagged 100x200 cup with a centered box."""

    def factory(
        image_id: str,
        class_name: str = "cup",
        *,
        box: PixelBox | None = DEFAULT_BOX,
        reasons: tuple[ReviewReason, ...] = (),
        sha256: str | None = None,
        exif_orientation: int = 1,
    ) -> BoxProposal:
        if box is None and ReviewReason.NO_BOX not in reasons:
            reasons = (ReviewReason.NO_BOX, *reasons)
        return BoxProposal(
            image_id=image_id,
            class_name=class_name,
            width_px=100,
            height_px=200,
            exif_orientation=exif_orientation,
            sha256=sha256 or f"hash-{image_id}",
            box=box,
            confidence=None if box is None else 0.9,
            detector_class=None if box is None else class_name,
            second_object_confidence=None,
            review_reasons=reasons,
        )

    return factory


@pytest.fixture
def kaggle_dir(tmp_path: Path) -> Path:
    """A three-image Kaggle-style folder with train.csv and images/."""
    source = tmp_path / "kaggle"
    (source / "images").mkdir(parents=True)
    rows = [("0001", "cup"), ("0002", "fork"), ("0003", "plate")]
    for image_id, _ in rows:
        Image.new("RGB", (100, 200), "white").save(source / "images" / f"{image_id}.jpg")
    lines = ["Id,label", *(f"{image_id},{label}" for image_id, label in rows)]
    (source / "train.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return source
