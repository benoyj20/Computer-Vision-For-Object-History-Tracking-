"""Shared pieces for writing YOLO-format detection datasets.

Layout: `images/<split>/`, `labels/<split>/`, `data.yaml`, `manifest.csv`, and `summary.json`.
The manifest marks a folder as an export, so `--overwrite` never deletes anything else.
"""

import os
import shutil
from pathlib import Path

import yaml

from objhist_detection.annotation import CLASS_NAMES, UPRIGHT_EXIF_ORIENTATION, load_upright_rgb

SPLITS: tuple[str, ...] = ("train", "val", "test")
MANIFEST_FILENAME = "manifest.csv"
SUMMARY_FILENAME = "summary.json"
REENCODED_JPEG_QUALITY = 95


class ExportError(ValueError):
    """The reviews do not support a complete, consistent export."""


def prepare_output_dir(out_dir: Path, overwrite: bool) -> None:
    """Create empty split folders, replacing a previous export only when asked."""
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise ExportError(f"{out_dir} is not empty; pass --overwrite to replace the export")
        if not (out_dir / MANIFEST_FILENAME).is_file():
            raise ExportError(
                f"{out_dir} has no {MANIFEST_FILENAME}, so it is not a previous export; "
                "refusing to delete it"
            )
        shutil.rmtree(out_dir)
    for split in SPLITS:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def place_upright_image(source: Path, destination: Path, exif_orientation: int) -> None:
    """Put an image into the export in the upright frame its boxes use."""
    if exif_orientation != UPRIGHT_EXIF_ORIENTATION:
        # Store rotated photos upright so training reads the same frame the boxes use.
        upright, _ = load_upright_rgb(source)
        upright.save(destination, quality=REENCODED_JPEG_QUALITY)
        return
    try:
        os.link(source, destination)
    except OSError:
        # Hard links fail across drives; a copy keeps the same bytes.
        shutil.copy2(source, destination)


def write_data_yaml(out_dir: Path) -> None:
    data_config = {
        "path": str(out_dir.resolve()),
        **{split: f"images/{split}" for split in SPLITS},
        "names": dict(enumerate(CLASS_NAMES)),
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_config, sort_keys=False), "utf-8")
