"""Draw detector output on photos to see how the detector does.

    uv run python -m objhist_detection.preview --weights <best.pt> data/raw/sequences/<session>

Writes one annotated JPEG per photo and `detections.json` (weights hash, settings, and every
detection) to `--out-dir`. Tiled inference is on by default; `--no-tiled` runs a single pass.
This is a viewing tool: it does not score against labels.
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from objhist_core import Detection
from objhist_detection.annotation import file_sha256, load_upright_rgb
from objhist_detection.counter_labels.sessions import PHOTO_SUFFIXES
from objhist_detection.inference_settings import (
    add_inference_arguments,
    inference_settings_from_args,
)

BOX_COLOR = (40, 220, 80)


def find_photos(paths: list[Path]) -> list[Path]:
    photos: list[Path] = []
    for path in paths:
        if path.is_dir():
            photos.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES))
        elif path.is_file() and path.suffix.lower() in PHOTO_SUFFIXES:
            photos.append(path)
        else:
            raise FileNotFoundError(f"{path} is not a photo or a folder of photos")
    if not photos:
        raise FileNotFoundError("no photos found")
    return photos


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    line_px = max(3, image.width // 300)
    font = ImageFont.load_default(size=max(14, image.width // 60))
    for detection in detections:
        box = detection.box
        draw.rectangle(
            [box.x_min, box.y_min, box.x_max, box.y_max], outline=BOX_COLOR, width=line_px
        )
        draw.text(
            (box.x_min + line_px, box.y_min + line_px),
            f"{detection.class_name} {detection.confidence:.2f}",
            fill=BOX_COLOR,
            font=font,
            stroke_width=line_px,
            stroke_fill="black",
        )
    return annotated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photos", nargs="+", type=Path, help="photos or folders of photos")
    parser.add_argument(
        "--weights", type=Path, required=True, help="detector weights from your own training run"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("OBJHIST_DATA_DIR", "data")) / "interim" / "previews",
    )
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    add_inference_arguments(parser)
    args = parser.parse_args(argv)

    try:
        settings = inference_settings_from_args(args)
        photos = find_photos(args.photos)
        if not args.weights.is_file():
            raise FileNotFoundError(f"{args.weights} not found")
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    # Every dependency is locked by uv; stop Ultralytics from pip-installing missing ones.
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    # Imported here so `--help` and argument errors do not load PyTorch.
    import ultralytics
    from ultralytics import YOLO

    from objhist_detection.ultralytics_predictor import ultralytics_predictor

    predict = ultralytics_predictor(YOLO(str(args.weights)), args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for photo in photos:
        image, _ = load_upright_rgb(photo)
        detections = settings.detect(image, predict)
        draw_detections(image, detections).save(
            args.out_dir / f"{photo.stem}_detections.jpg", quality=90
        )
        summary = ", ".join(f"{d.class_name} {d.confidence:.2f}" for d in detections) or "nothing"
        print(f"{photo.name}: {summary}")
        results.append(
            {
                "photo": str(photo),
                "width_px": image.width,
                "height_px": image.height,
                "tile_px": (
                    settings.tiling.tile_px_for(image.width, image.height)
                    if settings.tiled
                    else None
                ),
                "detections": [json.loads(d.model_dump_json()) for d in detections],
            }
        )
    record = {
        "created_at": datetime.now(UTC).isoformat(),
        "weights": str(args.weights),
        "weights_sha256": file_sha256(args.weights),
        "settings": settings.as_record(),
        "ultralytics_version": ultralytics.__version__,
        "photos": results,
    }
    (args.out_dir / "detections.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(photos)} annotated photos and detections.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
