"""Command line for auto-labeling counter-session photos.

Run from the repository root (see `data/README.md`):

    uv run python -m objhist_detection.counter_labels propose
    uv run python -m objhist_detection.counter_labels review-page
    uv run python -m objhist_detection.counter_labels split
    uv run python -m objhist_detection.counter_labels export

Photos are read from `$OBJHIST_DATA_DIR/raw/sequences/<session>/`, intermediate files go to
`$OBJHIST_DATA_DIR/interim/counter-labels`, and the dataset to
`$OBJHIST_DATA_DIR/processed/counter` (default `OBJHIST_DATA_DIR` is `data`).
"""

import argparse
import json
import os
import platform
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from objhist_detection.counter_labels import export, proposals, review, splits
from objhist_detection.counter_labels.sessions import SessionError, find_session_photos
from objhist_detection.yolo_dataset import ExportError

PHOTOS_FILENAME = "photos.csv"
BOXES_FILENAME = "proposals.csv"
PROPOSALS_METADATA_FILENAME = "proposals.meta.json"
SPLITS_FILENAME = "session_splits.csv"
REVIEWS_FILENAME = "reviews.json"
REVIEW_PAGE_FILENAME = "review.html"


def _data_dir() -> Path:
    return Path(os.environ.get("OBJHIST_DATA_DIR", "data"))


def _read_proposals(
    work_dir: Path,
) -> tuple[dict[str, proposals.PhotoRecord], dict[str, list[proposals.ProposedBox]]]:
    photos_path = work_dir / PHOTOS_FILENAME
    if not photos_path.is_file():
        raise FileNotFoundError(f"{photos_path} not found; run the propose step first")
    photos = proposals.read_photos(photos_path)
    return photos, proposals.read_boxes(work_dir / BOXES_FILENAME, photos)


def run_propose(
    sequences_dir: Path,
    work_dir: Path,
    session_ids: list[str] | None,
    model_path: str,
    device: str,
    force: bool,
) -> None:
    # Every dependency is locked by uv; stop Ultralytics from pip-installing missing ones.
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    # Imported here so the other steps run without loading PyTorch and Ultralytics.
    import torch
    import ultralytics

    from objhist_detection import yoloe_prompts
    from objhist_detection.annotation import file_sha256
    from objhist_detection.counter_labels import yoloe_boxes

    session_photos = find_session_photos(sequences_dir, session_ids)
    photos: dict[str, proposals.PhotoRecord] = {}
    boxes: dict[str, list[proposals.ProposedBox]] = {}
    if (work_dir / PHOTOS_FILENAME).is_file():
        photos, boxes = _read_proposals(work_dir)
    pending = []
    for photo in session_photos:
        existing = photos.get(photo.key)
        if existing is None or force:
            pending.append(photo)
        elif existing.sha256 != file_sha256(photo.path):
            raise FileExistsError(
                f"{photo.path} changed since it was proposed; pass --force to propose it again "
                "(its existing review becomes invalid)"
            )
    if not pending:
        print(f"all {len(session_photos)} photos already have proposals")
        return

    model = yoloe_prompts.load_prompted_model(model_path)
    for record, photo_boxes in yoloe_boxes.propose_photo_boxes(model, pending, device):
        photos[record.key] = record
        boxes[record.key] = photo_boxes
        classes = Counter(proposed.class_name for proposed in photo_boxes)
        print(f"{record.key}: {len(photo_boxes)} boxes {dict(sorted(classes.items()))}")
    keys = sorted(photos)
    proposals.write_photos(work_dir / PHOTOS_FILENAME, (photos[key] for key in keys))
    proposals.write_boxes(work_dir / BOXES_FILENAME, (box for key in keys for box in boxes[key]))

    metadata_path = work_dir / PROPOSALS_METADATA_FILENAME
    runs = json.loads(metadata_path.read_text("utf-8"))["runs"] if metadata_path.is_file() else []
    runs.append(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "photos": [photo.key for photo in pending],
            "model": model_path,
            "model_sha256": file_sha256(Path(model.ckpt_path)),
            "text_prompts": yoloe_prompts.YOLOE_TEXT_PROMPTS,
            "inference_image_size_px": yoloe_boxes.INFERENCE_IMAGE_SIZE_PX,
            "agnostic_nms": yoloe_boxes.AGNOSTIC_NMS,
            "proposal_confidence_floor": proposals.PROPOSAL_CONFIDENCE_FLOOR,
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "python_version": platform.python_version(),
        }
    )
    metadata_path.write_text(json.dumps({"runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(f"proposed {len(pending)} photos; wrote {work_dir / BOXES_FILENAME}")


def _reviews_if_present(
    work_dir: Path,
    photos: dict[str, proposals.PhotoRecord],
    boxes: dict[str, list[proposals.ProposedBox]],
) -> dict[str, review.PhotoReview]:
    path = work_dir / REVIEWS_FILENAME
    return review.read_reviews(path, photos, boxes) if path.is_file() else {}


def run_review_page(sequences_dir: Path, work_dir: Path) -> None:
    photos, boxes = _read_proposals(work_dir)
    reviews = _reviews_if_present(work_dir, photos, boxes)
    session_photos = {photo.key: photo for photo in find_session_photos(sequences_dir)}
    missing = sorted(set(photos) - set(session_photos))
    if missing:
        raise FileNotFoundError(
            f"photos with proposals are missing from {sequences_dir}: {missing[:5]}"
        )
    image_paths = {key: session_photos[key].path for key in photos}
    page_path = work_dir / REVIEW_PAGE_FILENAME
    page_path.write_text(
        review.render_review_page(photos, boxes, image_paths, reviews), encoding="utf-8"
    )
    print(
        f"{len(photos)} photos, {sum(map(len, boxes.values()))} proposed boxes, {len(reviews)} already reviewed"
    )
    print(f"open {page_path.resolve().as_uri()}")


def run_split(work_dir: Path, overrides: dict[str, str]) -> None:
    photos, _ = _read_proposals(work_dir)
    photos_per_session = Counter(photo.session_id for photo in photos.values())
    splits_path = work_dir / SPLITS_FILENAME
    existing = splits.read_splits(splits_path)
    assignments = splits.assign_new_sessions(existing, photos_per_session, overrides)
    splits.write_splits(splits_path, assignments)
    for session_id, split in sorted(assignments.items()):
        state = "frozen" if session_id in existing else "new"
        print(
            f"{session_id:30} {split:5} {photos_per_session.get(session_id, 0):4} photos ({state})"
        )
    print(f"wrote {splits_path}")


def run_export(sequences_dir: Path, work_dir: Path, out_dir: Path, overwrite: bool) -> None:
    photos, boxes = _read_proposals(work_dir)
    reviews_path = work_dir / REVIEWS_FILENAME
    if not reviews_path.is_file():
        raise FileNotFoundError(f"{reviews_path} not found; download it from the review page")
    reviews = review.read_reviews(reviews_path, photos, boxes)
    plan = export.plan_export(photos, splits.read_splits(work_dir / SPLITS_FILENAME), reviews)
    session_photos = {photo.key: photo for photo in find_session_photos(sequences_dir)}
    input_files = {
        name: work_dir / name
        for name in (PHOTOS_FILENAME, BOXES_FILENAME, SPLITS_FILENAME, REVIEWS_FILENAME)
    }
    summary = export.write_export(plan, boxes, session_photos, out_dir, input_files, overwrite)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'data.yaml'}")


def _parse_assignment(text: str) -> tuple[str, str]:
    session_id, separator, split = text.partition("=")
    if not separator or not session_id or not split:
        raise argparse.ArgumentTypeError(f"expected SESSION=SPLIT, got {text!r}")
    return session_id, split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sequences-dir",
        type=Path,
        default=_data_dir() / "raw" / "sequences",
        help="folder that holds one folder of photos per session",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=_data_dir() / "interim" / "counter-labels",
        help="folder for proposals, splits, and reviews",
    )
    steps = parser.add_subparsers(dest="step", required=True)

    propose_step = steps.add_parser("propose", help="propose every object box with YOLOE")
    propose_step.add_argument(
        "--session", action="append", dest="sessions", help="only this session (repeatable)"
    )
    propose_step.add_argument("--model", default="yoloe-26s-seg.pt")
    propose_step.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    propose_step.add_argument(
        "--force", action="store_true", help="propose already-proposed photos again"
    )

    steps.add_parser("review-page", help="write the HTML review page")

    split_step = steps.add_parser("split", help="assign new sessions to train, val, or test")
    split_step.add_argument(
        "--assign",
        action="append",
        type=_parse_assignment,
        default=[],
        metavar="SESSION=SPLIT",
        help="put a new session in a chosen split (repeatable)",
    )

    export_step = steps.add_parser("export", help="write the YOLO-format dataset")
    export_step.add_argument("--out-dir", type=Path, default=_data_dir() / "processed" / "counter")
    export_step.add_argument(
        "--overwrite", action="store_true", help="replace a previous export in --out-dir"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        match args.step:
            case "propose":
                run_propose(
                    args.sequences_dir,
                    args.work_dir,
                    args.sessions,
                    args.model,
                    args.device,
                    args.force,
                )
            case "review-page":
                run_review_page(args.sequences_dir, args.work_dir)
            case "split":
                run_split(args.work_dir, dict(args.assign))
            case "export":
                run_export(args.sequences_dir, args.work_dir, args.out_dir, args.overwrite)
    except (
        SessionError,
        ExportError,
        review.ReviewError,
        ValueError,
        FileExistsError,
        FileNotFoundError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
