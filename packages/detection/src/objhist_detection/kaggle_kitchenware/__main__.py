"""Command line for the Kaggle kitchenware auto-labeling pipeline.

Run the steps in order (see `data/README.md`):

    uv run python -m objhist_detection.kaggle_kitchenware split --source <kaggle-dir>
    uv run python -m objhist_detection.kaggle_kitchenware propose --source <kaggle-dir>
    uv run python -m objhist_detection.kaggle_kitchenware review-page --source <kaggle-dir>
    uv run python -m objhist_detection.kaggle_kitchenware export --source <kaggle-dir>

Intermediate files go to `$OBJHIST_DATA_DIR/interim/kaggle-kitchenware` and the dataset to
`$OBJHIST_DATA_DIR/processed/kitchen` (default `OBJHIST_DATA_DIR` is `data`).
"""

import argparse
import json
import os
import platform
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from objhist_detection.kaggle_kitchenware import export, proposals, review, splits
from objhist_detection.kaggle_kitchenware.source import (
    LABELS_FILENAME,
    SourceDatasetError,
    file_sha256,
    read_labeled_images,
)

PROPOSALS_FILENAME = "proposals.csv"
PROPOSALS_METADATA_FILENAME = "proposals.meta.json"
SPLITS_FILENAME = "splits.csv"
QUEUE_FILENAME = "review_queue.csv"
REVIEWS_FILENAME = "reviews.csv"
REVIEW_PAGE_FILENAME = "review.html"
PROGRESS_INTERVAL_IMAGES = 500


def _data_dir() -> Path:
    return Path(os.environ.get("OBJHIST_DATA_DIR", "data"))


def _refuse_overwrite(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass --force to replace it")


def run_split(source_dir: Path, work_dir: Path, force: bool) -> None:
    out_path = work_dir / SPLITS_FILENAME
    # Validation and test splits are frozen once written; replacing them is deliberate.
    _refuse_overwrite(out_path, force)
    images = read_labeled_images(source_dir)
    rows = [(image.image_id, image.class_name, file_sha256(image.path)) for image in images]
    assignments = splits.assign_splits(rows)
    splits.write_splits(out_path, assignments)
    for split, counts in splits.split_counts(assignments).items():
        print(f"{split:5} {sum(counts.values()):5}  {dict(sorted(counts.items()))}")
    print(f"wrote {out_path}")


def run_propose(
    source_dir: Path, work_dir: Path, model_path: str, device: str, batch_size: int, force: bool
) -> None:
    # Every dependency is locked by uv; stop Ultralytics from pip-installing missing ones.
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    # Imported here so the other steps run without loading PyTorch and Ultralytics.
    import torch
    import ultralytics

    from objhist_detection.kaggle_kitchenware import yoloe_boxes

    out_path = work_dir / PROPOSALS_FILENAME
    _refuse_overwrite(out_path, force)
    images = read_labeled_images(source_dir)
    model = yoloe_boxes.load_prompted_model(model_path)
    started_at = datetime.now(UTC)
    results: list[proposals.BoxProposal] = []
    for proposal in yoloe_boxes.propose_boxes(model, images, device, batch_size):
        results.append(proposal)
        if len(results) % PROGRESS_INTERVAL_IMAGES == 0:
            print(f"proposed {len(results)}/{len(images)}", flush=True)
    results = proposals.flag_duplicate_images(results)
    proposals.write_proposals(out_path, results)

    reasons = Counter(reason for proposal in results for reason in proposal.review_reasons)
    metadata = {
        "created_at": started_at.isoformat(),
        "source_labels_sha256": file_sha256(source_dir / LABELS_FILENAME),
        "image_count": len(results),
        "model": model_path,
        "model_sha256": file_sha256(Path(model.ckpt_path)),
        "text_prompts": yoloe_boxes.YOLOE_TEXT_PROMPTS,
        "inference_image_size_px": yoloe_boxes.INFERENCE_IMAGE_SIZE_PX,
        "candidate_confidence_floor": yoloe_boxes.CANDIDATE_CONFIDENCE_FLOOR,
        "low_confidence_threshold": proposals.LOW_CONFIDENCE_THRESHOLD,
        "second_object_confidence_threshold": proposals.SECOND_OBJECT_CONFIDENCE_THRESHOLD,
        "same_object_containment": proposals.SAME_OBJECT_CONTAINMENT,
        "review_reason_counts": dict(sorted(reasons.items())),
        "flagged_images": sum(proposal.needs_review for proposal in results),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "python_version": platform.python_version(),
    }
    metadata_path = work_dir / PROPOSALS_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ("review_reason_counts", "flagged_images")}))
    print(f"wrote {out_path} and {metadata_path}")


def _image_paths(source_dir: Path) -> dict[str, Path]:
    return {image.image_id: image.path for image in read_labeled_images(source_dir)}


def _read_reviews_if_present(
    work_dir: Path, queue: dict[str, review.ReviewCategory]
) -> dict[str, review.Review]:
    reviews_path = work_dir / REVIEWS_FILENAME
    return review.read_reviews(reviews_path, queue) if reviews_path.is_file() else {}


def run_review_page(source_dir: Path, work_dir: Path, spot_check_size: int, force: bool) -> None:
    box_proposals = proposals.read_proposals(work_dir / PROPOSALS_FILENAME)
    assignments = splits.read_splits(work_dir / SPLITS_FILENAME)
    queue = review.build_review_queue(box_proposals, assignments, spot_check_size)
    queue_path = work_dir / QUEUE_FILENAME
    queued_categories = {item.image_id: item.category for item in queue}
    if queue_path.is_file() and review.read_queue(queue_path) != queued_categories and not force:
        raise FileExistsError(
            f"{queue_path} differs from the queue built from the current proposals and splits; "
            "pass --force to replace it (existing reviews of dropped images become invalid)"
        )
    review.write_queue(queue_path, queue)
    reviews = _read_reviews_if_present(work_dir, queued_categories)
    page = review.render_review_page(
        queue, box_proposals, assignments, _image_paths(source_dir), reviews
    )
    page_path = work_dir / REVIEW_PAGE_FILENAME
    page_path.write_text(page, encoding="utf-8")
    counts = Counter(str(item.category) for item in queue)
    print(f"queued {len(queue)} images: {dict(counts)}; {len(reviews)} already reviewed")
    print(f"open {page_path.resolve().as_uri()}")


def run_export(source_dir: Path, work_dir: Path, out_dir: Path, overwrite: bool) -> None:
    input_files = {
        name: work_dir / name
        for name in (PROPOSALS_FILENAME, SPLITS_FILENAME, QUEUE_FILENAME, REVIEWS_FILENAME)
    }
    missing = [str(path) for path in input_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {', '.join(missing)}; run the earlier steps first")
    box_proposals = proposals.read_proposals(input_files[PROPOSALS_FILENAME])
    assignments = splits.read_splits(input_files[SPLITS_FILENAME])
    queue = review.read_queue(input_files[QUEUE_FILENAME])
    reviews = review.read_reviews(input_files[REVIEWS_FILENAME], queue)
    plan = export.plan_export(box_proposals, assignments, queue, reviews)
    summary = export.write_export(
        plan, queue, reviews, _image_paths(source_dir), out_dir, input_files, overwrite
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'data.yaml'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", type=Path, required=True, help="Kaggle folder with train.csv and images/"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=_data_dir() / "interim" / "kaggle-kitchenware",
        help="folder for proposals, splits, the review queue, and reviews",
    )
    steps = parser.add_subparsers(dest="step", required=True)

    split_step = steps.add_parser("split", help="freeze the train/val/test split")
    split_step.add_argument("--force", action="store_true", help="replace an existing split")

    propose_step = steps.add_parser("propose", help="propose one box per image with YOLOE")
    propose_step.add_argument("--model", default="yoloe-26s-seg.pt")
    propose_step.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    propose_step.add_argument("--batch-size", type=int, default=16)
    propose_step.add_argument("--force", action="store_true", help="replace existing proposals")

    page_step = steps.add_parser("review-page", help="write the HTML review page")
    page_step.add_argument("--spot-check-size", type=int, default=review.DEFAULT_SPOT_CHECK_SIZE)
    page_step.add_argument("--force", action="store_true", help="replace a changed queue")

    export_step = steps.add_parser("export", help="write the YOLO-format dataset")
    export_step.add_argument("--out-dir", type=Path, default=_data_dir() / "processed" / "kitchen")
    export_step.add_argument(
        "--overwrite", action="store_true", help="replace a previous export in --out-dir"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        match args.step:
            case "split":
                run_split(args.source, args.work_dir, args.force)
            case "propose":
                run_propose(
                    args.source, args.work_dir, args.model, args.device, args.batch_size, args.force
                )
            case "review-page":
                run_review_page(args.source, args.work_dir, args.spot_check_size, args.force)
            case "export":
                run_export(args.source, args.work_dir, args.out_dir, args.overwrite)
    except (SourceDatasetError, export.ExportError, FileExistsError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
