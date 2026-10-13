"""Score detector output files against the labels of an exported dataset split.

    uv run python -m objhist_evaluation.score_detections \\
        --dataset data/processed/counter-2026-10-12 --split test \\
        data/interim/detections/<detector version A> data/interim/detections/<detector version B>

Each positional argument is a folder of `SessionDetections` files from
`python -m objhist_detection.detect` (one detector configuration per folder). Every labeled
photo of the split must have a frame in each folder: an unprocessed photo is an error, not a
missed object. Prints overall and per-class counts and the classes confused with each other;
`--json-out` also writes them with the detector records and the matching rule.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from objhist_core import DetectorRun, FrameDetections, SessionDetections
from objhist_evaluation.detector_scoring import (
    IOU_MATCH_THRESHOLD,
    MATCHING_RULE_VERSION,
    DetectionCounts,
    LabeledObject,
    NormalizedBox,
    ScoredPrediction,
    score_photo,
)
from objhist_evaluation.yolo_labels import LabelError, read_split_labels


class ScoringError(ValueError):
    """Detection files cannot be scored against the labels as given."""


@dataclass(frozen=True)
class DetectorOutput:
    source: Path
    detector: DetectorRun
    frames: Mapping[str, FrameDetections]


def load_detector_output(source: Path) -> DetectorOutput:
    """Read every `SessionDetections` file in a folder (or one file); one detector only."""
    paths = sorted(source.glob("*.json")) if source.is_dir() else [source]
    if not paths or not paths[0].is_file():
        raise ScoringError(f"no detection files in {source}")
    detector: DetectorRun | None = None
    frames: dict[str, FrameDetections] = {}
    for path in paths:
        try:
            session = SessionDetections.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as error:
            raise ScoringError(f"{path} is not a valid detection file: {error}") from error
        if detector is not None and session.detector != detector:
            raise ScoringError(
                f"{path} comes from detector {session.detector.version}, not {detector.version}; "
                "keep one detector configuration per folder"
            )
        detector = session.detector
        for frame in session.frames:
            if frame.frame_id in frames:
                raise ScoringError(f"{frame.frame_id} appears in more than one file in {source}")
            frames[frame.frame_id] = frame
    if detector is None:
        raise ScoringError(f"no detection files in {source}")
    return DetectorOutput(source, detector, frames)


def normalized_predictions(frame: FrameDetections) -> list[ScoredPrediction]:
    width, height = frame.image_width_px, frame.image_height_px
    return [
        ScoredPrediction(
            detection.class_name,
            detection.confidence,
            NormalizedBox(
                detection.box.x_min / width,
                detection.box.y_min / height,
                detection.box.x_max / width,
                detection.box.y_max / height,
            ),
        )
        for detection in frame.detections
    ]


def score_output(
    output: DetectorOutput,
    labels: Mapping[str, Sequence[LabeledObject]],
    min_confidence: float | None,
) -> tuple[DetectionCounts, float]:
    """Counts over every labeled photo, and the confidence threshold that was applied."""
    recorded = output.detector.confidence_threshold
    threshold = recorded if min_confidence is None else min_confidence
    if threshold < recorded:
        raise ScoringError(
            f"{output.source} only holds detections at confidence {recorded} or above, "
            f"so it cannot be scored at {threshold}; rerun detect with --conf {threshold}"
        )
    missing = sorted(set(labels) - set(output.frames))
    if missing:
        raise ScoringError(
            f"{output.source} has no frame for {len(missing)} labeled photos, "
            f"for example {missing[:3]}; run detect on their sessions"
        )
    counts = DetectionCounts()
    for key, photo_labels in labels.items():
        counts.add(score_photo(photo_labels, normalized_predictions(output.frames[key]), threshold))
    return counts, threshold


def format_report(results: Sequence[tuple[DetectorRun, DetectionCounts, float]]) -> str:
    width = max(len("detector"), *(len(detector.version) for detector, _, _ in results))
    lines = [
        f"{'detector':{width}} {'objects':>7} {'correct':>7} {'wrong':>6} {'missed':>6} {'false':>6}"
    ]
    for detector, counts, _ in results:
        t = counts.totals()
        lines.append(
            f"{detector.version:{width}} {t['objects']:7} {t['correct']:7} {t['wrong_class']:6} "
            f"{t['missed']:6} {t['false_boxes']:6}"
        )
    for detector, counts, threshold in results:
        lines += ["", f"{detector.version} (confidence >= {threshold})"]
        lines.append(
            f"  {'class':8} {'objects':>7} {'correct':>7} {'wrong':>6} {'missed':>6} {'false':>6}"
        )
        for class_name in sorted(set(counts.objects) | set(counts.false_boxes)):
            lines.append(
                f"  {class_name:8} {counts.objects[class_name]:7} {counts.correct[class_name]:7} "
                f"{counts.wrong_class[class_name]:6} {counts.missed[class_name]:6} "
                f"{counts.false_boxes[class_name]:6}"
            )
        if counts.confusions:
            pairs = ", ".join(
                f"{true} as {predicted} x{n}"
                for (true, predicted), n in sorted(counts.confusions.items(), key=lambda kv: -kv[1])
            )
            lines.append(f"  confused: {pairs}")
    return "\n".join(lines)


def results_record(
    dataset_dir: Path,
    split: str,
    labels: Mapping[str, Sequence[LabeledObject]],
    results: Sequence[tuple[DetectorRun, DetectionCounts, float]],
) -> dict[str, object]:
    manifest = dataset_dir / "manifest.csv"
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "matching_rule": MATCHING_RULE_VERSION,
        "iou_match_threshold": IOU_MATCH_THRESHOLD,
        "dataset": str(dataset_dir),
        "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "split": split,
        "photos": len(labels),
        "results": [
            {
                "detector": json.loads(detector.model_dump_json()),
                "min_confidence": threshold,
                "totals": counts.totals(),
                "per_class": {
                    name: {
                        "objects": counts.objects[name],
                        "correct": counts.correct[name],
                        "wrong_class": counts.wrong_class[name],
                        "missed": counts.missed[name],
                        "false_boxes": counts.false_boxes[name],
                    }
                    for name in sorted(set(counts.objects) | set(counts.false_boxes))
                },
                "confusions": [
                    {"labeled": true, "predicted": predicted, "count": n}
                    for (true, predicted), n in sorted(counts.confusions.items())
                ],
            }
            for detector, counts, threshold in results
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("detections", nargs="+", type=Path, help="folders of detection files")
    parser.add_argument("--dataset", type=Path, required=True, help="exported dataset folder")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--min-confidence",
        type=float,
        help="score only detections at or above this (default: each file's own threshold)",
    )
    parser.add_argument("--json-out", type=Path, help="also write the results as JSON here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        labels = read_split_labels(args.dataset, args.split)
        results = []
        for source in args.detections:
            output = load_detector_output(source)
            counts, threshold = score_output(output, labels, args.min_confidence)
            results.append((output.detector, counts, threshold))
    except (LabelError, ScoringError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"{args.split} split of {args.dataset}: {len(labels)} photos, IoU >= {IOU_MATCH_THRESHOLD}"
    )
    print(format_report(results))
    if args.json_out:
        record = results_record(args.dataset, args.split, labels, results)
        args.json_out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
