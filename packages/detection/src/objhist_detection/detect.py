"""Run the trained detector on counter sessions and write the files the tracker reads.

    uv run python -m objhist_detection.detect --weights runs/detect/<run>/weights/best.pt

Writes one `objhist_core.SessionDetections` JSON file per session to
`--out-dir/<detector version>/<session>.json`, with every photo as a frame in capture order.
Capture times come from the session's `manifest.csv` or the photos' EXIF (see
`capture_times`), and are checked for every session before the model is loaded.
"""

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from objhist_core import CaptureTimeSource, DetectorRun, FrameDetections, SessionDetections
from objhist_detection.annotation import file_sha256, load_upright_rgb
from objhist_detection.capture_times import session_capture_times
from objhist_detection.counter_labels.sessions import SessionPhoto, find_session_photos
from objhist_detection.inference_settings import (
    InferenceSettings,
    add_inference_arguments,
    inference_settings_from_args,
)
from objhist_detection.tiled_inference import Predictor

WEIGHTS_DIRNAME = "weights"
"""Ultralytics saves weights as `<run>/weights/best.pt`; the run name identifies them."""


class DetectError(RuntimeError):
    """A photo could not be run through the detector."""


@dataclass(frozen=True)
class PlannedSession:
    """A session's photos in capture order, with their capture times, ready for detection."""

    session_id: str
    capture_time_source: CaptureTimeSource
    photos: tuple[tuple[SessionPhoto, datetime], ...]


def plan_sessions(photos: Sequence[SessionPhoto]) -> list[PlannedSession]:
    """Group photos by session and order each session by capture time."""
    by_session: dict[str, list[SessionPhoto]] = {}
    for photo in photos:
        by_session.setdefault(photo.session_id, []).append(photo)
    planned = []
    for session_id, session_photos in by_session.items():
        source, times = session_capture_times(session_photos[0].path.parent, session_photos)
        ordered = sorted(session_photos, key=lambda photo: times[photo.filename])
        planned.append(
            PlannedSession(
                session_id, source, tuple((photo, times[photo.filename]) for photo in ordered)
            )
        )
    return planned


def detector_version(weights: Path, weights_sha256: str, settings: InferenceSettings) -> str:
    """Run name, weights hash prefix, and inference mode, for example
    `yolo26n-kitchen-3f2a9c0d1e4b-tiled`."""
    parent = weights.resolve().parent
    run_name = parent.parent.name if parent.name == WEIGHTS_DIRNAME else weights.stem
    return f"{run_name}-{weights_sha256[:12]}-{settings.mode_label}"


def detect_session(
    session: PlannedSession,
    predict: Predictor,
    settings: InferenceSettings,
    detector: DetectorRun,
    created_at: datetime,
) -> SessionDetections:
    """Detections for every photo of `session`, in upright-photo pixels."""
    frames = []
    for photo, captured_at in session.photos:
        try:
            image, _ = load_upright_rgb(photo.path)
        except OSError as error:
            # An unreadable photo is not a photo with no objects; stop instead of skipping it.
            raise DetectError(f"cannot read {photo.path}: {error}") from error
        frames.append(
            FrameDetections(
                frame_id=photo.key,
                captured_at=captured_at,
                image_width_px=image.width,
                image_height_px=image.height,
                detector_version=detector.version,
                detections=tuple(settings.detect(image, predict)),
            )
        )
    return SessionDetections(
        session_id=session.session_id,
        created_at=created_at,
        capture_time_source=session.capture_time_source,
        detector=detector,
        frames=tuple(frames),
    )


def write_session(path: Path, session: SessionDetections) -> None:
    """Write through a temporary file so a failed run never leaves a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(session.model_dump_json(indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def build_parser() -> argparse.ArgumentParser:
    data_dir = Path(os.environ.get("OBJHIST_DATA_DIR", "data"))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--weights", type=Path, required=True, help="detector weights from your own training run"
    )
    parser.add_argument(
        "--session",
        action="append",
        dest="sessions",
        metavar="SESSION_ID",
        help="session to run (repeatable); default is every session",
    )
    parser.add_argument("--sequences-dir", type=Path, default=data_dir / "raw" / "sequences")
    parser.add_argument("--out-dir", type=Path, default=data_dir / "interim" / "detections")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--force", action="store_true", help="replace existing detection files")
    add_inference_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = inference_settings_from_args(args)
        if not args.weights.is_file():
            raise FileNotFoundError(f"{args.weights} not found")
        sessions = plan_sessions(find_session_photos(args.sequences_dir, args.sessions))
        weights_sha256 = file_sha256(args.weights)
        version = detector_version(args.weights, weights_sha256, settings)
        out_paths = {
            s.session_id: args.out_dir / version / f"{s.session_id}.json" for s in sessions
        }
        existing = [str(path) for path in out_paths.values() if path.exists()]
        if existing and not args.force:
            raise FileExistsError(
                f"detections already written: {', '.join(existing)}; pass --force to replace them"
            )
    except (ValueError, OSError) as error:
        # SessionError and CaptureTimeError are ValueErrors; FileNotFoundError is an OSError.
        print(f"error: {error}", file=sys.stderr)
        return 1

    # Every dependency is locked by uv; stop Ultralytics from pip-installing missing ones.
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    # Imported here so `--help` and input errors do not load PyTorch.
    import ultralytics
    from ultralytics import YOLO

    from objhist_detection.ultralytics_predictor import ultralytics_predictor

    model = YOLO(str(args.weights))
    detector = DetectorRun(
        version=version,
        weights_sha256=weights_sha256,
        class_names=tuple(model.names[index] for index in sorted(model.names)),
        confidence_threshold=settings.confidence_threshold,
        settings=settings.as_record(),
        framework_version=f"ultralytics {ultralytics.__version__}",
    )
    predict = ultralytics_predictor(model, args.device)
    for session in sessions:
        try:
            result = detect_session(session, predict, settings, detector, datetime.now(UTC))
        except DetectError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        write_session(out_paths[session.session_id], result)
        detection_count = sum(len(frame.detections) for frame in result.frames)
        print(
            f"{session.session_id}: {len(result.frames)} frames, {detection_count} detections "
            f"-> {out_paths[session.session_id]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
