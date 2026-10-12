# 0005: Detector output file for the tracker

Date: 2026-10-12. Status: proposed by Benoy; needs Justin's agreement before tracker code
depends on it.

## Context

The tracker (`packages/tracking`) consumes detector output, but only the per-frame
`FrameDetections` contract existed. The tracker also needs each frame's capture time from a
documented source, the frames in capture order, and the detector configuration that it must
record with every history (AGENTS.md). Running the detector inside the tracker would tie
tracking experiments to the GPU and to one detector, and would rerun the detector for every
association change.

## Decision

- The detector writes one JSON file per capture session, validated by
  `objhist_core.SessionDetections`, with `python -m objhist_detection.detect`. Files go to
  `$OBJHIST_DATA_DIR/interim/detections/<detector version>/<session>.json`, so outputs of
  different detectors sit side by side for comparison. The command does not replace an
  existing file without `--force`.
- The file holds `format_version` (1), `session_id`, `created_at`, `capture_time_source`
  (`manifest` or `exif`), a `detector` record (`DetectorRun`: version, weights SHA-256, class
  names, confidence threshold, inference settings, framework version), and `frames`, a list of
  `FrameDetections`.
- Invariants checked on load: at least one frame; frame IDs are unique and have the form
  `<session_id>/<photo filename>`; capture times strictly increase; every frame names the
  file's detector version; every class appears in the detector's class list.
- Every photo of the session is a frame. An empty `detections` list means the detector ran
  and found nothing; a photo that cannot be read stops the run, so a frame is never silently
  missing.
- Capture times come from the session's `manifest.csv` when present, otherwise from each
  photo's EXIF `DateTimeOriginal`, `SubsecTimeOriginal`, and `OffsetTimeOriginal`. A photo
  without a UTC offset, two photos with the same time, or a manifest that does not list every
  photo exactly once stops the run.
- The detector version is `<training run name>-<first 12 hex digits of the weights
  SHA-256>-<tiled | single<px>>`, for example `yolo26n-kitchen-4f404eb4be11-tiled`.

## Consequences

- The tracker depends only on `objhist-core`, not on Ultralytics or a GPU, and its tests can
  use small synthetic `SessionDetections` files.
- Boxes are in pixels of the upright photo. To crop a detection for an appearance embedding,
  load `$OBJHIST_DATA_DIR/raw/sequences/<frame_id>` and apply its EXIF orientation first (as
  `PIL.ImageOps.exif_transpose` does), or the crop will be from the wrong place.
- The camera is handheld and may move between frames (data/README.md), so equal pixel
  positions in two frames do not mean an object stayed put, and different positions do not
  mean it moved.
- Detections below the file's `confidence_threshold` are not in the file. A tracker that
  applies a higher threshold must record its own threshold with the history.
- Changing a field or invariant is a shared-interface change: increment `format_version`,
  and update the producer, the tracker, and their tests together.
