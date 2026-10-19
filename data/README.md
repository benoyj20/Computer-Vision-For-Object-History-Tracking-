# Local data

Everything in this directory except this file is ignored by Git. Set `OBJHIST_DATA_DIR` in
`.env` to keep data elsewhere. Record sources, versions, and hashes in
[the data card](../docs/data-card.md).

Suggested layout:

```text
data/
├── raw/          # downloads and provided sequences exactly as received
│   ├── roboflow-kitchen-items/
│   └── sequences/<session-id>/
├── interim/      # pipeline state: proposals, frozen splits, review queue and reviews
│   ├── kaggle-kitchenware/
│   └── counter-labels/
├── processed/    # YOLO-format datasets ready for training
│   └── kitchen/
│       ├── data.yaml
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
├── scenarios/    # scripted evaluation sequences and their ground truth
└── histories/    # generated object histories (SQLite)
```

## Provided snapshot sequences

The project does not capture images itself; Benoy provides them. Each session is one folder of
phone snapshots of the counter, plus a manifest:

```text
data/raw/sequences/<session-id>/
├── manifest.csv
├── 000001.jpg
├── 000002.jpg
└── ...
```

`manifest.csv` has one row per image, in capture order:

```csv
filename,captured_at
000001.jpg,2026-10-12T14:05:00-04:00
000002.jpg,2026-10-12T14:06:00-04:00
```

- `captured_at` is ISO 8601 with a UTC offset. Placement intervals and "how long has it been
  out" answers come from these times, so they must be the real capture times, not file
  modification times.
- The camera is handheld or loosely placed and may move between snapshots (Benoy's capture
  method, decided 2026-10-12). Tracking therefore cannot assume an object keeps its pixel
  position when nothing changed; association relies on appearance and on positions relative
  to the counter.
- Photos are JPEG (Android's default; an iPhone needs Camera > Formats > Most Compatible).
- Use a short, descriptive session ID such as `2026-10-12-evening`. A session belongs to exactly
  one split (train, validation, or test).
- For scripted evaluation scenarios, also include the list of changes that were made (what,
  where, and when) so ground truth can be annotated; see
  [the evaluation protocol](../docs/evaluation-protocol.md).

If capture times are only in the images' EXIF data, deliver the images and the manifest can be
generated from EXIF, provided the camera clock and time zone were correct.

The `2026-10-18-*` photos arrived without EXIF times, so their manifests come from the shared
zip's file times (offset -04:00). Benoy checked these against the shoot and confirmed them as the
real capture times (2026-10-18).

## Kaggle kitchenware auto-labels

The [Kaggle Kitchenware Classification](https://www.kaggle.com/competitions/kitchenware-classification/data)
images have class labels but no boxes. `objhist_detection.kaggle_kitchenware` proposes one box
per photo with YOLOE, a person reviews the proposals, and the reviewed set is exported in YOLO
format. Download and unzip the competition data anywhere (it holds `train.csv` and `images/`),
then run the steps in order from the repository root:

```sh
KAGGLE=path/to/kitchenware-classification
uv run python -m objhist_detection.kaggle_kitchenware --source $KAGGLE split
uv run python -m objhist_detection.kaggle_kitchenware --source $KAGGLE propose
uv run python -m objhist_detection.kaggle_kitchenware --source $KAGGLE review-page
# open the printed review.html, review, download reviews.csv into data/interim/kaggle-kitchenware/
uv run python -m objhist_detection.kaggle_kitchenware --source $KAGGLE export
make train DATA=data/processed/kitchen/data.yaml
```

- `split` freezes `splits.csv` (70/15/15 per class, seed 5330). It refuses to replace an
  existing split; `--force` is a deliberate unfreeze that invalidates earlier results.
- `propose` needs the CUDA build and downloads `yoloe-26s-seg.pt` and its text encoder into
  `weights/` on first use. It writes `proposals.csv` and `proposals.meta.json` (model hash,
  prompts, thresholds, versions).
- `review-page` queues every validation and test image, every flagged training image, and a
  300-image random spot-check of the other training images. Re-run it after saving
  `reviews.csv` to resume with those verdicts loaded; progress is also kept in the browser.
- `export` refuses to run until every validation and test image has a verdict. It writes
  `data.yaml`, `images/`, `labels/`, `manifest.csv`, and `summary.json` with review outcomes;
  record those in the [data card](../docs/data-card.md).

## Counter-session auto-labels

Counter photos show several small objects, which the Kaggle close-up model misses.
`objhist_detection.counter_labels` proposes every object box with YOLOE at 1280 px, a reviewer
confirms each box's class, rejects false boxes, and draws missed objects, and reviewed photos
are exported in YOLO format. From the repository root:

```sh
uv run python -m objhist_detection.counter_labels propose
uv run python -m objhist_detection.counter_labels review-page
# open the printed review.html, review every photo, download reviews.json into data/interim/counter-labels/
uv run python -m objhist_detection.counter_labels split
uv run python -m objhist_detection.counter_labels export
```

- `propose` reads every session under `data/raw/sequences/` and only processes photos added
  since the last run (`--session <id>` limits it; `--force` re-proposes). It writes
  `photos.csv`, `proposals.csv`, and `proposals.meta.json` (model hash, settings, versions).
- For **training** sessions, `propose --detector runs/detect/<run>/weights/<weights>.pt`
  proposes boxes with a detector trained on this project's labels instead of YOLOE. It runs a
  single pass at 1280 px with duplicates merged and a 0.15 floor, so review is mostly Keep: on
  test session `2026-10-12-1643` its proposals matched the seven-object list in 10 of 16 photos,
  against none for YOLOE. Validation and test labels must not start from the detector they
  measure, so `propose --detector` refuses sessions already assigned to val or test, and `split`
  refuses to hold out a session whose current proposals came from a trained detector. Assign
  such sessions with `split --assign <session>=train`, or re-propose them with YOLOE
  (`--force`). The detector's class list must match the project's.
- On the review page, every proposed box needs Keep or Reject with the right class, and every
  missed object needs a drawn box. A box encloses the visible part of its object: parts seen
  through clear or tinted glass count as visible, and parts hidden behind or inside something
  opaque (a ceramic mug, under the plate) are left out. Only photos marked done are exported. Re-run `review-page`
  after saving `reviews.json` to continue with those reviews loaded.
- **Adjust** on a box (proposed or drawn) zooms in and shows handles: drag them, or type the edges in
  pixels, to fit the box tightly. Tight boxes matter for thin cutlery and for mAP50-95. An adjusted
  proposal is saved with `adjusted: true`; any other change to a proposed box is rejected when
  `reviews.json` is read, so a slip cannot reach the dataset unnoticed.
- `split` assigns new sessions to train, val, or test by photo count (70/15/15) and never moves
  a session once assigned; `--assign <session>=test` chooses a split for a new session. The
  photos of validation and test sessions should be reviewed by a person, since they measure
  the detector.
- `export` refuses to run while any validation or test photo is unreviewed and writes
  `data/processed/counter/` with `data.yaml`, `manifest.csv`, and `summary.json`.

## Previewing the detector on photos

To see what a trained detector finds, draw its detections on photos:

```sh
uv run python -m objhist_detection.preview --weights runs/detect/<run>/weights/best.pt data/raw/sequences/<session>
```

It writes annotated copies and `detections.json` (weights hash, settings, every detection) to
`data/interim/previews/`. By default it uses tiled inference: one pass over the whole photo
plus evenly spaced, overlapping square tiles, merged into one set of detections, so small
cutlery is not shrunk out of sight. Tiles are 0.32 of the photo's long side (1280 px on a
4000x2252 photo, 614 px on 1920x1080; 12 tiles either way), so any resolution works.
`--no-tiled` runs a single pass at `--inference-px` instead. A single pass merges boxes that
overlap as one object, as tiled inference does, unless `--no-merge-duplicates` is given.
For `yolo26n-counter-kitchen-1280-v2`, which was trained at 1280 px, a merged single pass
matched tiled inference on the validation photos and is much faster.

## Detections for the tracker

To hand detector output to the tracker, run the detector over whole sessions:

```sh
uv run python -m objhist_detection.detect --weights runs/detect/<run>/weights/best.pt
```

It writes one file per session to
`data/interim/detections/<detector version>/<session>.json` (format in
[ADR 0005](../docs/adr/0005-detector-output-file-for-the-tracker.md)): every photo in capture
order with its detections, plus the detector version, weights hash, confidence threshold, and
inference settings. `--session <id>` (repeatable) limits the run, `--no-tiled`, `--inference-px`,
and `--conf` work as in `preview`, and `--force` replaces existing files.

- Capture times come from `manifest.csv` when the session has one, otherwise from each photo's
  EXIF original time, sub-seconds, and UTC offset. Phones that record no offset need a
  manifest. Every session is checked before the model loads.
- The run stops on a photo it cannot read rather than leaving it out; a frame with no
  detections means the detector ran and found nothing.
- Boxes are in pixels of the upright photo (EXIF orientation applied).

## Scoring detectors on a labeled split

To compare detectors on the frozen test split, run `detect` on the split's sessions once per
detector configuration, then score the output folders against the exported labels:

```sh
uv run python -m objhist_detection.detect --weights <best.pt> --session 2026-10-12-1643 --no-tiled
uv run python -m objhist_evaluation.score_detections --dataset data/processed/counter-2026-10-12 \
    --split test data/interim/detections/<detector version> [more folders...] \
    --json-out data/interim/detections/scores-test.json
```

- For each folder it reports labeled objects found with the right class, found with the wrong
  class, missed, and false boxes, per class, plus which classes were confused. An object
  counts as found at IoU 0.5 or more; predictions are matched most confident first. The rule is
  in `objhist_evaluation.detector_scoring`.
- Every labeled photo of the split must have a frame in each folder; a photo that `detect` did
  not process stops the scoring instead of counting as missed.
- `--min-confidence` scores only detections at or above a threshold. It cannot go below the
  threshold `detect` ran with, because weaker detections are not in the file.
- Pick thresholds and settings on `--split val`. Score the test split only for final results.

## Training

Train with the dataset's `data.yaml`, for example
`make train DATA=data/processed/kitchen/data.yaml`. Keep the class order in `data.yaml`
identical to the class list in the data card. Ultralytics writes training output to `runs/`,
which is also ignored.
