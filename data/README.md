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
│   └── kaggle-kitchenware/
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
snapshots from a single fixed camera position, plus a manifest:

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
- The camera does not move within a session. If it moves, start a new session.
- Images within a session have the same resolution.
- Use a short, descriptive session ID such as `2026-10-12-evening`. A session belongs to exactly
  one split (train, validation, or test).
- For scripted evaluation scenarios, also include the list of changes that were made (what,
  where, and when) so ground truth can be annotated; see
  [the evaluation protocol](../docs/evaluation-protocol.md).

If capture times are only in the images' EXIF data, deliver the images and the manifest can be
generated from EXIF, provided the camera clock and time zone were correct.

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

## Training

Train with the dataset's `data.yaml`, for example
`make train DATA=data/processed/kitchen/data.yaml`. Keep the class order in `data.yaml`
identical to the class list in the data card. Ultralytics writes training output to `runs/`,
which is also ignored.
