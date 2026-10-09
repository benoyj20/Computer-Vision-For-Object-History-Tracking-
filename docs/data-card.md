# Kitchen object dataset — data card

Status: draft. No dataset release exists yet. Datasets live under `OBJHIST_DATA_DIR`
(default `data/`) and are never committed; this card records where they came from and how they
are split so results can be reproduced.

## Intended use

Fine-tune a YOLO26 detector for objects on a shared kitchen counter seen from a fixed overhead
camera, and build scripted scenarios for evaluating the object history.

## Candidate sources

| Source | Content | Usable for detector training | Open questions |
| --- | --- | --- | --- |
| [Roboflow Kitchen Items](https://universe.roboflow.com/new-bywkn/kitchen-items-u5c2t) | 1,839 images, boxes for 9 classes: cup, plate, spoon, fork, knife, dish, kettle, pan, rice cooker | Yes | License and version to record at download; viewpoint differs from an overhead counter camera |
| [Kaggle Kitchenware Classification](https://www.kaggle.com/competitions/kitchenware-classification/data) | Image-level labels for 6 categories: cups, glasses, plates, spoons, forks, knives | Only after box annotation; see [below](#kaggle-kitchenware-auto-labels) | Competition rules not yet confirmed to allow use outside the competition |
| Provided snapshot sequences | Ordered images from a fixed counter camera, supplied by Benoy in the format in [data/README.md](../data/README.md) | Yes, after box labeling | Required for the target classes and viewpoint; see gaps below |
| COCO-pretrained YOLO26 | Pretrained classes include cup, bottle, bowl, knife, spoon, fork, wine glass, banana, apple, orange | Zero-shot baseline only | Useful to measure what fine-tuning adds |

## Kaggle kitchenware auto-labels

Status: boxes proposed, human review not started, no export yet. Commands are in
[data/README.md](../data/README.md#kaggle-kitchenware-auto-labels); the code is
`packages/detection/src/objhist_detection/kaggle_kitchenware/`.

**License: unresolved, blocks training.** The competition (DataTalks.Club, December 2022 to
February 2023, images collected with Toloka) states its data terms on the Kaggle Rules tab,
which could not be read without a Kaggle login. Older Kaggle competition rules limited data use
to the competition. Before training on or sharing anything derived from these images, a
contributor must read the Rules tab, record the license and any non-competition-use clause
here, and confirm that a course project is allowed.

### Source facts (checked 2026-10-09)

- `train.csv`: 5,559 labeled images; `test.csv`: 3,808 unlabeled images, not used.
  SHA-256 of `train.csv`: `db6d55c5d0d6fc8d1e7ea70fd41672215d795578ceed26dab538c9dcb7957719`.
- Labels per class: plate 1,227, cup 1,135, spoon 989, knife 909, glass 742, fork 557.
- One centered object per photo, mostly close-up. All images are RGB JPEG; 52 distinct sizes,
  most often 750x1000 (3,968 images). Three images carry an EXIF rotation; they are proposed and
  exported upright.
- Three pairs of byte-identical images have conflicting labels (knife/fork, cup/glass,
  spoon/fork). They are kept in the same split and flagged; at most one of each pair can be
  right.

### Method

1. **Split before labeling.** `splits.csv` assigns 70/15/15 per class with seed 5330 from the
   labels and image hashes only, so detector output cannot influence the split. Identical
   images share a split. Result: train 3,891, val 836, test 832.
   SHA-256 of `splits.csv`: `84819968ece3f71f0e61337f24b908785a452ee215a0a8a2db907e5b25954363`.
2. **Propose boxes.** YOLOE-26s-seg (`yoloe-26s-seg.pt`, official Ultralytics release) is
   prompted with the six class names ("drinking glass" for glass). The highest-confidence box
   becomes the object box, and its class always comes from `train.csv`. YOLOE's class for the box
   disagreed with the label on 1,602 images (29%), almost all one-directional (glass read as cup
   680 times, spoon as knife 307, fork as knife 250, plate as cup 187). That is a systematic bias,
   not evidence of mislabeling, so it is recorded but is not a review reason.
3. **Flag for review**, with thresholds fixed before any review:
   no box at confidence 0.05 or above; top box below confidence 0.4; a second box at
   confidence 0.4 or above with less than half its area inside the top box (a second object that
   would be left unlabeled); byte-identical duplicate. Result on 2026-10-09: 1,312 of 5,559
   images flagged (875 low confidence, 355 second object, 76 no box, 6 duplicates; no image had
   more than one reason). No-box images are mostly plates (35) and spoons (23) that YOLOE missed.
   SHA-256 of `proposals.csv`: `27240ced66d697175adf5c9883c39980e195bba50bb7be335922a3dc37187313`; model SHA-256 and versions are in
   `proposals.meta.json`.
4. **Human review** on the generated `review.html`: every validation and test image, every
   flagged training image, and a seeded random spot-check of 300 unflagged training images.
   Reviewers accept, reject (bad box, several objects, or not a listed class), or change the
   class. Queue: 2,872 images (1,668 validation/test, 904 flagged training, 300 spot-check).
5. **Export** to `data/processed/kitchen/` in YOLO format. Images without a box and rejected
   images are dropped; flagged training images are kept only if accepted. The export refuses to
   run while any validation or test image lacks a verdict. `summary.json` records the counts and
   review outcomes; copy them into the table below.

### Review outcomes

Not reviewed yet. Record per category: queued, reviewed, rejected, relabeled, and the resulting
error rates. `rejected` includes images without a box, which can only be rejected; subtract them
when reporting how often a proposed box was wrong. The spot-check error rate estimates label noise in the unreviewed training images.

### Limitations

- Close-up single-object photos differ from the overhead multi-object counter view the project
  targets; this set can only supplement in-domain data, and all evaluation stays on the provided
  sequences.
- Dropping images without a usable box removes hard cases from validation and test, so metrics
  on this set are optimistic.
- Near-duplicate photos (same object, slightly different shot) are not detected; only
  byte-identical images are grouped. Validation metrics may be inflated by such pairs.
- No milk, dish soap, or other scenario classes.

## Known gaps

- The motivating queries mention milk and dish soap, which no candidate dataset labels. The
  final class list must be chosen with the scenarios in mind, and those classes need team
  annotations on the provided sequences.
- Public images are mostly close-up or side views. The provided fixed-camera sequences are
  needed for in-domain training and must be used for all evaluation.
- Consecutive snapshots from one session are nearly identical, which inflates metrics if they
  are split randomly.

## Provided sequences

Record for each delivered batch: number of sessions, images per session, snapshot interval,
camera position, resolution, where capture times came from (EXIF or manifest), and the time
zone. Sessions without reliable capture times cannot support time-based answers.

## Class list

Not decided. The Kaggle export uses `cup, fork, glass, knife, plate, spoon` (indices 0-5,
alphabetical) as a provisional list; new classes should be appended so these indices stay valid. Record the final ordered class list here and keep it identical to the training
`data.yaml`; class indices are part of the shared contract with tracking and evaluation.

## Splits

- Group by capture session: every frame of one session or scripted scenario belongs to exactly
  one split.
- Freeze validation and test splits before any tuning; record their image lists and hashes.
- Scripted evaluation scenarios are held out from detector training entirely.

## Versioning

For each dataset version, record here: source URLs and export versions, download date, class
list, image counts per split, label format, and a SHA-256 of the archive or image list.

## Privacy

Images of a shared kitchen can include people. Avoid capturing faces where possible, keep raw
captures out of the repository and external services, and remove frames whose subjects have
not agreed to be recorded.
