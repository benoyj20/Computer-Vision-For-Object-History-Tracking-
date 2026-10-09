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
| [Kaggle Kitchenware Classification](https://www.kaggle.com/competitions/kitchenware-classification/data) | Image-level labels for 6 categories: cups, glasses, plates, spoons, forks, knives | Only after box annotation; see [below](#kaggle-kitchenware-auto-labels) | Training confirmed allowed (2026-10-09); no redistribution |
| Provided snapshot sequences | Ordered images from a fixed counter camera, supplied by Benoy in the format in [data/README.md](../data/README.md) | Yes, after box labeling | Required for the target classes and viewpoint; see gaps below |
| COCO-pretrained YOLO26 | Pretrained classes include cup, bottle, bowl, knife, spoon, fork, wine glass, banana, apple, orange | Zero-shot baseline only | Useful to measure what fine-tuning adds |

## Kaggle kitchenware auto-labels

Status: reviewed and exported as dataset version `kaggle-kitchenware-v1` (2026-10-09); one
YOLO26n model trained on it. Commands are in
[data/README.md](../data/README.md#kaggle-kitchenware-auto-labels); the code is
`packages/detection/src/objhist_detection/kaggle_kitchenware/`.

**License: training permitted.** On 2026-10-09 Benoy checked the competition terms
(DataTalks.Club, December 2022 to February 2023, images collected with Toloka) and confirmed
that training on the images for this course project is allowed. The images and anything derived
from them (boxes, exports, weights) stay local and are not redistributed; add the exact license
name from the Rules tab here when it is next opened.

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
4. **Review** (designed for people on the generated `review.html`; this release was reviewed by
   Claude, see below): every validation and test image, every
   flagged training image, and a seeded random spot-check of 300 unflagged training images.
   Reviewers accept, reject (bad box, several objects, or not a listed class), or change the
   class. Queue: 2,872 images (1,668 validation/test, 904 flagged training, 300 spot-check).
5. **Export** to `data/processed/kitchen/` in YOLO format. Images without a box and rejected
   images are dropped; flagged training images are kept only if accepted. The export refuses to
   run while any validation or test image lacks a verdict. `summary.json` records the counts and
   review outcomes; copy them into the table below.

### Review outcomes (2026-10-09)

**Who reviewed:** Claude (`claude-opus-5-5`), not a person. The model read 175 contact sheets of 16 images, each with its proposed box drawn on. I calibrated the rubric on two sheets myself; seven parallel reviewers using that rubric covered the rest. The reviewer column in `reviews.csv` says `claude-opus-5-5 visual review`. Images without a box were rejected automatically. Three duplicate pairs were decided by keeping the correctly labeled copy (0237 knife, 8532 glass, 7196 fork).

**Rubric:**
- Accept a box that encloses the whole object.
- Reject when the box covers only part of the object (a handle, a rim, or a cup handle more than about 15% outside), sits on background, another listed object is clearly visible, or the object is not a listed class (bowls, pans, ladles).
- Relabel only when the label is clearly wrong.

| Queue | Queued | Without a box | Rejected (box shown) | Relabeled | Error rate among images with a box |
| --- | --- | --- | --- | --- | --- |
| Validation and test | 1,668 | 26 | 215 | 10 | 13.7% (225/1,642) |
| Flagged training | 904 | 50 | 285 | 1 | 33.5% (286/854) |
| Training spot-check | 300 | 0 | 22 | 3 | 8.3% (25/300; 95% CI 5.7–12.0%) |

- **Unreviewed training images:** the spot-check estimates that about 8% of the 2,979 unreviewed training images carry a bad box or label. Most are partial boxes (a spoon handle without its bowl) or bowls labeled plate.
- **Audit:** I re-checked a random 32 of the reviewers' verdicts (16 accepts, 16 rejects) and agreed with 31. The miss was a deep bowl accepted as plate, so bowl-versus-plate leniency is the main known reviewer error.
- **Reviewer inconsistency:** reviewers differed on opaque handleless tumblers labeled glass. Some relabeled them cup (12 glass-to-cup relabels in total) and others kept glass. This leaves a little cup/glass label noise.
- **Before reporting test results externally**, a person should re-check the test split.

### Exported dataset `kaggle-kitchenware-v1`

`data/processed/kitchen/`; class order `cup, fork, glass, knife, plate, spoon`.

| Split | cup | fork | glass | knife | plate | spoon | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 737 | 377 | 490 | 604 | 718 | 608 | 3,534 |
| val | 153 | 82 | 99 | 127 | 134 | 126 | 721 |
| test | 154 | 77 | 98 | 125 | 128 | 124 | 706 |

- Excluded: 598 images (76 without a box, 522 rejected).
- `manifest.csv` SHA-256: `421bdcd1b41f7ba19ca661bd3efe3910ef09fe1e81a3bbce6fae0ce418d7d039`.
- `reviews.csv` SHA-256: `85cd382dd43c7ba31b94f36184fbd77ba7cad18e73c17ce3962c5a0195d8013b`.
- Input hashes are in `summary.json`.

### Training run `yolo26n-kitchen` (2026-10-09)

- **Command:** `make train DATA=data/processed/kitchen/data.yaml` (`yolo26n.pt`, 100 epochs, imgsz 640, batch 16, seed 0, `deterministic=True`, other Ultralytics 8.4.174 defaults).
- **Hardware:** RTX 5070 Laptop GPU 8 GB, torch 2.14.1+cu130. Training took 0.93 h.
- **Best checkpoint:** epoch 84; `best.pt` SHA-256 `4f404eb4be11a25e155d8b8849704d253a6c20d366cc5c559ff1ff13ab6f6392`.
  Output is in `runs/detect/runs/detect/yolo26n-kitchen/` (the Makefile has since been fixed to write `runs/detect/<name>/`).
- **Validation (721 images):** mAP50 0.985, mAP50-95 0.962.
- **Test (706 images), evaluated once after training:**
  `yolo detect val model=.../best.pt data=data/processed/kitchen/data.yaml split=test`

| Class | Images | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| all | 706 | 0.964 | 0.967 | 0.984 | 0.961 |
| cup | 154 | 0.936 | 0.957 | 0.981 | 0.969 |
| fork | 77 | 0.958 | 0.974 | 0.977 | 0.938 |
| glass | 98 | 0.928 | 0.922 | 0.964 | 0.952 |
| knife | 125 | 0.986 | 0.976 | 0.994 | 0.953 |
| plate | 128 | 0.988 | 0.992 | 0.995 | 0.987 |
| spoon | 124 | 0.988 | 0.984 | 0.995 | 0.969 |

Inference took 7.6 ms per image at 640 px.

**What this number means:** the test boxes come from the same YOLOE proposals that a reviewer accepted, and hard images were removed. The score therefore measures how well YOLO26n reproduces reviewed YOLOE boxes on single-object close-ups. It is not object-history accuracy and not counter-view performance. It is a single run with a single seed. A zero-shot COCO baseline has not been run yet; COCO has no plate class.

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
