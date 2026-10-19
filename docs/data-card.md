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

## Counter-session auto-labels

Status: dataset `counter-2026-10-12` exported from 100 photos reviewed by Benoy (2026-10-12),
and one YOLO26n model fine-tuned on it; see [below](#dataset-counter-2026-10-12).
Code: `packages/detection/src/objhist_detection/counter_labels/`; commands in
[data/README.md](../data/README.md#counter-session-auto-labels).

**Why it exists.** On Benoy's first two counter photos (`2026-10-12-counter-test`: Galaxy S25
Ultra, 4000x2252 JPEG, EXIF times with UTC offset -04:00, handheld from standing height), the
Kaggle-trained YOLO26n at its default 640 px found only the plate (1 of 6 and 1 of 4 objects at
confidence 0.25 or above, plus a wrong "cup" box on the plate in the first photo). Text-prompted YOLOE at 1280 px boxed every object.

**Method.**
1. YOLOE-26s-seg (same weights and prompts as the Kaggle pipeline) runs at 1280 px with
   class-agnostic NMS. Every box with confidence 0.15 or above becomes a proposal, so one glass
   is not proposed as both cup and glass.
2. A reviewer keeps or rejects each proposal, sets its class, and draws boxes for missed
   objects. A photo is exported only once every proposal has a verdict and it is marked done.
   Since 2026-10-13 a loose or cut-off box can be adjusted (drag handles or type edges); the
   review records it as `adjusted`, and the export's `kept_adjusted` counts it. Earlier
   reviews could only keep a proposed box as is or reject and redraw it.
3. Whole sessions are assigned to splits; validation and test sessions must be fully reviewed.

**First check (2026-10-12, 2 photos, pipeline test only).**

| Photo | Objects (from Benoy's log) | Proposed | Kept, same class | Kept, relabeled | Rejected | Drawn |
| --- | --- | --- | --- | --- | --- | --- |
| 20261012_134754 | 6 | 7 | 5 | 1 (glass proposed as cup) | 1 (stove grate as spoon) | 0 |
| 20261012_134837 | 4 | 8 | 3 | 1 (glass proposed as cup) | 4 (stove grates as spoon) | 0 |

- **Proposals:** every object in Benoy's log got a correct box; no box needed drawing. YOLOE's
  known glass-as-cup bias appeared on both glasses, and its false positives were low-confidence
  "spoons" on the stove grates (0.19-0.33).
- **Who reviewed:** Claude (`claude-opus-5-5 test review`) reviewed these two photos to test the
  pipeline end to end, not a person. Re-review them on the page before they count as training
  data.
- **Export check:** the exported labels were drawn back onto the images and matched the log.
  Ultralytics loaded the export (2 images, 10 boxes) for a one-epoch load check.
- **Too small to measure:** two photos are a smoke test, not a measurement. Detector accuracy
  on counter photos needs held-out sessions.

### Tiled inference with the Kaggle model (2026-10-12, same two photos)

The Kaggle model recognized every object when run on a crop around it (spoon 0.88, glass 0.98,
fork 0.96, plate 0.97, cup 0.95, knife 0.92). The misses on the full photos came from object
size: a knife is 21% x 79% of a Kaggle photo but 3% x 28% of a counter photo, about 19 px wide
at a 640 px input. `objhist_detection.tiled_inference` runs the model on overlapping tiles plus
the full photo and merges the results.

Scored against the reviewed labels above, at confidence 0.25 or above:

| Mode | Photo 1 (6 objects) | Photo 2 (4 objects) |
| --- | --- | --- |
| Single pass, 640 px | 1 correct, 1 false | 1 correct |
| Single pass, 1280 px | 4 correct | 2 correct |
| Tiled (default settings) | 5 correct, 1 wrong class (mug as glass), 1 false | 4 correct, 1 false |

- **False boxes:** both are a "knife" on the wall light switch (0.66 and 0.26).
- **Settings were adjusted after seeing these photos:** even tile spacing and the cross-class
  merge threshold (0.6). Treat these results as a demonstration, not an evaluation; the
  settings must be checked on held-out sessions.
- **Reference labels:** they come from YOLOE boxes reviewed by Claude, which favors box
  placements close to YOLOE's.

**Lower resolutions (2026-10-12).** Tiles were first a fixed 1280 px, which only suited 4000 px
photos: at 1920 px and below, tiling found no more than a single pass. Tiles are now 0.32 of
the photo's long side (1280 px at 4000 px, so results there are unchanged). The same two photos,
downscaled with Lanczos and scored against the scaled reviewed labels, correct of 10 objects:

| Long side (px) | 4000 | 2560 | 1920 | 1280 | 960 | 640 |
| --- | --- | --- | --- | --- | --- | --- |
| Single pass at 1280 px | 6 | 6 | 6 | 5 | 7 | 5 |
| Tiled, fixed 1280 px tiles | 9 | 9 | 6 | 5 | 7 | 5 |
| Tiled, relative tiles | 9 | 10 | 9 | 9 | 9 | 8 |

- **False boxes with relative tiles:** 0–3 per resolution (most at 2560 and 1920 px).
- **Wrong class with relative tiles:** at most 2.
- **Labeling tool:** YOLOE proposals covered all 10 objects at every resolution down to 640 px.
- **Same caveat:** these are downscaled copies of the same two photos, not independent data.

### Dataset `counter-2026-10-12`

`data/processed/counter-2026-10-12/`; class order `cup, fork, glass, knife, plate, spoon`.

- **Photos:** 100 handheld photos of one counter, shot 16:18:50–16:51:45 on 2026-10-12
  (-04:00). The same seven objects are in every photo: white plate, silver fork, silver knife,
  silver spoon, transparent glass, green mug, white design mug. The two mugs are both `cup`.
- **Sessions:** the photos arrived as one folder. They were split into six sessions wherever
  the shooting paused, which was every gap of at least 47 s (the median gap is 15 s). Each
  session folder holds byte-identical copies of its photos (SHA-256 checked) and its own
  `changes.xlsx`. The original folder `2026-10-12-counter-test` was deleted on 2026-10-12, after
  every photo and change-log row was confirmed to have an identical copy in the sessions.
- **Splits** (`session_splits.csv`, frozen before review, SHA-256
  `21c114dabd1274e6e90fce1646bcda5747da6d840f0ec1fca83debbe88afccfd`):

| Split | Sessions | Photos | cup | fork | glass | knife | plate | spoon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1618, 1628, 1635, 1647 | 75 | 150 | 75 | 75 | 75 | 75 | 75 |
| val | 1624 | 9 | 18 | 9 | 9 | 9 | 9 | 9 |
| test | 1643 | 16 | 32 | 16 | 16 | 16 | 16 | 16 |

- **Proposals:** YOLOE matched the known object list in none of the 100 photos. It called the
  glass "cup" in all 100, missed the fork in 48 and the knife in 29, and added 92 extra spoons.
- **Review:** Benoy reviewed all 100 photos on the review page: 475 boxes kept with the same
  class, 140 relabeled, 134 rejected, and 85 drawn. Each photo's labels were then compared with
  the known object list. That flagged 7 photos (a glass left as cup, an extra spoon, a missing
  mug, a fork labeled spoon), and Benoy corrected all 7. After the corrections all 100 photos
  match the list. The check counts labels only; it cannot catch a box that is in the wrong
  place.
- **Hashes:** `manifest.csv` `ec54ffb8f36917fb735a146939eaff00aeed96130ab7c6e2c363703e27dd4dbb`,
  `reviews.json` `fe5277059a01448265370e74272ac296f773b1637ec25df3ab30b85937feb6aa`. Input
  hashes are in `summary.json`.
- **Ultralytics rewrites these JPEGs:** the S25 adds data after the JPEG end marker, so
  Ultralytics treats each image as corrupt and saves a repaired copy the first time it scans
  the folder. It writes a new file, so the hard-linked raw photos are not changed (all 100 raw
  hashes were re-checked). A training list that names the same image more than once makes
  data-loader workers repair the same file at the same time; on Windows this fails and the
  image is skipped. Before training, scan each split once with unique paths.

### Training run `yolo26n-counter-kitchen-1280-v2` (2026-10-12)

- **Data:** `data/processed/counter-plus-kitchen-2026-10-12/data.yaml`. Training uses
  `kaggle-kitchenware-v1` train (3,534 images) plus `counter-2026-10-12` train, with each of
  its 75 photos listed 10 times, 4,284 entries in all. The repetition makes counter photos
  about 18% of each epoch instead of 2%. Validation and test use only `counter-2026-10-12`.
- **Command:** `make train DATA=data/processed/counter-plus-kitchen-2026-10-12/data.yaml
  MODEL=runs/detect/runs/detect/yolo26n-kitchen/weights/best.pt EPOCHS=30 IMGSZ=1280 BATCH=8
  RUN_NAME=yolo26n-counter-kitchen-1280-v2`. It starts from the `yolo26n-kitchen` weights and
  uses seed 0, `deterministic=True`, and the other Ultralytics 8.4.174 defaults.
- **Hardware:** RTX 5070 Laptop GPU 8 GB, torch 2.14.1+cu130; 30 epochs in 0.94 h.
- **Best checkpoint:** epoch 19, picked by Ultralytics fitness on the 9 validation photos.
  `best.pt` SHA-256 `cb28044be38eed795f63b156f9f2a414213b5ecd4de1073aaa3434c3bf30b99a`.
  Validation mAP50 0.923, mAP50-95 0.842. It reached about 0.95 / 0.83 by epoch 4 and then
  stayed between 0.86 and 0.94, which is about as stable as 9 photos allow.
- **Earlier attempt:** `yolo26n-counter-kitchen-1280` was stopped during epoch 1 because the
  repeated paths made Ultralytics skip 76 entries (see the JPEG note above). It is kept, not
  used.

**Test comparison (session `2026-10-12-1643`, 16 photos, 112 objects), scored once after
training.** "Correct" means the right class with IoU ≥ 0.5 at confidence ≥ 0.25. Predictions
are matched greedily, most confident first. A box over an object of another class counts as
wrong class. These counts come from `objhist_evaluation.score_detections` (matching rule
`detector-counts-v1`; commands in [data/README.md](../data/README.md#scoring-detectors-on-a-labeled-split)),
which reproduces the first, one-off script exactly. The YOLOE row comes from its proposals
file, which that command does not read. mAP comes from `yolo detect val ... split=test`.

| Detector | Correct | Wrong class | Missed | False boxes | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| `yolo26n-kitchen`, single pass 640 px | 22 | 3 | 87 | 1 | 0.334 | 0.295 |
| `yolo26n-kitchen`, single pass 1280 px | – | – | – | – | 0.365 | 0.337 |
| `yolo26n-kitchen`, tiled | 48 | 0 | 64 | 7 | – | – |
| YOLOE-26s zero-shot, 1280 px (pre-review proposals) | 67 | 27 | 18 | 7 | – | – |
| `yolo26n-counter-kitchen-1280-v2`, single pass 1280 px | 97 | 6 | 9 | 9 | 0.885 | 0.782 |
| `yolo26n-counter-kitchen-1280-v2`, tiled | 99 | 7 | 6 | 8 | – | – |
| `yolo26n-counter-kitchen-1280-v2`, single pass 1280 px, duplicates merged | 95 | 6 | 11 | 6 | – | – |

Per class, for the new model at 1280 px:

| | cup | fork | glass | knife | plate | spoon |
| --- | --- | --- | --- | --- | --- | --- |
| mAP50-95 | 0.918 | 0.619 | 0.995 | 0.613 | 0.916 | 0.632 |

- **Tiled settings were not checked on held-out data.** On these test photos the old model
  with tiling found 48 of 112 objects, below the 9 of 10 seen on the two photos the tiling
  settings were tuned on. With the new model, tiling adds 2 correct objects and removes 3
  misses, but it also adds 1 wrong class.
- **Cutlery is the weak point:** fork, knife, and spoon score about 0.62 mAP50-95. The other
  classes score at least 0.92. At 1280 px, every one of the new model's mistakes is cutlery:
  spoon read as fork 4 times, fork as spoon once, knife as spoon once, 9 cutlery objects
  missed (fork 3, knife 4, spoon 2), and 9 false cutlery boxes. All 32 cups, 16 glasses, and
  16 plates were found.
- **What this number means:** every session shows the same counter and the same seven objects
  in the same 33 minutes, so the test measures this setup, not other kitchens, lighting, or
  objects. The test labels started as YOLOE boxes, which may favor YOLOE's box placement. This
  is one run with one seed and 16 test photos.

**Validation error analysis (2026-10-12, session `2026-10-12-1624`, 9 photos).** The new model
at 1280 px, single pass, found 58 of 63 objects: 1 wrong class, 4 missed, 8 false boxes, all
cutlery. Claude viewed the 9 photos as downscaled 2000 px copies without metadata, with each box
drawn in a color for its outcome, under the AGENTS.md exception for Benoy's photos. Test photos
were not viewed. Causes:

- **Duplicate boxes on one object (5 of the 8 false boxes):** a second box of another class on
  the same spoon (fork 0.78, fork 0.31, 0.34), or a second fork box inside the fork's box.
  YOLO26 does not suppress overlapping boxes, and the single pass does not merge them; tiled
  inference already merges.
- **Boxes covering part of an object (3 pairs counted as both missed and false):** a knife
  whose black handle blends into the black granite was boxed only along its blade; a spoon
  lying across the fork was boxed together with it; a fork partly behind the glass was boxed
  only where it shows.
- **Partly hidden cutlery:** the fork's label runs through the clear glass, and the knife's
  label stops at the ceramic mug that hides its handle. That is one consistent rule, not two
  (see the label audit below); the model boxed the fork only where it shows and drew the
  knife's box into the mug.
- **Backlit spoon in the glass:** read as knife (0.47), with only its bright handle visible.
- No false boxes on the wall switches or the stove grates.

**Label audit for partly hidden objects (2026-10-12).** Rule, now on the review page and in
[data/README.md](../data/README.md#counter-session-auto-labels): a box encloses the visible part
of its object; parts seen through clear or tinted glass count as visible, and parts hidden
behind or inside something opaque (a ceramic mug, under the plate) are left out. Every photo
where a labeled fork, knife, or spoon box covers more than 10% of a mug or glass box was
checked: 40 of the 100 photos (31 train, 5 val, 4 test). Claude viewed downscaled crops without
metadata showing the labels only, with no detector output, plus 3 full photos where the crop
was too tight. All 40 already follow the rule, so no label was changed and no retraining is
needed for it. Cutlery under the plate is not covered by the 10% filter; the 3 full photos
showed none that broke the rule.

**Merging duplicate single-pass boxes (2026-10-12).** Single-pass output now goes through the
tiled merge rule by default (`merge_duplicates`, thresholds 0.5 same-class IoU, 0.7
containment, 0.6 cross-class IoU, all unchanged). The default was chosen on the validation
session: false boxes fell from 8 to 3 there, with 58 objects correct either way, and merged
single pass matched tiled inference (4 false boxes) at a fraction of the cost. Scored once on
the test session afterwards, it gave 95 correct, 6 wrong class, 11 missed, and 6 false boxes,
against 97 / 6 / 9 / 9 without merging. That is 3 fewer false boxes but 2 lost objects, a knife
and a spoon, most likely cutlery lying close enough to another piece to be merged into it. The
default was not changed on the basis of the test result. Revisit it, and the cross-class
threshold, on a larger validation set. `--no-merge-duplicates` keeps the old behavior.

### Dataset `counter-2026-10-13` (adds two night sessions)

`data/processed/counter-2026-10-13/`; same class order. It is `counter-2026-10-12` plus 45
photos shot at night (23:14–23:30, -04:00) with the same seven objects. They were split at a
45 s pause into `2026-10-12-2314` (20 photos, val) and `2026-10-12-2322` (25 photos, test), at
Benoy's choice. The earlier sessions keep their splits.

| Split | Photos | cup | fork | glass | knife | plate | spoon |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 75 | 150 | 75 | 75 | 75 | 75 | 75 |
| val | 29 | 58 | 29 | 29 | 29 | 29 | 29 |
| test | 41 | 82 | 41 | 41 | 41 | 41 | 41 |

- **Proposals:** YOLOE ran on the CPU while the GPU was training. It matched the object list in
  none of the 45 photos (glass as cup in all 45, 15 forks and 22 knives missing, 45 extra
  spoons).
- **Review:** Benoy reviewed all 45 (recorded once as `Benoy` and once as `benoy`, the same
  person). The label check flagged 9 photos, all of them slips: 5 glasses left as cup, 2 forks
  labeled spoon or knife, a knife standing in a mug left unboxed, and a glass left unboxed.
  Claude viewed the 9 as downscaled copies without metadata, with labels only, and Benoy
  corrected them. All 145 photos now match the list. Review totals over the 145 photos: 658
  kept, 203 relabeled, 230 rejected, 154 drawn.
- **Hashes:** `manifest.csv` `4ae66c5a60b6ab3ce60e1392ce8e5faa94b4be9d4cff04b3a3b748a047794dfd`,
  `reviews.json` `f92109162f7c509e103aae999b84c523aaf7c769428ba6d70175a04130b7e395`,
  `session_splits.csv` `25380b745b5020472c9eecc7265ace24fe6e0d5216cdfa119b3588358a12d33e`.

### Model comparison on the 29 validation photos (2026-10-13)

None of these models trained on the validation photos. Counts use `score_detections`
(`detector-counts-v1`, confidence ≥ 0.25). The cutlery columns count forks, knives, and spoons,
87 objects in all.

| Model | Mode | Correct (of 203) | Cutlery correct (of 87) | False cutlery boxes | Glass read as cup |
| --- | --- | --- | --- | --- | --- |
| `yolo26n-counter-kitchen-1280-v2` | single pass, merged | 184 | 68 | 4 | 0 |
| `yolo26n-counter-kitchen-1280-v2` | tiled | 189 | 73 | 10 | 0 |
| `yolo26s-counter-kitchen-1280-b4` | single pass, merged | 181 | 69 | 7 | 4 |
| `yolo26s-counter-kitchen-1280-b4` | tiled | 186 | 73 | 10 | 3 |

- **`yolo26s-counter-kitchen-1280-b4`:** official `yolo26s.pt` (SHA-256
  `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`) trained on
  `counter-plus-kitchen-2026-10-12` at 1280 px for 30 epochs, batch 4, seed 0; 1.94 h; `best.pt`
  `54746e5dd77c8f90a77aa833558d71d743d2fa87f971d1fd4aa3accd9ec36185`. A batch-8 attempt
  (`yolo26s-counter-kitchen-1280`) needed 10.7 GB, spilled into system memory at 18.7 s per
  step, and was stopped in epoch 1. **Result:** the larger model found no more cutlery than the
  nano and added glass-as-cup errors, so the nano was kept. It started from COCO weights rather
  than Kaggle-trained weights, so model size is not the only difference.
- **Confidence threshold (nano, single pass):** at thresholds from 0.05 to 0.50 the correct
  count ranged from 186 to 182, while false boxes went from 16 down to 4. The threshold stays at
  0.25.
- **`yolo26n-counter-kitchen-1600-e15`** (nano at 1600 px, 15 epochs, batch 4, data
  `counter-plus-kitchen-2026-10-13`): stopped after epoch 8 at Benoy's request. It had stayed
  at about 0.78–0.80 mAP50-95 for four epochs (best 0.798 at epoch 5; `best.pt`
  `09b66d9bb113aa707b3b7803310cda10d1cd5240e9bef962ee0f70f5f9c9f49b`), against 0.820 for the
  1280 px nano on the same 29 photos (`yolo detect val`). Higher resolution was not pursued.

### Final training run `yolo26n-counter-final-1280-e15` (2026-10-13)

- **Data:** `data/processed/counter-final-2026-10-13/data.yaml`. Kaggle train plus every
  counter photo except session `2026-10-12-1643`: train 75, val 29, and the night test session
  `2026-10-12-2322` (25), which was moved into training at Benoy's request. That is 129 photos,
  each listed 10 times, 4,824 entries in all. Because Ultralytics' val (session 1624) is also
  trained on, the epoch count was fixed at 15 in advance and the last epoch's weights are used.
- **Command:** `make train DATA=data/processed/counter-final-2026-10-13/data.yaml
  MODEL=runs/detect/runs/detect/yolo26n-kitchen/weights/best.pt EPOCHS=15 IMGSZ=1280 BATCH=8
  RUN_NAME=yolo26n-counter-final-1280-e15`. It used seed 0 and took 0.57 h on the RTX 5070
  Laptop GPU. `last.pt` (copied to `final.pt`) has SHA-256
  `3d2d29c4662893802f5b4752b10eb201b5f971f4c7cfdb6c4fd28aa7ccb19c89`.
- **Held-out test:** session `2026-10-12-1643`, 16 afternoon photos with 112 objects, 48 of
  them cutlery. No model has trained on it. Scored once.

| Model | Mode | Correct | Missed | False boxes | Cutlery correct (of 48) | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `yolo26n-counter-kitchen-1280-v2` | single pass, merged | 95 | 11 | 6 | 31 | 0.885 | 0.782 |
| `yolo26n-counter-kitchen-1280-v2` | tiled | 99 | 6 | 8 | 35 | – | – |
| `yolo26n-counter-final-1280-e15` | single pass, merged | **98** | **7** | **3** | **34** | **0.908** | **0.796** |
| `yolo26n-counter-final-1280-e15` | tiled | 99 | 7 | 8 | 35 | – | – |

Cutlery mAP50-95: fork 0.618 → 0.585, knife 0.613 → 0.645, spoon 0.632 → 0.705.

- **What changed:** with the 54 extra photos (45 of them night), the single pass found 3 more
  pieces of cutlery and drew half as many false boxes. It now matches the old model's tiled
  cutlery count without tiling's cost or its extra false boxes. Spoons improved most; forks got
  slightly worse (3 forks read as spoon).
- **Caveats:** these are 16 photos and 48 pieces of cutlery, so a 3-object difference is small.
  The test session is from the same afternoon, counter, and objects as the training data. Now
  that both night sessions are in training, no held-out night photos remain to measure night
  performance.

### Training run `yolo26s-counter-final-1280-e15` (2026-10-13)

- **Setup:** same data as the final nano (`counter-final-2026-10-13`, 129 counter photos
  including both night sessions). It starts from the official `yolo26s.pt` and trains at 1280
  px for 15 epochs, batch 4, seed 0, taking 1.10 h. `last.pt` SHA-256
  `f9e10aa9ec4d4e29a13f4e4a432f58cb03133d8f3ed100d4c2a0a1c61d3c7e51`.
- **Epoch count:** Benoy's rule was 15 epochs, more if the score was still rising. On the
  monitored validation photos (also trained on) mAP50-95 went 0.954, 0.953, 0.952 over epochs
  13–15, so it stopped at 15.
- **Held-out test (session 1643, 16 photos, 48 cutlery pieces):**

| Model | Mode | Correct (of 112) | Wrong class | Missed | False | Cutlery correct (of 48) | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `yolo26n-counter-final-1280-e15` | single pass, merged | 98 | 7 | 7 | 3 | 34 (71%) | 0.796 |
| `yolo26s-counter-final-1280-e15` | single pass, merged | **104** | **1** | 7 | 5 | **40 (83%)** | **0.830** |
| `yolo26s-counter-final-1280-e15` | tiled | 105 | 1 | 6 | 7 | 41 (85%) | – |

Cutlery mAP50-95 for YOLO26s: fork 0.658, knife 0.697, spoon 0.778 (nano: 0.585, 0.645,
0.705).

- **What changed:** the fork/spoon mix-ups nearly disappeared, from 7 wrong-class objects to 1.
  On the smaller training set (75 photos) YOLO26s was no better than the nano on validation;
  with 129 photos it is clearly better. The larger model needed the extra data.
- **Caveat:** no held-out validation photos remained, so this test session both measured the
  two models and chose between them. That makes 83% slightly optimistic. Confirm it on a new
  test session before relying on it, together with the tiling choice. 16 photos and one seed.
  Session `2026-10-18-1140` (48 photos from a second kitchen) is that new test session; see
  [Provided sequences](#provided-sequences).

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
zone.

- `2026-10-12-counter-test` (test batch): 2 photos, 43 s apart; handheld phone (Galaxy S25
  Ultra) from standing height, so the framing shifts between photos; 4000x2252 JPEG; capture
  times from EXIF `DateTimeOriginal` with offset -04:00; change log `changes.xlsx`. Sessions without reliable capture times cannot support time-based answers.
  Those two photos were later replaced by the 100 photos below, which were copied into six
  sessions; the folder was then deleted. The pipeline-test proposals and reviews for the two
  photos are kept in `data/interim/counter-labels/`.
- `2026-10-12-1618`, `-1624`, `-1628`, `-1635`, `-1643`, `-1647` (first labeled batch): 100
  photos in 6 sessions of 16, 9, 18, 28, 16, and 13 photos, about 15 s apart (median); same
  phone, handheld; 4000x2252 JPEG; capture times from EXIF `DateTimeOriginal`,
  `SubsecTimeOriginal`, and `OffsetTimeOriginal` (-04:00). Change logs list the same seven
  objects as present in every photo.
- `2026-10-18-1055`, `-1140`, `-1155` (second kitchen, 2026-10-18): 155 photos (IMG_6571–6726,
  IMG_6628 absent) taken by Amrutha in her kitchen with an iPhone, 4032x3024 JPEG (3
  portrait). Benoy delivered them as a Google Drive zip
  (`Amrutha Kitchen Images-20261019T032652Z-1-001.zip`). Amrutha agreed to their use for
  training, label review, and error analysis by coding agents (reported by Benoy,
  2026-10-18). The photos carry no EXIF capture time, camera model, or GPS.
  - **Capture times:** by Benoy's decision, the `-1140` and `-1155` manifests use the zip's
    file times with offset -04:00. These times are approximate: they are whole-second file times,
    not the camera clock. `-1055` has no manifest because 4 of its zip times are out of camera
    order (IMG_6602–6610), and training does not need it. Do not use these sessions to evaluate
    placement times.
  - **Splits** (by Claude, at Benoy's request), cut at a 16-minute pause in the shoot:
    `-1055` = train (84 photos, 10:56–11:24, proposed by `yolo26s-counter-final-1280-e15`);
    `-1140` = test (48 photos, 11:40–11:54) and `-1155` = validation (23 photos,
    11:56–12:02), both proposed by YOLOE.
  - **Domain shift:** before any review, the YOLO26s detector found exactly the seven-object
    list (2 cups, 1 each of fork, glass, knife, plate, spoon) in only 28 of the 155 photos.
    Missed glasses were the most common difference. Claude did not view the photos for this
    count.

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
