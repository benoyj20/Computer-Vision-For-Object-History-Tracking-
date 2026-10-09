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
| [Kaggle Kitchenware Classification](https://www.kaggle.com/competitions/kitchenware-classification/data) | Image-level labels for 6 categories: cups, glasses, plates, spoons, forks, knives | No, it has no boxes | Competition data rules may restrict use outside the competition; usable only after re-annotation, if allowed |
| Provided snapshot sequences | Ordered images from a fixed counter camera, supplied by Benoy in the format in [data/README.md](../data/README.md) | Yes, after box labeling | Required for the target classes and viewpoint; see gaps below |
| COCO-pretrained YOLO26 | Pretrained classes include cup, bottle, bowl, knife, spoon, fork, wine glass, banana, apple, orange | Zero-shot baseline only | Useful to measure what fine-tuning adds |

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

Not decided. Record the final ordered class list here and keep it identical to the training
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
