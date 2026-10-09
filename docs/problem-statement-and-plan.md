# Problem statement and project plan

Source: CS5330 project proposal, "Physical Version Control for Object History Tracking",
Justin Gubbens and Benoy Joseph, Khoury College of Computer Sciences, Northeastern University.
This document restates the proposal for contributors; the
[project checklist](project-checklist.md) tracks what is actually implemented.

## Problem

Shared kitchens become disorganized when several people use the same items and leave them in
different places. Perishable food gets left out, and people waste time searching for items and
guessing how long food has been sitting out.

## Goal: physical version control

Given consecutive images of a kitchen counter taken over time from a fixed camera, produce an
object-level history that records:

- when objects are added and removed,
- where they are positioned and how they move over time, and
- how long each object remains in a single position.

People can then ask what is on the counter, where an item is, and when it was put there, in
natural language, for example "How long has the milk been out?" or "Where is the dish soap?".
The system is intended as a backbone for physical version control of any space.

## Approach

1. **Detection.** An Ultralytics YOLO26 model fine-tuned on labeled kitchen images detects
   objects in each snapshot.
2. **Tracking.** Each detection is matched to the previous frame by position and appearance (DINOv2
   or CLIP image embeddings) so the system recognizes it as the same object.
3. **History.** Added, removed, and moved events are recorded per object. Because snapshots are
   sparse, a change is known to have happened between two frames; placement time is an interval,
   following the interval reasoning in Khronos [3].
4. **Generative AI interface.** A language model answers questions using queries over the
   history, similar to the tool-based querying in VideoAgent [2] and DAAAM [4].
5. **Evaluation.** Scripted scenarios with known changes measure precision and recall under a
   variety of conditions. See the [evaluation protocol](evaluation-protocol.md).

## Related work and how this project differs

| Work | What it does | Gap this project addresses |
| --- | --- | --- |
| Sachdeva and Zisserman [1] | Siamese network boxes objects added or removed between two images; robust to lighting and viewpoint | Two images only; no object classes, identity, or history |
| VideoAgent [2] | Object memory for long videos using detection, ByteTrack, and CLIP/DINOv2 re-identification; an LLM queries it with SQL | Needs continuous video; records appearance, not position, movement, or dwell time |
| Khronos [3] | 3D spatio-temporal map tagging objects as persistent, appearing, or disappearing, with change-time intervals | Mobile RGB-D sensing; cannot associate moved objects; unbounded memory |
| DAAAM [4] | Compact per-object records (description, position, observation timeline) queried by an LLM via tools | Records when objects were seen, not when they were placed |

## Datasets

- Roboflow Kitchen Items: 1,839 images, 9 classes (cup, plate, spoon, fork, knife, dish,
  kettle, pan, rice cooker). <https://universe.roboflow.com/new-bywkn/kitchen-items-u5c2t>
- Kaggle Kitchenware Classification: 6 categories (cups, glasses, plates, spoons, forks,
  knives). <https://www.kaggle.com/competitions/kitchenware-classification/data>

The [data card](data-card.md) records gaps between these datasets and the target scenarios.

## Deliverables for the final report

- A YOLO26 detector fine-tuned on labeled kitchen images.
- A tracking module that maintains object identity across frames and logs when objects are
  added, taken away, or moved.
- A generative AI user interface for accessing information from the logs.
- An evaluation based on scripted scenarios reporting precision and recall under a variety of
  conditions.

## Division of work

- Benoy leads training and optimizing the YOLO26 detector (`packages/detection`) and provides
  the image dataset; no capture hardware is built.
- Justin leads the tracking module and object history data structure (`packages/tracking`).
- The generative AI interface (`packages/assistant`) is a joint responsibility, as is the
  evaluation (`packages/evaluation`).

## References

1. R. Sachdeva and A. Zisserman, "The change you want to see," in *Proc. WACV*, 2023,
   pp. 3993–4002.
2. Y. Fan et al., "VideoAgent: A memory-augmented multimodal agent for video understanding," in
   *Proc. ECCV*, 2024, pp. 75–92.
3. L. Schmid, M. Abate, Y. Chang, and L. Carlone, "Khronos: A unified approach for
   spatio-temporal metric-semantic SLAM in dynamic environments," in *Proc. RSS*, 2024.
4. N. Gorlo, L. Schmid, and L. Carlone, "Describe anything anywhere at any moment," in
   *Proc. CVPR*, 2026, pp. 35002–35013.
