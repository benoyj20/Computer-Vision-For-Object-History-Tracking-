# Evaluation protocol

Status: draft. Both contributors must agree on the values marked **to confirm** and record the
date below before the first evaluation run. After that, changes need a new protocol version and
a reason; results under different versions are not compared directly.

Protocol version: `eval-v0` (draft, not frozen)

## What is evaluated

The system is evaluated on scripted scenarios: ordered snapshot sequences from the provided
dataset in which a known sequence of changes was performed. Detector quality and history
quality are reported separately, because accurate detection does not by itself produce a
correct history.

## Scenario ground truth

Each scenario records, for every scripted change:

- `event_type`: `added`, `removed`, or `moved`
- `object_id`: the ground-truth identity, stable across the scenario
- `class_name`: from the frozen class list in the [data card](data-card.md)
- `frame_index`: the first snapshot in which the change is visible
- `box`: the object's box after the change (not used for `removed`)
- the wall-clock time the change was actually made, for checking placement intervals

Scenarios are held out from detector training.

## Conditions to cover

- Baseline: good lighting, objects well separated
- Lighting change between snapshots
- Partial occlusion, including a person's arm in the frame
- Clutter and several objects added at once
- Duplicates: two identical items, such as two of the same cup
- Small moves near the movement threshold
- An object removed and a similar one added in the same interval

## Event matching

A predicted event matches a ground-truth event when all hold:

1. Same `event_type` and `class_name`.
2. Same `frame_index` (**to confirm**: allow ±1 frame tolerance or not).
3. For `added` and `moved`: box IoU ≥ 0.5 with the ground-truth box (**to confirm**).

Matching is one-to-one, choosing the highest-IoU pairs first. Unmatched predictions are false
positives; unmatched ground-truth events are false negatives.

A move is an object whose box center shifts by more than a threshold (**to confirm**, in pixels
or as a fraction of the box diagonal) while keeping its identity. The same threshold is used by
the tracker and by ground truth, and is recorded with the results.

## Metrics

- Event precision, recall, and F1 per event type, and micro-averaged over all events.
- Identity switches: ground-truth objects whose predicted identity changes during a scenario.
- Placement-interval coverage: fraction of `added` events whose true placement time lies
  inside the predicted interval, with the median interval width in seconds.
- Detector mAP50 and mAP50-95 on the held-out test split.
- Assistant: fraction of a fixed question set answered correctly from the history, with
  "unknown" scored correct only when the history truly lacks the answer.

## Baseline

Compare against IoU-only association (no appearance features) on the same scenarios and
detector, so the contribution of appearance matching is measured rather than assumed. Report
DINOv2 and CLIP embeddings as separate rows; choose between them on the validation scenarios
only, never on the test scenarios.

## Acceptance thresholds

To be set before the first evaluation run (**to confirm**). Record each threshold and the date
it was agreed. Do not adjust thresholds after seeing results; report failures as measured.

## Reporting

Record, for every result: protocol version, scenario list, detector weights and dataset version,
confidence threshold, tracker version and thresholds, and the command used. Keep failed and
inconclusive runs.
