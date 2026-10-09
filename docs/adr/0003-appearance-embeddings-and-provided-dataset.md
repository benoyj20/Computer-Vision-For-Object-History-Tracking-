# 0003: Appearance embeddings for association and a provided dataset

Date: 2026-10-09. Status: accepted.

## Context

The tracker matches each detection to an existing object. Position alone fails when an object
moves far between snapshots or when similar objects swap places, so each detection also needs an
appearance descriptor. Separately, the project will not build capture hardware.

## Decision

- **Appearance:** the tracker describes each detection crop with a deep image embedding and
  compares embeddings by cosine similarity, combined with position. DINOv2 is the default because
  its features separate individual object instances well; CLIP is the alternative. Both are
  evaluated on the validation scenarios, and the one used is recorded with the tracker version.
  Colour histograms are not used.
- **Data source:** Benoy provides the image dataset as ordered snapshot sequences from a fixed
  camera, in the format described in [data/README.md](../../data/README.md). No camera rig or
  capture software is part of the project. The snapshot interval is whatever the provided
  sequences use and is recorded in the [data card](../data-card.md).

## Consequences

- Association needs a second pretrained model on the GPU. Embeddings run once per detection crop;
  cache them per frame so evaluation reruns do not recompute them.
- The embedding dependency (for example `transformers` for DINOv2 and CLIP) is added to
  `packages/tracking` when that code is written, with weights downloaded from the official
  model repositories only.
- An IoU-only tracker remains the evaluation baseline, so the gain from embeddings is measured.
- Placement intervals depend on the capture times supplied with the dataset; sequences without
  reliable timezone-aware times cannot be used for time-based answers.
