"""Turn counter-session photos into a reviewed multi-object detection dataset.

A counter photo shows several objects at once, small in a wide frame, which the Kaggle
close-up model does not handle. The pipeline, run with
`python -m objhist_detection.counter_labels <step>`:

1. `propose`: run text-prompted YOLOE at 1280 px on every new photo and keep every box
   above a confidence floor. Re-running only processes photos added since.
2. `review-page`: write a local HTML page where a reviewer fixes each box's class, rejects
   false boxes, draws missed objects, and marks each photo done.
3. `split`: assign whole sessions to train, val, or test; assignments stay frozen.
4. `export`: write reviewed photos as a YOLO-format dataset with a manifest and summary.

See `data/README.md` for the commands and `docs/data-card.md` for the dataset facts.
"""
