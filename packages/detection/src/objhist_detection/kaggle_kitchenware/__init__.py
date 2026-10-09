"""Turn the Kaggle Kitchenware Classification images into a reviewed detection dataset.

The Kaggle set has one centered object per photo and an image-level class, but no boxes.
The pipeline, run with `python -m objhist_detection.kaggle_kitchenware <step>`:

1. `split`: freeze a class-stratified 70/15/15 split from the labels alone, keeping
   byte-identical images together.
2. `propose`: run open-vocabulary YOLOE once per image, keep its highest-confidence box,
   and give it the class from `train.csv`. YOLOE's own class guess is recorded for
   reference but never used as the label or as a review reason.
3. `review-page`: write a local HTML page that queues every validation and test image,
   every flagged training image, and a random spot-check sample of the remaining
   training images for a human verdict.
4. `export`: combine proposals and reviews into a YOLO-format dataset with a manifest
   and a summary of review outcomes.

See `docs/data-card.md` for the dataset facts and `data/README.md` for the commands.
"""
