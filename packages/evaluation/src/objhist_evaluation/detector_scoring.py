"""Count what a detector got right on labeled photos: correct, wrong class, missed, false boxes.

Complements mAP with numbers that map to the history: an object the detector missed or
misnamed is an object the tracker cannot follow. Boxes are compared in coordinates normalized
by the photo's size, so labels and detections need not share a pixel scale; IoU does not
change when both boxes are scaled by the same factors.

Matching rule (`detector-counts-v1`): predictions at or above the confidence threshold are
taken most confident first. Each takes the unmatched labeled object of its own class with the
highest IoU of at least `IOU_MATCH_THRESHOLD` (correct). Failing that, it takes the unmatched
object of any class with the highest such IoU (wrong class). Otherwise it is a false box.
Labeled objects left unmatched are missed. A second box on an already matched object is a
false box.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

MATCHING_RULE_VERSION = "detector-counts-v1"
IOU_MATCH_THRESHOLD = 0.5
"""Minimum IoU for a prediction to count as finding a labeled object, as in mAP50."""


@dataclass(frozen=True)
class NormalizedBox:
    """Box corners as fractions of the photo's width and height."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if not (0 <= self.x_min < self.x_max <= 1 and 0 <= self.y_min < self.y_max <= 1):
            raise ValueError(f"invalid normalized box {self}")

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def iou(self, other: "NormalizedBox") -> float:
        width = min(self.x_max, other.x_max) - max(self.x_min, other.x_min)
        height = min(self.y_max, other.y_max) - max(self.y_min, other.y_min)
        intersection = max(width, 0.0) * max(height, 0.0)
        return intersection / (self.area + other.area - intersection)


@dataclass(frozen=True)
class LabeledObject:
    class_name: str
    box: NormalizedBox


@dataclass(frozen=True)
class ScoredPrediction:
    class_name: str
    confidence: float
    box: NormalizedBox


@dataclass
class DetectionCounts:
    """Counts for one detector over a set of photos.

    `correct`, `wrong_class`, and `missed` are keyed by the labeled object's class, so they add
    up to `objects` per class. `false_boxes` is keyed by the predicted class. `confusions`
    counts (labeled class, predicted class) pairs of wrong-class matches.
    """

    objects: Counter[str] = field(default_factory=Counter)
    correct: Counter[str] = field(default_factory=Counter)
    wrong_class: Counter[str] = field(default_factory=Counter)
    missed: Counter[str] = field(default_factory=Counter)
    false_boxes: Counter[str] = field(default_factory=Counter)
    confusions: Counter[tuple[str, str]] = field(default_factory=Counter)

    def add(self, other: "DetectionCounts") -> None:
        for name in ("objects", "correct", "wrong_class", "missed", "false_boxes", "confusions"):
            getattr(self, name).update(getattr(other, name))

    def totals(self) -> dict[str, int]:
        return {
            "objects": self.objects.total(),
            "correct": self.correct.total(),
            "wrong_class": self.wrong_class.total(),
            "missed": self.missed.total(),
            "false_boxes": self.false_boxes.total(),
        }


def _best_unmatched(
    prediction: ScoredPrediction,
    labels: Sequence[LabeledObject],
    matched: set[int],
    same_class_only: bool,
) -> int | None:
    best_index, best_iou = None, IOU_MATCH_THRESHOLD
    for index, label in enumerate(labels):
        if index in matched or (same_class_only and label.class_name != prediction.class_name):
            continue
        overlap = prediction.box.iou(label.box)
        # >= keeps a prediction at exactly the threshold, as mAP50 does.
        if overlap >= best_iou and (best_index is None or overlap > best_iou):
            best_index, best_iou = index, overlap
    return best_index


def score_photo(
    labels: Sequence[LabeledObject],
    predictions: Iterable[ScoredPrediction],
    min_confidence: float,
) -> DetectionCounts:
    """Counts for one photo under the module's matching rule."""
    counts = DetectionCounts(objects=Counter(label.class_name for label in labels))
    # Ties are broken by position and class so the result does not depend on input order.
    ranked = sorted(
        (p for p in predictions if p.confidence >= min_confidence),
        key=lambda p: (-p.confidence, p.box.x_min, p.box.y_min, p.class_name),
    )
    matched: set[int] = set()
    for prediction in ranked:
        index = _best_unmatched(prediction, labels, matched, same_class_only=True)
        if index is not None:
            matched.add(index)
            counts.correct[prediction.class_name] += 1
            continue
        index = _best_unmatched(prediction, labels, matched, same_class_only=False)
        if index is not None:
            matched.add(index)
            true_class = labels[index].class_name
            counts.wrong_class[true_class] += 1
            counts.confusions[(true_class, prediction.class_name)] += 1
        else:
            counts.false_boxes[prediction.class_name] += 1
    for index, label in enumerate(labels):
        if index not in matched:
            counts.missed[label.class_name] += 1
    return counts
