"""Read the labeled objects of one split of an exported YOLO dataset.

An export (see `objhist_detection.yolo_dataset`) has `data.yaml` with the class names,
`labels/<split>/<export name stem>.txt`, and `manifest.csv`, which maps each source photo
(`key`, the same `<session>/<filename>` used as a frame ID) to its split and export name.
"""

import csv
import math
from pathlib import Path

import yaml

from objhist_evaluation.detector_scoring import LabeledObject, NormalizedBox

MANIFEST_COLUMNS = ("key", "split", "export_name")
ROUNDING_TOLERANCE = 1e-5
"""Label files store 6 decimals, so a box on the photo edge can end slightly outside [0, 1]."""


class LabelError(ValueError):
    """The dataset folder or a label file does not match the export layout."""


def read_class_names(dataset_dir: Path) -> tuple[str, ...]:
    path = dataset_dir / "data.yaml"
    if not path.is_file():
        raise LabelError(f"{path} not found; pass an exported dataset folder")
    names = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("names")
    if isinstance(names, dict) and sorted(names) == list(range(len(names))):
        return tuple(str(names[index]) for index in range(len(names)))
    if isinstance(names, list) and names:
        return tuple(str(name) for name in names)
    raise LabelError(f"{path} has no class names list")


def _clip_unit(value: float, line: str, path: Path) -> float:
    if not -ROUNDING_TOLERANCE <= value <= 1 + ROUNDING_TOLERANCE:
        raise LabelError(f"{path}: box outside the photo in line {line!r}")
    return min(max(value, 0.0), 1.0)


def parse_label_file(path: Path, class_names: tuple[str, ...]) -> list[LabeledObject]:
    """Objects in one YOLO label file: `class center_x center_y width height`, normalized."""
    objects = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        try:
            class_index = int(fields[0])
            center_x, center_y, width, height = (float(value) for value in fields[1:])
        except (ValueError, IndexError) as error:
            raise LabelError(f"{path}: cannot parse line {line!r}") from error
        if not 0 <= class_index < len(class_names):
            raise LabelError(f"{path}: class index {class_index} is not in data.yaml")
        if not all(map(math.isfinite, (center_x, center_y, width, height))):
            raise LabelError(f"{path}: non-finite value in line {line!r}")
        corners = (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        )
        x_min, y_min, x_max, y_max = (_clip_unit(value, line, path) for value in corners)
        try:
            box = NormalizedBox(x_min, y_min, x_max, y_max)
        except ValueError as error:
            raise LabelError(f"{path}: empty box in line {line!r}") from error
        objects.append(LabeledObject(class_names[class_index], box))
    return objects


def read_split_labels(dataset_dir: Path, split: str) -> dict[str, list[LabeledObject]]:
    """Labeled objects of every photo in `split`, keyed by source photo key (frame ID)."""
    class_names = read_class_names(dataset_dir)
    manifest = dataset_dir / "manifest.csv"
    if not manifest.is_file():
        raise LabelError(f"{manifest} not found; the dataset must come from an export")
    labels: dict[str, list[LabeledObject]] = {}
    with manifest.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing_columns = set(MANIFEST_COLUMNS) - set(reader.fieldnames or ())
        if missing_columns:
            raise LabelError(f"{manifest} lacks columns {sorted(missing_columns)}")
        for row in reader:
            if row["split"] != split:
                continue
            label_path = dataset_dir / "labels" / split / f"{Path(row['export_name']).stem}.txt"
            if not label_path.is_file():
                raise LabelError(f"{label_path} not found for {row['key']}")
            labels[row["key"]] = parse_label_file(label_path, class_names)
    if not labels:
        raise LabelError(f"{manifest} lists no photos in split {split!r}")
    return labels
