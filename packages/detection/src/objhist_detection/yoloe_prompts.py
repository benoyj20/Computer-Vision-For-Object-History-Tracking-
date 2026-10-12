"""Text-prompted YOLOE (built into Ultralytics) for proposing boxes to be reviewed.

Weights come from official Ultralytics releases; Ultralytics downloads them to its
`weights_dir` setting on first use. YOLOE's class guesses are review hints only: on the
Kaggle set it read glass as cup and spoons or forks as knife systematically.
"""

from ultralytics import YOLOE

from objhist_detection.annotation import CLASS_NAMES

DEFAULT_YOLOE_MODEL = "yoloe-26s-seg.pt"
YOLOE_TEXT_PROMPTS: dict[str, str] = {
    "cup": "cup",
    "fork": "fork",
    # "glass" alone also matches windows and glass surfaces.
    "glass": "drinking glass",
    "knife": "knife",
    "plate": "plate",
    "spoon": "spoon",
}


def load_prompted_model(model_path: str) -> YOLOE:
    """Load YOLOE prompted with the class list; `model.ckpt_path` is the resolved weights file.

    Class index `i` in the model's output corresponds to `CLASS_NAMES[i]`.
    """
    model = YOLOE(model_path)
    model.set_classes([YOLOE_TEXT_PROMPTS[name] for name in CLASS_NAMES])
    return model
