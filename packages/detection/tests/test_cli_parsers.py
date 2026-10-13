from pathlib import Path

import pytest

from objhist_detection import detect
from objhist_detection.counter_labels import __main__ as counter_cli
from objhist_detection.kaggle_kitchenware import __main__ as kaggle_cli


def test_kaggle_cli_parses_every_step() -> None:
    parser = kaggle_cli.build_parser()

    for step in ("split", "propose", "review-page", "export"):
        assert parser.parse_args(["--source", "kaggle", step]).step == step


def test_counter_cli_parses_steps_and_split_assignments() -> None:
    parser = counter_cli.build_parser()

    args = parser.parse_args(["split", "--assign", "2026-10-12-a=test"])

    assert args.assign == [("2026-10-12-a", "test")]
    for step in ("propose", "review-page", "export"):
        assert parser.parse_args([step]).step == step
    assert parser.parse_args(["propose", "--detector", "best.pt"]).detector == Path("best.pt")
    with pytest.raises(SystemExit):
        parser.parse_args(["propose", "--detector", "best.pt", "--model", "yoloe-26s-seg.pt"])
    with pytest.raises(SystemExit):
        parser.parse_args(["split", "--assign", "no-equals-sign"])


def test_detect_cli_takes_repeated_sessions_and_inference_options() -> None:
    parser = detect.build_parser()

    args = parser.parse_args(
        ["--weights", "best.pt", "--session", "a", "--session", "b", "--no-tiled", "--conf", "0.4"]
    )

    assert args.sessions == ["a", "b"]
    assert (args.no_tiled, args.conf, args.inference_px) == (True, 0.4, 1280)
    assert args.merge_duplicates is True
    assert (
        parser.parse_args(["--weights", "best.pt", "--no-merge-duplicates"]).merge_duplicates
        is False
    )
    assert parser.parse_args(["--weights", "best.pt"]).sessions is None
